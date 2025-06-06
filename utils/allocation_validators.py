"""
Allocation Validators - Business rules and validation for allocations
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
import logging

from utils.db import get_db_engine
from sqlalchemy import text

logger = logging.getLogger(__name__)


class AllocationValidator:
    """Validates allocation plans against business rules"""
    
    @staticmethod
    def validate_allocation_results(allocation_df: pd.DataFrame, 
                                  supply_df: pd.DataFrame) -> List[str]:
        """Validate allocation results against business rules"""
        warnings = []
        
        # 1. Check over-allocation
        over_allocation_warnings = AllocationValidator._check_over_allocation(
            allocation_df, supply_df
        )
        warnings.extend(over_allocation_warnings)
        
        # 2. Check credit limits
        credit_warnings = AllocationValidator._check_credit_limits(allocation_df)
        warnings.extend(credit_warnings)
        
        # 3. Check shelf life for perishables
        shelf_life_warnings = AllocationValidator._check_shelf_life(
            allocation_df, supply_df
        )
        warnings.extend(shelf_life_warnings)
        
        # 4. Check minimum order quantities
        moq_warnings = AllocationValidator._check_minimum_order_quantities(allocation_df)
        warnings.extend(moq_warnings)
        
        # 5. Check customer allocation limits
        customer_limit_warnings = AllocationValidator._check_customer_limits(allocation_df)
        warnings.extend(customer_limit_warnings)
        
        # 6. Check allocation timing
        timing_warnings = AllocationValidator._check_allocation_timing(allocation_df)
        warnings.extend(timing_warnings)
        
        return warnings
    
    @staticmethod
    def _check_over_allocation(allocation_df: pd.DataFrame, 
                              supply_df: pd.DataFrame) -> List[str]:
        """Check if allocation exceeds available supply"""
        warnings = []
        
        # Group allocations by product and entity
        allocation_summary = allocation_df.groupby(['pt_code', 'legal_entity']).agg({
            'allocated_qty': 'sum'
        }).reset_index()
        
        # Group supply by product and entity
        supply_summary = supply_df.groupby(['pt_code', 'legal_entity']).agg({
            'quantity': 'sum'
        }).reset_index()
        
        # Merge to compare
        check_df = allocation_summary.merge(
            supply_summary,
            on=['pt_code', 'legal_entity'],
            how='left',
            suffixes=('_allocated', '_available')
        )
        
        # Find over-allocations
        over_allocated = check_df[
            check_df['allocated_qty'] > check_df['quantity'].fillna(0)
        ]
        
        for _, row in over_allocated.iterrows():
            available = row['quantity'] if pd.notna(row['quantity']) else 0
            warnings.append(
                f"Over-allocation: {row['pt_code']} at {row['legal_entity']} - "
                f"Allocated: {row['allocated_qty']:.0f}, Available: {available:.0f}"
            )
        
        return warnings
    
    @staticmethod
    def _check_credit_limits(allocation_df: pd.DataFrame) -> List[str]:
        """Check if allocations exceed customer credit limits"""
        warnings = []
        
        try:
            engine = get_db_engine()
            
            # Get customer credit limits
            credit_query = text("""
                SELECT 
                    c.id as customer_id,
                    c.name as customer_name,
                    c.credit_limit,
                    COALESCE(
                        (SELECT SUM(total_amount) 
                         FROM sales_orders 
                         WHERE customer_id = c.id 
                           AND payment_status IN ('PENDING', 'PARTIAL')
                           AND status NOT IN ('CANCELLED', 'DELIVERED')
                        ), 0
                    ) as outstanding_amount
                FROM customers c
                WHERE c.credit_limit > 0
                  AND c.credit_limit IS NOT NULL
            """)
            
            credit_df = pd.read_sql(credit_query, engine)
            
            if credit_df.empty:
                return warnings
            
            # Calculate allocation values (assuming unit price in allocation_df)
            if 'unit_price' in allocation_df.columns:
                customer_allocations = allocation_df.groupby('customer').agg({
                    'allocated_qty': 'sum',
                    'unit_price': 'mean'
                }).reset_index()
                
                customer_allocations['allocation_value'] = (
                    customer_allocations['allocated_qty'] * 
                    customer_allocations['unit_price']
                )
                
                # Check against credit limits
                for _, customer in customer_allocations.iterrows():
                    credit_info = credit_df[
                        credit_df['customer_name'] == customer['customer']
                    ]
                    
                    if not credit_info.empty:
                        credit_limit = credit_info.iloc[0]['credit_limit']
                        outstanding = credit_info.iloc[0]['outstanding_amount']
                        new_exposure = outstanding + customer['allocation_value']
                        
                        if new_exposure > credit_limit:
                            warnings.append(
                                f"Credit limit exceeded for {customer['customer']}: "
                                f"Limit: ${credit_limit:,.0f}, "
                                f"Total exposure: ${new_exposure:,.0f}"
                            )
            
        except Exception as e:
            logger.error(f"Error checking credit limits: {str(e)}")
        
        return warnings
    
    @staticmethod
    def _check_shelf_life(allocation_df: pd.DataFrame, 
                         supply_df: pd.DataFrame) -> List[str]:
        """Check shelf life constraints for perishable products"""
        warnings = []
        
        try:
            engine = get_db_engine()
            
            # Get product shelf life info
            shelf_query = text("""
                SELECT 
                    pt_code,
                    product_name,
                    shelf_life_days,
                    is_perishable
                FROM products
                WHERE is_perishable = 1
                  AND shelf_life_days IS NOT NULL
            """)
            
            shelf_df = pd.read_sql(shelf_query, engine)
            
            if shelf_df.empty:
                return warnings
            
            # Check allocations for perishable products
            for _, product in shelf_df.iterrows():
                pt_code = product['pt_code']
                shelf_days = product['shelf_life_days']
                
                # Get allocations for this product
                product_allocations = allocation_df[
                    allocation_df['pt_code'] == pt_code
                ]
                
                if not product_allocations.empty:
                    # Check ETDs against shelf life
                    for _, alloc in product_allocations.iterrows():
                        if pd.notna(alloc['etd']):
                            etd = pd.to_datetime(alloc['etd'])
                            days_to_etd = (etd - datetime.now()).days
                            
                            if days_to_etd > shelf_days * 0.7:  # 70% shelf life rule
                                warnings.append(
                                    f"Shelf life risk for {pt_code}: "
                                    f"ETD in {days_to_etd} days exceeds 70% of "
                                    f"{shelf_days} days shelf life"
                                )
            
            # Check supply expiry dates if available
            if 'expiry_date' in supply_df.columns:
                expired_allocations = []
                
                for _, alloc in allocation_df.iterrows():
                    # Find matching supply
                    matching_supply = supply_df[
                        (supply_df['pt_code'] == alloc['pt_code']) &
                        (supply_df['legal_entity'] == alloc['legal_entity'])
                    ]
                    
                    if not matching_supply.empty:
                        earliest_expiry = matching_supply['expiry_date'].min()
                        if pd.notna(earliest_expiry) and pd.notna(alloc['etd']):
                            if pd.to_datetime(alloc['etd']) > pd.to_datetime(earliest_expiry):
                                expired_allocations.append(alloc['pt_code'])
                
                if expired_allocations:
                    warnings.append(
                        f"Products with expiry before ETD: {', '.join(set(expired_allocations))}"
                    )
            
        except Exception as e:
            logger.error(f"Error checking shelf life: {str(e)}")
        
        return warnings
    
    @staticmethod
    def _check_minimum_order_quantities(allocation_df: pd.DataFrame) -> List[str]:
        """Check if allocations meet minimum order quantities"""
        warnings = []
        
        try:
            engine = get_db_engine()
            
            # Get MOQ settings
            moq_query = text("""
                SELECT 
                    p.pt_code,
                    p.product_name,
                    p.minimum_order_qty,
                    p.order_multiple
                FROM products p
                WHERE p.minimum_order_qty > 0
                   OR p.order_multiple > 1
            """)
            
            moq_df = pd.read_sql(moq_query, engine)
            
            if moq_df.empty:
                return warnings
            
            # Check allocations against MOQ
            for _, product in moq_df.iterrows():
                pt_code = product['pt_code']
                moq = product['minimum_order_qty'] or 0
                multiple = product['order_multiple'] or 1
                
                # Get allocations for this product
                product_allocations = allocation_df[
                    (allocation_df['pt_code'] == pt_code) &
                    (allocation_df['allocated_qty'] > 0)
                ]
                
                for _, alloc in product_allocations.iterrows():
                    qty = alloc['allocated_qty']
                    
                    # Check MOQ
                    if moq > 0 and qty < moq:
                        warnings.append(
                            f"Below MOQ for {pt_code} to {alloc['customer']}: "
                            f"Allocated {qty:.0f}, MOQ is {moq:.0f}"
                        )
                    
                    # Check order multiple
                    if multiple > 1 and qty % multiple != 0:
                        warnings.append(
                            f"Not in order multiple for {pt_code} to {alloc['customer']}: "
                            f"Allocated {qty:.0f}, should be multiple of {multiple}"
                        )
            
        except Exception as e:
            logger.error(f"Error checking MOQ: {str(e)}")
        
        return warnings
    
    @staticmethod
    def _check_customer_limits(allocation_df: pd.DataFrame) -> List[str]:
        """Check customer-specific allocation limits"""
        warnings = []
        
        try:
            engine = get_db_engine()
            
            # Get customer allocation limits (if any)
            limit_query = text("""
                SELECT 
                    c.name as customer_name,
                    cal.product_id,
                    p.pt_code,
                    cal.max_quantity_per_order,
                    cal.max_quantity_per_month
                FROM customer_allocation_limits cal
                JOIN customers c ON cal.customer_id = c.id
                JOIN products p ON cal.product_id = p.id
                WHERE cal.is_active = 1
            """)
            
            # Check if table exists first
            table_check = text("""
                SELECT COUNT(*) as count
                FROM information_schema.tables 
                WHERE table_schema = DATABASE()
                  AND table_name = 'customer_allocation_limits'
            """)
            
            result = engine.execute(table_check).fetchone()
            
            if result['count'] == 0:
                return warnings  # Table doesn't exist
            
            limit_df = pd.read_sql(limit_query, engine)
            
            if limit_df.empty:
                return warnings
            
            # Check allocations against limits
            for _, limit in limit_df.iterrows():
                customer = limit['customer_name']
                pt_code = limit['pt_code']
                
                # Get allocations for this customer-product
                customer_product_alloc = allocation_df[
                    (allocation_df['customer'] == customer) &
                    (allocation_df['pt_code'] == pt_code)
                ]
                
                if not customer_product_alloc.empty:
                    total_allocated = customer_product_alloc['allocated_qty'].sum()
                    
                    # Check per-order limit
                    if pd.notna(limit['max_quantity_per_order']):
                        for _, alloc in customer_product_alloc.iterrows():
                            if alloc['allocated_qty'] > limit['max_quantity_per_order']:
                                warnings.append(
                                    f"Exceeds per-order limit for {customer} - {pt_code}: "
                                    f"Allocated {alloc['allocated_qty']:.0f}, "
                                    f"limit is {limit['max_quantity_per_order']:.0f}"
                                )
                    
                    # Check monthly limit
                    if pd.notna(limit['max_quantity_per_month']):
                        if total_allocated > limit['max_quantity_per_month']:
                            warnings.append(
                                f"Exceeds monthly limit for {customer} - {pt_code}: "
                                f"Total allocated {total_allocated:.0f}, "
                                f"limit is {limit['max_quantity_per_month']:.0f}"
                            )
            
        except Exception as e:
            # Table might not exist, which is OK
            if "doesn't exist" not in str(e):
                logger.error(f"Error checking customer limits: {str(e)}")
        
        return warnings
    
    @staticmethod
    def _check_allocation_timing(allocation_df: pd.DataFrame) -> List[str]:
        """Check allocation timing constraints"""
        warnings = []
        
        # Check for past ETDs
        if 'etd' in allocation_df.columns:
            past_etd = allocation_df[
                (pd.to_datetime(allocation_df['etd']) < datetime.now()) &
                (allocation_df['allocated_qty'] > 0)
            ]
            
            if not past_etd.empty:
                past_count = len(past_etd)
                earliest = past_etd['etd'].min()
                warnings.append(
                    f"Found {past_count} allocations with past ETD "
                    f"(earliest: {earliest})"
                )
        
        # Check for very far future ETDs (potential data error)
        if 'etd' in allocation_df.columns:
            far_future = allocation_df[
                (pd.to_datetime(allocation_df['etd']) > datetime.now() + timedelta(days=365)) &
                (allocation_df['allocated_qty'] > 0)
            ]
            
            if not far_future.empty:
                warnings.append(
                    f"Found {len(far_future)} allocations with ETD more than 1 year in future"
                )
        
        return warnings
    
    @staticmethod
    def validate_before_approval(allocation_id: int) -> Tuple[bool, List[str]]:
        """Comprehensive validation before approval"""
        errors = []
        
        try:
            engine = get_db_engine()
            
            # Get allocation details
            details_query = text("""
                SELECT 
                    ad.*,
                    ap.status as plan_status
                FROM allocation_details ad
                JOIN allocation_plans ap ON ad.allocation_plan_id = ap.id
                WHERE ad.allocation_plan_id = :allocation_id
            """)
            
            details_df = pd.read_sql(details_query, engine, 
                                   params={'allocation_id': allocation_id})
            
            if details_df.empty:
                errors.append("No allocation details found")
                return False, errors
            
            # Check plan status
            if details_df.iloc[0]['plan_status'] != 'DRAFT':
                errors.append(f"Cannot approve - status is {details_df.iloc[0]['plan_status']}")
                return False, errors
            
            # Check if all allocations are valid
            zero_allocations = details_df[details_df['allocated_qty'] <= 0]
            if len(zero_allocations) == len(details_df):
                errors.append("All allocation quantities are zero")
                return False, errors
            
            # Check for duplicate allocations
            duplicates = details_df[
                details_df.duplicated(
                    subset=['product_id', 'customer_id', 'etd'], 
                    keep=False
                )
            ]
            
            if not duplicates.empty:
                errors.append(
                    f"Found {len(duplicates)} duplicate allocation entries"
                )
            
            # Additional business rule checks can be added here
            
            return len(errors) == 0, errors
            
        except Exception as e:
            logger.error(f"Error validating allocation: {str(e)}")
            errors.append(f"Validation error: {str(e)}")
            return False, errors
    
    @staticmethod
    def get_allocation_constraints(product_codes: List[str] = None) -> Dict:
        """Get all applicable constraints for products"""
        constraints = {
            'credit_limits': {},
            'moq': {},
            'shelf_life': {},
            'customer_limits': {}
        }
        
        try:
            engine = get_db_engine()
            
            # Get all relevant constraints from database
            # This can be used to pre-load constraints for the allocation process
            
            # Credit limits
            credit_query = text("""
                SELECT name, credit_limit 
                FROM customers 
                WHERE credit_limit > 0
            """)
            credit_df = pd.read_sql(credit_query, engine)
            constraints['credit_limits'] = dict(zip(
                credit_df['name'], 
                credit_df['credit_limit']
            ))
            
            # MOQ and multiples
            product_filter = ""
            params = {}
            if product_codes:
                product_filter = "WHERE pt_code IN :product_codes"
                params['product_codes'] = tuple(product_codes)
            
            moq_query = text(f"""
                SELECT pt_code, minimum_order_qty, order_multiple
                FROM products
                {product_filter}
            """)
            moq_df = pd.read_sql(moq_query, engine, params=params)
            
            for _, row in moq_df.iterrows():
                constraints['moq'][row['pt_code']] = {
                    'min_qty': row['minimum_order_qty'] or 0,
                    'multiple': row['order_multiple'] or 1
                }
            
            # Shelf life
            shelf_query = text(f"""
                SELECT pt_code, shelf_life_days, is_perishable
                FROM products
                WHERE is_perishable = 1
                {' AND ' + product_filter if product_filter else ''}
            """)
            shelf_df = pd.read_sql(shelf_query, engine, params=params)
            
            for _, row in shelf_df.iterrows():
                constraints['shelf_life'][row['pt_code']] = row['shelf_life_days']
            
        except Exception as e:
            logger.error(f"Error getting constraints: {str(e)}")
        
        return constraints