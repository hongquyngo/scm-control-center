"""
Allocation Validators - Business rules and validation for allocations
"""

import pandas as pd
import numpy as np
from datetime import datetime
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
        """Validate allocation results against business rules
        
        Currently validates:
        1. Over-allocation - Check if allocation exceeds available supply
        2. Credit limits - Check if allocations exceed customer credit limits
        """
        warnings = []
        
        # 1. Check over-allocation
        over_allocation_warnings = AllocationValidator._check_over_allocation(
            allocation_df, supply_df
        )
        warnings.extend(over_allocation_warnings)
        
        # 2. Check credit limits
        credit_warnings = AllocationValidator._check_credit_limits(allocation_df)
        warnings.extend(credit_warnings)
        
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
            # Use DataManager
            from utils.data_manager import DataManager
            data_manager = DataManager()
            
            # 1. Get customer credit limits
            customer_df = data_manager.load_customer_master()
            customer_df = customer_df[customer_df['credit_limit_usd'] > 0]
            
            if customer_df.empty:
                return warnings
            
            # 2. Get current outstanding from OC
            oc_df = data_manager.load_demand_oc()
            current_outstanding = pd.DataFrame()
            
            if not oc_df.empty:
                current_outstanding = oc_df.groupby('customer').agg({
                    'outstanding_amount_usd': 'sum'
                }).reset_index()
            
            # 3. Calculate allocation value
            # Unit value = total value / demand quantity
            allocation_df['unit_value'] = (
                allocation_df['value_in_usd'] / allocation_df['demand_quantity']
            ).fillna(0)
            
            allocation_df['allocation_value_usd'] = (
                allocation_df['allocated_qty'] * allocation_df['unit_value']
            )
            
            # 4. Group by customer
            customer_allocations = allocation_df.groupby('customer').agg({
                'allocation_value_usd': 'sum',
                'allocated_qty': 'sum'
            }).reset_index()
            
            # 5. Check credit limits
            for _, customer_alloc in customer_allocations.iterrows():
                customer_name = customer_alloc['customer']
                
                # Get credit limit
                credit_info = customer_df[customer_df['customer_name'] == customer_name]
                if credit_info.empty:
                    continue
                    
                credit_limit = credit_info.iloc[0]['credit_limit_usd']
                
                # Get current outstanding
                outstanding = 0
                if not current_outstanding.empty:
                    cust_outstanding = current_outstanding[
                        current_outstanding['customer'] == customer_name
                    ]
                    if not cust_outstanding.empty:
                        outstanding = cust_outstanding.iloc[0]['outstanding_amount_usd']
                
                # Calculate total exposure
                new_allocation = customer_alloc['allocation_value_usd']
                total_exposure = outstanding + new_allocation
                
                # Check if over limit
                if total_exposure > credit_limit:
                    available = credit_limit - outstanding
                    
                    warnings.append(
                        f"Credit limit exceeded for {customer_name}: "
                        f"Limit: ${credit_limit:,.0f}, "
                        f"Outstanding: ${outstanding:,.0f}, "
                        f"New allocation: ${new_allocation:,.0f}, "
                        f"Total: ${total_exposure:,.0f} "
                        f"(Over by ${total_exposure - credit_limit:,.0f})"
                    )
                    
                    if available <= 0:
                        warnings.append(
                            f"⚠️ {customer_name} already over limit! "
                            f"No credit available."
                        )
        
        except Exception as e:
            logger.error(f"Error checking credit limits: {str(e)}")
            warnings.append(f"Credit check error: {str(e)}")
        
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
            
            return len(errors) == 0, errors
            
        except Exception as e:
            logger.error(f"Error validating allocation: {str(e)}")
            errors.append(f"Validation error: {str(e)}")
            return False, errors
    
    @staticmethod
    def get_allocation_constraints(product_codes: List[str] = None) -> Dict:
        """Get all applicable constraints for products
        
        Currently returns:
        - credit_limits: Customer credit limits
        
        Future constraints to be added:
        - moq: Minimum order quantities
        - shelf_life: Shelf life constraints
        - customer_limits: Customer-specific limits
        """
        constraints = {
            'credit_limits': {}
        }
        
        try:
            engine = get_db_engine()
            
            # Credit limits only for now
            from utils.data_manager import DataManager
            data_manager = DataManager()
            
            customer_df = data_manager.load_customer_master()
            customer_df = customer_df[customer_df['credit_limit_usd'] > 0]
            
            constraints['credit_limits'] = dict(zip(
                customer_df['customer_name'], 
                customer_df['credit_limit_usd']
            ))
            
        except Exception as e:
            logger.error(f"Error getting constraints: {str(e)}")
        
        return constraints