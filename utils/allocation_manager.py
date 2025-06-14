"""
Allocation Manager - Core allocation plan management
"""
import streamlit as st
import pandas as pd
from datetime import datetime
from sqlalchemy import text
from typing import List, Dict, Optional, Tuple, Any
import logging

from utils.db import get_db_engine
from utils.formatters import generate_allocation_number
from utils.allocation_cancellation import AllocationCancellationManager

logger = logging.getLogger(__name__)


class AllocationManager:
    """Manages allocation plans and operations"""
    
    def __init__(self):
        self.engine = get_db_engine()
        self.cancellation_manager = AllocationCancellationManager()  # Add this
    
    def get_active_allocations(self, status_filter: List[str] = None,
                            date_from: datetime = None, date_to: datetime = None,
                            search_text: str = None) -> pd.DataFrame:
        """Get active allocations for display
        
        Args:
            status_filter: List of delivery statuses to filter by
            date_from: Start date for allocated_etd filter
            date_to: End date for allocated_etd filter
            search_text: Text to search in pt_code, customer_name, or allocation_number
            
        Returns:
            pd.DataFrame: Active allocations with details
        """

        try:
            query = """
                SELECT * FROM active_allocations_view
                WHERE 1=1
            """
            
            params = {}
            
            # Status filter
            if status_filter:
                placeholders = ', '.join([f':status{i}' for i in range(len(status_filter))])
                query += f" AND delivery_status IN ({placeholders})"
                for i, status in enumerate(status_filter):
                    params[f'status{i}'] = status
            
            # Date filter on ETD
            if date_from:
                query += " AND allocated_etd >= :date_from"
                params['date_from'] = date_from
                
            if date_to:
                query += " AND allocated_etd <= :date_to"
                params['date_to'] = date_to
                
            # Search filter
            if search_text:
                query += """ AND (
                    pt_code LIKE :search_text OR 
                    customer_name LIKE :search_text OR
                    allocation_number LIKE :search_text
                )"""
                params['search_text'] = f"%{search_text}%"
            
            # Order by ETD and allocation number
            query += " ORDER BY allocated_etd ASC, allocation_number, pt_code"
            
            df = pd.read_sql(text(query), self.engine, params=params)
            
            return df
            
        except Exception as e:
            logger.error(f"Error getting active allocations: {str(e)}")
            return pd.DataFrame()

    def get_allocation_plans(self, status_filter: List[str] = None, 
                            date_from: datetime = None, date_to: datetime = None,
                            search_text: str = None) -> pd.DataFrame:
        """Get allocation plans using v_allocation_plans_summary view
        
        Args:
            status_filter: List of statuses to filter by
            date_from: Start date for allocation_date filter
            date_to: End date for allocation_date filter  
            search_text: Text to search in allocation_number or notes
            
        Returns:
            pd.DataFrame: Allocation plans with summary information
        """
        try:
            # Base query
            query = """
                SELECT 
                    vaps.*,
                    u.username as creator_name,
                    JSON_UNQUOTE(JSON_EXTRACT(vaps.allocation_context, '$.allocation_method')) as allocation_method
                FROM v_allocation_plans_summary vaps
                LEFT JOIN users u ON vaps.creator_id = u.id
                WHERE 1=1
            """
            
            params = {}
            
            # Status filter
            if status_filter:
                status_mapping = {
                    'DRAFT': ['ALL_DRAFT', 'MIXED_DRAFT'],
                    'ALLOCATED': ['IN_PROGRESS'],
                    'PARTIAL_DELIVERED': ['IN_PROGRESS'],
                    'DELIVERED': ['ALL_DELIVERED'],
                    'CANCELLED': ['ALL_CANCELLED']
                }
                
                display_statuses = []
                for status in status_filter:
                    display_statuses.extend(status_mapping.get(status, [status]))
                
                # Remove duplicates
                display_statuses = list(set(display_statuses))
                
                if display_statuses:
                    # Create parameterized placeholders
                    placeholders = ', '.join([f':status{i}' for i in range(len(display_statuses))])
                    query += f" AND vaps.display_status IN ({placeholders})"
                    
                    # Add each status as individual parameter
                    for i, status in enumerate(display_statuses):
                        params[f'status{i}'] = status
            
            # Date filters
            if date_from:
                query += " AND DATE(vaps.allocation_date) >= :date_from"
                params['date_from'] = date_from
                
            if date_to:
                query += " AND DATE(vaps.allocation_date) <= :date_to"
                params['date_to'] = date_to
                
            # Search filter
            if search_text:
                query += """ AND (vaps.allocation_number LIKE :search_text 
                            OR vaps.notes LIKE :search_text)"""
                params['search_text'] = f"%{search_text}%"
            
            query += " ORDER BY vaps.allocation_date DESC"
            
            # Execute with text() and dict params
            df = pd.read_sql(text(query), self.engine, params=params)
            
            return df
            
        except Exception as e:
            logger.error(f"Error getting allocation plans: {str(e)}")
            return pd.DataFrame()

    def validate_before_allocation(self, allocation_id: int) -> Tuple[bool, List[str]]:
        """Validate allocation plan before changing from DRAFT to ALLOCATED
        
        This method performs comprehensive validation to ensure the allocation plan
        is ready to be activated (allocated).
        
        Args:
            allocation_id: ID of the allocation plan to validate
            
        Returns:
            Tuple[bool, List[str]]: (is_valid, list_of_errors)
            - is_valid: True if plan can be allocated, False otherwise
            - list_of_errors: List of validation error messages
        """
        errors = []
        
        try:
            with self.engine.connect() as conn:
                # 1. Check if plan exists and get basic info
                plan_query = text("""
                    SELECT 
                        ap.id,
                        ap.allocation_number,
                        vaps.display_status,
                        vaps.total_count,
                        vaps.draft_count,
                        vaps.allocated_count,
                        ap.notes,
                        JSON_UNQUOTE(JSON_EXTRACT(ap.allocation_context, '$.allocation_info.type')) as allocation_type
                    FROM allocation_plans ap
                    LEFT JOIN v_allocation_plans_summary vaps ON ap.id = vaps.id
                    WHERE ap.id = :allocation_id
                """)
                
                result = conn.execute(plan_query, {'allocation_id': allocation_id})
                plan = result.fetchone()
                
                if not plan:
                    errors.append("Allocation plan not found")
                    return False, errors
                
                plan_dict = dict(plan._mapping) if hasattr(plan, '_mapping') else dict(zip(result.keys(), plan))
                
                # 2. Check if plan status allows allocation
                if plan_dict['display_status'] not in ['ALL_DRAFT', 'MIXED_DRAFT']:
                    errors.append(f"Plan status '{plan_dict['display_status']}' does not allow allocation. Only DRAFT plans can be allocated.")
                    return False, errors
                
                # 3. Check if plan has any details
                if plan_dict['total_count'] == 0:
                    errors.append("Plan has no allocation details")
                    return False, errors
                
                # 4. Get detailed validation data
                details_query = text("""
                    SELECT 
                        ad.id,
                        ad.pt_code,
                        ad.product_id,
                        ad.customer_code,
                        ad.customer_name,
                        ad.allocated_qty,
                        ad.requested_qty,
                        ad.allocation_mode,
                        ad.supply_source_type,
                        ad.supply_source_id,
                        ad.status,
                        ad.etd,
                        ad.allocated_etd,
                        p.name as product_name
                    FROM allocation_details ad
                    LEFT JOIN products p ON ad.product_id = p.id
                    WHERE ad.allocation_plan_id = :allocation_id
                    AND ad.status = 'DRAFT'
                """)
                
                details_result = conn.execute(details_query, {'allocation_id': allocation_id})
                details = details_result.fetchall()
                
                if not details:
                    errors.append("No DRAFT details found to allocate")
                    return False, errors
                
                # Convert to list of dicts for easier processing
                details_list = []
                for row in details:
                    details_list.append(dict(row._mapping) if hasattr(row, '_mapping') else dict(zip(details_result.keys(), row)))
                
                # 5. Validate each detail
                zero_allocation_count = 0
                missing_customer_count = 0
                missing_product_count = 0
                over_allocation_items = []
                hard_allocation_issues = []
                missing_dates = []
                
                for detail in details_list:
                    # Check for zero allocations
                    if detail['allocated_qty'] <= 0:
                        zero_allocation_count += 1
                    
                    # Check for missing customer mapping
                    if not detail['customer_code']:
                        missing_customer_count += 1
                    
                    # Check for missing product
                    if not detail['product_id']:
                        missing_product_count += 1
                    
                    # Check for over-allocation
                    if detail['requested_qty'] > 0 and detail['allocated_qty'] > detail['requested_qty']:
                        over_allocation_items.append(
                            f"{detail['pt_code']} - {detail['customer_name']}: "
                            f"allocated {detail['allocated_qty']} > requested {detail['requested_qty']}"
                        )
                    
                    # Check HARD allocation requirements
                    if detail['allocation_mode'] == 'HARD':
                        if not detail['supply_source_type'] or not detail['supply_source_id']:
                            hard_allocation_issues.append(
                                f"{detail['pt_code']} - {detail['customer_name']}: "
                                f"HARD allocation missing supply mapping"
                            )
                    
                    # Check dates
                    if not detail['etd'] or not detail['allocated_etd']:
                        missing_dates.append(f"{detail['pt_code']} - {detail['customer_name']}")
                
                # 6. Compile validation errors
                if zero_allocation_count > 0:
                    errors.append(f"{zero_allocation_count} items have zero allocated quantity")
                
                if missing_customer_count > 0:
                    errors.append(f"{missing_customer_count} items missing customer code mapping")
                
                if missing_product_count > 0:
                    errors.append(f"{missing_product_count} items missing product ID")
                
                if over_allocation_items:
                    errors.append(f"Over-allocation found in {len(over_allocation_items)} items")
                    # Add first 3 examples
                    for item in over_allocation_items[:3]:
                        errors.append(f"  - {item}")
                    if len(over_allocation_items) > 3:
                        errors.append(f"  ... and {len(over_allocation_items) - 3} more")
                
                if hard_allocation_issues:
                    errors.append(f"HARD allocation issues in {len(hard_allocation_issues)} items")
                    for issue in hard_allocation_issues[:3]:
                        errors.append(f"  - {issue}")
                    if len(hard_allocation_issues) > 3:
                        errors.append(f"  ... and {len(hard_allocation_issues) - 3} more")
                
                if missing_dates:
                    errors.append(f"{len(missing_dates)} items missing ETD or allocated ETD")
                
                # 7. Check allocation type specific validations
                allocation_type = plan_dict.get('allocation_type', 'SOFT')
                
                if allocation_type in ['HARD', 'MIXED']:
                    # For HARD/MIXED, validate supply availability
                    supply_validation_query = text("""
                        SELECT 
                            ad.id,
                            ad.pt_code,
                            ad.allocated_qty,
                            ad.supply_source_type,
                            ad.supply_source_id,
                            -- Check if supply source still exists and has enough quantity
                            CASE 
                                WHEN ad.supply_source_type = 'Inventory' THEN (
                                    SELECT ih.quantity_actual 
                                    FROM inventory_history ih 
                                    WHERE ih.id = ad.supply_source_id 
                                    AND ih.delete_flag = 0
                                )
                                WHEN ad.supply_source_type = 'Pending PO' THEN (
                                    SELECT pol.quantity - COALESCE(pol.received_quantity, 0)
                                    FROM purchase_order_lines pol 
                                    WHERE pol.id = ad.supply_source_id 
                                    AND pol.delete_flag = 0
                                )
                                -- Add other supply types as needed
                                ELSE NULL
                            END as available_qty
                        FROM allocation_details ad
                        WHERE ad.allocation_plan_id = :allocation_id
                        AND ad.allocation_mode = 'HARD'
                        AND ad.status = 'DRAFT'
                    """)
                    
                    supply_result = conn.execute(supply_validation_query, {'allocation_id': allocation_id})
                    supply_issues = []
                    
                    for row in supply_result:
                        supply_dict = dict(row._mapping) if hasattr(row, '_mapping') else dict(zip(supply_result.keys(), row))
                        
                        if supply_dict['available_qty'] is None:
                            supply_issues.append(
                                f"{supply_dict['pt_code']}: Supply source not found"
                            )
                        elif supply_dict['available_qty'] < supply_dict['allocated_qty']:
                            supply_issues.append(
                                f"{supply_dict['pt_code']}: Insufficient supply "
                                f"(need {supply_dict['allocated_qty']}, available {supply_dict['available_qty']})"
                            )
                    
                    if supply_issues:
                        errors.append(f"Supply availability issues for HARD allocations:")
                        for issue in supply_issues[:5]:
                            errors.append(f"  - {issue}")
                        if len(supply_issues) > 5:
                            errors.append(f"  ... and {len(supply_issues) - 5} more")
                
                # 8. Business rule validations
                # Check if all items have same allocation mode for SOFT/HARD types
                if allocation_type in ['SOFT', 'HARD']:
                    mode_check_query = text("""
                        SELECT COUNT(DISTINCT allocation_mode) as mode_count
                        FROM allocation_details
                        WHERE allocation_plan_id = :allocation_id
                        AND status = 'DRAFT'
                    """)
                    
                    mode_result = conn.execute(mode_check_query, {'allocation_id': allocation_id})
                    mode_row = mode_result.fetchone()
                    
                    if mode_row and mode_row[0] > 1:
                        errors.append(
                            f"Inconsistent allocation modes found. "
                            f"Plan type is '{allocation_type}' but details have mixed modes."
                        )
                
                # Return validation result
                is_valid = len(errors) == 0
                
                if is_valid:
                    logger.info(f"Allocation plan {allocation_id} passed validation")
                else:
                    logger.warning(f"Allocation plan {allocation_id} failed validation with {len(errors)} errors")
                
                return is_valid, errors
                
        except Exception as e:
            logger.error(f"Error validating allocation plan {allocation_id}: {str(e)}")
            errors.append(f"Validation error: {str(e)}")
            return False, errors

    def get_allocation_details(self, allocation_id: int) -> Tuple[Dict, pd.DataFrame]:
        """Get allocation plan and its details with computed status
        
        Args:
            allocation_id: ID of the allocation plan
            
        Returns:
            Tuple[Dict, pd.DataFrame]: Plan dict and details DataFrame
        """
        try:
            # Get plan header with JSON extraction
            plan_query = """
                SELECT 
                    ap.*, 
                    u.username as creator_name,
                    -- Extract allocation_method from multiple possible JSON paths
                    COALESCE(
                        JSON_UNQUOTE(JSON_EXTRACT(ap.allocation_context, '$.allocation_method')),
                        JSON_UNQUOTE(JSON_EXTRACT(ap.allocation_context, '$.allocation_info.method')),
                        'Manual'
                    ) as allocation_method,
                    -- Extract allocation_type from multiple possible JSON paths
                    COALESCE(
                        JSON_UNQUOTE(JSON_EXTRACT(ap.allocation_context, '$.allocation_type')),
                        JSON_UNQUOTE(JSON_EXTRACT(ap.allocation_context, '$.allocation_info.type')),
                        'SOFT'
                    ) as allocation_type
                FROM allocation_plans ap
                LEFT JOIN users u ON ap.creator_id = u.id
                WHERE ap.id = :allocation_id
            """
            
            plan_df = pd.read_sql(text(plan_query), self.engine, 
                                params={'allocation_id': allocation_id})
            
            if plan_df.empty:
                logger.warning(f"No allocation plan found with ID: {allocation_id}")
                return None, pd.DataFrame()
            
            plan = plan_df.iloc[0].to_dict()
            
            # Ensure allocation_method and allocation_type are not NULL/NaN
            if pd.isna(plan.get('allocation_method')) or plan.get('allocation_method') == 'null':
                plan['allocation_method'] = 'Manual'
            if pd.isna(plan.get('allocation_type')) or plan.get('allocation_type') == 'null':
                plan['allocation_type'] = 'SOFT'
            
            # Get details with computed status from view
            details_query = """
                SELECT 
                    ads.*,
                    p.name as product_name,
                    p.package_size,
                    p.uom as standard_uom,
                    -- Use computed effective_status instead of stored status
                    ads.effective_status as status,
                    -- For backward compatibility
                    ads.allocated_qty as effective_allocated_qty,
                    ads.remaining_qty as cancellable_qty
                FROM allocation_delivery_status_view ads
                LEFT JOIN products p ON ads.product_id = p.id
                WHERE ads.allocation_plan_id = :allocation_id
                ORDER BY ads.pt_code, ads.allocated_etd
            """
            
            details_df = pd.read_sql(text(details_query), self.engine,
                                params={'allocation_id': allocation_id})
            
            # Add cancellation summary to plan
            plan['cancellation_summary'] = self.cancellation_manager.get_plan_cancellation_summary(allocation_id)
            
            # Log successful retrieval
            logger.info(f"Retrieved allocation plan {allocation_id} with {len(details_df)} details")
            
            return plan, details_df
            
        except Exception as e:
            logger.error(f"Error getting allocation details: {str(e)}")
            return None, pd.DataFrame()

    def get_available_supply_for_hard_allocation(self, product_codes: List[str], 
                                            legal_entities: List[str]) -> pd.DataFrame:
        """Get available supply details for HARD allocation mapping from GAP Analysis results"""
        try:
            # Import streamlit here to avoid circular dependency
            import streamlit as st
            
            # Get supply data from session state (already filtered by GAP Analysis)
            supply_filtered = st.session_state.get('supply_filtered', pd.DataFrame())
            
            if supply_filtered.empty:
                logger.warning("No supply data found in session state. Please run GAP Analysis first.")
                return pd.DataFrame()
            
            # Filter for selected products and entities
            mask = (
                supply_filtered['pt_code'].isin(product_codes) & 
                supply_filtered['legal_entity'].isin(legal_entities) &
                supply_filtered['quantity'] > 0
            )
            
            filtered_supply = supply_filtered[mask].copy()
            
            # Rename columns for consistency
            filtered_supply = filtered_supply.rename(columns={
                'quantity': 'available_qty',
                'supply_number': 'reference'
            })
            
            # Add source_id based on source_type
            def get_source_id(row):
                """Extract source ID from supply_number or other identifiers"""
                if row['source_type'] == 'Inventory':
                    return row.get('inventory_history_id', row.get('supply_number', ''))
                elif row['source_type'] == 'Pending CAN':
                    return row.get('arrival_note_line_id', row.get('supply_number', ''))
                elif row['source_type'] == 'Pending PO':
                    return row.get('po_line_id', row.get('supply_number', ''))
                else:
                    return row.get('supply_number', '')
            
            filtered_supply['source_id'] = filtered_supply.apply(get_source_id, axis=1)
            
            # Select relevant columns
            columns_to_keep = [
                'source_type', 'source_id', 'pt_code', 'product_name',
                'legal_entity', 'available_qty', 'reference'
            ]
            
            # Add optional columns if they exist
            optional_cols = ['batch_number', 'expiry_date', 'origin_country', 
                            'date_ref', 'vendor', 'from_warehouse', 'to_warehouse']
            
            for col in optional_cols:
                if col in filtered_supply.columns:
                    columns_to_keep.append(col)
            
            # Rename date_ref to expected_date for consistency
            if 'date_ref' in filtered_supply.columns:
                filtered_supply['expected_date'] = filtered_supply['date_ref']
                columns_to_keep.append('expected_date')
            
            result_df = filtered_supply[columns_to_keep].copy()
            
            # Sort by source type and expected date
            sort_cols = ['source_type']
            if 'expected_date' in result_df.columns:
                sort_cols.append('expected_date')
            
            result_df = result_df.sort_values(sort_cols)
            
            logger.info(f"Found {len(result_df)} supply items for HARD allocation from GAP Analysis data")
            
            return result_df
            
        except Exception as e:
            logger.error(f"Error getting available supply for HARD allocation: {str(e)}")
            return pd.DataFrame()

    def create_allocation_plan(self, plan_data: Dict, allocation_details: pd.DataFrame, 
                            supply_mapping: Dict = None, data_prepared: bool = False) -> Optional[int]:
        """Create new allocation plan with details and supply mapping
        
        Args:
            plan_data: Plan header information
            allocation_details: DataFrame with allocation line details
            supply_mapping: Dict mapping demand_line_id to supply source (for HARD allocation)
            data_prepared: If True, skip data preparation steps (customer mapping, etc.)
            
        Returns:
            Optional[int]: Allocation plan ID if successful, None otherwise
        """
        conn = self.engine.connect()
        trans = conn.begin()
        
        try:
            # Generate allocation number if not provided
            if 'allocation_number' in plan_data:
                allocation_number = plan_data['allocation_number']
            else:
                allocation_number = generate_allocation_number()
            
            # Use SCM system user (id=1) as default creator
            creator_id = plan_data.get('creator_id', 1)
            
            # Convert allocation_context to JSON string
            import json
            import numpy as np
            
            allocation_context_json = None
            if 'allocation_context' in plan_data and plan_data['allocation_context']:
                # Clean the context data before JSON encoding
                def clean_for_json(obj):
                    if isinstance(obj, dict):
                        return {k: clean_for_json(v) for k, v in obj.items()}
                    elif isinstance(obj, list):
                        return [clean_for_json(item) for item in obj]
                    elif isinstance(obj, float):
                        if np.isnan(obj) or np.isinf(obj):
                            return None
                        return obj
                    elif isinstance(obj, np.floating):
                        if np.isnan(obj) or np.isinf(obj):
                            return None
                        return float(obj)
                    elif isinstance(obj, np.integer):
                        return int(obj)
                    elif pd.isna(obj):
                        return None
                    else:
                        return obj
                
                cleaned_context = clean_for_json(plan_data['allocation_context'])
                allocation_context_json = json.dumps(cleaned_context)
            
            # Insert allocation plan header
            plan_insert = text("""
                INSERT INTO allocation_plans 
                (allocation_number, allocation_date, creator_id, created_date, 
                notes, allocation_context)
                VALUES 
                (:allocation_number, NOW(), :creator_id, NOW(), 
                :notes, :allocation_context)
            """)
            
            result = conn.execute(plan_insert, {
                'allocation_number': allocation_number,
                'creator_id': creator_id,
                'notes': plan_data.get('notes', ''),
                'allocation_context': allocation_context_json
            })
            
            allocation_plan_id = result.lastrowid
            logger.info(f"Created allocation plan header with ID {allocation_plan_id}")
            
            # === PREPARE DATA IF NEEDED ===
            if not data_prepared:
                # Original data preparation logic
                logger.info("Preparing allocation data...")
                
                # Get customer mapping
                from utils.data_manager import DataManager
                data_manager = DataManager()
                customers_df = data_manager.load_customer_master()
                
                if not customers_df.empty:
                    customer_mapping = dict(zip(
                        customers_df['customer_name'].str.strip(), 
                        customers_df['customer_code'].str.strip()
                    ))
                else:
                    customer_mapping = {}
                
                # Apply mappings and prepare data
                if 'customer' in allocation_details.columns:
                    allocation_details['customer_name'] = allocation_details['customer'].str.strip()
                    allocation_details['customer_code'] = allocation_details['customer_name'].map(customer_mapping)
                
                # Add other preparations as needed...
                
            # Get product mapping
            product_map = self._get_product_mapping()
            
            if not product_map:
                raise ValueError("Product master data not available - please refresh data from main dashboard")
            
            # Validate that all required products exist
            missing_products = []
            if not allocation_details.empty and 'pt_code' in allocation_details.columns:
                for pt_code in allocation_details['pt_code'].dropna().unique():
                    pt_code_str = str(pt_code).strip()
                    if pt_code_str not in product_map:
                        missing_products.append(pt_code_str)
                
                if missing_products:
                    logger.error(f"Products not found in master: {missing_products}")
                    raise ValueError(f"Products not found in master data: {', '.join(missing_products[:5])}" + 
                                (f" and {len(missing_products) - 5} more" if len(missing_products) > 5 else ""))
            
            logger.info("All products validated against master data")
            
            # Insert allocation details
            details_inserted = 0
            failed_rows = []
            
            # Convert DataFrame to list of dicts for easier access
            details_list = allocation_details.to_dict('records')
            
            for idx, row_dict in enumerate(details_list):
                try:
                    # Check if this is a HARD allocation from supply_mapping
                    allocation_mode = row_dict.get('allocation_mode', 'SOFT')
                    supply_source_type = row_dict.get('supply_source_type')
                    supply_source_id = row_dict.get('supply_source_id')
                    
                    # Override with supply_mapping if exists
                    demand_line_id = row_dict.get('demand_line_id', '')
                    if supply_mapping and str(demand_line_id) in supply_mapping:
                        allocation_mode = 'HARD'
                        mapping_info = supply_mapping[str(demand_line_id)]
                        supply_source_type = mapping_info.get('source_type')
                        supply_source_id = mapping_info.get('source_id')
                    
                    # Get product_id from pt_code
                    pt_code = row_dict.get('pt_code')
                    product_id = None
                    
                    if pt_code and pd.notna(pt_code):
                        pt_code_str = str(pt_code).strip()
                        product_id = product_map.get(pt_code_str)
                        
                        if not product_id:
                            logger.error(f"Product not found for pt_code: {pt_code_str}")
                            failed_rows.append(f"Row {idx}: Product {pt_code_str} not found")
                            continue
                    else:
                        logger.error(f"No pt_code for row {idx}")
                        failed_rows.append(f"Row {idx}: Missing pt_code")
                        continue
                    
                    # Determine initial status based on plan status
                    detail_status = row_dict.get('status', 'DRAFT')
                    
                    # Helper function to clean values
                    def clean_value(val):
                        if pd.isna(val):
                            return None
                        if isinstance(val, (np.floating, float)) and (np.isnan(val) or np.isinf(val)):
                            return None
                        return val
                    
                    # Process dates
                    etd_value = row_dict.get('etd')
                    allocated_etd_value = row_dict.get('allocated_etd', etd_value)
                    
                    # Convert dates to string format for MySQL
                    if etd_value and pd.notna(etd_value):
                        if isinstance(etd_value, pd.Timestamp):
                            etd_value = etd_value.strftime('%Y-%m-%d')
                        elif isinstance(etd_value, str):
                            try:
                                etd_value = pd.to_datetime(etd_value).strftime('%Y-%m-%d')
                            except:
                                etd_value = None
                    else:
                        etd_value = None
                    
                    if allocated_etd_value and pd.notna(allocated_etd_value):
                        if isinstance(allocated_etd_value, pd.Timestamp):
                            allocated_etd_value = allocated_etd_value.strftime('%Y-%m-%d')
                        elif isinstance(allocated_etd_value, str):
                            try:
                                allocated_etd_value = pd.to_datetime(allocated_etd_value).strftime('%Y-%m-%d')
                            except:
                                allocated_etd_value = etd_value
                    else:
                        allocated_etd_value = etd_value
                    
                    # Insert detail record
                    detail_insert = text("""
                        INSERT INTO allocation_details
                        (allocation_plan_id, allocation_mode, status,
                        demand_type, demand_reference_id, demand_number,
                        product_id, pt_code, customer_code, customer_name,
                        legal_entity_name, requested_qty, allocated_qty, delivered_qty,
                        etd, allocated_etd, notes,
                        supply_source_type, supply_source_id)
                        VALUES
                        (:allocation_plan_id, :allocation_mode, :status,
                        :demand_type, :demand_reference_id, :demand_number,
                        :product_id, :pt_code, :customer_code, :customer_name,
                        :legal_entity_name, :requested_qty, :allocated_qty, 0,
                        :etd, :allocated_etd, :notes,
                        :supply_source_type, :supply_source_id)
                    """)
                    
                    detail_params = {
                        'allocation_plan_id': allocation_plan_id,
                        'allocation_mode': allocation_mode,
                        'status': detail_status,
                        'demand_type': clean_value(row_dict.get('demand_type', 'OC')),
                        'demand_reference_id': clean_value(row_dict.get('demand_reference_id')),
                        'demand_number': clean_value(row_dict.get('demand_number', '')),
                        'product_id': int(product_id),
                        'pt_code': pt_code_str,
                        'customer_code': clean_value(row_dict.get('customer_code')),
                        'customer_name': clean_value(row_dict.get('customer_name', '')),
                        'legal_entity_name': clean_value(row_dict.get('legal_entity_name', '')),
                        'requested_qty': float(clean_value(row_dict.get('requested_qty', 0)) or 0),
                        'allocated_qty': float(clean_value(row_dict.get('allocated_qty', 0)) or 0),
                        'etd': etd_value,
                        'allocated_etd': allocated_etd_value,
                        'notes': clean_value(row_dict.get('notes', '')),
                        'supply_source_type': supply_source_type,
                        'supply_source_id': supply_source_id
                    }
                    
                    conn.execute(detail_insert, detail_params)
                    details_inserted += 1
                    
                except Exception as detail_error:
                    logger.error(f"Error inserting detail row {idx}: {str(detail_error)}")
                    logger.error(f"Row data: {row_dict}")
                    failed_rows.append(f"Row {idx}: {str(detail_error)}")
                    continue
            
            # Check if any details were inserted
            if details_inserted == 0:
                raise ValueError("No allocation details were inserted successfully")
            
            # Commit transaction
            trans.commit()
            
            logger.info(f"Created allocation plan {allocation_number} with ID {allocation_plan_id}")
            logger.info(f"Successfully inserted {details_inserted}/{len(allocation_details)} detail rows")
            
            if failed_rows:
                logger.warning(f"Failed rows: {', '.join(failed_rows[:5])}")
                if len(failed_rows) > 5:
                    logger.warning(f"... and {len(failed_rows) - 5} more failed rows")
            
            # Log allocation summary
            if supply_mapping:
                logger.info(f"Including {len(supply_mapping)} HARD allocations")
            
            return allocation_plan_id
            
        except Exception as e:
            trans.rollback()
            logger.error(f"Error creating allocation plan: {str(e)}")
            logger.error(f"Error type: {type(e).__name__}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            
            return None
        finally:
            conn.close()

  
    def _get_product_mapping(self) -> Dict[str, int]:
        """Get product mapping from cached data
        
        Returns:
            Dict[str, int]: Mapping of pt_code to product_id
        """
        try:
            # Get from DataManager cache
            from utils.data_manager import DataManager
            data_manager = DataManager()
            
            # This will use cached data if available (TTL=3600s)
            products_df = data_manager.load_product_master()
            
            if not products_df.empty:
                # Create mapping: pt_code -> product_id
                product_map = dict(zip(products_df['pt_code'], products_df['product_id']))
                logger.info(f"Retrieved {len(product_map)} products from cache")
                return product_map
            else:
                logger.warning("No products found in cache")
                return {}
                
        except Exception as e:
            logger.error(f"Error getting product mapping: {str(e)}")
            return {}

    def update_allocation_plan(self, allocation_id: int, updates: Dict) -> bool:
        """Update allocation plan"""
        try:
            update_query = text("""
                UPDATE allocation_plans
                SET notes = :notes,
                    updated_date = NOW()
                WHERE id = :allocation_id AND status = 'DRAFT'
            """)
            
            with self.engine.connect() as conn:
                result = conn.execute(update_query, {
                    'allocation_id': allocation_id,
                    'notes': updates.get('notes', '')
                })
                conn.commit()
                
            return result.rowcount > 0
            
        except Exception as e:
            logger.error(f"Error updating allocation plan: {str(e)}")
            return False
    
    def approve_allocation(self, allocation_id: int, approved_by: str) -> bool:
        """Approve allocation plan"""
        try:
            update_query = text("""
                UPDATE allocation_plans
                SET status = 'APPROVED',
                    approved_by = :approved_by,
                    approved_date = NOW()
                WHERE id = :allocation_id AND status = 'DRAFT'
            """)
            
            with self.engine.connect() as conn:
                result = conn.execute(update_query, {
                    'allocation_id': allocation_id,
                    'approved_by': approved_by
                })
                conn.commit()
                
            return result.rowcount > 0
            
        except Exception as e:
            logger.error(f"Error approving allocation: {str(e)}")
            return False


    def cancel_allocation_plan(self, allocation_id: int, reason: str = None, 
                            reason_category: str = 'BUSINESS_DECISION',
                            user_id: int = None) -> bool:
        """Cancel allocation plan - handles both DRAFT and ALLOCATED plans
        
        Args:
            allocation_id: ID of allocation plan to cancel
            reason: Cancellation reason (optional)
            reason_category: Category of cancellation (default: BUSINESS_DECISION)
            user_id: User performing cancellation (default: system user)
            
        Returns:
            bool: True if successful, False otherwise
        """
        conn = self.engine.connect()
        trans = conn.begin()
        
        try:
            # Get user info for audit
            if user_id is None:
                user_id = 1  # System user default
                
            # 1. Get plan summary to determine status
            plan_query = text("""
                SELECT 
                    vaps.*,
                    ap.allocation_number,
                    CONCAT(COALESCE(e.first_name, ''), ' ', COALESCE(e.last_name, '')) as creator_name
                FROM v_allocation_plans_summary vaps
                JOIN allocation_plans ap ON vaps.id = ap.id
                LEFT JOIN employees e ON vaps.creator_id = e.id
                WHERE vaps.id = :allocation_id
            """)
            
            result = conn.execute(plan_query, {'allocation_id': allocation_id})
            plan = result.fetchone()
            
            if not plan:
                logger.error(f"Allocation plan {allocation_id} not found")
                return False
            
            # Convert Row to dict for easier access
            plan_dict = dict(plan._mapping) if hasattr(plan, '_mapping') else dict(zip(result.keys(), plan))
            
            logger.info(f"Cancelling allocation plan {plan_dict['allocation_number']} with status {plan_dict['display_status']}")
            
            # 2. Get all details for this plan
            details_query = text("""
                SELECT 
                    ad.id,
                    ad.status as original_status,
                    ad.allocated_qty,
                    ad.delivered_qty,
                    ad.pt_code,
                    ad.customer_name,
                    COALESCE(
                        ad.allocated_qty - ad.delivered_qty - 
                        COALESCE((
                            SELECT SUM(ac.cancelled_qty) 
                            FROM allocation_cancellations ac 
                            WHERE ac.allocation_detail_id = ad.id 
                            AND ac.status = 'ACTIVE'
                        ), 0), 
                        0
                    ) as cancellable_qty
                FROM allocation_details ad
                WHERE ad.allocation_plan_id = :allocation_id
            """)
            
            details = conn.execute(details_query, {'allocation_id': allocation_id}).fetchall()
            
            if not details:
                logger.warning(f"No details found for allocation plan {allocation_id}")
                return True  # Empty plan, consider it cancelled
            
            # 3. Process based on plan status
            cancelled_count = 0
            draft_deleted_count = 0
            
            # Set default reason if not provided
            if not reason:
                reason = f"Allocation plan {plan_dict['allocation_number']} cancelled"
            
            # Get user name for audit
            user_query = text("""
                SELECT CONCAT(COALESCE(first_name, ''), ' ', COALESCE(last_name, '')) as full_name 
                FROM employees 
                WHERE id = :user_id AND delete_flag = 0
            """)
            user_result = conn.execute(user_query, {'user_id': user_id})
            user_row = user_result.fetchone()
            cancelled_by_name = user_row['full_name'].strip() if user_row and user_row['full_name'] else f'User {user_id}'
            
            for detail in details:
                detail_dict = dict(detail._mapping) if hasattr(detail, '_mapping') else dict(zip(details_query.keys(), detail))
                
                # Handle DRAFT details
                if detail_dict['original_status'] == 'DRAFT':
                    # For DRAFT, we can either delete or mark as cancelled
                    # Option 1: Delete (clean removal)
                    delete_query = text("""
                        DELETE FROM allocation_details 
                        WHERE id = :detail_id
                    """)
                    conn.execute(delete_query, {'detail_id': detail_dict['id']})
                    draft_deleted_count += 1
                    
                    # Option 2: Create cancellation record even for DRAFT (for audit trail)
                    # Uncomment if you want to keep DRAFT records with cancellation
                    """
                    cancel_insert = text('''
                        INSERT INTO allocation_cancellations 
                        (allocation_detail_id, allocation_plan_id, cancelled_qty, 
                        reason, reason_category, cancelled_by_user_id, status)
                        VALUES (:detail_id, :plan_id, :qty, 
                                :reason, :category, :user_id, 'ACTIVE')
                    ''')
                    
                    conn.execute(cancel_insert, {
                        'detail_id': detail_dict['id'],
                        'plan_id': allocation_id,
                        'qty': detail_dict['allocated_qty'],
                        'reason': f"{reason} (Draft item)",
                        'category': reason_category,
                        'user_id': user_id
                    })
                    """
                    
                # Handle ALLOCATED details with cancellable quantity
                elif detail_dict['original_status'] == 'ALLOCATED' and detail_dict['cancellable_qty'] > 0:
                    # Create cancellation record
                    cancel_insert = text("""
                        INSERT INTO allocation_cancellations 
                        (allocation_detail_id, allocation_plan_id, cancelled_qty, 
                        reason, reason_category, cancelled_by_user_id, status)
                        VALUES (:detail_id, :plan_id, :qty, 
                                :reason, :category, :user_id, 'ACTIVE')
                    """)
                    
                    conn.execute(cancel_insert, {
                        'detail_id': detail_dict['id'],
                        'plan_id': allocation_id,
                        'qty': detail_dict['cancellable_qty'],
                        'reason': reason,
                        'category': reason_category,
                        'user_id': user_id
                    })
                    cancelled_count += 1
                    
                    logger.info(f"Cancelled {detail_dict['cancellable_qty']} qty for {detail_dict['pt_code']} - {detail_dict['customer_name']}")
                
                # Skip if already delivered or no cancellable quantity
                elif detail_dict['delivered_qty'] >= detail_dict['allocated_qty']:
                    logger.info(f"Skipping {detail_dict['pt_code']} - already delivered")
                elif detail_dict['cancellable_qty'] <= 0:
                    logger.info(f"Skipping {detail_dict['pt_code']} - no cancellable quantity")
            
            # 4. Add plan-level cancellation note
            if plan_dict.get('notes'):
                updated_notes = f"{plan_dict['notes']}\n\n[CANCELLED {datetime.now().strftime('%Y-%m-%d %H:%M')}] {reason}"
            else:
                updated_notes = f"[CANCELLED {datetime.now().strftime('%Y-%m-%d %H:%M')}] {reason}"
            
            # Update plan notes
            update_plan = text("""
                UPDATE allocation_plans
                SET notes = :notes,
                    updated_date = NOW()
                WHERE id = :allocation_id
            """)
            
            conn.execute(update_plan, {
                'allocation_id': allocation_id,
                'notes': updated_notes
            })
            
            # 5. Commit transaction
            trans.commit()
            
            # Log summary
            logger.info(f"""
                Allocation plan {plan_dict['allocation_number']} cancelled successfully:
                - Draft items deleted: {draft_deleted_count}
                - Allocated items cancelled: {cancelled_count}
                - Status was: {plan_dict['display_status']}
                - Cancelled by: {cancelled_by_name}
                - Reason: {reason}
            """)
            
            return True
            
        except Exception as e:
            trans.rollback()
            logger.error(f"Error cancelling allocation plan {allocation_id}: {str(e)}")
            logger.error(f"Stack trace: ", exc_info=True)
            return False
            
        finally:
            conn.close()


    def bulk_cancel_allocation_plans(self, plan_ids: List[int], reason: str = None,
                                    reason_category: str = 'BUSINESS_DECISION',
                                    user_id: int = None) -> Tuple[int, List[str]]:
        """Cancel multiple allocation plans
        
        Args:
            plan_ids: List of allocation plan IDs to cancel
            reason: Cancellation reason
            reason_category: Category of cancellation
            user_id: User performing cancellation
            
        Returns:
            Tuple[int, List[str]]: (success_count, list of errors)
        """
        success_count = 0
        errors = []
        
        for plan_id in plan_ids:
            try:
                if self.cancel_allocation_plan(plan_id, reason, reason_category, user_id):
                    success_count += 1
                else:
                    errors.append(f"Plan ID {plan_id}: Cancellation failed")
            except Exception as e:
                errors.append(f"Plan ID {plan_id}: {str(e)}")
        
        return success_count, errors

    def validate_before_cancel(self, allocation_id: int) -> Tuple[bool, Dict[str, Any]]:
        """Validate and get impact analysis before cancelling
        
        Args:
            allocation_id: Allocation plan ID
            
        Returns:
            Tuple[bool, Dict]: (can_cancel, impact_analysis)
        """
        try:
            with self.engine.connect() as conn:
                # Get plan info
                plan_query = text("""
                    SELECT 
                        vaps.*,
                        ap.allocation_number
                    FROM v_allocation_plans_summary vaps
                    JOIN allocation_plans ap ON vaps.id = ap.id
                    WHERE vaps.id = :allocation_id
                """)
                
                plan = conn.execute(plan_query, {'allocation_id': allocation_id}).fetchone()
                
                if not plan:
                    return False, {'error': 'Plan not found'}
                
                plan_dict = dict(plan._mapping) if hasattr(plan, '_mapping') else dict(zip(plan.keys(), plan))
                
                # Get impact analysis
                impact_query = text("""
                    SELECT 
                        COUNT(DISTINCT ad.id) as total_details,
                        COUNT(DISTINCT CASE WHEN ad.status = 'DRAFT' THEN ad.id END) as draft_count,
                        COUNT(DISTINCT CASE WHEN ad.status = 'ALLOCATED' THEN ad.id END) as allocated_count,
                        COUNT(DISTINCT CASE WHEN ad.delivered_qty > 0 THEN ad.id END) as partial_delivered_count,
                        SUM(ad.allocated_qty) as total_allocated_qty,
                        SUM(ad.delivered_qty) as total_delivered_qty,
                        SUM(
                            CASE 
                                WHEN ad.status = 'DRAFT' THEN ad.allocated_qty
                                WHEN ad.status = 'ALLOCATED' THEN 
                                    ad.allocated_qty - ad.delivered_qty - 
                                    COALESCE((
                                        SELECT SUM(ac.cancelled_qty) 
                                        FROM allocation_cancellations ac 
                                        WHERE ac.allocation_detail_id = ad.id 
                                        AND ac.status = 'ACTIVE'
                                    ), 0)
                                ELSE 0
                            END
                        ) as total_cancellable_qty,
                        COUNT(DISTINCT ad.pt_code) as affected_products,
                        COUNT(DISTINCT ad.customer_name) as affected_customers
                    FROM allocation_details ad
                    WHERE ad.allocation_plan_id = :allocation_id
                """)
                
                impact = conn.execute(impact_query, {'allocation_id': allocation_id}).fetchone()
                impact_dict = dict(impact._mapping) if hasattr(impact, '_mapping') else dict(zip(impact.keys(), impact))
                
                # Check if can cancel
                can_cancel = True
                warnings = []
                
                # Cannot cancel if all delivered
                if impact_dict['total_delivered_qty'] >= impact_dict['total_allocated_qty']:
                    can_cancel = False
                    warnings.append("All quantities already delivered")
                
                # Warning if partial deliveries exist
                if impact_dict['partial_delivered_count'] > 0:
                    warnings.append(f"{impact_dict['partial_delivered_count']} items have partial deliveries")
                
                # Build impact analysis
                analysis = {
                    'can_cancel': can_cancel,
                    'plan_info': {
                        'allocation_number': plan_dict['allocation_number'],
                        'status': plan_dict['display_status'],
                        'total_items': impact_dict['total_details']
                    },
                    'impact': {
                        'draft_items_to_delete': impact_dict['draft_count'],
                        'allocated_items_to_cancel': impact_dict['allocated_count'],
                        'quantity_to_release': float(impact_dict['total_cancellable_qty'] or 0),
                        'affected_products': impact_dict['affected_products'],
                        'affected_customers': impact_dict['affected_customers']
                    },
                    'warnings': warnings
                }
                
                return can_cancel, analysis
                
        except Exception as e:
            logger.error(f"Error validating cancellation: {str(e)}")
            return False, {'error': str(e)}


    def cancel_allocation_detail(self, detail_id: int, quantity: float, 
                            reason: str, reason_category: str, user_id: int = None) -> Tuple[bool, str]:
        """Cancel PARTIAL quantity on allocation detail với audit trail"""
        # Import streamlit here to avoid circular dependency
        import streamlit as st
        
        if user_id is None:
            user_id = st.session_state.get('user_id', 1)
        
        return self.cancellation_manager.cancel_quantity(
            detail_id, quantity, reason, reason_category, user_id
        )

    def bulk_cancel_details(self, detail_ids: List[int], reason: str, 
                        reason_category: str, user_id: int = None) -> Tuple[int, List[str]]:
        """Bulk cancel multiple allocation details"""
        # Import streamlit here to avoid circular dependency
        import streamlit as st
        
        if user_id is None:
            user_id = st.session_state.get('user_id', 1)
            
        return self.cancellation_manager.bulk_cancel(
            detail_ids, reason, reason_category, user_id
        )


    def reverse_cancellation(self, cancellation_id: int, reason: str, user_id: int = None) -> Tuple[bool, str]:
        """Reverse a cancellation"""
        # Import streamlit here to avoid circular dependency
        import streamlit as st
        
        if user_id is None:
            user_id = st.session_state.get('user_id', 1)
            
        return self.cancellation_manager.reverse_cancellation(
            cancellation_id, user_id, reason
        )

    def get_cancellation_history(self, plan_id: int = None, detail_id: int = None) -> pd.DataFrame:
        """Get cancellation history"""
        return self.cancellation_manager.get_cancellation_history(plan_id, detail_id)

    def get_cancellable_details(self, allocation_id: int) -> pd.DataFrame:
        """Get all details that can be cancelled"""
        try:
            # Use view for consistent calculation
            query = text("""
                SELECT 
                    ads.id as detail_id,
                    ads.pt_code,
                    ads.product_id,
                    p.product_name,
                    ads.customer_name,
                    ads.original_allocated_qty as allocated_qty,
                    ads.delivered_qty,
                    ads.cancelled_qty,
                    ads.remaining_qty as cancellable_qty,
                    ads.detail_status as status,
                    ads.allocation_mode
                FROM allocation_delivery_status_view ads
                LEFT JOIN products p ON ads.product_id = p.id
                WHERE ads.allocation_plan_id = :allocation_id
                AND ads.detail_status = 'ALLOCATED'  -- Only ALLOCATED can be cancelled
                AND ads.remaining_qty > 0
                ORDER BY ads.pt_code, ads.customer_name
            """)
            
            df = pd.read_sql(query, self.engine, params={'allocation_id': allocation_id})
            return df
            
        except Exception as e:
            logger.error(f"Error getting cancellable details: {str(e)}")
            return pd.DataFrame()

    def get_active_allocations_summary(self) -> pd.DataFrame:
        """Get summary of active allocations for supply adjustment"""
        try:
            # Sử dụng view đã optimize
            query = """
                SELECT 
                    pt_code,
                    legal_entity_name as legal_entity,
                    SUM(undelivered_qty) as undelivered_qty
                FROM active_allocations_view
                WHERE undelivered_qty > 0
                GROUP BY pt_code, legal_entity_name
            """
            
            df = pd.read_sql(text(query), self.engine)
            return df
            
        except Exception as e:
            logger.error(f"Error getting active allocations summary: {str(e)}")
            return pd.DataFrame()


    def get_allocation_performance_metrics(self, date_from: datetime = None, 
                                         date_to: datetime = None) -> Dict:
        """Get allocation performance metrics"""
        try:
            params = {}
            date_filter = ""
            
            if date_from:
                date_filter += " AND ap.allocation_date >= :date_from"
                params['date_from'] = date_from
            
            if date_to:
                date_filter += " AND ap.allocation_date <= :date_to"
                params['date_to'] = date_to
            
            # Overall metrics
            overall_query = text(f"""
                SELECT 
                    COUNT(DISTINCT ap.id) as total_plans,
                    COUNT(DISTINCT CASE WHEN ap.status = 'EXECUTED' THEN ap.id END) as executed_plans,
                    AVG(CASE WHEN ap.status = 'EXECUTED' 
                        THEN ad.delivered_qty / NULLIF(ad.allocated_qty, 0) * 100 
                        END) as avg_fulfillment_rate,
                    SUM(ad.allocated_qty) as total_allocated,
                    SUM(ad.delivered_qty) as total_delivered
                FROM allocation_plans ap
                JOIN allocation_details ad ON ap.id = ad.allocation_plan_id
                WHERE 1=1 {date_filter}
            """)
            
            overall_df = pd.read_sql(overall_query, self.engine, params=params)
            
            # By method metrics
            method_query = text(f"""
                SELECT 
                    ap.allocation_method,
                    COUNT(DISTINCT ap.id) as plan_count,
                    AVG(ad.delivered_qty / NULLIF(ad.allocated_qty, 0) * 100) as avg_fulfillment
                FROM allocation_plans ap
                JOIN allocation_details ad ON ap.id = ad.allocation_plan_id
                WHERE ap.status = 'EXECUTED' {date_filter}
                GROUP BY ap.allocation_method
            """)
            
            method_df = pd.read_sql(method_query, self.engine, params=params)
            
            return {
                'overall': overall_df.iloc[0].to_dict() if not overall_df.empty else {},
                'by_method': method_df.to_dict('records')
            }
            
        except Exception as e:
            logger.error(f"Error getting performance metrics: {str(e)}")
            return {'overall': {}, 'by_method': []}
    
    def execute_allocation(self, allocation_id: int) -> bool:
        """Mark allocation as executed (ready for delivery)"""
        try:
            update_query = text("""
                UPDATE allocation_plans
                SET status = 'EXECUTED'
                WHERE id = :allocation_id AND status = 'APPROVED'
            """)
            
            with self.engine.connect() as conn:
                result = conn.execute(update_query, {'allocation_id': allocation_id})
                conn.commit()
                
            return result.rowcount > 0
            
        except Exception as e:
            logger.error(f"Error executing allocation: {str(e)}")
            return False

    def bulk_update_allocation_status(self, allocation_id: int, new_status: str) -> bool:
        """Update all DRAFT details in a plan to new status
        
        Note: Only updates from DRAFT -> ALLOCATED
        Other statuses are computed based on delivery/cancellation
        
        Args:
            allocation_id: Plan ID
            new_status: New status (only 'ALLOCATED' is valid)
            
        Returns:
            bool: Success status
        """
        conn = self.engine.connect()
        trans = conn.begin()
        
        try:
            # Validate status transition
            if new_status not in ['ALLOCATED']:
                logger.warning(f"Invalid status update requested: {new_status}")
                return False
            
            # Update all draft details to allocated
            update_query = text("""
                UPDATE allocation_details
                SET status = :new_status
                WHERE allocation_plan_id = :allocation_id
                AND status = 'DRAFT'
            """)
            
            result = conn.execute(update_query, {
                'allocation_id': allocation_id,
                'new_status': new_status
            })
            
            trans.commit()
            
            updated_count = result.rowcount
            logger.info(f"Updated {updated_count} details to {new_status} for plan {allocation_id}")
            
            return updated_count > 0
            
        except Exception as e:
            trans.rollback()
            error_msg = str(e)
            
            # Check for permission error
            if "command denied" in error_msg.lower():
                logger.error(f"Permission denied: User does not have UPDATE permission on allocation_details")
                st.error("⚠️ Permission Error: Cannot update allocation status. Please contact administrator.")
            else:
                logger.error(f"Error updating allocation status: {error_msg}")
                
            return False
        finally:
            conn.close()


    def validate_hard_allocation_supply(self, supply_mapping: Dict, 
                                    allocation_details: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Validate HARD allocation supply mapping using GAP Analysis data
        
        Args:
            supply_mapping: Dict mapping demand_line_id to supply source
            allocation_details: DataFrame with allocation details
            
        Returns:
            Tuple[bool, List[str]]: (is_valid, list_of_errors)
        """
        errors = []
        
        try:
            # Get supply data from session state (already filtered from GAP Analysis)
            import streamlit as st
            supply_filtered = st.session_state.get('supply_filtered', pd.DataFrame())
            
            if supply_filtered.empty:
                errors.append("No supply data found from GAP Analysis. Please run GAP Analysis first.")
                return False, errors
            
            # Group allocations by supply source
            supply_usage = {}
            
            for demand_line_id, mapping in supply_mapping.items():
                source_key = f"{mapping['source_type']}_{mapping['source_id']}"
                
                # Find allocation details for this demand line
                alloc_row = allocation_details[
                    allocation_details['demand_line_id'].astype(str) == str(demand_line_id)
                ]
                
                if alloc_row.empty:
                    errors.append(f"Demand line {demand_line_id} not found in allocation details")
                    continue
                
                allocated_qty = float(alloc_row.iloc[0]['allocated_qty'])
                
                if source_key not in supply_usage:
                    supply_usage[source_key] = {
                        'source_type': mapping['source_type'],
                        'source_id': mapping['source_id'],
                        'total_allocated': 0,
                        'demand_lines': []
                    }
                
                supply_usage[source_key]['total_allocated'] += allocated_qty
                supply_usage[source_key]['demand_lines'].append(demand_line_id)
            
            # Validate each supply source against GAP data
            for source_key, usage in supply_usage.items():
                # Find supply in GAP data
                source_type = usage['source_type']
                source_id = str(usage['source_id'])
                
                # Filter supply data based on source type and ID
                supply_match = supply_filtered[
                    (supply_filtered['source_type'] == source_type)
                ]
                
                # Additional filtering based on source_id
                if source_type == 'Inventory' and 'inventory_history_id' in supply_match.columns:
                    supply_match = supply_match[supply_match['inventory_history_id'].astype(str) == source_id]
                elif source_type == 'Pending CAN' and 'arrival_note_line_id' in supply_match.columns:
                    supply_match = supply_match[supply_match['arrival_note_line_id'].astype(str) == source_id]
                elif source_type == 'Pending PO':
                    # Handle PO line ID extraction from supply_number
                    if 'supply_number' in supply_match.columns:
                        # supply_number format: "PO123_L456"
                        supply_match['extracted_line_id'] = supply_match['supply_number'].str.extract(r'_L(\d+)')[0]
                        supply_match = supply_match[supply_match['extracted_line_id'] == source_id]
                    elif 'po_line_id' in supply_match.columns:
                        supply_match = supply_match[supply_match['po_line_id'].astype(str) == source_id]
                elif source_type == 'Pending WH Transfer' and 'warehouse_transfer_line_id' in supply_match.columns:
                    supply_match = supply_match[supply_match['warehouse_transfer_line_id'].astype(str) == source_id]
                else:
                    # Fallback: try to match by supply_number
                    if 'supply_number' in supply_match.columns:
                        supply_match = supply_match[supply_match['supply_number'].astype(str) == source_id]
                
                if supply_match.empty:
                    errors.append(
                        f"Supply source {source_type} ID: {source_id} not found in GAP Analysis data"
                    )
                    continue
                
                # Get available quantity from GAP data
                available_qty = float(supply_match['quantity'].sum())
                
                # Check for existing HARD allocations on this supply
                existing_allocations = self._get_existing_hard_allocations(source_type, source_id)
                net_available = available_qty - existing_allocations
                
                if usage['total_allocated'] > net_available:
                    errors.append(
                        f"Over-allocation for {source_type} ID: {source_id}: "
                        f"Allocated {usage['total_allocated']:.2f} but only {net_available:.2f} available "
                        f"(Total: {available_qty:.2f}, Already allocated: {existing_allocations:.2f})"
                    )
            
            return len(errors) == 0, errors
            
        except Exception as e:
            logger.error(f"Error validating HARD allocation: {str(e)}")
            errors.append(f"Validation error: {str(e)}")
            return False, errors


    def _get_existing_hard_allocations(self, source_type: str, source_id: str) -> float:
        """Get existing HARD allocations for a supply source
        
        This still needs to query DB to check for existing allocations
        that might not be visible in current GAP analysis
        """
        try:
            query = text("""
                SELECT SUM(ad.allocated_qty - ad.delivered_qty) as allocated
                FROM allocation_details ad
                JOIN allocation_plans ap ON ad.allocation_plan_id = ap.id
                WHERE ad.supply_source_type = :source_type
                AND ad.supply_source_id = :source_id
                AND ad.allocation_mode = 'HARD'
                AND ad.status = 'ALLOCATED'
                AND ad.allocated_qty > ad.delivered_qty
            """)
            
            with self.engine.connect() as conn:
                result = conn.execute(query, {
                    'source_type': source_type,
                    'source_id': source_id
                })
                row = result.fetchone()
                
                if row and row['allocated'] is not None:
                    return float(row['allocated'])
            
            return 0.0
            
        except Exception as e:
            logger.error(f"Error checking existing allocations: {str(e)}")
            return 0.0