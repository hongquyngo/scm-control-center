"""
Allocation Manager - Core allocation plan management
"""
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from sqlalchemy import text
from typing import List, Dict, Optional, Tuple
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
        """Get active allocations with enhanced information"""
        try:
            query = """
                SELECT * FROM active_allocations_view
                WHERE 1=1
            """
            
            params = {}
            
            # Status filter
            if status_filter:
                placeholders = ', '.join([f':status_{i}' for i in range(len(status_filter))])
                query += f" AND delivery_status IN ({placeholders})"
                for i, status in enumerate(status_filter):
                    params[f'status_{i}'] = status
            
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
        """Get allocation plans using v_allocation_plans_summary view"""
        try:
            # Base query - extract allocation_method and allocation_type from JSON
            query = """
                SELECT 
                    ap.*,
                    vaps.*,
                    u.name as creator_name,
                    -- Extract from JSON context
                    JSON_UNQUOTE(JSON_EXTRACT(ap.allocation_context, '$.allocation_method')) as allocation_method,
                    JSON_UNQUOTE(JSON_EXTRACT(ap.allocation_context, '$.allocation_type')) as allocation_type
                FROM allocation_plans ap
                JOIN v_allocation_plans_summary vaps ON ap.id = vaps.id
                LEFT JOIN users u ON ap.creator_id = u.id
                WHERE 1=1
            """
            
            params = {}
            
            # Filter by display_status (computed) thay vì plan status
            if status_filter:
                # Map UI status to display_status values
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
                
                if display_statuses:
                    # Fix SQL injection issue - use proper parameterized query
                    placeholders = ','.join(['%s'] * len(display_statuses))
                    query += f" AND vaps.display_status IN ({placeholders})"
                    # SQLAlchemy will handle the list properly
                    params = display_statuses
            
            if date_from:
                query += " AND DATE(ap.allocation_date) >= %(date_from)s"
                if not isinstance(params, list):
                    params['date_from'] = date_from
            
            if date_to:
                query += " AND DATE(ap.allocation_date) <= %(date_to)s"
                if not isinstance(params, list):
                    params['date_to'] = date_to
            
            if search_text:
                query += """ AND (ap.allocation_number LIKE %(search_text)s 
                            OR ap.notes LIKE %(search_text)s)"""
                if not isinstance(params, list):
                    params['search_text'] = f"%{search_text}%"
            
            query += " ORDER BY ap.allocation_date DESC"
            
            # Execute query with proper parameter handling
            if isinstance(params, list):
                # For IN clause with list
                df = pd.read_sql(query, self.engine, params=params)
            else:
                # For named parameters
                df = pd.read_sql(text(query), self.engine, params=params)
            
            return df
            
        except Exception as e:
            logger.error(f"Error getting allocation plans: {str(e)}")
            return pd.DataFrame()

    def get_allocation_details(self, allocation_id: int) -> Tuple[Dict, pd.DataFrame]:
        """Get allocation plan and its details with computed status"""
        try:
            # Get plan header
            plan_query = """
                SELECT ap.*, u.name as creator_name
                FROM allocation_plans ap
                LEFT JOIN users u ON ap.creator_id = u.id
                WHERE ap.id = :allocation_id
            """
            
            plan_df = pd.read_sql(text(plan_query), self.engine, 
                                params={'allocation_id': allocation_id})
            
            if plan_df.empty:
                return None, pd.DataFrame()
            
            plan = plan_df.iloc[0].to_dict()
            
            # Get details with computed status from view
            details_query = """
                SELECT 
                    ads.*,
                    p.product_name,
                    p.package_size,
                    p.standard_uom,
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
                            supply_mapping: Dict = None) -> Optional[int]:
        """Create new allocation plan with details and supply mapping for HARD allocation"""
        conn = self.engine.connect()
        trans = conn.begin()
        
        try:
            # Generate allocation number
            allocation_number = generate_allocation_number()
            
            # Use SCM system user (id=1) as default creator
            creator_id = plan_data.get('creator_id', 1)
            
            # Convert allocation_context to JSON string - handle NaN values
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
                    else:
                        return obj
                
                cleaned_context = clean_for_json(plan_data['allocation_context'])
                allocation_context_json = json.dumps(cleaned_context)
            
            # Insert allocation plan
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
            
            # Pre-load product mapping
            product_map = {}
            if not allocation_details.empty and 'pt_code' in allocation_details.columns:
                pt_codes = allocation_details['pt_code'].dropna().unique().tolist()
                if pt_codes:
                    # Use pandas read_sql to get proper DataFrame
                    product_query = """
                        SELECT id, pt_code 
                        FROM products 
                        WHERE pt_code IN %s
                    """
                    product_df = pd.read_sql(
                        product_query, 
                        conn,
                        params=[tuple(pt_codes)]
                    )
                    product_map = dict(zip(product_df['pt_code'], product_df['id']))
                    logger.info(f"Loaded {len(product_map)} product mappings")
            
            # Insert allocation details
            details_inserted = 0
            failed_rows = []
            
            # Convert DataFrame to list of dicts for easier access
            details_list = allocation_details.to_dict('records')
            
            for idx, row_dict in enumerate(details_list):
                try:
                    # Check if this is a HARD allocation from supply_mapping
                    allocation_mode = 'SOFT'
                    supply_source_type = None
                    supply_source_id = None
                    
                    demand_line_id = row_dict.get('demand_line_id', '')
                    if supply_mapping and str(demand_line_id) in supply_mapping:
                        allocation_mode = 'HARD'
                        mapping_info = supply_mapping[str(demand_line_id)]
                        supply_source_type = mapping_info.get('source_type')
                        supply_source_id = mapping_info.get('source_id')
                    
                    # Map demand type from source_type
                    demand_type = 'OC'  # default
                    if row_dict.get('source_type') == 'Forecast':
                        demand_type = 'FORECAST'
                    
                    # Extract demand reference ID from demand_line_id
                    demand_reference_id = None
                    if demand_line_id and pd.notna(demand_line_id):
                        try:
                            demand_reference_id = int(str(demand_line_id).split('_')[0])
                        except:
                            pass
                    
                    # Get product_id from pt_code using pre-loaded map
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
                    detail_status = 'DRAFT' if plan_data.get('status', 'DRAFT') == 'DRAFT' else 'ALLOCATED'
                    
                    # Clean values before insert
                    def clean_value(val):
                        if pd.isna(val):
                            return None
                        if isinstance(val, (np.floating, float)) and (np.isnan(val) or np.isinf(val)):
                            return None
                        return val
                    
                    # Get dates and ensure proper format
                    etd_value = row_dict.get('etd')
                    allocated_etd_value = row_dict.get('allocated_etd', etd_value)
                    
                    # Convert Timestamp to string
                    if etd_value and pd.notna(etd_value):
                        if isinstance(etd_value, pd.Timestamp):
                            etd_value = etd_value.strftime('%Y-%m-%d')
                        elif isinstance(etd_value, str):
                            # Parse and reformat to ensure YYYY-MM-DD
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
                    
                    # Get quantities with proper cleaning
                    requested_qty = float(clean_value(
                        row_dict.get('requested_qty', row_dict.get('demand_quantity', 0))
                    ) or 0)
                    allocated_qty = float(clean_value(row_dict.get('allocated_qty', 0)) or 0)
                    
                    detail_insert = text("""
                        INSERT INTO allocation_details
                        (allocation_plan_id, allocation_mode, status,
                        demand_type, demand_reference_id, demand_number,
                        product_id, pt_code, customer_id, customer_name,
                        legal_entity_id, legal_entity_name,
                        requested_qty, allocated_qty, delivered_qty,
                        etd, allocated_etd, notes,
                        supply_source_type, supply_source_id)
                        VALUES
                        (:allocation_plan_id, :allocation_mode, :status,
                        :demand_type, :demand_reference_id, :demand_number,
                        :product_id, :pt_code, :customer_id, :customer_name,
                        :legal_entity_id, :legal_entity_name,
                        :requested_qty, :allocated_qty, 0,
                        :etd, :allocated_etd, :notes,
                        :supply_source_type, :supply_source_id)
                    """)
                    
                    detail_params = {
                        'allocation_plan_id': allocation_plan_id,
                        'allocation_mode': allocation_mode,
                        'status': detail_status,
                        'demand_type': demand_type,
                        'demand_reference_id': demand_reference_id,
                        'demand_number': clean_value(row_dict.get('demand_number', '')),
                        'product_id': int(product_id),
                        'pt_code': pt_code_str,
                        'customer_id': None,  # GAP data doesn't have customer_id
                        'customer_name': clean_value(row_dict.get('customer', row_dict.get('customer_name', ''))),
                        'legal_entity_id': None,  # GAP data doesn't have legal_entity_id
                        'legal_entity_name': clean_value(row_dict.get('legal_entity', row_dict.get('legal_entity_name', ''))),
                        'requested_qty': requested_qty,
                        'allocated_qty': allocated_qty,
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
            
            if details_inserted == 0:
                raise ValueError("No allocation details were inserted successfully")
            
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
            
            # Log debug info
            if not allocation_details.empty:
                logger.error(f"Allocation details columns: {allocation_details.columns.tolist()}")
                logger.error(f"First row sample: {allocation_details.iloc[0].to_dict() if len(allocation_details) > 0 else 'No rows'}")
            
            return None
        finally:
            conn.close()

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
    
    def cancel_allocation_plan(self, allocation_id: int) -> bool:
        """Cancel allocation plan by creating cancellation records
        
        This creates cancellation records for all undelivered quantities
        instead of updating status fields (which no longer exist)
        """
        try:
            with self.engine.begin() as conn:
                # Get undelivered quantities using view
                details_query = text("""
                    SELECT 
                        id as detail_id,
                        allocation_plan_id,
                        remaining_qty as cancellable_qty
                    FROM allocation_delivery_status_view
                    WHERE allocation_plan_id = :allocation_id
                    AND detail_status = 'ALLOCATED'  -- Only ALLOCATED can be cancelled
                    AND remaining_qty > 0
                """)
                
                details = conn.execute(details_query, {'allocation_id': allocation_id}).fetchall()
                
                if not details:
                    logger.info(f"No cancellable details found for plan {allocation_id}")
                    return True
                
                # Bulk insert cancellations
                for detail in details:
                    if detail['cancellable_qty'] > 0:
                        cancel_insert = text("""
                            INSERT INTO allocation_cancellations 
                            (allocation_detail_id, allocation_plan_id, cancelled_qty, 
                            reason, reason_category, cancelled_by_user_id, cancelled_by_name, status)
                            VALUES (:detail_id, :plan_id, :qty, 
                                    'Plan cancelled', 'BUSINESS_DECISION', 1, 'System', 'ACTIVE')
                        """)
                        
                        conn.execute(cancel_insert, {
                            'detail_id': detail['detail_id'],
                            'plan_id': allocation_id,
                            'qty': detail['cancellable_qty']
                        })
                
                logger.info(f"Created {len(details)} cancellation records for plan {allocation_id}")
                return True
                
        except Exception as e:
            logger.error(f"Error cancelling allocation plan: {str(e)}")
            return False

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
        
        Note: Chỉ update từ DRAFT -> ALLOCATED
        Các status khác (PARTIAL_DELIVERED, DELIVERED, CANCELLED) là computed
        """
        conn = self.engine.connect()
        trans = conn.begin()
        
        try:
            # Chỉ cho phép update DRAFT -> ALLOCATED
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
            
            logger.info(f"Updated {result.rowcount} details to {new_status} for plan {allocation_id}")
            return result.rowcount > 0
            
        except Exception as e:
            trans.rollback()
            logger.error(f"Error updating allocation status: {str(e)}")
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