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
    
    def get_allocation_plans(self, status_filter: List[str] = None, 
                       date_from: datetime = None, date_to: datetime = None,
                       search_text: str = None) -> pd.DataFrame:
        """Get allocation plans using v_allocation_plans_summary view"""
        try:
            # Use the view which already calculates display_status and allocation_type
            query = """
                SELECT 
                    ap.*,
                    vaps.draft_count,
                    vaps.allocated_count,
                    vaps.partial_count,
                    vaps.delivered_count,
                    vaps.cancelled_count,
                    vaps.total_count,
                    vaps.allocation_type,
                    vaps.display_status,
                    vaps.total_requested,
                    vaps.total_allocated_original,
                    vaps.total_allocated_effective,
                    vaps.total_delivered,
                    vaps.total_cancelled,
                    vaps.fulfillment_rate,
                    vaps.hard_allocation_count,
                    u.name as creator_name
                FROM allocation_plans ap
                JOIN v_allocation_plans_summary vaps ON ap.id = vaps.id
                LEFT JOIN users u ON ap.creator_id = u.id
                WHERE 1=1
            """
            
            params = {}
            
            # Filter by display_status instead of plan status
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
                    query += " AND vaps.display_status IN :display_statuses"
                    params['display_statuses'] = tuple(display_statuses)
            
            if date_from:
                query += " AND DATE(ap.allocation_date) >= :date_from"
                params['date_from'] = date_from
            
            if date_to:
                query += " AND DATE(ap.allocation_date) <= :date_to"
                params['date_to'] = date_to
            
            if search_text:
                query += """ AND (ap.allocation_number LIKE :search_text 
                            OR ap.notes LIKE :search_text)"""
                params['search_text'] = f"%{search_text}%"
            
            query += " ORDER BY ap.allocation_date DESC"
            
            df = pd.read_sql(text(query), self.engine, params=params)
            return df
            
        except Exception as e:
            logger.error(f"Error getting allocation plans: {str(e)}")
            return pd.DataFrame()

    def get_allocation_details(self, allocation_id: int) -> Tuple[Dict, pd.DataFrame]:
        """Get allocation plan and its details"""
        try:
            # Phần plan_query giữ nguyên
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
            
            # UPDATE details_query để thêm cancellation columns
            details_query = """
                SELECT 
                    ad.*,
                    p.product_name,
                    p.package_size,
                    p.standard_uom,
                    c.name as customer_name,
                    le.name as legal_entity_name,
                    -- Thêm cancellation columns
                    COALESCE(ac.cancelled_qty, 0) as cancelled_qty,
                    COALESCE(ac.cancellation_count, 0) as cancellation_count,
                    (ad.allocated_qty - COALESCE(ac.cancelled_qty, 0)) as effective_allocated_qty,
                    GREATEST(0, ad.allocated_qty - COALESCE(ac.cancelled_qty, 0) - ad.delivered_qty) as cancellable_qty
                FROM allocation_details ad
                LEFT JOIN products p ON ad.product_id = p.id
                LEFT JOIN customers c ON ad.customer_id = c.id
                LEFT JOIN legal_entities le ON ad.legal_entity_id = le.id
                LEFT JOIN (
                    SELECT 
                        allocation_detail_id,
                        SUM(CASE WHEN status = 'ACTIVE' THEN cancelled_qty ELSE 0 END) as cancelled_qty,
                        COUNT(CASE WHEN status = 'ACTIVE' THEN 1 END) as cancellation_count
                    FROM allocation_cancellations
                    GROUP BY allocation_detail_id
                ) ac ON ad.id = ac.allocation_detail_id
                WHERE ad.allocation_plan_id = :allocation_id
                ORDER BY ad.pt_code, ad.allocated_etd
            """
            
            details_df = pd.read_sql(text(details_query), self.engine,
                                params={'allocation_id': allocation_id})
            
            # Thêm cancellation summary vào plan
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
            
            # Convert allocation_context to JSON string
            import json
            allocation_context_json = None
            if 'allocation_context' in plan_data and plan_data['allocation_context']:
                allocation_context_json = json.dumps(plan_data['allocation_context'])
            
            # Insert allocation plan with allocation_context
            plan_insert = text("""
                INSERT INTO allocation_plans 
                (allocation_number, allocation_date, allocation_method, allocation_type,
                status, creator_id, approved_by, approved_date, notes, allocation_context)
                VALUES 
                (:allocation_number, NOW(), :allocation_method, :allocation_type,
                :status, :creator_id, :approved_by, :approved_date, :notes, :allocation_context)
            """)
            
            result = conn.execute(plan_insert, {
                'allocation_number': allocation_number,
                'allocation_method': plan_data.get('allocation_method', 'MANUAL'),
                'allocation_type': plan_data.get('allocation_type', 'SOFT'),
                'status': plan_data.get('status', 'DRAFT'),
                'creator_id': plan_data.get('creator_id', 1),
                'approved_by': plan_data.get('approved_by'),
                'approved_date': plan_data.get('approved_date'),
                'notes': plan_data.get('notes', ''),
                'allocation_context': allocation_context_json
            })
            
            allocation_plan_id = result.lastrowid
            
            # Insert allocation details
            for _, row in allocation_details.iterrows():
                # Check if this is a HARD allocation from supply_mapping
                allocation_mode = 'SOFT'
                supply_source_type = None
                supply_source_id = None
                
                if supply_mapping and str(row.get('demand_line_id', '')) in supply_mapping:
                    allocation_mode = 'HARD'
                    mapping_info = supply_mapping[str(row.get('demand_line_id', ''))]
                    supply_source_type = mapping_info.get('source_type')
                    supply_source_id = mapping_info.get('source_id')
                
                detail_insert = text("""
                    INSERT INTO allocation_details
                    (allocation_plan_id, allocation_mode, demand_type, demand_reference_id,
                    product_id, pt_code, customer_id, customer_name,
                    legal_entity_id, legal_entity_name,
                    requested_qty, allocated_qty, delivered_qty,
                    etd, allocated_etd, status, notes,
                    supply_source_type, supply_source_id)
                    VALUES
                    (:allocation_plan_id, :allocation_mode, :demand_type, :demand_reference_id,
                    :product_id, :pt_code, :customer_id, :customer_name,
                    :legal_entity_id, :legal_entity_name,
                    :requested_qty, :allocated_qty, 0,
                    :etd, :allocated_etd, 'ALLOCATED', :notes,
                    :supply_source_type, :supply_source_id)
                """)
                
                conn.execute(detail_insert, {
                    'allocation_plan_id': allocation_plan_id,
                    'allocation_mode': allocation_mode,
                    'demand_type': row.get('demand_type', 'OC'),
                    'demand_reference_id': row.get('demand_reference_id'),
                    'product_id': row.get('product_id'),
                    'pt_code': row.get('pt_code'),
                    'customer_id': row.get('customer_id'),
                    'customer_name': row.get('customer', row.get('customer_name')),
                    'legal_entity_id': row.get('legal_entity_id'),
                    'legal_entity_name': row.get('legal_entity', row.get('legal_entity_name')),
                    'requested_qty': row.get('requested_qty', row.get('demand_quantity', 0)),
                    'allocated_qty': row.get('allocated_qty', 0),
                    'etd': row.get('etd'),
                    'allocated_etd': row.get('allocated_etd', row.get('etd')),
                    'notes': row.get('notes', ''),
                    'supply_source_type': supply_source_type,
                    'supply_source_id': supply_source_id
                })
            
            trans.commit()
            logger.info(f"Created allocation plan {allocation_number} with ID {allocation_plan_id}")
            
            # Log HARD allocations if any
            if supply_mapping:
                logger.info(f"Created {len(supply_mapping)} HARD allocations")
            
            return allocation_plan_id
            
        except Exception as e:
            trans.rollback()
            logger.error(f"Error creating allocation plan: {str(e)}")
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
        """Cancel allocation plan"""
        try:
            with self.engine.begin() as conn:
                # Update plan status
                plan_update = text("""
                    UPDATE allocation_plans
                    SET status = 'CANCELLED'
                    WHERE id = :allocation_id AND status IN ('DRAFT', 'APPROVED')
                """)
                
                result = conn.execute(plan_update, {'allocation_id': allocation_id})
                
                if result.rowcount > 0:
                    # Update details status
                    detail_update = text("""
                        UPDATE allocation_details
                        SET status = 'CANCELLED'
                        WHERE allocation_plan_id = :allocation_id
                          AND delivered_qty = 0
                    """)
                    
                    conn.execute(detail_update, {'allocation_id': allocation_id})
                    return True
                
            return False
            
        except Exception as e:
            logger.error(f"Error cancelling allocation: {str(e)}")
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
            query = text("""
                SELECT 
                    ad.id as detail_id,
                    ad.pt_code,
                    ad.product_id,
                    p.product_name,
                    ad.customer_name,
                    ad.allocated_qty,
                    ad.delivered_qty,
                    COALESCE(ac.cancelled_qty, 0) as cancelled_qty,
                    GREATEST(0, ad.allocated_qty - COALESCE(ac.cancelled_qty, 0) - ad.delivered_qty) as cancellable_qty,
                    ad.status,
                    ad.allocation_mode
                FROM allocation_details ad
                LEFT JOIN products p ON ad.product_id = p.id
                LEFT JOIN (
                    SELECT 
                        allocation_detail_id,
                        SUM(CASE WHEN status = 'ACTIVE' THEN cancelled_qty ELSE 0 END) as cancelled_qty
                    FROM allocation_cancellations
                    GROUP BY allocation_detail_id
                ) ac ON ad.id = ac.allocation_detail_id
                WHERE ad.allocation_plan_id = :allocation_id
                AND ad.status NOT IN ('DELIVERED', 'CANCELLED')
                AND ad.allocated_qty - COALESCE(ac.cancelled_qty, 0) > ad.delivered_qty
                ORDER BY ad.pt_code, ad.customer_name
            """)
            
            df = pd.read_sql(query, self.engine, params={'allocation_id': allocation_id})
            return df
            
        except Exception as e:
            logger.error(f"Error getting cancellable details: {str(e)}")
            return pd.DataFrame()

    def get_active_allocations_summary(self) -> pd.DataFrame:
        """Get summary of active allocations for supply adjustment"""
        try:
            # UPDATE query để trừ cancelled quantity
            query = """
                SELECT 
                    pt_code,
                    legal_entity_name as legal_entity,
                    SUM(ad.allocated_qty - COALESCE(ac.cancelled_qty, 0) - ad.delivered_qty) as undelivered_qty
                FROM allocation_details ad
                JOIN allocation_plans ap ON ad.allocation_plan_id = ap.id
                LEFT JOIN (
                    SELECT 
                        allocation_detail_id,
                        SUM(CASE WHEN status = 'ACTIVE' THEN cancelled_qty ELSE 0 END) as cancelled_qty
                    FROM allocation_cancellations
                    GROUP BY allocation_detail_id
                ) ac ON ad.id = ac.allocation_detail_id
                WHERE ap.status IN ('APPROVED', 'EXECUTED')
                AND ad.status IN ('ALLOCATED', 'PARTIAL_DELIVERED')
                AND ad.allocated_qty - COALESCE(ac.cancelled_qty, 0) > ad.delivered_qty
                GROUP BY pt_code, legal_entity_name
                HAVING undelivered_qty > 0
            """
            
            df = pd.read_sql(text(query), self.engine)
            return df
            
        except Exception as e:
            logger.error(f"Error getting active allocations: {str(e)}")
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
        """Update all DRAFT details in a plan to new status"""
        conn = self.engine.connect()
        trans = conn.begin()
        
        try:
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

    def update_delivery_status(self, allocation_detail_id: int, 
                             delivered_qty: float, delivery_id: int) -> bool:
        """Update delivery status for allocation detail"""
        conn = self.engine.connect()
        trans = conn.begin()
        
        try:
            # Get current allocation detail
            detail_query = text("""
                SELECT allocated_qty, delivered_qty
                FROM allocation_details
                WHERE id = :detail_id
            """)
            
            detail = conn.execute(detail_query, {'detail_id': allocation_detail_id}).fetchone()
            
            if not detail:
                return False
            
            new_delivered_qty = detail['delivered_qty'] + delivered_qty
            new_status = 'DELIVERED' if new_delivered_qty >= detail['allocated_qty'] else 'PARTIAL_DELIVERED'
            
            # Update allocation detail
            update_detail = text("""
                UPDATE allocation_details
                SET delivered_qty = :delivered_qty,
                    status = :status
                WHERE id = :detail_id
            """)
            
            conn.execute(update_detail, {
                'delivered_qty': new_delivered_qty,
                'status': new_status,
                'detail_id': allocation_detail_id
            })
            
            # Create delivery link
            link_insert = text("""
                INSERT INTO allocation_delivery_links
                (allocation_detail_id, delivery_detail_id, delivered_qty)
                VALUES (:allocation_detail_id, :delivery_detail_id, :delivered_qty)
            """)
            
            conn.execute(link_insert, {
                'allocation_detail_id': allocation_detail_id,
                'delivery_detail_id': delivery_id,
                'delivered_qty': delivered_qty
            })
            
            trans.commit()
            return True
            
        except Exception as e:
            trans.rollback()
            logger.error(f"Error updating delivery status: {str(e)}")
            return False
        finally:
            conn.close()