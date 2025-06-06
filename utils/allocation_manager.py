"""
Allocation Manager - Core allocation plan management
"""

import pandas as pd
import numpy as np
from datetime import datetime
from sqlalchemy import text
from typing import List, Dict, Optional, Tuple
import logging

from utils.db import get_db_engine
from utils.formatters import generate_allocation_number

logger = logging.getLogger(__name__)


class AllocationManager:
    """Manages allocation plans and operations"""
    
    def __init__(self):
        self.engine = get_db_engine()
    
    def get_allocation_plans(self, status_filter: List[str] = None, 
                           date_from: datetime = None, date_to: datetime = None,
                           search_text: str = None) -> pd.DataFrame:
        """Get allocation plans with filters"""
        try:
            query = """
                SELECT 
                    ap.*,
                    COUNT(DISTINCT ad.id) as item_count,
                    SUM(ad.allocated_qty) as total_allocated,
                    SUM(ad.delivered_qty) as total_delivered,
                    AVG(ad.delivered_qty / NULLIF(ad.allocated_qty, 0) * 100) as fulfillment_rate,
                    u.name as creator_name
                FROM allocation_plans ap
                LEFT JOIN allocation_details ad ON ap.id = ad.allocation_plan_id
                LEFT JOIN users u ON ap.creator_id = u.id
                WHERE 1=1
            """
            
            params = {}
            
            if status_filter:
                query += " AND ap.status IN :status_filter"
                params['status_filter'] = tuple(status_filter)
            
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
            
            query += """
                GROUP BY ap.id
                ORDER BY ap.allocation_date DESC
            """
            
            df = pd.read_sql(text(query), self.engine, params=params)
            return df
            
        except Exception as e:
            logger.error(f"Error getting allocation plans: {str(e)}")
            return pd.DataFrame()
    
    def get_allocation_details(self, allocation_id: int) -> Tuple[Dict, pd.DataFrame]:
        """Get allocation plan and its details"""
        try:
            # Get plan info
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
            
            # Get details
            details_query = """
                SELECT 
                    ad.*,
                    p.product_name,
                    p.package_size,
                    p.standard_uom,
                    c.name as customer_name,
                    le.name as legal_entity_name
                FROM allocation_details ad
                LEFT JOIN products p ON ad.product_id = p.id
                LEFT JOIN customers c ON ad.customer_id = c.id
                LEFT JOIN legal_entities le ON ad.legal_entity_id = le.id
                WHERE ad.allocation_plan_id = :allocation_id
                ORDER BY ad.pt_code, ad.allocated_etd
            """
            
            details_df = pd.read_sql(text(details_query), self.engine,
                                   params={'allocation_id': allocation_id})
            
            return plan, details_df
            
        except Exception as e:
            logger.error(f"Error getting allocation details: {str(e)}")
            return None, pd.DataFrame()
    
    def create_allocation_plan(self, plan_data: Dict, allocation_details: pd.DataFrame) -> Optional[int]:
        """Create new allocation plan with details"""
        conn = self.engine.connect()
        trans = conn.begin()
        
        try:
            # Generate allocation number
            allocation_number = generate_allocation_number()
            
            # Insert allocation plan
            plan_insert = text("""
                INSERT INTO allocation_plans 
                (allocation_number, allocation_date, allocation_method, status, 
                 creator_id, approved_by, approved_date, notes)
                VALUES 
                (:allocation_number, NOW(), :allocation_method, :status,
                 :creator_id, :approved_by, :approved_date, :notes)
            """)
            
            result = conn.execute(plan_insert, {
                'allocation_number': allocation_number,
                'allocation_method': plan_data.get('allocation_method', 'MANUAL'),
                'status': plan_data.get('status', 'DRAFT'),
                'creator_id': plan_data.get('creator_id', 1),
                'approved_by': plan_data.get('approved_by'),
                'approved_date': plan_data.get('approved_date'),
                'notes': plan_data.get('notes', '')
            })
            
            allocation_plan_id = result.lastrowid
            
            # Insert allocation details
            for _, row in allocation_details.iterrows():
                detail_insert = text("""
                    INSERT INTO allocation_details
                    (allocation_plan_id, demand_type, demand_reference_id,
                     product_id, pt_code, customer_id, customer_name,
                     legal_entity_id, legal_entity_name,
                     requested_qty, allocated_qty, delivered_qty,
                     etd, allocated_etd, status)
                    VALUES
                    (:allocation_plan_id, :demand_type, :demand_reference_id,
                     :product_id, :pt_code, :customer_id, :customer_name,
                     :legal_entity_id, :legal_entity_name,
                     :requested_qty, :allocated_qty, 0,
                     :etd, :allocated_etd, 'ALLOCATED')
                """)
                
                conn.execute(detail_insert, {
                    'allocation_plan_id': allocation_plan_id,
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
                    'allocated_etd': row.get('allocated_etd', row.get('etd'))
                })
            
            trans.commit()
            logger.info(f"Created allocation plan {allocation_number} with ID {allocation_plan_id}")
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
    
    def cancel_allocation(self, allocation_id: int) -> bool:
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
    
    def get_active_allocations_summary(self) -> pd.DataFrame:
        """Get summary of active allocations for supply adjustment"""
        try:
            query = """
                SELECT 
                    pt_code,
                    legal_entity_name as legal_entity,
                    SUM(allocated_qty - delivered_qty) as undelivered_qty
                FROM allocation_details ad
                JOIN allocation_plans ap ON ad.allocation_plan_id = ap.id
                WHERE ap.status IN ('APPROVED', 'EXECUTED')
                  AND ad.status IN ('ALLOCATED', 'PARTIAL_DELIVERED')
                  AND ad.allocated_qty > ad.delivered_qty
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