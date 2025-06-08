"""
Allocation Cancellation Manager - Handles partial cancellation and reversal
"""

import pandas as pd
import numpy as np
from datetime import datetime
from sqlalchemy import text
from typing import Dict, List, Optional, Tuple
import logging


from utils.db import get_db_engine

logger = logging.getLogger(__name__)


class AllocationCancellationManager:
    """Manages allocation cancellations with audit trail"""
    
    def __init__(self):
        self.engine = get_db_engine()
    
    def cancel_quantity(self, detail_id: int, quantity: float, 
                       reason: str, reason_category: str, user_id: int) -> Tuple[bool, str]:
        """
        Cancel allocated quantity with audit trail
        
        Returns:
            Tuple[bool, str]: (success, message)
        """
        conn = self.engine.connect()
        trans = conn.begin()
        
        try:
            # Get current allocation detail with cancellation info
            detail_query = text("""
                SELECT 
                    ad.*,
                    ap.allocation_number,
                    COALESCE(SUM(CASE WHEN ac.status = 'ACTIVE' THEN ac.cancelled_qty ELSE 0 END), 0) as total_cancelled
                FROM allocation_details ad
                JOIN allocation_plans ap ON ad.allocation_plan_id = ap.id
                LEFT JOIN allocation_cancellations ac ON ad.id = ac.allocation_detail_id
                WHERE ad.id = :detail_id
                GROUP BY ad.id
            """)
            
            result = conn.execute(detail_query, {'detail_id': detail_id})
            detail = result.fetchone()
            
            if not detail:
                return False, "Allocation detail not found"
            
            # Calculate cancellable quantity
            cancellable = float(detail['allocated_qty']) - float(detail['delivered_qty']) - float(detail['total_cancelled'])
            
            if quantity > cancellable:
                return False, f"Cannot cancel {quantity:.2f}. Only {cancellable:.2f} available to cancel"
            
            if quantity <= 0:
                return False, "Cancel quantity must be greater than 0"
            
            # Insert cancellation record
            cancel_insert = text("""
                INSERT INTO allocation_cancellations 
                (allocation_detail_id, allocation_plan_id, cancelled_qty, 
                 reason, reason_category, cancelled_by_user_id, status)
                VALUES (:detail_id, :plan_id, :qty, :reason, :category, :user_id, 'ACTIVE')
            """)
            
            conn.execute(cancel_insert, {
                'detail_id': detail_id,
                'plan_id': detail['allocation_plan_id'],
                'qty': quantity,
                'reason': reason,
                'category': reason_category,
                'user_id': user_id
            })
            
            # Update detail status if needed
            new_effective_qty = float(detail['allocated_qty']) - float(detail['total_cancelled']) - quantity
            
            if new_effective_qty <= 0:
                # Fully cancelled
                status_update = text("""
                    UPDATE allocation_details 
                    SET status = 'CANCELLED' 
                    WHERE id = :detail_id
                """)
                conn.execute(status_update, {'detail_id': detail_id})
                status_msg = "Allocation fully cancelled"
            elif float(detail['delivered_qty']) > 0 and new_effective_qty > float(detail['delivered_qty']):
                # Partial cancelled but still has undelivered
                status_msg = "Partial cancellation applied"
            else:
                status_msg = "Cancellation applied"
            
            trans.commit()
            
            logger.info(f"Cancelled {quantity} for allocation detail {detail_id}")
            return True, f"{status_msg}. Cancelled quantity: {quantity:.2f}"
            
        except Exception as e:
            trans.rollback()
            logger.error(f"Error cancelling allocation: {str(e)}")
            return False, f"Error: {str(e)}"
        finally:
            conn.close()
    
    def get_cancellable_quantity(self, detail_id: int) -> float:
        """Get quantity available for cancellation"""
        try:
            query = text("""
                SELECT 
                    ad.allocated_qty,
                    ad.delivered_qty,
                    COALESCE(SUM(CASE WHEN ac.status = 'ACTIVE' THEN ac.cancelled_qty ELSE 0 END), 0) as total_cancelled
                FROM allocation_details ad
                LEFT JOIN allocation_cancellations ac ON ad.id = ac.allocation_detail_id
                WHERE ad.id = :detail_id
                GROUP BY ad.id
            """)
            
            with self.engine.connect() as conn:
                result = conn.execute(query, {'detail_id': detail_id})
                row = result.fetchone()
                
                if row:
                    return float(row['allocated_qty']) - float(row['delivered_qty']) - float(row['total_cancelled'])
                return 0.0
                
        except Exception as e:
            logger.error(f"Error getting cancellable quantity: {str(e)}")
            return 0.0
    
    def reverse_cancellation(self, cancellation_id: int, user_id: int, reason: str) -> Tuple[bool, str]:
        """Reverse a cancellation (restore allocated quantity)"""
        conn = self.engine.connect()
        trans = conn.begin()
        
        try:
            # Get cancellation record
            cancel_query = text("""
                SELECT ac.*, ad.delivered_qty, ad.status as detail_status
                FROM allocation_cancellations ac
                JOIN allocation_details ad ON ac.allocation_detail_id = ad.id
                WHERE ac.id = :cancel_id AND ac.status = 'ACTIVE'
            """)
            
            result = conn.execute(cancel_query, {'cancel_id': cancellation_id})
            cancellation = result.fetchone()
            
            if not cancellation:
                return False, "Cancellation not found or already reversed"
            
            # Check if any quantity has been delivered since cancellation
            if float(cancellation['delivered_qty']) > 0:
                return False, "Cannot reverse - some quantity already delivered"
            
            # Reverse the cancellation
            reverse_update = text("""
                UPDATE allocation_cancellations
                SET status = 'REVERSED',
                    reversed_by_user_id = :user_id,
                    reversed_date = NOW(),
                    reversal_reason = :reason
                WHERE id = :cancel_id
            """)
            
            conn.execute(reverse_update, {
                'cancel_id': cancellation_id,
                'user_id': user_id,
                'reason': reason
            })
            
            # Update detail status if it was cancelled
            if cancellation['detail_status'] == 'CANCELLED':
                status_update = text("""
                    UPDATE allocation_details
                    SET status = 'ALLOCATED'
                    WHERE id = :detail_id
                """)
                conn.execute(status_update, {'detail_id': cancellation['allocation_detail_id']})
            
            trans.commit()
            
            logger.info(f"Reversed cancellation {cancellation_id}")
            return True, f"Cancellation reversed. Restored quantity: {cancellation['cancelled_qty']:.2f}"
            
        except Exception as e:
            trans.rollback()
            logger.error(f"Error reversing cancellation: {str(e)}")
            return False, f"Error: {str(e)}"
        finally:
            conn.close()
    
    def get_cancellation_history(self, plan_id: Optional[int] = None, 
                               detail_id: Optional[int] = None) -> pd.DataFrame:
        """Get cancellation history for a plan or detail"""
        try:
            query = """
                SELECT 
                    ac.*,
                    ad.pt_code,
                    ad.customer_name,
                    ad.allocated_qty as original_allocated,
                    ad.delivered_qty,
                    u1.name as cancelled_by,
                    u2.name as reversed_by
                FROM allocation_cancellations ac
                JOIN allocation_details ad ON ac.allocation_detail_id = ad.id
                LEFT JOIN users u1 ON ac.cancelled_by_user_id = u1.id
                LEFT JOIN users u2 ON ac.reversed_by_user_id = u2.id
                WHERE 1=1
            """
            
            params = {}
            
            if plan_id:
                query += " AND ac.allocation_plan_id = :plan_id"
                params['plan_id'] = plan_id
            
            if detail_id:
                query += " AND ac.allocation_detail_id = :detail_id"
                params['detail_id'] = detail_id
            
            query += " ORDER BY ac.cancelled_date DESC"
            
            df = pd.read_sql(text(query), self.engine, params=params)
            
            # Format dates
            date_cols = ['cancelled_date', 'reversed_date']
            for col in date_cols:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col])
            
            return df
            
        except Exception as e:
            logger.error(f"Error getting cancellation history: {str(e)}")
            return pd.DataFrame()
    
    def validate_cancellation(self, detail_id: int, quantity: float) -> Tuple[bool, str]:
        """Validate if cancellation is allowed"""
        try:
            # Get current state
            query = text("""
                SELECT 
                    ad.*,
                    COALESCE(SUM(CASE WHEN ac.status = 'ACTIVE' THEN ac.cancelled_qty ELSE 0 END), 0) as total_cancelled
                FROM allocation_details ad
                LEFT JOIN allocation_cancellations ac ON ad.id = ac.allocation_detail_id
                WHERE ad.id = :detail_id
                GROUP BY ad.id
            """)
            
            with self.engine.connect() as conn:
                result = conn.execute(query, {'detail_id': detail_id})
                detail = result.fetchone()
                
                if not detail:
                    return False, "Allocation detail not found"
                
                if detail['status'] == 'DELIVERED':
                    return False, "Cannot cancel delivered allocation"
                
                cancellable = float(detail['allocated_qty']) - float(detail['delivered_qty']) - float(detail['total_cancelled'])
                
                if quantity > cancellable:
                    return False, f"Requested quantity ({quantity:.2f}) exceeds cancellable amount ({cancellable:.2f})"
                
                if quantity <= 0:
                    return False, "Cancel quantity must be positive"
                
                return True, "Cancellation valid"
                
        except Exception as e:
            logger.error(f"Error validating cancellation: {str(e)}")
            return False, f"Validation error: {str(e)}"
    
    def get_plan_cancellation_summary(self, plan_id: int) -> Dict:
        """Get cancellation summary for a plan"""
        try:
            query = text("""
                SELECT 
                    COUNT(DISTINCT ac.id) as total_cancellations,
                    COUNT(DISTINCT CASE WHEN ac.status = 'ACTIVE' THEN ac.id END) as active_cancellations,
                    COUNT(DISTINCT CASE WHEN ac.status = 'REVERSED' THEN ac.id END) as reversed_cancellations,
                    COALESCE(SUM(CASE WHEN ac.status = 'ACTIVE' THEN ac.cancelled_qty ELSE 0 END), 0) as total_cancelled_qty,
                    COUNT(DISTINCT ac.allocation_detail_id) as affected_lines,
                    COUNT(DISTINCT ad.customer_id) as affected_customers,
                    COUNT(DISTINCT ad.pt_code) as affected_products
                FROM allocation_cancellations ac
                JOIN allocation_details ad ON ac.allocation_detail_id = ad.id
                WHERE ac.allocation_plan_id = :plan_id
            """)
            
            with self.engine.connect() as conn:
                result = conn.execute(query, {'plan_id': plan_id})
                row = result.fetchone()
                
                if row:
                    return {
                        'total_cancellations': row['total_cancellations'],
                        'active_cancellations': row['active_cancellations'],
                        'reversed_cancellations': row['reversed_cancellations'],
                        'total_cancelled_qty': float(row['total_cancelled_qty']),
                        'affected_lines': row['affected_lines'],
                        'affected_customers': row['affected_customers'],
                        'affected_products': row['affected_products']
                    }
                
                return {
                    'total_cancellations': 0,
                    'active_cancellations': 0,
                    'reversed_cancellations': 0,
                    'total_cancelled_qty': 0,
                    'affected_lines': 0,
                    'affected_customers': 0,
                    'affected_products': 0
                }
                
        except Exception as e:
            logger.error(f"Error getting cancellation summary: {str(e)}")
            return {}
    
    def bulk_cancel(self, detail_ids: List[int], reason: str, 
                   reason_category: str, user_id: int) -> Tuple[int, List[str]]:
        """Bulk cancel multiple allocation details"""
        success_count = 0
        errors = []
        
        for detail_id in detail_ids:
            # Get cancellable quantity for each
            cancellable = self.get_cancellable_quantity(detail_id)
            
            if cancellable > 0:
                success, message = self.cancel_quantity(
                    detail_id, cancellable, reason, reason_category, user_id
                )
                
                if success:
                    success_count += 1
                else:
                    errors.append(f"Detail {detail_id}: {message}")
            else:
                errors.append(f"Detail {detail_id}: No cancellable quantity")
        
        return success_count, errors