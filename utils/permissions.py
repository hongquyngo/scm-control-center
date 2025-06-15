# utils/permissions.py
"""
Permission Management Module for SCM Control Center
Simple role-based access control for pages and database operations
"""

import streamlit as st
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)

class PermissionManager:
    """Centralized permission management"""
    
    # Page access mapping by role
    PAGE_ACCESS = {
        'admin': ['all'],  # Full access
        'manager': ['all'],  # Full access
        'user': [  # Limited access - no allocation/settings
            'main',
            '0_login',
            '1_demand_analysis', 
            '2_supply_analysis',
            '3_gap_analysis',
            '5_po_suggestions',
            '7_user_guide'
        ],
        'viewer': [  # Read-only access
            'main',
            '0_login',
            '1_demand_analysis',
            '2_supply_analysis', 
            '3_gap_analysis',
            '5_po_suggestions',
            '7_user_guide'
        ]
    }
    
    # Database operation permissions by role
    DB_PERMISSIONS = {
        'admin': ['create', 'update', 'delete', 'approve'],
        'manager': ['create', 'update', 'approve'],  # No delete
        'user': [],  # View only
        'viewer': []  # View only
    }
    
    # Friendly page names for display
    PAGE_NAMES = {
        'main': 'Dashboard',
        '0_login': 'Login',
        '1_demand_analysis': 'Demand Analysis',
        '2_supply_analysis': 'Supply Analysis',
        '3_gap_analysis': 'GAP Analysis',
        '4_allocation_plan': 'Allocation Plan',
        '5_po_suggestions': 'PO Suggestions',
        '6_data_adjustment_settings': 'Settings',
        '7_user_guide': 'User Guide'
    }
    
    @classmethod
    def check_page_access(cls, page_identifier: str, user_role: str = None) -> bool:
        """
        Check if user role has access to a specific page
        
        Args:
            page_identifier: Page name or identifier (e.g., 'allocation_plan', '4_allocation')
            user_role: User role from session state
            
        Returns:
            bool: True if user has access, False otherwise
        """
        # Get role from session if not provided
        if user_role is None:
            user_role = st.session_state.get('user_role', 'viewer').lower()
        else:
            user_role = user_role.lower()
        
        # Admin and manager have full access
        if user_role in ['admin', 'manager']:
            return True
        
        # Get allowed pages for role
        allowed_pages = cls.PAGE_ACCESS.get(user_role, [])
        
        # Check if 'all' access
        if 'all' in allowed_pages:
            return True
        
        # Normalize page identifier (handle different formats)
        page_key = cls._normalize_page_identifier(page_identifier)
        
        # Check access
        return page_key in allowed_pages
    
    @classmethod
    def check_db_permission(cls, operation: str, user_role: str = None) -> bool:
        """
        Check if user role has permission for database operation
        
        Args:
            operation: Operation type ('create', 'update', 'delete', 'approve')
            user_role: User role from session state
            
        Returns:
            bool: True if user has permission, False otherwise
        """
        # Get role from session if not provided
        if user_role is None:
            user_role = st.session_state.get('user_role', 'viewer').lower()
        else:
            user_role = user_role.lower()
        
        # Get permissions for role
        permissions = cls.DB_PERMISSIONS.get(user_role, [])
        
        # Check permission
        has_permission = operation.lower() in permissions
        
        # Log permission check
        logger.info(f"Permission check: {user_role} - {operation} = {has_permission}")
        
        return has_permission
    
    @classmethod
    def get_accessible_pages(cls, user_role: str = None) -> List[str]:
        """
        Get list of accessible pages for user role
        
        Args:
            user_role: User role from session state
            
        Returns:
            List[str]: List of accessible page identifiers
        """
        # Get role from session if not provided
        if user_role is None:
            user_role = st.session_state.get('user_role', 'viewer').lower()
        else:
            user_role = user_role.lower()
        
        # Get allowed pages
        allowed_pages = cls.PAGE_ACCESS.get(user_role, [])
        
        # Return all pages if 'all' access
        if 'all' in allowed_pages:
            return list(cls.PAGE_NAMES.keys())
        
        return allowed_pages
    
    @classmethod
    def show_permission_error(cls, operation: str = None, page: str = None):
        """
        Display standardized permission error message
        
        Args:
            operation: Database operation attempted
            page: Page access attempted
        """
        user_role = st.session_state.get('user_role', 'viewer')
        
        if operation:
            st.error(f"""
            ❌ **Permission Denied**
            
            Your role **{user_role.upper()}** does not have permission to **{operation}**.
            
            Please contact your administrator if you need this access.
            """)
        elif page:
            page_name = cls.PAGE_NAMES.get(page, page)
            st.warning(f"""
            ⚠️ **Access Restricted**
            
            Your role **{user_role.upper()}** does not have access to **{page_name}**.
            
            Redirecting to dashboard...
            """)
    
    @classmethod
    def require_page_access(cls, page_identifier: str):
        """
        Decorator/function to require page access
        Use at the beginning of each page
        
        Args:
            page_identifier: Page identifier to check
        """
        if not cls.check_page_access(page_identifier):
            cls.show_permission_error(page=page_identifier)
            st.stop()  # Stop execution
            # Note: Caller should handle redirect
    
    @classmethod
    def require_db_permission(cls, operation: str) -> bool:
        """
        Check and display error for database operations
        
        Args:
            operation: Operation to check
            
        Returns:
            bool: True if permitted, False otherwise
        """
        if not cls.check_db_permission(operation):
            cls.show_permission_error(operation=operation)
            return False
        return True
    
    @classmethod
    def _normalize_page_identifier(cls, identifier: str) -> str:
        """
        Normalize page identifier to match PAGE_ACCESS keys
        
        Args:
            identifier: Raw page identifier
            
        Returns:
            str: Normalized identifier
        """
        # Remove .py extension
        identifier = identifier.replace('.py', '')
        
        # Convert different formats
        if 'login' in identifier.lower():
            return '0_login'
        elif 'demand' in identifier.lower():
            return '1_demand_analysis'
        elif 'supply' in identifier.lower():
            return '2_supply_analysis'
        elif 'gap' in identifier.lower():
            return '3_gap_analysis'
        elif 'allocation' in identifier.lower():
            return '4_allocation_plan'
        elif 'po' in identifier.lower():
            return '5_po_suggestions'
        elif 'setting' in identifier.lower() or 'adjustment' in identifier.lower():
            return '6_data_adjustment_settings'
        elif 'guide' in identifier.lower() or 'help' in identifier.lower():
            return '7_user_guide'
        elif 'main' in identifier.lower() or 'dashboard' in identifier.lower():
            return 'main'
        
        # Try to extract number prefix
        parts = identifier.split('_')
        if parts[0].isdigit():
            # Already in correct format
            return identifier.lower()
        
        return identifier.lower()
    
    @classmethod
    def get_user_permissions_summary(cls) -> Dict:
        """
        Get current user's permissions summary for display
        
        Returns:
            Dict with role, accessible pages, and db permissions
        """
        user_role = st.session_state.get('user_role', 'viewer').lower()
        
        return {
            'role': user_role,
            'role_display': user_role.upper(),
            'accessible_pages': cls.get_accessible_pages(user_role),
            'db_permissions': cls.DB_PERMISSIONS.get(user_role, []),
            'has_create': 'create' in cls.DB_PERMISSIONS.get(user_role, []),
            'has_update': 'update' in cls.DB_PERMISSIONS.get(user_role, []),
            'has_delete': 'delete' in cls.DB_PERMISSIONS.get(user_role, []),
            'has_approve': 'approve' in cls.DB_PERMISSIONS.get(user_role, [])
        }

# Convenience functions for direct import
def check_page_access(page_identifier: str, user_role: str = None) -> bool:
    """Check page access permission"""
    return PermissionManager.check_page_access(page_identifier, user_role)

def check_db_permission(operation: str, user_role: str = None) -> bool:
    """Check database operation permission"""
    return PermissionManager.check_db_permission(operation, user_role)

def require_page_access(page_identifier: str):
    """Require page access - stops execution if not permitted"""
    PermissionManager.require_page_access(page_identifier)

def require_db_permission(operation: str) -> bool:
    """Require database permission - shows error if not permitted"""
    return PermissionManager.require_db_permission(operation)