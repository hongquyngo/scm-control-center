import streamlit as st
import pandas as pd
from typing import Dict, List, Any

class BusinessRulesManager:
    """Manage business rules for allocation and PO suggestions"""
    
    @staticmethod
    def render_allocation_rules():
        """Render allocation rules UI"""
        # Get current settings
        allocation_settings = st.session_state.business_settings.get('allocation_rules', {})
        
        # Basic Rules Section
        st.markdown("#### 📋 Basic Allocation Rules")
        
        # Allocation Method
        current_method = allocation_settings.get('method', 'FIFO')
        allocation_method = st.radio(
            "Allocation Method",
            options=['FIFO', 'FEFO', 'PRIORITY'],
            index=['FIFO', 'FEFO', 'PRIORITY'].index(current_method),
            horizontal=True,
            help="""
            - FIFO: First In First Out (by stock-in date)
            - FEFO: First Expired First Out (by expiry date)
            - PRIORITY: Based on customer priority scores
            """
        )
        
        # Shelf Life Rule
        st.markdown("#### 📅 Shelf Life Requirements")
        min_shelf_life = st.slider(
            "Minimum Shelf Life at Delivery (%)",
            min_value=0,
            max_value=100,
            value=allocation_settings.get('min_shelf_life_percent', 70),
            step=5,
            help="Products must have at least this percentage of shelf life remaining when delivered"
        )
        
        # Show info based on selection
        if allocation_method == 'PRIORITY':
            st.info("⚠️ Customer Priority mode selected. You can set customer priorities below.")
            
            # Customer Priority Section
            st.markdown("#### 👥 Customer Priorities")
            st.markdown("Set priority scores for customers (1-100, higher = more priority)")
            
            # Get current priorities
            current_priorities = allocation_settings.get('customer_priorities', {})
            
            # Simple input for demo (in production, this would be a data editor)
            col1, col2 = st.columns([2, 1])
            with col1:
                st.caption("Customer priorities can be imported from Excel or set individually")
            with col2:
                if st.button("📥 Import from Excel", use_container_width=True):
                    st.info("Import functionality coming soon")
            
            # Example customers (in production, load from database)
            st.markdown("##### Example Priority Settings")
            example_customers = {
                "Customer A": current_priorities.get("Customer A", 100),
                "Customer B": current_priorities.get("Customer B", 80),
                "Customer C": current_priorities.get("Customer C", 60)
            }
            
            updated_priorities = {}
            for customer, default_priority in example_customers.items():
                priority = st.number_input(
                    customer,
                    min_value=1,
                    max_value=100,
                    value=default_priority,
                    key=f"priority_{customer}"
                )
                updated_priorities[customer] = priority
        else:
            updated_priorities = allocation_settings.get('customer_priorities', {})
        
        # Action buttons
        st.markdown("---")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("💾 Save Allocation Rules", type="primary", use_container_width=True):
                # Save to session state
                st.session_state.business_settings['allocation_rules'] = {
                    'method': allocation_method,
                    'min_shelf_life_percent': min_shelf_life,
                    'customer_priorities': updated_priorities
                }
                st.success("✅ Allocation rules saved successfully!")
        
        with col2:
            if st.button("🔄 Reset to Default", use_container_width=True, key="reset_allocation"):
                # Reset to defaults
                st.session_state.business_settings['allocation_rules'] = {
                    'method': 'FIFO',
                    'min_shelf_life_percent': 70,
                    'customer_priorities': {}
                }
                st.rerun()
    
    @staticmethod
    def render_po_rules():
        """Render PO suggestion rules UI"""
        # Get current settings
        po_settings = st.session_state.business_settings.get('po_rules', {})
        
        # Reorder Parameters
        st.markdown("#### 📊 Reorder Parameters")
        
        col1, col2 = st.columns(2)
        
        with col1:
            reorder_point = st.number_input(
                "Reorder Point (days of stock)",
                min_value=1,
                max_value=365,
                value=po_settings.get('reorder_point_days', 30),
                help="Create PO when stock falls below this many days of demand"
            )
            
            min_coverage = st.number_input(
                "Minimum Coverage Days",
                min_value=1,
                max_value=365,
                value=po_settings.get('min_coverage_days', 30),
                help="Minimum days of demand to order"
            )
        
        with col2:
            max_coverage = st.number_input(
                "Maximum Coverage Days",
                min_value=1,
                max_value=365,
                value=po_settings.get('max_coverage_days', 90),
                help="Maximum days of demand to order"
            )
            
            # Validation
            if min_coverage > max_coverage:
                st.error("Minimum coverage cannot be greater than maximum coverage!")
        
        # Order Constraints
        st.markdown("#### 📦 Order Constraints")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            default_moq = st.number_input(
                "Default MOQ",
                min_value=1,
                value=po_settings.get('default_moq', 1),
                help="Default Minimum Order Quantity when product-specific MOQ is not available"
            )
        
        with col2:
            default_spq = st.number_input(
                "Default SPQ",
                min_value=1,
                value=po_settings.get('default_spq', 1),
                help="Default Standard Pack Quantity"
            )
        
        with col3:
            round_to_spq = st.checkbox(
                "Round up to SPQ",
                value=po_settings.get('round_to_spq', True),
                help="Round order quantities up to the nearest SPQ"
            )
        
        # Action buttons
        st.markdown("---")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("💾 Save PO Rules", type="primary", use_container_width=True):
                # Validate before saving
                if min_coverage <= max_coverage:
                    st.session_state.business_settings['po_rules'] = {
                        'reorder_point_days': reorder_point,
                        'min_coverage_days': min_coverage,
                        'max_coverage_days': max_coverage,
                        'default_moq': default_moq,
                        'default_spq': default_spq,
                        'round_to_spq': round_to_spq
                    }
                    st.success("✅ PO rules saved successfully!")
                else:
                    st.error("Please fix validation errors before saving!")
        
        with col2:
            if st.button("🔄 Reset to Default", use_container_width=True, key="reset_po"):
                # Reset to defaults
                st.session_state.business_settings['po_rules'] = {
                    'reorder_point_days': 30,
                    'min_coverage_days': 30,
                    'max_coverage_days': 90,
                    'default_moq': 1,
                    'default_spq': 1,
                    'round_to_spq': True
                }
                st.rerun()