import streamlit as st
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from utils.auth import AuthManager

# Authentication check
auth_manager = AuthManager()
if not auth_manager.check_session():
    st.switch_page("pages/0_🔐_Login.py")
    st.stop()

import pandas as pd
from datetime import datetime, timedelta
import numpy as np
from sqlalchemy import text
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict
import logging

# Import modules
from utils.display_components import DisplayComponents
from utils.formatters import format_number, generate_allocation_number
from utils.helpers import (
     get_from_session_state,
    convert_df_to_excel, export_multiple_sheets
)
from utils.session_state import initialize_session_state

# Import allocation specific modules
from utils.allocation_manager import AllocationManager
from utils.allocation_methods import AllocationMethods
from utils.allocation_components import AllocationComponents
from utils.allocation_validators import AllocationValidator

# Import allocation wizard components
from utils.allocation_wizard_components import (
    get_allocation_data,
    prepare_products_data,
    show_no_data_message,
    show_no_products_message,
    show_filter_options,
    apply_smart_filters,
    show_no_filtered_data_message,
    prepare_product_summary,
    show_summary_metrics,
    ensure_valid_current_page,
    show_pagination_controls,
    show_product_selection,
    show_step1_next_button
)

# Configure logging
logger = logging.getLogger(__name__)


# === Functions Implementation ===

# Update the show_allocation_list function in 4_🧩_Allocation_Plan.py

def show_allocation_list():
    """Display allocation management with tabs"""
    # Header with refresh button
    header_col1, header_col2 = st.columns([4, 1])
    
    with header_col1:
        st.markdown("### 📋 Allocation Management")
    
    with header_col2:
        if st.button("🔄 Refresh", help="Refresh allocation data", key="refresh_allocation_tabs"):
            st.rerun()
    
    # Create tabs
    tab1, tab2 = st.tabs(["🔥 Active Allocations", "📋 All Plans"])
    
    with tab1:
        show_active_allocations()
        
    with tab2:
        show_all_allocation_plans()

def show_active_allocations():
    """Display active allocations list"""
    # Filters
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        status_filter = st.multiselect(
            "Delivery Status", 
            options=['ALLOCATED', 'PARTIAL_DELIVERED', 'DELIVERED', 'CANCELLED'],
            default=['ALLOCATED', 'PARTIAL_DELIVERED']
        )
    
    with col2:
        # Get current year
        current_year = datetime.now().year
        # First day of current year
        year_start = datetime(current_year, 1, 1)
        date_from = st.date_input("ETD From", value=year_start)
    
    with col3:
        year_end = datetime(current_year, 12, 31)
        date_to = st.date_input("ETD To", value=year_end)
    
    with col4:
        search_text = st.text_input("Search", placeholder="Product, Customer, Allocation #")
    
    # Load active allocations
    allocations = allocation_manager.get_active_allocations(
        status_filter=status_filter,
        date_from=date_from,
        date_to=date_to,
        search_text=search_text
    )
    
    if allocations.empty:
        st.info("No active allocations found")
        return
    
    # Summary metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Total Items", len(allocations))
    with col2:
        allocated_count = len(allocations[allocations['delivery_status'] == 'ALLOCATED'])
        st.metric("Pending", allocated_count)
    with col3:
        partial_count = len(allocations[allocations['delivery_status'] == 'PARTIAL_DELIVERED'])
        st.metric("In Progress", partial_count)
    with col4:
        delivered_count = len(allocations[allocations['delivery_status'] == 'DELIVERED'])
        st.metric("Delivered", delivered_count)
    with col5:
        total_undelivered = allocations['undelivered_qty'].sum()
        st.metric("Undelivered Qty", format_number(total_undelivered))
    
    # Group by allocation plan for display
    plan_groups = allocations.groupby(['allocation_number', 'allocation_date', 'creator_name', 'allocation_plan_id'])
    
    for (allocation_number, allocation_date, creator_name, plan_id), group in plan_groups:
        with st.expander(f"📦 {allocation_number} - {allocation_date.strftime('%Y-%m-%d')} by {creator_name}", expanded=False):
            # Plan summary
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.write(f"**Method:** {group.iloc[0]['allocation_method']}")
                st.write(f"**Items:** {len(group)}")
            with col2:
                st.write(f"**Total Allocated:** {format_number(group['total_allocated_qty'].sum())}")
                st.write(f"**Delivered:** {format_number(group['total_delivered_qty'].sum())}")
            with col3:
                avg_fulfillment = group['avg_fulfillment_rate'].mean()
                st.write(f"**Fulfillment:** {avg_fulfillment:.1f}%")
                st.write(f"**Undelivered:** {format_number(group['undelivered_qty'].sum())}")
            with col4:
                if st.button("👁️ View Plan", key=f"view_active_{plan_id}"):
                    st.session_state['allocation_mode'] = 'view'
                    st.session_state['selected_allocation_id'] = plan_id
                    st.rerun()
            
            # Show allocation details
            display_cols = ['pt_code', 'customer_name', 'allocated_etd', 
                           'total_allocated_qty', 'total_delivered_qty', 
                           'undelivered_qty', 'delivery_status']
            
            display_df = group[display_cols].copy()
            display_df['allocated_etd'] = pd.to_datetime(display_df['allocated_etd']).dt.strftime('%Y-%m-%d')
            
            # Format quantities
            for col in ['total_allocated_qty', 'total_delivered_qty', 'undelivered_qty']:
                display_df[col] = display_df[col].apply(format_number)
            
            # Status color coding
            status_colors = {
                'ALLOCATED': '🔵',
                'PARTIAL_DELIVERED': '🟡',
                'DELIVERED': '🟢',
                'CANCELLED': '🔴'
            }
            display_df['delivery_status'] = display_df['delivery_status'].apply(
                lambda x: f"{status_colors.get(x, '')} {x}"
            )
            
            # Rename columns for display
            display_df = display_df.rename(columns={
                'pt_code': 'Product',
                'customer_name': 'Customer',
                'allocated_etd': 'ETD',
                'total_allocated_qty': 'Allocated',
                'total_delivered_qty': 'Delivered',
                'undelivered_qty': 'Pending',
                'delivery_status': 'Status'
            })
            
            st.dataframe(display_df, use_container_width=True, hide_index=True)

def show_all_allocation_plans():
    """Display all allocation plans"""
    # Filters
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        status_filter = st.multiselect(
            "Plan Status", 
            options=['DRAFT', 'ALLOCATED', 'PARTIAL_DELIVERED', 'DELIVERED', 'CANCELLED'],
            default=['DRAFT', 'ALLOCATED', 'PARTIAL_DELIVERED']
        )
    
    with col2:
        date_from = st.date_input("Created From", value=datetime.now() - timedelta(days=30), key="plan_date_from")
    
    with col3:
        date_to = st.date_input("Created To", value=datetime.now(), key="plan_date_to")
    
    with col4:
        search_text = st.text_input("Search", placeholder="Allocation number or notes", key="plan_search")
    
    # Load allocation plans
    allocations = allocation_manager.get_allocation_plans(
        status_filter=status_filter,
        date_from=date_from,
        date_to=date_to,
        search_text=search_text
    )
    
    if allocations.empty:
        st.info("No allocation plans found")
        return
    
    # Summary metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Total Plans", len(allocations))
    with col2:
        draft_count = len(allocations[allocations['display_status'] == 'ALL_DRAFT'])
        st.metric("Draft", draft_count)
    with col3:
        in_progress = len(allocations[allocations['display_status'] == 'IN_PROGRESS'])
        st.metric("In Progress", in_progress)
    with col4:
        delivered = len(allocations[allocations['display_status'] == 'ALL_DELIVERED'])
        st.metric("Delivered", delivered)
    with col5:
        cancelled = len(allocations[allocations['display_status'] == 'ALL_CANCELLED'])
        st.metric("Cancelled", cancelled)
    
    # Display table with actions
    for idx, row in allocations.iterrows():
        with st.container():
            col1, col2, col3, col4, col5, col6 = st.columns([2, 2, 2, 1, 1, 2])
            
            with col1:
                st.write(f"**{row['allocation_number']}**")
                allocation_type = row.get('allocation_type', 'SOFT')
                type_icon = {'SOFT': '🌊', 'HARD': '🔒', 'MIXED': '🔀'}.get(allocation_type, '❓')
                st.caption(f"{row['allocation_method']} | {type_icon} {allocation_type}")
            
            with col2:
                st.write(f"Date: {row['allocation_date'].strftime('%Y-%m-%d')}")
                st.caption(f"Items: {row.get('total_count', 0)}")
                if row.get('hard_allocation_count', 0) > 0:
                    st.caption(f"🔒 HARD: {row['hard_allocation_count']}")
            
            with col3:
                # Map display_status to color
                status_colors = {
                    'ALL_DRAFT': 'gray',
                    'IN_PROGRESS': 'blue',
                    'ALL_DELIVERED': 'green',
                    'ALL_CANCELLED': 'red',
                    'MIXED_DRAFT': 'orange',
                    'MIXED': 'violet'
                }
                status_color = status_colors.get(row['display_status'], 'gray')
                
                # Display user-friendly status
                status_display = {
                    'ALL_DRAFT': 'Draft',
                    'IN_PROGRESS': 'In Progress',
                    'ALL_DELIVERED': 'Delivered',
                    'ALL_CANCELLED': 'Cancelled',
                    'MIXED_DRAFT': 'Mixed (Draft)',
                    'MIXED': 'Mixed Status',
                    'EMPTY': 'Empty'
                }.get(row['display_status'], row['display_status'])
                
                st.markdown(f"Status: :{status_color}[{status_display}]")
                
                # Show fulfillment rate if in progress or delivered
                if row['display_status'] in ['IN_PROGRESS', 'ALL_DELIVERED', 'MIXED']:
                    st.caption(f"Fulfillment: {row.get('fulfillment_rate', 0):.1f}%")
            
            with col4:
                if st.button("👁️ View", key=f"view_plan_{row['id']}"):
                    st.session_state['allocation_mode'] = 'view'
                    st.session_state['selected_allocation_id'] = row['id']
                    st.rerun()
            
            with col5:
                # Only allow edit for draft plans
                if row['display_status'] in ['ALL_DRAFT', 'MIXED_DRAFT']:
                    if st.button("✏️ Edit", key=f"edit_{row['id']}"):
                        st.session_state['allocation_mode'] = 'edit'
                        st.session_state['selected_allocation_id'] = row['id']
                        st.rerun()

            with col6:
                # Actions based on status
                if row['display_status'] == 'ALL_DRAFT':
                    col6_1, col6_2 = st.columns(2)
                    with col6_1:
                        if st.button("✅ Allocate", key=f"allocate_{row['id']}", type="primary"):
                            # Validate before allocating
                            is_valid, errors = allocation_manager.validate_before_allocation(row['id'])
                            
                            if not is_valid:
                                st.session_state[f'allocate_errors_{row["id"]}'] = errors
                            else:
                                # Set flag to show confirmation
                                st.session_state[f'show_allocate_confirm_{row["id"]}'] = True
                            st.rerun()
                    
                    with col6_2:
                        if st.button("❌ Cancel", key=f"cancel_plan_{row['id']}"):
                            # Set flag to show cancel confirmation
                            st.session_state[f'show_cancel_confirm_{row["id"]}'] = True
                            st.rerun()

                elif row['display_status'] in ['IN_PROGRESS', 'MIXED']:
                    col6_1, col6_2 = st.columns(2)
                    with col6_1:
                        # Show delivery progress
                        delivered_pct = (row.get('total_delivered', 0) / row.get('total_allocated_effective', 1) * 100) if row.get('total_allocated_effective', 0) > 0 else 0
                        st.progress(delivered_pct / 100)
                        st.caption(f"{delivered_pct:.0f}% delivered")
                    with col6_2:
                        # Add cancel button for IN_PROGRESS plans
                        if st.button("❌ Cancel", key=f"cancel_plan_{row['id']}", type="secondary"):
                            st.session_state[f'show_cancel_confirm_{row["id"]}'] = True
                            st.rerun()

            st.divider()

    # Handle error messages (outside the columns)
    for idx, row in allocations.iterrows():
        error_key = f'allocate_errors_{row["id"]}'
        if error_key in st.session_state:
            with st.container():
                st.error(f"Cannot allocate plan {row['allocation_number']}:")
                for error in st.session_state[error_key]:
                    st.write(f"- {error}")
                if st.button("OK", key=f"error_ok_{row['id']}"):
                    del st.session_state[error_key]
                    st.rerun()
                st.markdown("---")

    # Show allocation confirmation dialog if needed (outside the columns)
    for idx, row in allocations.iterrows():
        show_confirm_key = f'show_allocate_confirm_{row["id"]}'
        if st.session_state.get(show_confirm_key, False):
            with st.container():
                st.markdown("---")
                st.warning(f"⚠️ Confirm Allocation for {row['allocation_number']}")
                st.write("Once allocated, the plan cannot be edited. Only cancellation is allowed.")
                st.write(f"Allocating {row.get('total_count', 0)} items")
                
                confirm_col1, confirm_col2, confirm_col3 = st.columns([1, 1, 1])
                with confirm_col1:
                    if st.button("Confirm Allocate", key=f"confirm_alloc_{row['id']}", type="primary"):
                        if allocation_manager.bulk_update_allocation_status(row['id'], 'ALLOCATED'):
                            st.success("✅ Plan allocated successfully!")
                            # Clear the confirmation flag
                            del st.session_state[show_confirm_key]
                            
                            # Optional: Send notifications
                            # notification_manager.notify_allocation_approved(row['id'])
                            
                            st.rerun()
                        else:
                            st.error("❌ Failed to allocate plan. Please check logs.")
                
                with confirm_col2:
                    if st.button("Cancel", key=f"cancel_alloc_{row['id']}"):
                        # Clear the confirmation flag
                        del st.session_state[show_confirm_key]
                        st.rerun()
                st.markdown("---")
    
    # Show cancel confirmation dialog if needed (outside the columns)
    for idx, row in allocations.iterrows():
        show_cancel_key = f'show_cancel_confirm_{row["id"]}'
        if st.session_state.get(show_cancel_key, False):
            with st.container():
                st.markdown("---")
                # Validate first
                can_cancel, impact = allocation_manager.validate_before_cancel(row['id'])
                
                if not can_cancel:
                    st.error("Cannot cancel this plan:")
                    for warning in impact.get('warnings', []):
                        st.write(f"- {warning}")
                    if st.button("OK", key=f"cancel_error_ok_{row['id']}"):
                        del st.session_state[show_cancel_key]
                        st.rerun()
                else:
                    st.warning(f"⚠️ Cancel Allocation Plan {row['allocation_number']}")
                    
                    # Show impact
                    impact_col1, impact_col2 = st.columns(2)
                    with impact_col1:
                        st.metric("Items to Cancel", impact['impact']['allocated_items_to_cancel'])
                        st.metric("Draft Items to Delete", impact['impact']['draft_items_to_delete'])
                    with impact_col2:
                        st.metric("Quantity to Release", format_number(impact['impact']['quantity_to_release']))
                        st.metric("Affected Customers", impact['impact']['affected_customers'])
                    
                    # Warnings
                    if impact['warnings']:
                        st.info("⚠️ Warnings:")
                        for warning in impact['warnings']:
                            st.write(f"- {warning}")
                    
                    # Reason input
                    reason = st.text_area("Cancellation Reason", 
                                        placeholder="Please provide reason for cancellation...",
                                        key=f"cancel_reason_{row['id']}")
                    
                    # Action buttons
                    cancel_col1, cancel_col2 = st.columns(2)
                    with cancel_col1:
                        if st.button("Confirm Cancel", key=f"confirm_cancel_{row['id']}", type="primary"):
                            if allocation_manager.cancel_allocation_plan(
                                row['id'], 
                                reason or "Cancelled by user",
                                user_id=st.session_state.get('user_id', 1)
                            ):
                                st.success("✅ Plan cancelled successfully")
                                del st.session_state[show_cancel_key]
                                st.rerun()
                            else:
                                st.error("❌ Failed to cancel plan")
                    
                    with cancel_col2:
                        if st.button("Keep Plan", key=f"keep_plan_{row['id']}"):
                            del st.session_state[show_cancel_key]
                            st.rerun()
                st.markdown("---")

def show_create_allocation_wizard():
    """Multi-step wizard for creating allocation plan"""
    st.markdown("### ➕ Create New Allocation Plan")
    
    # Initialize draft_allocation if not exists
    if 'draft_allocation' not in st.session_state or st.session_state['draft_allocation'] is None:
        st.session_state['draft_allocation'] = {}
    
    # Adjust total steps based on allocation type
    allocation_type = st.session_state['draft_allocation'].get('allocation_type', 'SOFT')
    total_steps = 6 if allocation_type == 'HARD' else 5
    
    # Progress indicator
    progress = st.session_state['allocation_step'] / total_steps
    st.progress(progress)
    
    # Step navigation
    col1, col2, col3 = st.columns([1, 3, 1])
    with col1:
        if st.session_state['allocation_step'] > 1:
            if st.button("⬅️ Previous"):
                st.session_state['allocation_step'] -= 1
                st.rerun()
    
    with col2:
        if allocation_type == 'HARD':
            steps = ['Select Products', 'Choose Method', 'Set Parameters', 'Preview & Adjust', 'Map Supply', 'Confirm']
        else:
            steps = ['Select Products', 'Choose Method', 'Set Parameters', 'Preview & Adjust', 'Confirm']
        current_step_idx = min(st.session_state['allocation_step'] - 1, len(steps) - 1)
        st.markdown(f"**Step {st.session_state['allocation_step']}/{total_steps}: {steps[current_step_idx]}**")
    
    # Step content
    if st.session_state['allocation_step'] == 1:
        show_step1_select_products()
    elif st.session_state['allocation_step'] == 2:
        show_step2_choose_method()
    elif st.session_state['allocation_step'] == 3:
        show_step3_set_parameters()
    elif st.session_state['allocation_step'] == 4:
        show_step4_preview()  # Updated with zero allocation handling
    elif st.session_state['allocation_step'] == 5:
        if allocation_type == 'HARD':
            show_step5_map_supply()  # Updated with filtering
        else:
            show_step5_confirm()  # Updated with excluded lines info
    elif st.session_state['allocation_step'] == 6:
        show_step6_final_confirm()  # Updated with excluded lines info

def show_step1_select_products():
    """Step 1: Select products for allocation with pagination and smart filters"""
    st.markdown("#### 📦 Select Products for Allocation")
    
    # Initialize product_summary at the very beginning
    product_summary = pd.DataFrame()
    
    try:
        # Get and validate data
        gap_data, demand_filtered, supply_filtered = get_allocation_data()
        
        if gap_data.empty:
            show_no_data_message()
            return
        
        # Debug GAP data if debug mode is on
        if st.session_state.get('debug_mode', False):
            with st.expander("🐛 Debug - GAP Data Info", expanded=False):
                st.write(f"**GAP data shape:** {gap_data.shape}")
                st.write(f"**Columns:** {list(gap_data.columns)[:10]}...")  # Show first 10
                
                # Check for required columns
                required_cols = ['pt_code', 'total_demand_qty', 'total_available']
                missing_cols = [col for col in required_cols if col not in gap_data.columns]
                if missing_cols:
                    st.error(f"Missing required columns: {missing_cols}")
                
                # Sample data
                if not gap_data.empty:
                    st.write("**Sample GAP data (first 3 rows):**")
                    display_cols = [col for col in ['pt_code', 'period', 'total_demand_qty', 'total_available', 'gap_quantity'] 
                                   if col in gap_data.columns]
                    st.dataframe(gap_data[display_cols].head(3))
        
        # Calculate fulfillment and filter products
        products_with_supply = prepare_products_data(gap_data, demand_filtered, supply_filtered)
        
        if products_with_supply.empty:
            show_no_products_message()
            return
        
        # Show filter options (checkbox and items per page only)
        use_smart_filters, items_per_page = show_filter_options()
        
        # Start with all products
        filtered_data = products_with_supply.copy()
        filter_type = 'All'  # Default
        
        # Apply filters based on settings
        if use_smart_filters:
            # apply_smart_filters now returns both filtered_data and filter_type
            filtered_data, filter_type = apply_smart_filters(filtered_data, products_with_supply)
        
        # Check if we have data after filters
        if filtered_data.empty:
            show_no_filtered_data_message(filter_type, use_smart_filters)
            return
        
        # Prepare product summary - THIS IS WHERE product_summary GETS ASSIGNED
        product_summary = prepare_product_summary(filtered_data)
        
        # Check if product_summary is valid
        if product_summary.empty:
            st.warning("No products available after processing. This might be due to:")
            st.write("- All products have 0 demand")
            st.write("- Data processing error")
            
            if st.session_state.get('debug_mode', False):
                st.write("🐛 Debug - filtered_data shape:", filtered_data.shape)
                st.write("🐛 Debug - product_summary shape:", product_summary.shape)
            return
        
        # Show summary metrics
        show_summary_metrics(product_summary)
        
        # Handle pagination
        total_products = len(product_summary)
        total_pages = max(1, (total_products + items_per_page - 1) // items_per_page)
        ensure_valid_current_page(total_pages)
        
        # Show pagination controls
        show_pagination_controls(total_pages, total_products)
        
        # Get current page data
        start_idx = (st.session_state.get('alloc_current_page', 1) - 1) * items_per_page
        end_idx = min(start_idx + items_per_page, total_products)
        page_products = product_summary.iloc[start_idx:end_idx]
        
        # Show selection interface
        selected_products = show_product_selection(page_products, filtered_data)
        
        # Show next button
        show_step1_next_button(selected_products)
        
    except Exception as e:
        st.error(f"Error in product selection: {str(e)}")
        
        if st.session_state.get('debug_mode', False):
            import traceback
            with st.expander("🐛 Full Error Traceback", expanded=True):
                st.code(traceback.format_exc())
                
                # Show session state info
                st.write("**Relevant Session State:**")
                relevant_keys = ['gap_analysis_result', 'demand_filtered', 'supply_filtered', 
                               'gap_analysis_data', 'selected_allocation_products']
                for key in relevant_keys:
                    if key in st.session_state:
                        value = st.session_state[key]
                        if isinstance(value, pd.DataFrame):
                            st.write(f"- {key}: DataFrame with shape {value.shape}")
                        elif value is None:
                            st.write(f"- {key}: None")
                        else:
                            st.write(f"- {key}: {type(value).__name__}")
                    else:
                        st.write(f"- {key}: Not in session state")

def show_step2_choose_method():
    """Step 2: Choose allocation method and type"""
    st.markdown("#### 🎯 Choose Allocation Method & Type")

    # Add general info
    st.info("""
    💡 **Important**: All allocation methods provide initial calculations to help you distribute quantities. 
    You can always manually adjust the allocated amounts in the preview step before saving.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Allocation Method
        st.markdown("##### Allocation Method")
        selected_method = st.radio(
            "Select allocation method:",
            options=list(ALLOCATION_METHODS.keys()),
            format_func=lambda x: ALLOCATION_METHODS[x],
            index=0
        )
        
        # Method description
        st.info(AllocationComponents.get_method_description(selected_method))
    
    with col2:
        # Allocation Type
        st.markdown("##### Allocation Type")
        allocation_type = st.radio(
            "Select allocation type:",
            options=['SOFT', 'HARD'],  # REMOVED MIXED
            format_func=lambda x: ALLOCATION_TYPES[x],
            index=0
        )
        
        # Type description
        if allocation_type == 'SOFT':
            st.info("""
            **Soft Allocation** (Recommended for 90% cases)
            - Allocates quantities only
            - Flexible fulfillment at delivery time
            - System chooses best available supply
            - Easy to manage and adjust
            """)
        elif allocation_type == 'HARD':
            st.info("""
            **Hard Allocation** (Special cases only)
            - Locks specific supply batches to orders
            - Required for origin/quality requirements
            - Cannot be changed after approval
            - Use only when customer needs specific batches
            """)
        # REMOVED MIXED description
    
    # Store selections
    if 'draft_allocation' not in st.session_state or st.session_state['draft_allocation'] is None:
        st.session_state['draft_allocation'] = {}
    
    st.session_state['draft_allocation']['method'] = selected_method
    st.session_state['draft_allocation']['allocation_type'] = allocation_type
    
    # Next button
    st.markdown("---")
    col1, col2, col3 = st.columns([2, 1, 2])
    with col2:
        if st.button("Next ➡️", type="primary", use_container_width=True):
            st.session_state['allocation_step'] = 3
            st.rerun()


def show_step3_set_parameters():
    """Step 3: Set allocation parameters based on method"""
    st.markdown("#### ⚙️ Set Allocation Parameters")
    
    method = st.session_state['draft_allocation'].get('method', 'FCFS')
    
    # Get demand details for selected products
    gap_data = get_from_session_state('gap_analysis_result', pd.DataFrame())
    demand_data = get_from_session_state('demand_filtered', pd.DataFrame())
    supply_data = get_from_session_state('supply_filtered', pd.DataFrame())
    
    selected_products = st.session_state.get('selected_allocation_products', [])
    
    # Filter data for selected products
    filtered_demand = demand_data[demand_data['pt_code'].isin(selected_products)].copy()
    filtered_supply = supply_data[supply_data['pt_code'].isin(selected_products)].copy()
    
    # Show parameters based on method
    parameters = {}
    
    if method == 'FCFS':
        st.info("📅 Orders will be allocated based on earliest ETD first (First Come First Served)")
        parameters['method_params'] = {'sort_by': 'etd', 'ascending': True}
        
    elif method == 'PRIORITY':
        st.markdown("##### Set Customer Priorities")
        
        # Get unique customers
        customers = filtered_demand['customer'].unique()
        
        # Priority input
        customer_priorities = {}
        for customer in customers:
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(customer)
            with col2:
                priority = st.number_input(
                    "Priority",
                    min_value=1,
                    max_value=10,
                    value=5,
                    key=f"priority_{customer}"
                )
                customer_priorities[customer] = priority
        
        parameters['method_params'] = {'customer_priorities': customer_priorities}
        
    elif method == 'PRO_RATA':
        st.info("⚖️ Supply will be distributed proportionally based on demand quantity")
        
        # Option to set minimum allocation percentage
        min_allocation = st.slider(
            "Minimum allocation % per order",
            min_value=0,
            max_value=100,
            value=50,
            help="Orders will receive at least this percentage of their demand if possible"
        )
        parameters['method_params'] = {'min_allocation_percent': min_allocation}
        
    elif method == 'MANUAL':
        st.info("✋ You will manually adjust allocations in the next step")
        parameters['method_params'] = {}
    
    # Additional parameters
    st.markdown("##### Additional Settings")
    
    col1, col2 = st.columns(2)
    with col1:
        respect_credit = st.checkbox("Respect customer credit limits", value=True)
        parameters['respect_credit_limit'] = respect_credit
    
    with col2:
        partial_allocation = st.checkbox("Allow partial allocation", value=True)
        parameters['allow_partial'] = partial_allocation
    
    # Notes
    notes = st.text_area("Notes", placeholder="Add any notes about this allocation plan")
    parameters['notes'] = notes
    
    # Store parameters
    st.session_state['draft_allocation']['parameters'] = parameters
    
    # Next button
    st.markdown("---")
    col1, col2, col3 = st.columns([2, 1, 2])
    with col2:
        if st.button("Next ➡️", type="primary", use_container_width=True):
            # Store time adjustments and filters for snapshot
            st.session_state['draft_allocation']['time_adjustments'] = get_from_session_state('time_adjustments', {})
            st.session_state['draft_allocation']['filters'] = {
                'entities': get_from_session_state('selected_entities', []),
                'products': st.session_state.get('selected_allocation_products', [])
            }
            # Just move to next step - calculation will be done in preview            
            st.session_state['allocation_step'] = 4
            st.rerun()

def show_step4_preview():
    """Step 4: Preview allocation results with dynamic updates and zero allocation warnings"""
    st.markdown("#### 👀 Preview Allocation Results")
    
    # Get data and parameters
    method = st.session_state['draft_allocation'].get('method', 'FCFS')
    parameters = st.session_state['draft_allocation'].get('parameters', {})
    selected_products = st.session_state.get('selected_allocation_products', [])
    
    # Load demand and supply data
    demand_data = get_from_session_state('demand_filtered', pd.DataFrame())
    supply_data = get_from_session_state('supply_filtered', pd.DataFrame())
    
    # Filter for selected products
    filtered_demand = demand_data[demand_data['pt_code'].isin(selected_products)].copy()
    filtered_supply = supply_data[supply_data['pt_code'].isin(selected_products)].copy()
    
    # Calculate allocations
    with st.spinner("Calculating allocations..."):
        allocation_results = AllocationMethods.calculate_allocation(
            demand_df=filtered_demand,
            supply_df=filtered_supply,
            method=method,
            parameters=parameters
        )
    
    if allocation_results.empty:
        st.error("Failed to calculate allocations")
        return
    
    # Round allocated quantities to whole numbers
    allocation_results['allocated_qty'] = allocation_results['allocated_qty'].round(0).astype(int)
    
    # Ensure package_size column exists
    if 'package_size' not in allocation_results.columns:
        allocation_results['package_size'] = ''
    
    # Store initial results
    st.session_state['draft_allocation']['results'] = allocation_results
    
    # === NEW: Check for zero allocations ===
    zero_allocations = allocation_results[allocation_results['allocated_qty'] <= 0]
    valid_allocations = allocation_results[allocation_results['allocated_qty'] > 0]
    
    # Warning about zero allocations
    if not zero_allocations.empty:  # ✅ FIXED: Use .empty instead of .any()
        st.warning(f"""
        ⚠️ **{len(zero_allocations)} line(s) have 0 allocated quantity**
        
        These lines will be **excluded** from the allocation plan:
        """)
        
        # Show zero allocation details
        with st.expander("View excluded lines", expanded=False):
            zero_display = zero_allocations[['pt_code', 'product_name', 'customer', 'requested_qty', 'allocated_qty']].copy()
            zero_display = zero_display.rename(columns={
                'pt_code': 'Product Code',
                'product_name': 'Product Name',
                'customer': 'Customer',
                'requested_qty': 'Requested',
                'allocated_qty': 'Allocated'
            })
            st.dataframe(zero_display, use_container_width=True, hide_index=True)
            
            # Explain why
            st.info(f"""
            **Possible reasons for 0 allocation:**
            - Insufficient supply for these products
            - Lower priority in {method} method
            - Supply already allocated to higher priority orders
            """)
    
    # === Summary section with valid allocations only ===
    st.markdown("---")
    st.markdown("### 📊 Allocation Summary")
    
    if valid_allocations.empty:
        st.error("❌ No valid allocations. All lines have 0 quantity.")
        st.info("Please check your supply data or adjust allocation parameters.")
        return
    
    # Create a container for dynamic metrics
    metrics_container = st.container()
    
    # Allocation details table - ONLY VALID ALLOCATIONS
    st.markdown("##### Allocation Details (Valid Lines Only)")
    st.info("💡 **Note**: Only showing lines with allocated quantity > 0. Adjust 'Actual Allocated' and 'Allocated ETD' as needed.")
    
    # Allow editing - pass only valid allocations
    edited_df = AllocationComponents.show_editable_allocation_table(valid_allocations)
    
    # Update session state with edited valid allocations only
    st.session_state['draft_allocation']['results'] = edited_df
    st.session_state['draft_allocation']['excluded_lines'] = zero_allocations  # Store for reference
    
    # Update metrics dynamically based on edited values
    with metrics_container:
        col1, col2, col3, col4, col5 = st.columns(5)
        
        total_demand = allocation_results['requested_qty'].sum()  # Original total
        valid_demand = edited_df['requested_qty'].sum()  # Valid lines only
        total_allocated = edited_df['allocated_qty'].sum()
        total_orders = len(allocation_results)  # Original count
        valid_orders = len(edited_df)  # Valid count
        
        # Calculate fulfillment based on actual allocated
        if valid_demand > 0:
            avg_fulfillment = (total_allocated / valid_demand) * 100
        else:
            avg_fulfillment = 0
        
        with col1:
            st.metric("Total Demand", format_number(total_demand))
            if not zero_allocations.empty:  # ✅ FIXED: Use .empty
                st.caption(f"({format_number(valid_demand)} valid)")
        
        with col2:
            st.metric(
                "Total Allocated", 
                format_number(total_allocated),
                delta=f"{total_allocated - edited_df['calculated_allocation'].sum():.0f} vs calculated" if 'calculated_allocation' in edited_df.columns else None
            )
        
        with col3:
            st.metric("Total Orders", total_orders)
            if not zero_allocations.empty:  # ✅ FIXED: Use .empty
                st.caption(f"({valid_orders} valid)")
        
        with col4:
            st.metric("Excluded", len(zero_allocations), delta_color="inverse")
        
        with col5:
            st.metric("Avg Fulfillment", f"{avg_fulfillment:.1f}%")
            st.caption("(valid lines only)")
    
    # Visualization based on valid allocations only
    col1, col2 = st.columns(2)
    
    with col1:
        # Fulfillment by product using actual allocated
        fig1 = AllocationComponents.create_fulfillment_chart_by_product(edited_df)
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        # Fulfillment by customer using actual allocated
        fig2 = AllocationComponents.create_fulfillment_chart_by_customer(edited_df)
        st.plotly_chart(fig2, use_container_width=True)
    
    # Validation warnings
    warnings = AllocationValidator.validate_allocation_results(edited_df, filtered_supply)
    if warnings:
        st.warning("⚠️ Validation Warnings:")
        for warning in warnings:
            st.write(f"- {warning}")
    
    # Next button
    st.markdown("---")
    col1, col2, col3 = st.columns([2, 1, 2])
    with col2:
        # Check allocation type to show appropriate button text
        allocation_type = st.session_state['draft_allocation'].get('allocation_type', 'SOFT')
        
        button_text = "Next ➡️" if allocation_type == 'HARD' else "Confirm ➡️"
        button_disabled = valid_allocations.empty  # Disable if no valid allocations
        
        if st.button(button_text, type="primary", use_container_width=True, disabled=button_disabled):
            if allocation_type == 'HARD':
                st.session_state['allocation_step'] = 5  # Go to map supply
            else:
                st.session_state['allocation_step'] = 5  # Go to confirm
            st.rerun()


def show_step5_map_supply():
    """Step 5: Map supply sources for HARD allocation with advanced features"""
    st.markdown("#### 🔗 Map Supply to Demand (HARD Allocation)")
    
    # Get allocation results from session state (already filtered in step 4)
    allocation_results = st.session_state.get('draft_allocation', {}).get('results', pd.DataFrame())
    excluded_lines = st.session_state.get('draft_allocation', {}).get('excluded_lines', pd.DataFrame())
    
    if allocation_results.empty:
        st.error("No valid allocation results found. Please go back to previous step.")
        return
    
    # Initialize supply mapping details if not exists
    if 'supply_mapping_details' not in st.session_state:
        st.session_state['supply_mapping_details'] = {}
    
    # === Summary of what we're processing ===
    st.info(f"""
        📊 **Supply Mapping Instructions:**
        - Processing {len(allocation_results)} valid allocation lines
        - Excluded {len(excluded_lines)} lines with 0 quantity
        - Map supply sources to fulfill each allocation line
        - System will auto-suggest quantities based on availability
        - You can adjust quantities and ETD dates as needed
        - Total mapped quantity must equal allocated quantity
    """)
    
    # Get supply data
    supply_filtered = st.session_state.get('supply_filtered', pd.DataFrame())
    
    if supply_filtered.empty:
        st.error("No supply data found. Please run GAP Analysis first.")
        return
    
    # Get unique products and entities
    product_codes = allocation_results['pt_code'].unique().tolist()
    legal_entities = allocation_results['legal_entity'].unique().tolist()
    
    # Get available supply
    available_supply = allocation_manager.get_available_supply_for_hard_allocation(
        product_codes, legal_entities
    )
    
    if available_supply.empty:
        st.error("No available supply found for HARD allocation")
        if st.button("← Go Back to Previous Step"):
            st.session_state['allocation_step'] = 4
            st.rerun()
        return
    
    # Process each allocation line
    all_mappings_valid = True
    
    for idx, alloc_row in allocation_results.iterrows():
        with st.expander(
            f"📦 {alloc_row['pt_code']} - {alloc_row['customer']} ({alloc_row['allocated_qty']:.0f} units)", 
            expanded=True
        ):
            # Create unique key for this allocation line
            line_key = f"{alloc_row['demand_line_id']}"
            
            # Get available supply for this product and entity
            product_supply = available_supply[
                (available_supply['pt_code'] == alloc_row['pt_code']) &
                (available_supply['legal_entity'] == alloc_row['legal_entity'])
            ].copy()
            
            if product_supply.empty:
                st.error(f"No supply available for {alloc_row['pt_code']}")
                all_mappings_valid = False
                continue
            
            # Initialize mapping for this line if not exists
            if line_key not in st.session_state['supply_mapping_details']:
                st.session_state['supply_mapping_details'][line_key] = {
                    'allocations': [],
                    'total_mapped': 0
                }
            
            # Display allocation info
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                st.write(f"**Product:** {alloc_row['pt_code']} - {alloc_row.get('product_name', '')}")
                st.write(f"**Customer:** {alloc_row['customer']}")
                if 'package_size' in alloc_row and pd.notna(alloc_row['package_size']):
                    st.caption(f"📦 Package: {alloc_row['package_size']}")
            with col2:
                st.metric("Required Qty", f"{alloc_row['allocated_qty']:.0f}")
            with col3:
                current_mapped = st.session_state['supply_mapping_details'][line_key]['total_mapped']
                remaining = alloc_row['allocated_qty'] - current_mapped
                
                # Color coding for progress
                if remaining == 0:
                    delta_color = "off"
                    status_icon = "✅"
                elif remaining < 0:
                    delta_color = "inverse"
                    status_icon = "❌"
                else:
                    delta_color = "normal"
                    status_icon = "⚠️"
                
                st.metric(
                    "Remaining", 
                    f"{remaining:.0f}",
                    delta=f"-{current_mapped:.0f} mapped" if current_mapped > 0 else None,
                    delta_color=delta_color
                )
                st.caption(status_icon)
            
            # Supply mapping interface
            st.markdown("##### Select Supply Sources")
            
            # Get already selected supply IDs for this line
            used_supply_ids = [
                alloc['supply_id'] 
                for alloc in st.session_state['supply_mapping_details'][line_key]['allocations']
            ]
            
            # Filter out already used supplies
            available_for_selection = product_supply[
                ~product_supply['source_id'].isin(used_supply_ids)
            ]
            
            # Add new supply mapping
            if not available_for_selection.empty and remaining > 0:
                st.markdown("**➕ Add Supply Source:**")
                add_col1, add_col2, add_col3, add_col4 = st.columns([3, 1.5, 1.5, 1])
                
                with add_col1:
                    # Supply selection
                    supply_options = []
                    for _, supply_row in available_for_selection.iterrows():
                        # Calculate available after existing allocations
                        already_allocated = sum([
                            alloc['quantity'] 
                            for alloc in st.session_state['supply_mapping_details'][line_key]['allocations']
                            if alloc['supply_id'] == supply_row['source_id']
                        ])
                        net_available = supply_row['available_qty'] - already_allocated
                        
                        option_text = (
                            f"{supply_row['source_type']} - "
                            f"{supply_row['reference']} - "
                            f"Available: {net_available:.0f}/{supply_row['available_qty']:.0f} units"
                        )
                        if pd.notna(supply_row.get('date_ref_adjusted')):
                            option_text += f" (ETA: {pd.to_datetime(supply_row['date_ref_adjusted']).strftime('%Y-%m-%d')})"
                        
                        supply_options.append({
                            'text': option_text,
                            'supply_row': supply_row,
                            'net_available': net_available
                        })
                    
                    if supply_options:
                        selected_idx = st.selectbox(
                            "Select Supply",
                            range(len(supply_options)),
                            format_func=lambda x: supply_options[x]['text'],
                            key=f"supply_select_{line_key}_{idx}"
                        )
                    else:
                        st.warning("No more supply sources available")
                        selected_idx = None
                
                with add_col2:
                    # Quantity input with smart suggestion
                    if selected_idx is not None:
                        selected_option = supply_options[selected_idx]
                        selected_supply = selected_option['supply_row']
                        net_available = selected_option['net_available']
                        
                        # Smart quantity suggestion
                        max_qty = min(net_available, remaining)
                        suggested_qty = max_qty  # Auto-suggest maximum possible
                        
                        qty_input = st.number_input(
                            "Quantity",
                            min_value=0.0,
                            max_value=float(max_qty),
                            value=float(suggested_qty),
                            step=1.0,
                            key=f"qty_input_{line_key}_{idx}",
                            help=f"Max: {max_qty:.0f} (min of available {net_available:.0f} and needed {remaining:.0f})"
                        )
                
                with add_col3:
                    # ETD adjustment
                    if selected_idx is not None:
                        default_etd = pd.to_datetime(alloc_row.get('allocated_etd', alloc_row['etd']))
                        
                        # Get supply ETA as reference
                        supply_eta = None
                        if pd.notna(selected_supply.get('date_ref_adjusted')):
                            supply_eta = pd.to_datetime(selected_supply['date_ref_adjusted'])
                        elif pd.notna(selected_supply.get('date_ref')):
                            supply_eta = pd.to_datetime(selected_supply['date_ref'])
                        
                        # Default to later of allocation ETD or supply ETA
                        if supply_eta and supply_eta > default_etd:
                            suggested_etd = supply_eta
                            st.caption("📅 Adjusted to supply ETA")
                        else:
                            suggested_etd = default_etd
                        
                        adjusted_etd = st.date_input(
                            "ETD",
                            value=suggested_etd,
                            key=f"etd_input_{line_key}_{idx}",
                            help="Delivery ETD for this supply allocation"
                        )
                
                with add_col4:
                    # Add button
                    if selected_idx is not None and st.button("➕ Add", key=f"add_btn_{line_key}_{idx}"):
                        if qty_input > 0:
                            # Add to mapping
                            new_allocation = {
                                'supply_id': selected_supply['source_id'],
                                'source_type': selected_supply['source_type'],
                                'reference': selected_supply['reference'],
                                'quantity': qty_input,
                                'etd': adjusted_etd,
                                'available_qty': selected_supply['available_qty']
                            }
                            st.session_state['supply_mapping_details'][line_key]['allocations'].append(new_allocation)
                            st.session_state['supply_mapping_details'][line_key]['total_mapped'] += qty_input
                            st.rerun()
            
            # Show current mappings
            if st.session_state['supply_mapping_details'][line_key]['allocations']:
                st.markdown("**📋 Current Mappings:**")
                
                for i, mapping in enumerate(st.session_state['supply_mapping_details'][line_key]['allocations']):
                    map_col1, map_col2, map_col3, map_col4 = st.columns([3, 1.5, 1.5, 1])
                    
                    with map_col1:
                        st.write(f"• {mapping['source_type']} - {mapping['reference']}")
                    with map_col2:
                        st.write(f"📦 {mapping['quantity']:.0f} units")
                    with map_col3:
                        st.write(f"📅 {mapping['etd'].strftime('%Y-%m-%d')}")
                    with map_col4:
                        if st.button("🗑️", key=f"remove_{line_key}_{i}", help="Remove this mapping"):
                            # Remove mapping
                            st.session_state['supply_mapping_details'][line_key]['total_mapped'] -= mapping['quantity']
                            st.session_state['supply_mapping_details'][line_key]['allocations'].pop(i)
                            st.rerun()
            
            # Validation for this line
            total_mapped = st.session_state['supply_mapping_details'][line_key]['total_mapped']
            required_qty = alloc_row['allocated_qty']
            
            # Progress bar for this line
            progress = min(total_mapped / required_qty, 1.0) if required_qty > 0 else 0
            st.progress(progress)
            
            if total_mapped < required_qty:
                st.warning(f"⚠️ Incomplete: {total_mapped:.0f}/{required_qty:.0f} mapped")
                all_mappings_valid = False
            elif total_mapped > required_qty:
                st.error(f"❌ Over-mapped: {total_mapped:.0f}/{required_qty:.0f}")
                all_mappings_valid = False
            else:
                st.success(f"✅ Complete: {total_mapped:.0f}/{required_qty:.0f} mapped")
    
    # Convert to final supply_mapping format for database
    if all_mappings_valid:
        # Build supply_mapping for save
        st.session_state['supply_mapping'] = {}
        
        for line_key, details in st.session_state['supply_mapping_details'].items():
            if details['allocations']:
                # For simple case with one supply per line
                if len(details['allocations']) == 1:
                    alloc = details['allocations'][0]
                    st.session_state['supply_mapping'][line_key] = {
                        'source_type': alloc['source_type'],
                        'source_id': alloc['supply_id']
                    }
                else:
                    # For multiple supplies per line
                    # Note: Current database structure may need enhancement to support this
                    # For now, we'll use the first supply as primary
                    primary_alloc = details['allocations'][0]
                    st.session_state['supply_mapping'][line_key] = {
                        'source_type': primary_alloc['source_type'],
                        'source_id': primary_alloc['supply_id']
                    }
                    
                    # Store additional info in session for future enhancement
                    st.session_state['supply_mapping'][line_key]['multiple_sources'] = [
                        {
                            'source_type': alloc['source_type'],
                            'source_id': alloc['supply_id'],
                            'quantity': alloc['quantity'],
                            'etd': alloc['etd']
                        }
                        for alloc in details['allocations']
                    ]
    
    # Summary section
    st.markdown("---")
    st.markdown("### 📊 Mapping Summary")
    
    summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)
    
    with summary_col1:
        total_lines = len(allocation_results)
        mapped_lines = len([
            k for k, v in st.session_state['supply_mapping_details'].items() 
            if v['total_mapped'] > 0
        ])
        complete_lines = len([
            k for k, v in st.session_state['supply_mapping_details'].items() 
            if v['total_mapped'] == allocation_results[allocation_results['demand_line_id'].astype(str) == k]['allocated_qty'].sum()
        ])
        
        st.metric("Lines Mapped", f"{mapped_lines}/{total_lines}")
        if complete_lines == total_lines:
            st.caption("✅ All complete")
        else:
            st.caption(f"✅ {complete_lines} complete")
    
    with summary_col2:
        total_required = allocation_results['allocated_qty'].sum()
        total_mapped = sum([
            v['total_mapped'] 
            for v in st.session_state['supply_mapping_details'].values()
        ])
        
        # Color coding
        if total_mapped == total_required:
            st.success(f"Total: {total_mapped:.0f}/{total_required:.0f}")
        elif total_mapped < total_required:
            st.warning(f"Total: {total_mapped:.0f}/{total_required:.0f}")
        else:
            st.error(f"Total: {total_mapped:.0f}/{total_required:.0f}")
    
    with summary_col3:
        # Count supply sources used
        unique_supplies = set()
        for details in st.session_state['supply_mapping_details'].values():
            for alloc in details['allocations']:
                unique_supplies.add(f"{alloc['source_type']}_{alloc['supply_id']}")
        
        st.metric("Supply Sources", len(unique_supplies))
        st.caption("Unique sources used")
    
    with summary_col4:
        if all_mappings_valid and mapped_lines == total_lines:
            st.success("✅ Ready to proceed")
            st.caption("All validations passed")
        else:
            st.error("❌ Incomplete mappings")
            incomplete_count = total_lines - complete_lines
            if incomplete_count > 0:
                st.caption(f"{incomplete_count} lines incomplete")
    
    # Navigation buttons
    st.markdown("---")
    col1, col2, col3 = st.columns([2, 1, 2])
    
    with col1:
        if st.button("← Previous", use_container_width=True):
            st.session_state['allocation_step'] = 4
            st.rerun()
    
    with col3:
        if st.button(
            "Next ➡️", 
            type="primary", 
            use_container_width=True,
            disabled=not all_mappings_valid
        ):
            st.session_state['allocation_step'] = 6
            st.rerun()

def show_step5_confirm():
    """Step 5: Confirm and save allocation (for SOFT allocation)"""
    st.markdown("#### ✅ Confirm Allocation Plan")
    
    # Get draft data
    draft = st.session_state.get('draft_allocation', {})
    results = draft.get('results', pd.DataFrame())
    excluded_lines = draft.get('excluded_lines', pd.DataFrame())

    if results.empty:
        st.error("No allocation results found")
        return
    
    # Generate allocation number if not exists
    if 'allocation_number' not in st.session_state:
        st.session_state['allocation_number'] = generate_allocation_number()
    
    allocation_number = st.session_state['allocation_number']
    
    # === SECTION 1: Allocation Summary ===
    st.markdown("##### 📊 Allocation Summary")
    
    # Show excluded lines summary if any
    if not excluded_lines.empty:
        st.info(f"""
        ℹ️ **Note**: {len(excluded_lines)} line(s) with 0 allocation were excluded from this plan.
        Only {len(results)} valid lines will be saved.
        """)

    col1, col2 = st.columns(2)
    
    with col1:
        st.write(f"**Allocation Number:** `{allocation_number}`")
        st.write(f"**Method:** {draft.get('method', 'Unknown')}")
        st.write(f"**Type:** {draft.get('allocation_type', 'SOFT')}")
        st.write(f"**Products:** {len(results['pt_code'].unique())}")
        st.write(f"**Valid Orders:** {len(results)}")
        if not excluded_lines.empty:
            st.write(f"**Excluded Orders:** {len(excluded_lines)}")
    
    with col2:
        st.write(f"**Total Allocated:** {format_number(results['allocated_qty'].sum())}")
        avg_fulfillment = results['fulfillment_rate'].mean() if 'fulfillment_rate' in results.columns else 0
        st.write(f"**Avg Fulfillment:** {avg_fulfillment:.1f}%")
        if draft.get('parameters', {}).get('notes'):
            st.write(f"**Notes:** {draft['parameters']['notes']}")
    
    # === SECTION 2: Database Preview ===
    st.markdown("---")
    st.markdown("##### 🗄️ Data to be Saved")
    
    # Tab layout for better organization
    tab1, tab2, tab3, tab4 = st.tabs(["📋 Plan Header", "📝 Allocation Details", "📸 Snapshot Context", "🔍 Full Preview"])
    
    with tab1:
        show_plan_header_tab(allocation_number, draft)
    
    with tab2:
        show_allocation_details_tab(results, draft)
    
    with tab3:
        show_snapshot_context_tab(draft, results)
    
    with tab4:
        show_full_preview_tab(results)
    
    # === SECTION 3: Validation Summary ===
    st.markdown("---")
    show_validation_summary(results)
    
    # === SECTION 4: Action Buttons ===
    st.markdown("---")
    show_action_buttons(results, draft, allocation_number)


def show_plan_header_tab(allocation_number: str, draft: Dict):
    """Show plan header information"""
    st.markdown("**Allocation Plan Table:**")
    
    # Get user info from session state
    user_id = st.session_state.get('user_id', 1)
    username = st.session_state.get('username', 'System')
    
    plan_data = {
        'Field': [
            'allocation_number',
            'allocation_date', 
            'creator_id',
            'creator_name',
            'notes',
            'allocation_context'
        ],
        'Value': [
            allocation_number,
            datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            user_id,
            username,
            draft.get('parameters', {}).get('notes', ''),
            'JSON object (see Snapshot Context tab)'
        ],
        'Description': [
            'Unique allocation number',
            'Current timestamp',
            'User ID creating the allocation',
            'Username for reference',
            'Notes from parameters',
            'Complete snapshot of GAP Analysis context'
        ]
    }
    
    plan_df = pd.DataFrame(plan_data)
    st.dataframe(plan_df, use_container_width=True, hide_index=True)


def show_allocation_details_tab(results: pd.DataFrame, draft: Dict):
    """Show allocation details preview"""
    st.markdown("**Allocation Details Table:**")
    st.info(f"Total {len(results)} records will be inserted")
    
    # Get customer mapping from cached data
    customer_mapping = get_customer_mapping_from_cache()
    
    # Prepare detail data
    detail_preview = prepare_allocation_details_for_db(results, draft, customer_mapping)
    
    # Show sample records
    st.markdown("**Sample Records (first 5):**")
    
    # Select important columns for preview
    preview_columns = [
        'pt_code',
        'product_name',
        'customer_code', 
        'customer_name',
        'legal_entity_name',
        'requested_qty',
        'allocated_qty',
        'etd',
        'allocated_etd',
        'allocation_mode',
        'demand_type'
    ]
    
    # Filter to available columns
    available_cols = [col for col in preview_columns if col in detail_preview.columns]
    sample_preview = detail_preview[available_cols].head(5)
    
    st.dataframe(sample_preview, use_container_width=True, hide_index=True)
    
    # Show field mapping
    with st.expander("View Database Field Mapping"):
        show_field_mapping_table(draft, detail_preview)


def get_customer_mapping_from_cache() -> Dict[str, str]:
    """Get customer name to code mapping from cached data"""
    try:
        # Check if customer data is in session state from DataManager
        if 'all_data_loaded' in st.session_state and st.session_state.get('all_data_loaded'):
            # Try to get from DataManager cache
            from utils.data_manager import DataManager
            data_manager = DataManager()
            customers_df = data_manager.load_customer_master()
            
            if not customers_df.empty:
                # Debug: Check column names
                if st.session_state.get('debug_mode', False):
                    st.write("🐛 Customer master columns:", list(customers_df.columns))
                    st.write("🐛 Sample customer data:")
                    st.dataframe(customers_df[['customer_name', 'customer_code']].head(3))
                
                # Create mapping from customer_name to customer_code
                # NOTE: View uses 'customer_name' not 'company_name'
                mapping = {}
                for _, row in customers_df.iterrows():
                    if pd.notna(row.get('customer_name')) and pd.notna(row.get('customer_code')):
                        # Store mapping with exact name match
                        mapping[str(row['customer_name']).strip()] = str(row['customer_code']).strip()
                
                # Debug: Show mapping sample
                if st.session_state.get('debug_mode', False):
                    st.write(f"🐛 Total customer mappings created: {len(mapping)}")
                    sample_mappings = dict(list(mapping.items())[:5])
                    st.write("🐛 Sample mappings:", sample_mappings)
                
                return mapping
        
        # Fallback to empty mapping
        return {}
        
    except Exception as e:
        logger.error(f"Error getting customer mapping from cache: {str(e)}")
        return {}

def prepare_allocation_details_for_db(results: pd.DataFrame, draft: Dict, 
                                     customer_mapping: Dict[str, str]) -> pd.DataFrame:
    """Prepare allocation details dataframe for database insertion
    
    Args:
        results: Raw allocation results from previous steps
        draft: Draft allocation data from session state
        customer_mapping: Dict mapping customer_name to customer_code
        
    Returns:
        pd.DataFrame: Prepared data ready for database insertion
    """
    df = results.copy()
    
    # Debug: Check what columns we have
    if st.session_state.get('debug_mode', False):
        st.write("🐛 Debug - prepare_allocation_details_for_db")
        st.write(f"- Input shape: {df.shape}")
        st.write(f"- Available columns: {list(df.columns)}")
        if 'customer' in df.columns:
            st.write(f"- Unique customers: {df['customer'].nunique()}")
            st.write(f"- Sample customers: {list(df['customer'].unique()[:3])}")
    
    # === 1. CUSTOMER MAPPING ===
    # Ensure customer_name column exists
    if 'customer' in df.columns:
        # Clean customer names - remove extra spaces, normalize
        df['customer_name'] = df['customer'].astype(str).str.strip().str.replace(r'\s+', ' ', regex=True)
    elif 'customer_name' not in df.columns:
        df['customer_name'] = ''
    
    # Apply customer code mapping
    if st.session_state.get('debug_mode', False):
        st.write("🐛 Customer mapping process:")
        st.write(f"- Total mappings available: {len(customer_mapping)}")
        if customer_mapping:
            st.write(f"- Sample mappings: {dict(list(customer_mapping.items())[:3])}")
    
    # Direct mapping first
    df['customer_code'] = df['customer_name'].map(customer_mapping)
    
    # Handle unmapped customers
    unmapped_mask = df['customer_code'].isna() & df['customer_name'].notna() & (df['customer_name'] != '')
    
    if unmapped_mask.any():
        unmapped_customers = df.loc[unmapped_mask, 'customer_name'].unique()
        
        if st.session_state.get('debug_mode', False):
            st.write(f"🐛 First pass - unmapped customers: {len(unmapped_customers)}")
            st.write(f"- Sample unmapped: {list(unmapped_customers[:5])}")
        
        # Try case-insensitive mapping as fallback
        if customer_mapping:
            # Create case-insensitive mapping
            ci_mapping = {k.upper(): v for k, v in customer_mapping.items()}
            
            # Apply case-insensitive mapping for unmapped only
            for idx in df[unmapped_mask].index:
                customer_upper = str(df.at[idx, 'customer_name']).upper()
                if customer_upper in ci_mapping:
                    df.at[idx, 'customer_code'] = ci_mapping[customer_upper]
                    
                    if st.session_state.get('debug_mode', False):
                        st.write(f"✅ Mapped '{df.at[idx, 'customer_name']}' via case-insensitive match")
        
        # Final check for still unmapped
        still_unmapped_mask = df['customer_code'].isna() & df['customer_name'].notna() & (df['customer_name'] != '')
        
        if still_unmapped_mask.any():
            final_unmapped = df.loc[still_unmapped_mask, 'customer_name'].unique()
            logger.warning(f"Final unmapped customers ({len(final_unmapped)}): {list(final_unmapped)[:10]}")
            
            # Set to None for unmapped (FK constraint allows NULL)
            df.loc[still_unmapped_mask, 'customer_code'] = None
    
    # === 2. LEGAL ENTITY MAPPING ===
    if 'legal_entity' in df.columns:
        df['legal_entity_name'] = df['legal_entity'].astype(str).str.strip()
    elif 'legal_entity_name' not in df.columns:
        df['legal_entity_name'] = ''
    
    # === 3. ALLOCATION MODE ===
    # Set uniform allocation_mode for SOFT/HARD (no more MIXED)
    df['allocation_mode'] = draft.get('allocation_type', 'SOFT')
    
    # === 4. DEMAND TYPE MAPPING ===
    if 'source_type' in df.columns:
        df['demand_type'] = df['source_type'].map({
            'OC': 'OC',
            'Forecast': 'FORECAST'
        }).fillna('OC')
    elif 'demand_type' not in df.columns:
        df['demand_type'] = 'OC'
    
    # === 5. DEMAND REFERENCE ID ===
    if 'demand_line_id' in df.columns:
        # Extract numeric ID from format like "123_OC" or "456_FC"
        df['demand_reference_id'] = pd.to_numeric(
            df['demand_line_id'].astype(str).str.extract(r'(\d+)_')[0], 
            errors='coerce'
        )
    else:
        df['demand_reference_id'] = None
    
    # === 6. DEMAND NUMBER ===
    if 'demand_number' not in df.columns:
        # Try to get from OC or Forecast number
        if 'oc_number' in df.columns:
            df['demand_number'] = df['oc_number'].astype(str)
        elif 'forecast_number' in df.columns:
            df['demand_number'] = df['forecast_number'].astype(str)
        elif 'demand_line_id' in df.columns:
            # Extract from demand_line_id if no other source
            df['demand_number'] = df['demand_line_id'].astype(str).str.split('_').str[0]
        else:
            df['demand_number'] = ''
    
    # === 7. QUANTITY FIELDS ===
    # Ensure requested_qty exists
    if 'requested_qty' not in df.columns:
        if 'demand_quantity' in df.columns:
            df['requested_qty'] = pd.to_numeric(df['demand_quantity'], errors='coerce').fillna(0)
        else:
            df['requested_qty'] = 0
    else:
        df['requested_qty'] = pd.to_numeric(df['requested_qty'], errors='coerce').fillna(0)
    
    # Ensure allocated_qty is numeric
    if 'allocated_qty' in df.columns:
        df['allocated_qty'] = pd.to_numeric(df['allocated_qty'], errors='coerce').fillna(0)
    else:
        df['allocated_qty'] = 0
    
    # Set delivered_qty to 0 (initial state)
    df['delivered_qty'] = 0
    
    # === 8. DATE FIELDS ===
    # Ensure ETD exists and is datetime
    if 'etd' in df.columns:
        df['etd'] = pd.to_datetime(df['etd'], errors='coerce')
    else:
        df['etd'] = pd.NaT
    
    # Ensure allocated_etd exists
    if 'allocated_etd' not in df.columns:
        df['allocated_etd'] = df['etd']
    else:
        df['allocated_etd'] = pd.to_datetime(df['allocated_etd'], errors='coerce')
    
    # === 9. STATUS FIELD ===
    # Status will be set based on save action (DRAFT or ALLOCATED)
    df['status'] = 'DRAFT'  # Default, will be updated in save_allocation
    
    # === 10. NOTES FIELD ===
    if 'notes' not in df.columns:
        df['notes'] = ''
    
    # === 11. SUPPLY SOURCE FIELDS (for HARD allocation) ===
    if draft.get('allocation_type') == 'SOFT':
        # SOFT allocation doesn't have supply mapping
        df['supply_source_type'] = None
        df['supply_source_id'] = None
    elif 'supply_source_type' not in df.columns:
        # HARD/MIXED but no supply mapping yet (will be added later)
        df['supply_source_type'] = None
        df['supply_source_id'] = None
    
    # === 12. FINAL VALIDATION ===
    # Ensure all required fields exist
    required_fields = [
        'pt_code', 'customer_code', 'customer_name', 'legal_entity_name',
        'requested_qty', 'allocated_qty', 'delivered_qty',
        'etd', 'allocated_etd', 'allocation_mode', 'demand_type',
        'status', 'notes', 'supply_source_type', 'supply_source_id',
        'demand_reference_id', 'demand_number'
    ]
    
    for field in required_fields:
        if field not in df.columns:
            logger.warning(f"Missing required field: {field}")
            df[field] = None
    
    # === 13. DATA TYPE CLEANUP ===
    # Ensure correct data types for database
    # Numeric fields
    for col in ['requested_qty', 'allocated_qty', 'delivered_qty']:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # String fields - convert to string and handle None
    string_fields = ['pt_code', 'customer_code', 'customer_name', 'legal_entity_name', 
                    'demand_number', 'notes', 'supply_source_type']
    for col in string_fields:
        df[col] = df[col].apply(lambda x: str(x) if pd.notna(x) and x != 'None' else None)
    
    # Integer fields
    int_fields = ['demand_reference_id', 'supply_source_id']
    for col in int_fields:
        df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')
        # Convert pandas Int64 NA to Python None for database
        df[col] = df[col].apply(lambda x: None if pd.isna(x) else int(x))
    
    # Date fields - ensure datetime
    date_fields = ['etd', 'allocated_etd']
    for col in date_fields:
        df[col] = pd.to_datetime(df[col], errors='coerce')
    
    # === 14. FINAL DEBUG OUTPUT ===
    if st.session_state.get('debug_mode', False):
        st.write("🐛 Final prepared data:")
        st.write(f"- Output shape: {df.shape}")
        st.write(f"- Columns: {list(df.columns)}")
        
        # Check mapping results
        total_rows = len(df)
        mapped_customers = df['customer_code'].notna().sum()
        st.write(f"- Customer mapping: {mapped_customers}/{total_rows} mapped")
        
        # Show sample of final data
        sample_cols = ['pt_code', 'customer_name', 'customer_code', 'allocated_qty', 'allocation_mode']
        if all(col in df.columns for col in sample_cols):
            st.write("- Sample prepared data:")
            st.dataframe(df[sample_cols].head(3))
    
    return df

def show_field_mapping_table(draft: Dict, detail_preview: pd.DataFrame):
    """Show database field mapping table"""
    # Get sample row for values
    sample_row = detail_preview.iloc[0] if not detail_preview.empty else {}
    
    field_mapping = {
        'Database Field': [
            'allocation_plan_id',
            'allocation_mode',
            'status',
            'demand_type',
            'demand_reference_id',
            'demand_number',
            'product_id',
            'pt_code',
            'customer_code',
            'customer_name',
            'legal_entity_name',
            'requested_qty',
            'allocated_qty',
            'delivered_qty',
            'etd',
            'allocated_etd',
            'notes',
            'supply_source_type',
            'supply_source_id'
        ],
        'Data Type': [
            'int',
            'enum(SOFT,HARD)',
            'enum(ALLOCATED,DRAFT)',
            'enum(OC,FORECAST)',
            'int',
            'varchar(50)',
            'bigint',
            'varchar(50)',
            'varchar(255)',
            'varchar(200)',
            'varchar(100)',
            'decimal(15,2)',
            'decimal(15,2)',
            'decimal(15,2)',
            'date',
            'date',
            'text',
            'varchar(50)',
            'int'
        ],
        'Source': [
            'Auto-generated after plan insert',
            f"'{draft.get('allocation_type', 'SOFT')}'",
            "'DRAFT' or 'ALLOCATED'",
            'From demand source',
            'Extracted from demand_line_id',
            'From OC/Forecast number',
            'Looked up from products table',
            'From GAP Analysis data',
            'Mapped from customer master',
            'From GAP Analysis data',
            'From GAP Analysis data',
            'From demand quantity',
            'User adjusted value',
            '0 (initial)',
            'From demand ETD',
            'User adjusted or same as ETD',
            'User notes per line',
            'NULL for SOFT allocation',
            'NULL for SOFT allocation'
        ],
        'Sample Value': [
            '(auto)',
            sample_row.get('allocation_mode', draft.get('allocation_type', 'SOFT')),
            'DRAFT',
            sample_row.get('demand_type', 'OC'),
            sample_row.get('demand_reference_id', ''),
            sample_row.get('demand_number', ''),
            '(lookup)',
            sample_row.get('pt_code', ''),
            sample_row.get('customer_code', ''),
            sample_row.get('customer_name', ''),
            sample_row.get('legal_entity_name', ''),
            f"{sample_row.get('requested_qty', 0):.2f}",
            f"{sample_row.get('allocated_qty', 0):.2f}",
            '0.00',
            str(sample_row.get('etd', ''))[:10] if 'etd' in sample_row else '',
            str(sample_row.get('allocated_etd', ''))[:10] if 'allocated_etd' in sample_row else '',
            sample_row.get('notes', ''),
            sample_row.get('supply_source_type'),
            sample_row.get('supply_source_id')
        ]
    }
    
    mapping_df = pd.DataFrame(field_mapping)
    st.dataframe(mapping_df, use_container_width=True, hide_index=True)


def show_snapshot_context_tab(draft: Dict, results: pd.DataFrame):
    """Show snapshot context JSON"""
    st.markdown("**Allocation Context (JSON):**")
    st.info("This captures the complete state of GAP Analysis at the time of allocation creation")
    
    # Build context
    gap_context = build_allocation_context(draft, results)
    
    # Display as formatted JSON
    st.json(gap_context)
    
    # Add download button for JSON
    import json
    json_str = json.dumps(gap_context, indent=2)
    st.download_button(
        label="📥 Download Context JSON",
        data=json_str,
        file_name=f"allocation_context_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json"
    )


def show_full_preview_tab(results: pd.DataFrame):
    """Show full data preview with search and pagination"""
    st.markdown("**Complete Allocation Details:**")
    
    # Summary statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Records", len(results))
    with col2:
        st.metric("Unique Products", results['pt_code'].nunique())
    with col3:
        st.metric("Unique Customers", results['customer'].nunique() if 'customer' in results.columns else 0)
    with col4:
        total_value = results.get('value_in_usd', pd.Series()).sum()
        if total_value > 0:
            st.metric("Total Value", f"${total_value:,.0f}")
        else:
            st.metric("Total Quantity", format_number(results['allocated_qty'].sum()))
    
    # Full data preview
    st.markdown("**All Records:**")
    
    # Add search/filter
    search_term = st.text_input("🔍 Search in preview", placeholder="Product code, customer...", key="preview_search")
    
    if search_term:
        mask = results.astype(str).apply(
            lambda x: x.str.contains(search_term, case=False, na=False)
        ).any(axis=1)
        filtered_results = results[mask]
    else:
        filtered_results = results
    
    # Pagination
    items_per_page = st.selectbox("Items per page:", [10, 25, 50, 100], index=1, key="preview_page_size")
    total_pages = max(1, (len(filtered_results) + items_per_page - 1) // items_per_page)
    current_page = st.number_input("Page", min_value=1, max_value=total_pages, value=1, key="preview_page")
    
    start_idx = (current_page - 1) * items_per_page
    end_idx = min(start_idx + items_per_page, len(filtered_results))
    
    st.info(f"Showing {start_idx + 1} to {end_idx} of {len(filtered_results)} records")
    
    # Display paginated data
    page_data = filtered_results.iloc[start_idx:end_idx]
    
    # Select columns to display
    display_columns = [
        'pt_code', 'product_name', 'customer', 'legal_entity',
        'requested_qty', 'allocated_qty', 'fulfillment_rate',
        'etd', 'allocated_etd'
    ]
    available_display_cols = [col for col in display_columns if col in page_data.columns]
    
    st.dataframe(
        page_data[available_display_cols], 
        use_container_width=True, 
        height=400
    )


def show_validation_summary(results: pd.DataFrame):
    """Show pre-save validation summary"""
    st.markdown("##### ✅ Pre-Save Validation")
    
    # Get customer mapping to check validation
    customer_mapping = get_customer_mapping_from_cache()
    
    validation_cols = st.columns(4)
    
    with validation_cols[0]:
        # Check for zero allocations
        zero_allocs = len(results[results['allocated_qty'] <= 0])
        if zero_allocs > 0:
            st.warning(f"⚠️ {zero_allocs} items with zero allocation")
        else:
            st.success("✅ All items have allocation > 0")
    
    with validation_cols[1]:
        # Check for over-allocation
        if 'requested_qty' in results.columns:
            over_allocs = len(results[results['allocated_qty'] > results['requested_qty']])
            if over_allocs > 0:
                st.warning(f"⚠️ {over_allocs} items over-allocated")
            else:
                st.success("✅ No over-allocations")
        else:
            st.info("ℹ️ Cannot check over-allocation")
    
    with validation_cols[2]:
        # Check customer mapping
        if 'customer' in results.columns:
            unmapped_count = 0
            for customer in results['customer'].unique():
                if pd.notna(customer) and str(customer) not in customer_mapping:
                    unmapped_count += 1
            
            if unmapped_count > 0:
                st.warning(f"⚠️ {unmapped_count} unmapped customers")
            else:
                st.success("✅ All customers mapped")
        else:
            st.warning("⚠️ No customer data")
    
    with validation_cols[3]:
        # Check required fields
        required_fields = ['pt_code', 'allocated_qty']
        missing_fields = []
        
        for field in required_fields:
            if field not in results.columns:
                missing_fields.append(field)
            elif results[field].isna().any():
                missing_fields.append(f"{field} (has nulls)")
        
        if missing_fields:
            st.error(f"❌ Issues: {', '.join(missing_fields)}")
        else:
            st.success("✅ All required fields OK")


def show_action_buttons(results: pd.DataFrame, draft: Dict, allocation_number: str):
    """Show action buttons for save/export/cancel"""
    col1, col2, col3, col4 = st.columns(4)
    
    # Get customer mapping for preparing data
    customer_mapping = get_customer_mapping_from_cache()
    
    with col1:
        if st.button("💾 Save as Draft", type="secondary", use_container_width=True):
            # Prepare data before saving
            prepared_data = prepare_allocation_details_for_db(results, draft, customer_mapping)
            gap_context = build_allocation_context(draft, results)
            
            # Store in session state
            st.session_state['prepared_allocation_data'] = prepared_data
            st.session_state['prepared_allocation_context'] = gap_context
            st.session_state['prepared_allocation_number'] = allocation_number
            
            save_allocation('DRAFT')
    
    with col2:
        if st.button("✅ Save & Approve", type="primary", use_container_width=True):
            # Prepare data before saving
            prepared_data = prepare_allocation_details_for_db(results, draft, customer_mapping)
            gap_context = build_allocation_context(draft, results)
            
            # Store in session state
            st.session_state['prepared_allocation_data'] = prepared_data
            st.session_state['prepared_allocation_context'] = gap_context
            st.session_state['prepared_allocation_number'] = allocation_number
            
            save_allocation('APPROVED')
    
    with col3:
        if st.button("📤 Export Preview", use_container_width=True):
            # Prepare data for export
            prepared_data = prepare_allocation_details_for_db(results, draft, customer_mapping)
            gap_context = build_allocation_context(draft, results)
            export_allocation_preview(prepared_data, gap_context)
    
    with col4:
        if st.button("❌ Cancel", use_container_width=True):
            # Clear allocation number and data
            for key in ['allocation_number', 'prepared_allocation_data', 
                       'prepared_allocation_context', 'prepared_allocation_number']:
                if key in st.session_state:
                    del st.session_state[key]
            
            # Reset to list view
            st.session_state['allocation_mode'] = 'list'
            st.session_state['allocation_step'] = 1
            st.session_state['draft_allocation'] = {}
            st.session_state['selected_allocation_products'] = []
            st.rerun()


def build_allocation_context(draft: Dict, results: pd.DataFrame) -> Dict:
    """Build complete allocation context for snapshot"""
    return {
        'snapshot_datetime': datetime.now().isoformat(),
        'gap_analysis': {
            'run_date': st.session_state.get('gap_analysis_run_date', datetime.now()).isoformat(),
            'total_products': len(results['pt_code'].unique()),
            'total_customers': len(results.get('customer', pd.Series()).unique()),
            'total_allocated': float(results['allocated_qty'].sum()),
            'total_requested': float(results.get('requested_qty', results.get('demand_quantity', 0)).sum()),
            'avg_fulfillment': float(results.get('fulfillment_rate', pd.Series()).mean() or 0)
        },
        'time_adjustments': st.session_state.get('time_adjustments', {
            'etd_offset_days': 0,
            'supply_arrival_offset': 0,
            'wh_transfer_lead_time': 2,
            'transportation_time': 3
        }),
        'filters': {
            'entities': st.session_state.get('selected_entities', []),
            'products': st.session_state.get('selected_allocation_products', []),
            'brands': st.session_state.get('selected_brands', []),
            'customers': st.session_state.get('selected_customers', []),
            'date_range': {
                'start': st.session_state.get('date_from', datetime.now()).strftime('%Y-%m-%d'),
                'end': st.session_state.get('date_to', datetime.now() + timedelta(days=30)).strftime('%Y-%m-%d')
            }
        },
        'data_sources': {
            'demand_sources': st.session_state.get('selected_demand_sources', ['OC']),
            'supply_sources': st.session_state.get('selected_supply_sources', ['Inventory']),
            'include_converted_forecasts': st.session_state.get('include_converted_forecasts', False)
        },
        'allocation_info': {
            'method': draft.get('method', 'MANUAL'),
            'type': draft.get('allocation_type', 'SOFT'),
            'parameters': draft.get('parameters', {}),
            'selected_products': st.session_state.get('selected_allocation_products', []),
            'total_orders': len(results)
        }
    }


def show_step6_final_confirm():
    """Step 6: Final confirmation for HARD allocation"""
    st.markdown("#### ✅ Confirm Allocation Plan (with Supply Mapping)")
    
    # Get data
    allocation_results = st.session_state.get('draft_allocation', {}).get('results', pd.DataFrame())
    excluded_lines = st.session_state.get('draft_allocation', {}).get('excluded_lines', pd.DataFrame())
    supply_mapping = st.session_state.get('supply_mapping', {})
    
    if allocation_results.empty:
        st.error("No allocation results found")
        return
    
    # Show excluded lines info if any
    if not excluded_lines.empty:
        st.info(f"""
        ℹ️ **Note**: {len(excluded_lines)} line(s) with 0 allocation were excluded.
        Only {len(allocation_results)} valid lines with supply mapping will be saved.
        """)
    
    # Add supply mapping info to results
    allocation_results = allocation_results.copy()
    
    # Update with supply mapping
    allocation_results['allocation_mode'] = 'HARD'  # All are HARD
    allocation_results['supply_source_type'] = None
    allocation_results['supply_source_id'] = None
    
    for demand_key, mapping in supply_mapping.items():
        mask = allocation_results['demand_line_id'].astype(str) == str(demand_key)
        allocation_results.loc[mask, 'supply_source_type'] = mapping.get('source_type')
        allocation_results.loc[mask, 'supply_source_id'] = mapping.get('source_id')
    
    # Store in draft
    st.session_state['draft_allocation']['results'] = allocation_results
    
    # Use the same confirmation interface as SOFT
    show_step5_confirm()
    
    # Add HARD allocation specific section
    st.markdown("---")
    st.markdown("##### 🔒 HARD Allocation Mapping Summary")
    
    mapping_summary = []
    for demand_id, mapping in supply_mapping.items():
        detail = allocation_results[allocation_results['demand_line_id'].astype(str) == str(demand_id)]
        if not detail.empty:
            row = detail.iloc[0]
            mapping_summary.append({
                'Product': row['pt_code'],
                'Customer': row.get('customer', row.get('customer_name', '')),
                'Allocated Qty': f"{row['allocated_qty']:.0f}",
                'Supply Type': mapping['source_type'],
                'Supply ID': mapping['source_id'],
                'Mode': '🔒 HARD'
            })
    
    if mapping_summary:
        mapping_df = pd.DataFrame(mapping_summary)
        st.dataframe(mapping_df, use_container_width=True, hide_index=True)
        
        st.success(f"✅ {len(mapping_df)} items will be HARD allocated to specific supply sources")


def export_allocation_preview(results: pd.DataFrame, context: Dict):
    """Export allocation preview to Excel"""
    import io
    
    # Create Excel file with multiple sheets
    output = io.BytesIO()
    
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        # Sheet 1: Allocation Details
        results.to_excel(writer, sheet_name='Allocation Details', index=False)
        
        # Sheet 2: Summary
        summary_data = {
            'Metric': [
                'Total Products',
                'Total Orders', 
                'Total Allocated Quantity',
                'Average Fulfillment Rate',
                'Allocation Method',
                'Allocation Type'
            ],
            'Value': [
                results['pt_code'].nunique(),
                len(results),
                results['allocated_qty'].sum(),
                f"{results['fulfillment_rate'].mean():.1f}%",
                context['allocation_info']['method'],
                context['allocation_info']['type']
            ]
        }
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
        
        # Sheet 3: Context
        context_df = pd.DataFrame([{'Context': 'See JSON', 'Value': str(context)}])
        context_df.to_excel(writer, sheet_name='Context', index=False)
    
    output.seek(0)
    
    st.download_button(
        label="📥 Download Preview",
        data=output,
        file_name=f"allocation_preview_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )


def save_allocation(status):
    """Save allocation plan to database using prepared data"""
    try:
        # Get prepared data from session state
        allocation_number = st.session_state.get('prepared_allocation_number')
        detail_data = st.session_state.get('prepared_allocation_data')
        gap_context = st.session_state.get('prepared_allocation_context')
        
        if not allocation_number or detail_data is None or detail_data.empty:
            st.error("Missing prepared allocation data. Please go back and try again.")
            return
        
        # Update status in detail data based on save action
        detail_data['status'] = 'DRAFT' if status == 'DRAFT' else 'ALLOCATED'
        
        # Prepare plan data with pre-generated allocation number
        plan_data = {
            'allocation_number': allocation_number,  # Use the pre-generated number
            'creator_id': st.session_state.get('user_id', 1),
            'notes': st.session_state.get('draft_allocation', {}).get('parameters', {}).get('notes', ''),
            'allocation_context': gap_context,
            'status': status  # This determines initial detail status
        }
        
        # Get supply mapping if exists (for HARD allocation)
        supply_mapping = st.session_state.get('supply_mapping', {})
        
        # Log save action
        logger.info(f"Saving allocation {allocation_number} with status: {status}")
        logger.info(f"Total items: {len(detail_data)}")
        
        # Progress indicator
        progress_placeholder = st.empty()
        progress_placeholder.info(f"💾 Saving allocation plan {allocation_number}...")
        
        # Save to database with data_prepared=True flag
        allocation_id = allocation_manager.create_allocation_plan(
            plan_data, 
            detail_data,
            supply_mapping,
            data_prepared=True  # ← KEY CHANGE: Tell method data is already prepared
        )

        if allocation_id:
            progress_placeholder.success(f"✅ Allocation plan {allocation_number} saved successfully! (ID: {allocation_id})")
            
            # If APPROVED, update all details from DRAFT to ALLOCATED
            if status == 'APPROVED':
                with st.spinner("📝 Activating allocation..."):
                    success = allocation_manager.bulk_update_allocation_status(
                        allocation_id, 'ALLOCATED'
                    )
                    if success:
                        st.info("✅ All items marked as ALLOCATED and ready for delivery")
                    else:
                        st.warning("⚠️ Plan created but some items may not have been activated")
            
            # Show success summary
            show_save_success_summary(allocation_id, allocation_number, detail_data)
            
            # Clear temporary data
            cleanup_allocation_session_state()
            
            # Show next actions
            show_post_save_actions(allocation_id)
            
        else:
            progress_placeholder.error(f"❌ Failed to save allocation plan {allocation_number}")
            
    except Exception as e:
        st.error(f"❌ Error saving allocation: {str(e)}")
        logger.error(f"Error in save_allocation: {str(e)}", exc_info=True)
        
        if st.session_state.get('debug_mode', False):
            import traceback
            with st.expander("🐛 Error Details", expanded=True):
                st.code(traceback.format_exc())



def cleanup_allocation_session_state():
    """Clean up allocation-related session state after save"""
    keys_to_remove = [
        'allocation_mode',
        'allocation_step', 
        'draft_allocation',
        'selected_allocation_products',
        'supply_mapping',
        'temp_allocation_results',
        'selected_hard_items',
        'alloc_current_page',
        'allocation_filter_mode',
        'allocation_items_per_page',
        'prepared_allocation_data',
        'prepared_allocation_context',
        'prepared_allocation_number',
        'allocation_number',
        # Smart filter states
        'alloc_smart_filters',
        'alloc_filter_type',
        'alloc_entity_selection',
        'alloc_customer_selection',
        'alloc_brand_selection',
        'alloc_product_selection'
    ]
    
    for key in keys_to_remove:
        if key in st.session_state:
            del st.session_state[key]
    
    # Reset to default state
    st.session_state['allocation_mode'] = 'view'
    st.session_state['allocation_step'] = 1
    st.session_state['draft_allocation'] = {}
    st.session_state['selected_allocation_products'] = []


def show_save_success_summary(allocation_id: int, allocation_number: str, detail_data: pd.DataFrame):
    """Show summary after successful save"""
    with st.expander("📊 Allocation Summary", expanded=True):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Allocation ID", allocation_id)
            st.metric("Allocation Number", allocation_number)
        
        with col2:
            st.metric("Total Products", detail_data['pt_code'].nunique())
            st.metric("Total Orders", len(detail_data))
        
        with col3:
            st.metric("Total Quantity", format_number(detail_data['allocated_qty'].sum()))
            
            # Show HARD allocation count if any
            if 'allocation_mode' in detail_data.columns:
                hard_count = len(detail_data[detail_data['allocation_mode'] == 'HARD'])
                if hard_count > 0:
                    st.metric("HARD Allocations", hard_count)


def show_post_save_actions(allocation_id: int):
    """Show action buttons after successful save"""
    st.markdown("---")
    st.markdown("### 🎯 What's Next?")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### 📋 View Your Plan")
        st.write("Review allocation details and track delivery")
        if st.button("View Plan", type="primary", use_container_width=True):
            st.session_state['selected_allocation_id'] = allocation_id
            st.rerun()
    
    with col2:
        st.markdown("#### ➕ Create Another")
        st.write("Start a new allocation plan")
        if st.button("Create Another", use_container_width=True):
            st.session_state['allocation_mode'] = 'create'
            st.session_state['selected_allocation_id'] = None
            st.rerun()
    
    with col3:
        st.markdown("#### 📊 Back to Analysis")
        st.write("Return to GAP Analysis")
        if st.button("GAP Analysis", use_container_width=True):
            st.switch_page("pages/3_📊_GAP_Analysis.py")


def export_allocation_results(results_df):
    """Export allocation results to Excel"""
    excel_data = convert_df_to_excel(results_df)
    
    st.download_button(
        label="📥 Download Allocation Results",
        data=excel_data,
        file_name=f"allocation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )


def show_view_allocation():
    """View allocation plan details with cancellation support"""
    allocation_id = st.session_state.get('selected_allocation_id')
    
    if not allocation_id:
        st.error("No allocation selected")
        if st.button("← Back to List"):
            st.session_state['allocation_mode'] = 'list'
            st.rerun()
        return
    
    # Load allocation details
    plan, details = allocation_manager.get_allocation_details(allocation_id)
    
    if plan is None:
        st.error(f"Allocation plan {allocation_id} not found")
        if st.button("← Back to List"):
            st.session_state['allocation_mode'] = 'list'
            st.rerun()
        return
    
    # Header
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown(f"### 📋 {plan.get('allocation_number', 'Unknown')}")
    with col2:
        if st.button("← Back to List"):
            st.session_state['allocation_mode'] = 'list'
            st.rerun()
    
    # Plan info with safe gets
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        # Format date safely
        allocation_date = plan.get('allocation_date')
        if allocation_date:
            if isinstance(allocation_date, str):
                date_str = allocation_date
            else:
                date_str = allocation_date.strftime('%Y-%m-%d %H:%M')
        else:
            date_str = 'Unknown'
            
        st.write("**Date:**", date_str)
        st.write("**Method:**", plan.get('allocation_method', 'Manual'))
    
    with col2:
        # Get status from plan data - use display_status from view
        status = plan.get('display_status', 'Unknown')
        status_color = ALLOCATION_STATUS_COLORS.get(status, 'gray')
        
        # User-friendly status display
        status_display = {
            'ALL_DRAFT': 'Draft',
            'IN_PROGRESS': 'In Progress',
            'ALL_DELIVERED': 'Delivered',
            'ALL_CANCELLED': 'Cancelled',
            'MIXED_DRAFT': 'Mixed (Draft)',
            'MIXED': 'Mixed Status',
            'EMPTY': 'Empty'
        }.get(status, status)
        
        st.markdown(f"**Status:** :{status_color}[{status_display}]")
        st.write("**Created by:**", plan.get('creator_name', 'System'))
    
    with col3:
        allocation_type = plan.get('allocation_type', 'SOFT')
        type_icon = {'SOFT': '🌊', 'HARD': '🔒', 'MIXED': '🔀'}.get(allocation_type, '❓')
        st.write(f"**Type:** {type_icon} {allocation_type}")
        
        # Count HARD allocations if column exists
        if not details.empty and 'allocation_mode' in details.columns:
            hard_count = len(details[details['allocation_mode'] == 'HARD'])
            if hard_count > 0:
                st.write(f"**HARD allocations:** {hard_count}")
    
    with col4:
        # Additional info if available
        if plan.get('approved_by'):
            st.write("**Approved by:**", plan['approved_by'])
            if plan.get('approved_date'):
                approved_date_str = plan['approved_date'].strftime('%Y-%m-%d')
                st.write("**Approved:**", approved_date_str)
    
    with col5:
        # Calculate metrics with safe defaults
        if not details.empty:
            if 'effective_allocated_qty' in details.columns:
                total_allocated = details['effective_allocated_qty'].sum()
            else:
                total_allocated = details.get('allocated_qty', 0).sum()
            
            total_delivered = details.get('delivered_qty', 0).sum()
            
            st.metric("Total Allocated", format_number(total_allocated))
            
            if status in ['IN_PROGRESS', 'ALL_DELIVERED', 'MIXED']:
                delivery_pct = (total_delivered / total_allocated * 100) if total_allocated > 0 else 0
                st.metric("Delivered", format_number(total_delivered), 
                         f"{delivery_pct:.1f}%")
        else:
            st.metric("Total Allocated", "0")
    
    # Show cancellation summary if exists
    cancellation_summary = plan.get('cancellation_summary', {})
    if cancellation_summary.get('total_cancellations', 0) > 0:
        with st.expander("🚫 Cancellation Summary", expanded=False):
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Cancellations", cancellation_summary.get('total_cancellations', 0))
            with col2:
                st.metric("Active", cancellation_summary.get('active_cancellations', 0))
            with col3:
                st.metric("Reversed", cancellation_summary.get('reversed_cancellations', 0))
            with col4:
                st.metric("Cancelled Qty", format_number(cancellation_summary.get('total_cancelled_qty', 0)))
    
    # Show snapshot context if available
    if plan.get('allocation_context'):
        try:
            import json
            context = plan['allocation_context']
            if isinstance(context, str):
                context = json.loads(context)
            
            with st.expander("📸 Snapshot Context", expanded=False):
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Snapshot Time:**", context.get('snapshot_datetime', 'N/A'))
                    if context.get('time_adjustments'):
                        adjustments = context['time_adjustments']
                        st.write("**Time Adjustments:**")
                        st.write(f"- ETD Offset: {adjustments.get('etd_offset_days', 0)} days")
                        st.write(f"- Supply Offset: {adjustments.get('supply_arrival_offset', 0)} days")
                    
                with col2:
                    if context.get('gap_analysis'):
                        gap_info = context['gap_analysis']
                        st.write("**GAP Analysis Info:**")
                        st.write(f"- Total Products: {gap_info.get('total_products', 0)}")
                        st.write(f"- Shortage Products: {gap_info.get('shortage_products', 0)}")
                        st.write(f"- Fulfillment Rate: {gap_info.get('fulfillment_rate', 0):.1f}%")
        except Exception as e:
            logger.error(f"Error parsing allocation context: {str(e)}")
    
    # Allocation details section
    # st.markdown("---")
    st.markdown("#### Allocation Details")
    
    # Tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Summary", "📋 Details", "🚫 Cancellations", "📈 Analytics"])
    
    with tab1:
        # Summary by product
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("##### By Product")
            if not details.empty:
                # Build aggregation dict based on available columns
                agg_dict = {}
                
                # Check and add columns if they exist
                if 'original_allocated_qty' in details.columns:
                    agg_dict['original_allocated_qty'] = 'sum'
                elif 'allocated_qty' in details.columns:
                    agg_dict['allocated_qty'] = 'sum'
                    
                if 'allocated_qty' in details.columns:
                    agg_dict['allocated_qty'] = 'sum'
                    
                if 'delivered_qty' in details.columns:
                    agg_dict['delivered_qty'] = 'sum'
                    
                if 'cancelled_qty' in details.columns:
                    agg_dict['cancelled_qty'] = 'sum'
                
                # Only proceed if we have columns to aggregate
                if agg_dict:
                    product_summary = details.groupby(['pt_code', 'product_name']).agg(agg_dict).reset_index()
                    
                    # Ensure we have required columns for chart
                    if 'original_allocated_qty' in product_summary.columns:
                        product_summary['requested_qty'] = product_summary['original_allocated_qty']
                    elif 'allocated_qty' in product_summary.columns:
                        product_summary['requested_qty'] = product_summary['allocated_qty']
                    else:
                        product_summary['requested_qty'] = 0
                    
                    # Create summary chart
                    fig1 = AllocationComponents.create_allocation_summary_chart(product_summary, 'product')
                    st.plotly_chart(fig1, use_container_width=True)
                else:
                    st.info("No numeric data to summarize")
            else:
                st.info("No details to display")

    with tab2:
        # Details table with cancellation actions
        st.markdown("##### Detail Lines")
        
        # Filter options
        col1, col2, col3 = st.columns(3)
        with col1:
            filter_status = st.selectbox(
                "Filter by Status",
                options=['All', 'Allocated', 'Partial Delivered', 'Delivered', 'Cancelled'],
                index=0,
                key="detail_filter_status"
            )
        
        with col2:
            if st.checkbox("Show cancellable only", value=False, key="show_cancellable"):
                if 'cancellable_qty' in details.columns:
                    details = details[details['cancellable_qty'] > 0]
        
        with col3:
            search_term = st.text_input("Search", placeholder="Product or Customer", key="detail_search")
        
        # Apply filters
        filtered_details = details.copy()
        
        if filter_status != 'All':
            status_map = {
                'Allocated': 'ALLOCATED',
                'Partial Delivered': 'PARTIAL_DELIVERED',
                'Delivered': 'DELIVERED',
                'Cancelled': 'CANCELLED'
            }
            if 'status' in filtered_details.columns:
                filtered_details = filtered_details[filtered_details['status'] == status_map.get(filter_status, filter_status)]
        
        if search_term:
            mask = (
                filtered_details['pt_code'].str.contains(search_term, case=False, na=False) |
                filtered_details['customer_name'].str.contains(search_term, case=False, na=False)
            )
            filtered_details = filtered_details[mask]
        
        # Display details with actions
        if not filtered_details.empty:
            # Add computed columns
            filtered_details['fulfillment_status'] = filtered_details.apply(
                lambda x: '✅ Delivered' if x.get('delivered_qty', 0) >= x.get('effective_allocated_qty', x.get('allocated_qty', 0))
                else '🔄 Partial' if x.get('delivered_qty', 0) > 0 
                else '⏳ Pending', axis=1
            )
            
            # Calculate cancellable quantity if not exists
            if 'cancellable_qty' not in filtered_details.columns:
                filtered_details['cancellable_qty'] = (
                    filtered_details['allocated_qty'] - filtered_details['delivered_qty']
                ).clip(lower=0)
            
            # Bulk actions if plan is active
            if plan.get('display_status') in ['IN_PROGRESS', 'MIXED']:
                cancellable_details = filtered_details[filtered_details['cancellable_qty'] > 0]
                
                if not cancellable_details.empty:
                    with st.expander("🎯 Bulk Actions", expanded=False):
                        selected_ids = st.multiselect(
                            "Select items for bulk action:",
                            options=cancellable_details['id'].tolist(),
                            format_func=lambda x: f"{cancellable_details[cancellable_details['id']==x]['pt_code'].iloc[0]} - {cancellable_details[cancellable_details['id']==x]['customer_name'].iloc[0]}",
                            key="bulk_select_items"
                        )
                        
                        if selected_ids:
                            col1, col2 = st.columns(2)
                            with col1:
                                if st.button("🚫 Bulk Cancel Selected", type="secondary"):
                                    st.session_state['show_bulk_cancel'] = True
                                    st.session_state['selected_cancel_ids'] = selected_ids
            
            # Display each detail row WITH CANCEL BUTTON
            for idx, row in filtered_details.iterrows():
                with st.container():
                    col1, col2, col3, col4, col5 = st.columns([3, 2, 1, 1, 1])
                    
                    with col1:
                        st.write(f"**{row.get('pt_code', '')}** - {row.get('product_name', '')}")
                        st.caption(f"Customer: {row.get('customer_name', '')}")
                        if 'allocation_mode' in row:
                            mode_icon = '🔒' if row['allocation_mode'] == 'HARD' else '🌊'
                            st.caption(f"Mode: {mode_icon} {row['allocation_mode']}")
                    
                    with col2:
                        subcol1, subcol2 = st.columns(2)
                        with subcol1:
                            st.metric("Requested", format_number(row.get('requested_qty', 0)))
                        with subcol2:
                            effective_alloc = row.get('effective_allocated_qty', row.get('allocated_qty', 0))
                            st.metric("Allocated", format_number(effective_alloc))
                            if 'cancelled_qty' in row and row.get('cancelled_qty', 0) > 0:
                                st.caption(f"Cancelled: {format_number(row['cancelled_qty'])}")
                    
                    with col3:
                        st.metric("Delivered", format_number(row.get('delivered_qty', 0)))
                        st.caption(row.get('fulfillment_status', ''))
                    
                    with col4:
                        etd = row.get('allocated_etd')
                        if etd:
                            if isinstance(etd, str):
                                st.write(f"**ETD:** {etd}")
                            else:
                                st.write(f"**ETD:** {etd.strftime('%Y-%m-%d')}")
                        
                        if row.get('cancellable_qty', 0) > 0:
                            st.caption(f"Cancellable: {format_number(row['cancellable_qty'])}")
                    
                    with col5:
                        # ACTION BUTTONS - THIS IS THE KEY PART
                        if plan.get('display_status') in ['IN_PROGRESS', 'MIXED']:
                            # Cancel button
                            if row.get('cancellable_qty', 0) > 0:
                                if st.button("🚫", key=f"cancel_{row['id']}", 
                                        help=f"Cancel {row['cancellable_qty']} units"):
                                    st.session_state[f'show_cancel_{row["id"]}'] = True
                            
                            # Delivery button (optional)
                            if row.get('delivered_qty', 0) < row.get('effective_allocated_qty', row.get('allocated_qty', 0)):
                                if st.button("🚚", key=f"deliver_{row['id']}", 
                                        help="Mark as delivered", disabled=True):
                                    st.info("Delivery function to be implemented")
                    
                    # Show cancel dialog if triggered
                    if st.session_state.get(f'show_cancel_{row["id"]}', False):
                        show_cancellation_dialog(row['id'], row.to_dict())
                    
                    st.divider()
        else:
            st.info("No details to display")

    with tab3:
        # Cancellation history
        show_cancellation_history(allocation_id)

    with tab4:
        # Analytics
        st.markdown("##### Allocation Performance")
        
        if not details.empty:
            # Performance metrics - FIX column names
            col1, col2, col3, col4 = st.columns(4)
            
            # Use original_allocated_qty instead of requested_qty
            total_requested = details.get('original_allocated_qty', 0).sum()
            total_allocated_orig = details.get('original_allocated_qty', 0).sum()
            total_cancelled = details.get('cancelled_qty', 0).sum()
            total_delivered = details.get('delivered_qty', 0).sum()
            
            # Current effective allocated
            total_allocated_effective = details.get('allocated_qty', 0).sum()
            
            with col1:
                # Since we don't have requested_qty, show original allocation
                st.metric("Original Allocated", format_number(total_allocated_orig))
            
            with col2:
                if total_cancelled > 0:
                    cancel_rate = (total_cancelled / total_allocated_orig * 100) if total_allocated_orig > 0 else 0
                    st.metric("Cancellation Rate", f"{cancel_rate:.1f}%", 
                            delta=f"-{format_number(total_cancelled)} units", delta_color="inverse")
                else:
                    st.metric("Cancellation Rate", "0%")
            
            with col3:
                delivery_rate = (total_delivered / total_allocated_effective * 100) if total_allocated_effective > 0 else 0
                st.metric("Delivery Rate", f"{delivery_rate:.1f}%",
                        help="Delivered vs effective allocated")
            
            with col4:
                pending = total_allocated_effective - total_delivered
                st.metric("Pending Delivery", format_number(pending))   
 
    # Export button
    # st.markdown("---")
    if st.button("📤 Export Details", key="export_allocation_details"):
        export_allocation_details(plan, details)
    
    # Handle bulk cancel dialog
    if st.session_state.get('show_bulk_cancel', False):
        show_bulk_cancel_dialog()


def show_bulk_cancel_dialog():
    """Show bulk cancellation dialog"""
    selected_ids = st.session_state.get('selected_cancel_ids', [])
    
    if not selected_ids:
        st.error("No items selected")
        return
    
    with st.form("bulk_cancel_form"):
        st.markdown("### 🚫 Bulk Cancel Allocations")
        st.info(f"Cancelling {len(selected_ids)} allocation(s)")
        
        reason_category = st.selectbox(
            "Reason Category",
            options=['CUSTOMER_REQUEST', 'SUPPLY_ISSUE', 'QUALITY_ISSUE', 'BUSINESS_DECISION', 'OTHER'],
            format_func=lambda x: x.replace('_', ' ').title()
        )
        
        reason = st.text_area(
            "Detailed Reason (applies to all)",
            placeholder="Please provide reason for bulk cancellation...",
            height=100
        )
        
        col1, col2 = st.columns(2)
        with col1:
            if st.form_submit_button("Confirm Bulk Cancel", type="primary"):
                if reason:
                    success_count, errors = allocation_manager.bulk_cancel_details(
                        selected_ids, reason, reason_category
                    )
                    
                    if success_count > 0:
                        st.success(f"Successfully cancelled {success_count} items")
                    
                    if errors:
                        st.error("Errors occurred:")
                        for error in errors:
                            st.write(f"- {error}")
                    
                    # Clean up session state
                    del st.session_state['show_bulk_cancel']
                    del st.session_state['selected_cancel_ids']
                    st.rerun()
                else:
                    st.error("Please provide a reason")
        
        with col2:
            if st.form_submit_button("Cancel"):
                del st.session_state['show_bulk_cancel']
                del st.session_state['selected_cancel_ids']
                st.rerun()


def export_allocation_details(plan, details):
    """Export allocation plan with details"""
    # Create summary sheet
    summary_data = {
        'Field': ['Allocation Number', 'Date', 'Method', 'Type', 'Status', 
                  'Total Allocated', 'Total Delivered', 'HARD Allocations', 'SOFT Allocations'],
        'Value': [
            plan['allocation_number'],
            plan['allocation_date'].strftime('%Y-%m-%d %H:%M'),
            plan['allocation_method'],
            plan.get('allocation_type', 'SOFT'),
            plan['status'],
            details['allocated_qty'].sum(),
            details['delivered_qty'].sum(),
            len(details[details.get('allocation_mode', 'SOFT') == 'HARD']) if 'allocation_mode' in details.columns else 0,
            len(details[details.get('allocation_mode', 'SOFT') == 'SOFT']) if 'allocation_mode' in details.columns else len(details)
        ]
    }
    summary_df = pd.DataFrame(summary_data)
    
    # Add snapshot context if available
    if plan.get('snapshot_context'):
        snapshot_df = pd.DataFrame([{
            'Snapshot Time': plan['snapshot_context'].get('snapshot_datetime'),
            'Total Products': plan['snapshot_context'].get('summary', {}).get('total_products'),
            'Shortage Products': plan['snapshot_context'].get('summary', {}).get('shortage_products'),
            'Fulfillment Rate': plan['snapshot_context'].get('summary', {}).get('fulfillment_rate')
        }])
    else:
        snapshot_df = pd.DataFrame()
    
    # Export multiple sheets
    sheets = {
        'Summary': summary_df,
        'Details': details,
        'By Product': details.groupby(['pt_code', 'product_name']).agg({
            'requested_qty': 'sum',
            'allocated_qty': 'sum',
            'delivered_qty': 'sum'
        }).reset_index(),
        'By Customer': details.groupby('customer_name').agg({
            'requested_qty': 'sum',
            'allocated_qty': 'sum',
            'delivered_qty': 'sum'
        }).reset_index()
    }
    
    if not snapshot_df.empty:
        sheets['Snapshot Context'] = snapshot_df
    
    # Add HARD allocation details if any
    if 'allocation_mode' in details.columns and len(details[details['allocation_mode'] == 'HARD']) > 0:
        hard_details = details[details['allocation_mode'] == 'HARD'].copy()
        if 'supply_reference' in hard_details.columns:
            sheets['HARD Allocations'] = hard_details[[
                'pt_code', 'customer_name', 'allocated_qty', 
                'supply_source_type', 'supply_reference'
            ]]
    
    excel_data = export_multiple_sheets(sheets)
    
    st.download_button(
        label="📥 Download Allocation Report",
        data=excel_data,
        file_name=f"allocation_{plan['allocation_number']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

def show_edit_allocation():
    """Edit draft allocation plan"""
    st.markdown("### ✏️ Edit Allocation Plan")
    st.info("Edit functionality will be implemented based on specific requirements")
    
    if st.button("← Back to List"):
        st.session_state['allocation_mode'] = 'list'
        st.rerun()

def show_cancellation_history(plan_id: int):
    """Show cancellation history for a plan"""
    st.markdown("### 📜 Cancellation History")
    
    history_df = allocation_manager.cancellation_manager.get_cancellation_history(plan_id=plan_id)
    
    if history_df.empty:
        st.info("No cancellations recorded")
        return
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    active_cancellations = history_df[history_df['status'] == 'ACTIVE']
    reversed_cancellations = history_df[history_df['status'] == 'REVERSED']
    
    with col1:
        st.metric("Total Cancellations", len(history_df))
    with col2:
        st.metric("Active", len(active_cancellations))
    with col3:
        st.metric("Reversed", len(reversed_cancellations))
    with col4:
        st.metric("Total Cancelled Qty", f"{active_cancellations['cancelled_qty'].sum():,.0f}")
    
    # History table
    for idx, row in history_df.iterrows():
        with st.container():
            col1, col2, col3, col4 = st.columns([2, 2, 1, 1])
            
            with col1:
                st.write(f"**{row['pt_code']}** - {row['customer_name']}")
                st.caption(f"Cancelled: {row['cancelled_qty']:,.0f} units")
            
            with col2:
                status_color = "🟢" if row['status'] == 'ACTIVE' else "🔴"
                st.write(f"{status_color} {row['status']}")
                st.caption(f"Date: {row['cancelled_date'].strftime('%Y-%m-%d %H:%M')}")
            
            with col3:
                st.write(f"**Reason:** {row['reason_category']}")
                with st.expander("Details"):
                    st.write(row['reason'])
            
            with col4:
                if row['status'] == 'ACTIVE' and row['delivered_qty'] == 0:
                    if st.button("↩️ Reverse", key=f"reverse_{row['id']}"):
                        # Show reversal form
                        with st.form(f"reverse_form_{row['id']}"):
                            reversal_reason = st.text_input("Reversal Reason")
                            if st.form_submit_button("Confirm Reverse"):
                                success, msg = allocation_manager.cancellation_manager.reverse_cancellation(
                                    row['id'], 
                                    st.session_state.get('user_id', 1),
                                    reversal_reason
                                )
                                if success:
                                    st.success(msg)
                                    st.rerun()
                                else:
                                    st.error(msg)
            
            if row['status'] == 'REVERSED':
                st.info(f"Reversed by {row['reversed_by']} on {row['reversed_date'].strftime('%Y-%m-%d')}")
                st.caption(f"Reversal reason: {row['reversal_reason']}")
            
            st.divider()

# Thêm function để show cancellation dialog
def show_cancellation_dialog(detail_id: int, detail_info: Dict):
    """Show cancellation dialog"""
    with st.form(f"cancel_form_{detail_id}"):
        st.markdown("### 🚫 Cancel Allocation")
        
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Product:** {detail_info['pt_code']} - {detail_info['product_name']}")
            st.write(f"**Customer:** {detail_info['customer_name']}")
        
        with col2:
            st.metric("Allocated", f"{detail_info['allocated_qty']:,.0f}")
            st.metric("Delivered", f"{detail_info['delivered_qty']:,.0f}")
            st.metric("Available to Cancel", f"{detail_info['cancellable_qty']:,.0f}")
        
        # Cancellation inputs
        cancel_qty = st.number_input(
            "Quantity to Cancel",
            min_value=0.0,
            max_value=float(detail_info['cancellable_qty']),
            value=float(detail_info['cancellable_qty']),
            step=1.0
        )
        
        reason_category = st.selectbox(
            "Reason Category",
            options=['CUSTOMER_REQUEST', 'SUPPLY_ISSUE', 'QUALITY_ISSUE', 'BUSINESS_DECISION', 'OTHER'],
            format_func=lambda x: x.replace('_', ' ').title()
        )
        
        reason = st.text_area(
            "Detailed Reason",
            placeholder="Please provide detailed reason for cancellation...",
            height=100
        )
        
        st.warning("⚠️ This action cannot be undone after delivery starts")
        
        col1, col2 = st.columns(2)
        with col1:
            submitted = st.form_submit_button("Confirm Cancel", type="primary")
        with col2:
            cancelled = st.form_submit_button("Close")
        
        if submitted and reason:
            # Process cancellation
            success, message = allocation_manager.cancellation_manager.cancel_quantity(
                detail_id=detail_id,
                quantity=cancel_qty,
                reason=reason,
                reason_category=reason_category,
                user_id=st.session_state.get('user_id', 1)
            )
            
            if success:
                st.success(message)
                st.rerun()
            else:
                st.error(message)
        elif submitted and not reason:
            st.error("Please provide a reason for cancellation")

# ==== Main Execution ====
# === Page Config ===
st.set_page_config(
    page_title="Allocation Plan - SCM",
    page_icon="🧩",
    layout="wide"
)

# === Initialize Session State ===
initialize_session_state()

# Initialize allocation session state
if 'allocation_mode' not in st.session_state:
    st.session_state['allocation_mode'] = 'list'  # list, create, edit, view
if 'allocation_step' not in st.session_state:
    st.session_state['allocation_step'] = 1
if 'draft_allocation' not in st.session_state:
    st.session_state['draft_allocation'] = {}  # Initialize as empty dict instead of None
if 'selected_allocation_products' not in st.session_state:
    st.session_state['selected_allocation_products'] = []

# === Constants ===
ALLOCATION_METHODS = {
    'FCFS': '📅 First Come First Served - Prioritize earliest ETD',
    'PRIORITY': '⭐ Priority Based - Based on customer priority',
    'PRO_RATA': '⚖️ Pro Rata - Proportional distribution',
    'MANUAL': '✋ Manual - Custom allocation'
}

ALLOCATION_TYPES = {
    'SOFT': '🌊 Soft Allocation - Flexible quantity allocation',
    'HARD': '🔒 Hard Allocation - Lock specific supply batches',
    # 'MIXED': '🔀 Mixed - Combination of soft and hard'
}

# Update ALLOCATION_STATUS_COLORS constant
ALLOCATION_STATUS_COLORS = {
    'ALL_DRAFT': 'gray',
    'IN_PROGRESS': 'blue', 
    'ALL_DELIVERED': 'green',
    'ALL_CANCELLED': 'red',
    'MIXED_DRAFT': 'orange',
    'MIXED': 'violet',
    'EMPTY': 'gray'
}

# === Initialize Components ===
@st.cache_resource
def get_allocation_manager():
    return AllocationManager()

allocation_manager = get_allocation_manager()

# === Debug Mode Toggle ===
col_debug1, col_debug2 = st.columns([6, 1])
with col_debug2:
    debug_mode = st.checkbox("🐛 Debug Mode", value=False, key="debug_mode")

if debug_mode:
    st.info("🐛 Debug Mode is ON - Additional information will be displayed")
    
    # Show current allocation state
    with st.expander("🔍 Current Session State (Allocation)", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Mode & Step:**")
            st.write(f"- allocation_mode: {st.session_state.get('allocation_mode', 'N/A')}")
            st.write(f"- allocation_step: {st.session_state.get('allocation_step', 'N/A')}")
            st.write(f"- selected_allocation_id: {st.session_state.get('selected_allocation_id', 'N/A')}")
        
        with col2:
            st.write("**Data State:**")
            draft = st.session_state.get('draft_allocation', {})
            st.write(f"- draft_allocation keys: {list(draft.keys()) if draft else 'Empty'}")
            st.write(f"- selected products: {len(st.session_state.get('selected_allocation_products', []))} items")
            
            # Check if we have GAP data
            gap_data = st.session_state.get('gap_analysis_result')
            if gap_data is not None and hasattr(gap_data, 'shape'):
                st.write(f"- GAP data available: {gap_data.shape}")
            else:
                st.write("- GAP data: Not available")

# === Header ===
DisplayComponents.show_page_header(
    title="Allocation Plan Management",
    icon="🧩",
    prev_page="pages/3_📊_GAP_Analysis.py",
    next_page="pages/5_📌_PO_Suggestions.py"
)

# === Additional Debug Info ===
if debug_mode:
    # Show timestamp for debugging
    st.caption(f"🕐 Page loaded at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}")

# === Mode Selection ===
col1, col2, col3, col4 = st.columns(4)
with col1:
    if st.button("📋 Allocation List", use_container_width=True):
        st.session_state['allocation_mode'] = 'list'
        st.session_state['allocation_step'] = 1

with col2:
    # Check if we have data (GAP analysis or demand data)
    gap_data = get_from_session_state('gap_analysis_result', pd.DataFrame())
    demand_data = get_from_session_state('demand_filtered', pd.DataFrame())
    has_data = not gap_data.empty or not demand_data.empty
    
    if st.button("➕ Create New", type="primary" if has_data else "secondary", 
                 use_container_width=True, disabled=not has_data):
        st.session_state['allocation_mode'] = 'create'
        st.session_state['allocation_step'] = 1
        st.session_state['draft_allocation'] = {}  # Initialize as empty dict instead of None
        st.session_state['selected_allocation_products'] = []
        st.session_state['supply_mapping'] = {}
        st.session_state['temp_allocation_results'] = None
        st.rerun()

if not has_data and st.session_state['allocation_mode'] == 'list':
    st.info("💡 Run GAP Analysis or load demand data to create allocation plans")

# === Main Content Based on Mode ===
if st.session_state['allocation_mode'] == 'list':
    show_allocation_list()
    
elif st.session_state['allocation_mode'] == 'create':
    show_create_allocation_wizard()
    
elif st.session_state['allocation_mode'] == 'edit':
    show_edit_allocation()
    
elif st.session_state['allocation_mode'] == 'view':
    show_view_allocation()

# === Help Section ===
DisplayComponents.show_help_section(
    "Allocation Plan Guide",
    """
    ### Allocation Types
    
    **SOFT Allocation (90% of cases)**
    - Allocates quantities only, not specific batches
    - System automatically selects best supply at delivery time
    - Flexible - can adjust supply sources as needed
    - Recommended for most scenarios
    
    **HARD Allocation (10% special cases)**  
    - Locks specific supply batches to customer orders
    - Required when customers need specific origin/quality
    - Cannot change supply source after approval
    - Use only when absolutely necessary
    
    **MIXED Allocation**
    - Combination of SOFT and HARD in same plan
    - Configure allocation mode per product/order
    - Maximum flexibility for complex scenarios
    
    ### Allocation Methods
    
    **FCFS (First Come First Served)**
    - Prioritizes orders with earliest ETD
    - Ensures older orders are fulfilled first
    - Best for time-sensitive products
    
    **Priority Based**
    - Allocates based on customer importance
    - Set priority scores (1-10) for each customer
    - Higher priority customers get preference
    
    **Pro Rata**
    - Distributes proportionally based on demand
    - Fair distribution across all orders
    - Can set minimum allocation percentage
    
    **Manual**
    - Full control over allocation quantities
    - Drag and adjust allocations as needed
    - Best for complex scenarios
    
    ### Workflow
    
    **For SOFT Allocation (5 steps):**
    1. **Select Products**: Choose any products for allocation (shortage, available, or all)
    2. **Choose Method & Type**: Select allocation method (FCFS/Priority/Pro-rata/Manual) and SOFT type
    3. **Set Parameters**: Configure method-specific settings (priorities, minimum %, etc.)
    4. **Preview & Adjust**: Review allocations, manually adjust if needed
    5. **Confirm & Save**: Save as draft or approve immediately
    
    **For HARD/MIXED Allocation (6 steps):**
    1. **Select Products**: Choose any products for allocation
    2. **Choose Method & Type**: Select allocation method and HARD/MIXED type
    3. **Set Parameters**: Configure method-specific settings
    4. **Map Supply Sources**: Link specific batches/lots to customer orders
    5. **Preview with Mapping**: Review allocations with supply assignments
    6. **Confirm & Save**: Save as draft or approve immediately
    
    ### Best Practices
    - Allocation can be created for ALL products, not just shortage ones
    - Use filters to focus on specific product groups (shortage/available)
    - SOFT allocation suitable for most cases (90%)
    - Use HARD allocation only when specific batch requirements exist
    - Consider customer credit limits
    - Allow partial allocations for flexibility
    - Document reasoning in notes
    - Review allocation performance regularly
    
    ### Snapshot Context
    - Each allocation captures GAP Analysis context
    - Includes time adjustments, filters, and summary
    - Provides audit trail and reproducibility
    - Helps understand allocation decisions later
    """
)

# Footer
st.markdown("---")
st.caption(f"Allocation Plan Module | Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")