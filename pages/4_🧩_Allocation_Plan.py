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
from utils.formatters import format_number
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

def show_allocation_list():
    """Display allocation management with tabs"""
    st.markdown("### 📋 Allocation Management")
    
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
        date_from = st.date_input("ETD From", value=datetime.now() - timedelta(days=7))
    
    with col3:
        date_to = st.date_input("ETD To", value=datetime.now() + timedelta(days=30))
    
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
                            if allocation_manager.bulk_update_allocation_status(row['id'], 'ALLOCATED'):
                                st.success("Plan allocated successfully!")
                                st.rerun()
                    with col6_2:
                        if st.button("❌ Cancel", key=f"cancel_plan_{row['id']}"):
                            if allocation_manager.cancel_allocation_plan(row['id']):
                                st.info("Plan cancelled")
                                st.rerun()
                
                elif row['display_status'] in ['IN_PROGRESS', 'MIXED']:
                    # Show delivery progress
                    delivered_pct = (row.get('total_delivered', 0) / row.get('total_allocated_effective', 1) * 100) if row.get('total_allocated_effective', 0) > 0 else 0
                    st.progress(delivered_pct / 100)
                    st.caption(f"{delivered_pct:.0f}% delivered")

            st.divider()


def show_create_allocation_wizard():
    """Multi-step wizard for creating allocation plan"""
    st.markdown("### ➕ Create New Allocation Plan")
    
    # Initialize draft_allocation if not exists
    if 'draft_allocation' not in st.session_state or st.session_state['draft_allocation'] is None:
        st.session_state['draft_allocation'] = {}
    
    # Adjust total steps based on allocation type
    allocation_type = st.session_state['draft_allocation'].get('allocation_type', 'SOFT')
    total_steps = 6 if allocation_type in ['HARD', 'MIXED'] else 5
    
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
        if allocation_type in ['HARD', 'MIXED']:
            steps = ['Select Products', 'Choose Method', 'Set Parameters', 'Map Supply', 'Preview', 'Confirm']
        else:
            steps = ['Select Products', 'Choose Method', 'Set Parameters', 'Preview', 'Confirm']
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
        if allocation_type in ['HARD', 'MIXED']:
            show_step4_map_supply()
        else:
            show_step4_preview()
    elif st.session_state['allocation_step'] == 5:
        if allocation_type in ['HARD', 'MIXED']:
            show_step5_preview_with_mapping()
        else:
            show_step5_confirm()
    elif st.session_state['allocation_step'] == 6:
        show_step6_confirm_with_mapping()

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
            options=list(ALLOCATION_TYPES.keys()),
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
        else:  # MIXED
            st.info("""
            **Mixed Allocation**
            - Some products use SOFT allocation
            - Some products use HARD allocation
            - Configure per product in next steps
            - For complex scenarios
            """)
    
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
            
            # Check allocation type to determine next step
            allocation_type = st.session_state['draft_allocation'].get('allocation_type', 'SOFT')
            if allocation_type in ['HARD', 'MIXED']:
                # Calculate allocation first for HARD mapping
                method = st.session_state['draft_allocation'].get('method', 'FCFS')
                parameters = st.session_state['draft_allocation'].get('parameters', {})
                selected_products = st.session_state.get('selected_allocation_products', [])
                
                demand_data = get_from_session_state('demand_filtered', pd.DataFrame())
                supply_data = get_from_session_state('supply_filtered', pd.DataFrame())
                
                filtered_demand = demand_data[demand_data['pt_code'].isin(selected_products)].copy()
                filtered_supply = supply_data[supply_data['pt_code'].isin(selected_products)].copy()
                
                allocation_results = AllocationMethods.calculate_allocation(
                    demand_df=filtered_demand,
                    supply_df=filtered_supply,
                    method=method,
                    parameters=parameters
                )
                st.session_state['temp_allocation_results'] = allocation_results
            
            st.session_state['allocation_step'] = 4
            st.rerun()

def show_step4_map_supply():
    """Step 4: Map supply sources for HARD allocation"""
    st.markdown("#### 🔗 Map Supply to Demand (HARD Allocation)")
    
    allocation_type = st.session_state['draft_allocation'].get('allocation_type', 'SOFT')
    
    # Get allocation results from previous calculation
    allocation_results = st.session_state.get('temp_allocation_results')
    if allocation_results is None:
        st.error("Allocation results not found. Please go back to previous step.")
        return
    
    # Get supply data from session state (from GAP Analysis)
    supply_filtered = st.session_state.get('supply_filtered', pd.DataFrame())
    
    if supply_filtered.empty:
        st.error("No supply data found. Please run GAP Analysis first.")
        return
    
    # Get unique products and entities from allocation results
    product_codes = allocation_results['pt_code'].unique().tolist()
    legal_entities = allocation_results['legal_entity'].unique().tolist()
    
    # Get available supply using the corrected function
    available_supply = allocation_manager.get_available_supply_for_hard_allocation(
        product_codes, legal_entities
    )
    
    if available_supply.empty:
        st.error("No available supply found for HARD allocation")
        # Allow user to go back
        if st.button("← Go Back to Previous Step"):
            st.session_state['allocation_step'] = 3
            st.rerun()
        return
    
    # Show info about data source
    st.info(f"""
        📊 Using supply data from GAP Analysis:
        - Total supply items: {len(supply_filtered)}
        - Filtered for selected products: {len(available_supply)}
        - Time adjustments applied: {st.session_state.get('time_adjustments', {}).get('supply_arrival_offset', 0)} days
    """)
    
    # Initialize supply mapping in session state
    if 'supply_mapping' not in st.session_state:
        st.session_state['supply_mapping'] = {}
    
    # Group allocations by product for easier mapping
    allocation_by_product = allocation_results.groupby('pt_code')
    
    # For MIXED allocation, allow selection of which items are HARD
    if allocation_type == 'MIXED':
        st.markdown("##### Select items for HARD allocation")
        st.info("Items not selected will use SOFT allocation (flexible supply)")
        
        hard_allocation_items = []
        for _, row in allocation_results.iterrows():
            key = f"hard_{row['demand_line_id']}"
            if st.checkbox(
                f"{row['pt_code']} - {row['customer']} ({row['allocated_qty']:.0f} units)",
                key=key,
                value=key in st.session_state.get('selected_hard_items', [])
            ):
                hard_allocation_items.append(row['demand_line_id'])
        
        st.session_state['selected_hard_items'] = hard_allocation_items
        
        # Filter to show only selected items for mapping
        if hard_allocation_items:
            allocation_to_map = allocation_results[
                allocation_results['demand_line_id'].isin(hard_allocation_items)
            ]
        else:
            st.warning("No items selected for HARD allocation")
            allocation_to_map = pd.DataFrame()
    else:
        # For pure HARD allocation, map all items
        allocation_to_map = allocation_results
    
    if not allocation_to_map.empty:
        st.markdown("##### Map Supply Sources")
        
        # Create mapping interface
        for pt_code, product_allocations in allocation_to_map.groupby('pt_code'):
            with st.expander(f"📦 {pt_code}", expanded=True):
                # Get available supply for this product
                product_supply = available_supply[available_supply['pt_code'] == pt_code]
                
                if product_supply.empty:
                    st.warning(f"No supply available for {pt_code}")
                    continue
                
                # Show supply summary
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.metric("Available Supply", f"{product_supply['available_qty'].sum():.0f}")
                    st.metric("Allocation Needed", f"{product_allocations['allocated_qty'].sum():.0f}")
                
                with col2:
                    # Supply breakdown by source
                    supply_by_source = product_supply.groupby('source_type')['available_qty'].sum()
                    for source, qty in supply_by_source.items():
                        st.write(f"- {source}: {qty:.0f} units")
                
                # Map each allocation line
                for _, alloc_row in product_allocations.iterrows():
                    st.markdown(f"**Customer: {alloc_row['customer']} - {alloc_row['allocated_qty']:.0f} units**")
                    
                    # Create supply options
                    supply_options = []
                    for _, supply_row in product_supply.iterrows():
                        option_text = (
                            f"{supply_row['source_type']} - "
                            f"{supply_row['reference']} - "
                            f"{supply_row['available_qty']:.0f} units"
                        )
                        if 'expected_date' in supply_row and pd.notna(supply_row['expected_date']):
                            option_text += f" (ETA: {supply_row['expected_date'].strftime('%Y-%m-%d')})"
                        
                        supply_options.append({
                            'text': option_text,
                            'value': f"{supply_row['source_type']}_{supply_row['source_id']}",
                            'source_type': supply_row['source_type'],
                            'source_id': supply_row['source_id']
                        })
                    
                    # Supply selection with better error handling
                    selected_key = st.selectbox(
                        "Select Supply Source",
                        options=[opt['value'] for opt in supply_options],
                        format_func=lambda x: next((opt['text'] for opt in supply_options if opt['value'] == x), ''),
                        key=f"supply_{alloc_row['demand_line_id']}"
                    )
                    
                    # Store mapping with error handling
                    if selected_key:
                        # Safe iteration with default
                        selected_supply = next(
                            (opt for opt in supply_options if opt['value'] == selected_key), 
                            None
                        )
                        
                        if selected_supply:
                            st.session_state['supply_mapping'][str(alloc_row['demand_line_id'])] = {
                                'source_type': selected_supply['source_type'],
                                'source_id': selected_supply['source_id']
                            }
                        else:
                            logger.warning(f"Could not find supply option for key: {selected_key}")
                    
                    st.divider()
    
    # Validation
    if allocation_type == 'HARD' or (allocation_type == 'MIXED' and st.session_state.get('selected_hard_items')):
        mapped_count = len(st.session_state.get('supply_mapping', {}))
        required_count = len(allocation_to_map) if allocation_type == 'HARD' else len(st.session_state.get('selected_hard_items', []))
        
        if mapped_count < required_count:
            st.warning(f"Please map all allocations. Mapped: {mapped_count}/{required_count}")
    
    # Next button
    st.markdown("---")
    col1, col2, col3 = st.columns([2, 1, 2])
    with col2:
        # Check if all required mappings are done
        can_proceed = True
        if allocation_type == 'HARD':
            can_proceed = len(st.session_state.get('supply_mapping', {})) >= len(allocation_to_map)
        elif allocation_type == 'MIXED' and st.session_state.get('selected_hard_items'):
            can_proceed = all(
                str(item_id) in st.session_state.get('supply_mapping', {})
                for item_id in st.session_state.get('selected_hard_items', [])
            )
        
        if st.button("Next ➡️", type="primary", use_container_width=True, disabled=not can_proceed):
            st.session_state['allocation_step'] = 5
            st.rerun()

def show_step4_preview():
    """Step 4: Preview allocation results with dynamic updates"""
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
    
    # Create a container for dynamic metrics
    metrics_container = st.container()
    
    # Allocation details table
    st.markdown("##### Allocation Details")
    st.info("💡 **Note**: System calculated allocations are shown. Adjust 'Actual Allocated' and 'Allocated ETD' as needed.")
    
    # Allow editing
    edited_df = AllocationComponents.show_editable_allocation_table(allocation_results)
    st.session_state['draft_allocation']['results'] = edited_df
    
    # Update metrics dynamically based on edited values
    with metrics_container:
        col1, col2, col3, col4 = st.columns(4)
        
        total_demand = edited_df['requested_qty'].sum()
        total_allocated = edited_df['allocated_qty'].sum()  # Use actual allocated
        total_orders = len(edited_df)
        
        # Calculate fulfillment based on actual allocated
        if total_demand > 0:
            avg_fulfillment = (total_allocated / total_demand) * 100
        else:
            avg_fulfillment = 0
        
        with col1:
            st.metric("Total Demand", format_number(total_demand))
        with col2:
            st.metric(
                "Total Allocated", 
                format_number(total_allocated),
                delta=f"{total_allocated - edited_df['calculated_allocation'].sum():.0f} vs calculated" if 'calculated_allocation' in edited_df.columns else None
            )
        with col3:
            st.metric("Orders", total_orders)
        with col4:
            st.metric("Avg Fulfillment", f"{avg_fulfillment:.1f}%")
    
    # Visualization based on actual allocated values
    col1, col2 = st.columns(2)
    
    with col1:
        # Fulfillment by product using actual allocated
        fig1 = AllocationComponents.create_fulfillment_chart_by_product(edited_df)
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        # Fulfillment by customer using actual allocated
        fig2 = AllocationComponents.create_fulfillment_chart_by_customer(edited_df)
        st.plotly_chart(fig2, use_container_width=True)
    
    # Show allocation variance analysis
    if 'calculated_allocation' in edited_df.columns:
        with st.expander("📊 Allocation Variance Analysis", expanded=False):
            variance_df = edited_df[['pt_code', 'customer', 'calculated_allocation', 'allocated_qty']].copy()
            variance_df['variance'] = variance_df['allocated_qty'] - variance_df['calculated_allocation']
            variance_df['variance_pct'] = (variance_df['variance'] / variance_df['calculated_allocation'] * 100).fillna(0)
            
            # Show only items with variance
            variance_df = variance_df[variance_df['variance'] != 0]
            
            if not variance_df.empty:
                st.dataframe(
                    variance_df.style.format({
                        'calculated_allocation': '{:.0f}',
                        'allocated_qty': '{:.0f}',
                        'variance': '{:+.0f}',
                        'variance_pct': '{:+.1f}%'
                    }),
                    use_container_width=True
                )
            else:
                st.info("No variance - all allocations match system calculations")
    
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
        if st.button("Next ➡️", type="primary", use_container_width=True):
            st.session_state['allocation_step'] = 5
            st.rerun()


def show_step5_preview_with_mapping():
    """Step 5: Preview allocation with supply mapping (for HARD allocation)"""
    st.markdown("#### 👀 Preview Allocation Results")
    
    # Get allocation results
    allocation_results = st.session_state.get('temp_allocation_results', pd.DataFrame())
    supply_mapping = st.session_state.get('supply_mapping', {})
    
    if allocation_results.empty:
        st.error("No allocation results found")
        return
    
    # Add supply mapping info to results
    allocation_results['allocation_mode'] = 'SOFT'
    allocation_results['supply_reference'] = None
    
    for demand_key, mapping in supply_mapping.items():
        mask = allocation_results['demand_line_id'].astype(str) == str(demand_key)
        allocation_results.loc[mask, 'allocation_mode'] = 'HARD'
        allocation_results.loc[mask, 'supply_reference'] = mapping.get('source_type', '') + ' - ' + str(mapping.get('source_id', ''))
    
    # Store updated results
    st.session_state['draft_allocation']['results'] = allocation_results
    
    # Summary metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    total_demand = allocation_results['requested_qty'].sum()
    total_allocated = allocation_results['allocated_qty'].sum()
    total_orders = len(allocation_results)
    hard_orders = len(allocation_results[allocation_results['allocation_mode'] == 'HARD'])
    avg_fulfillment = (allocation_results['allocated_qty'] / allocation_results['requested_qty']).mean() * 100
    
    with col1:
        st.metric("Total Demand", format_number(total_demand))
    with col2:
        st.metric("Total Allocated", format_number(total_allocated))
    with col3:
        st.metric("Total Orders", total_orders)
    with col4:
        st.metric("HARD Orders", hard_orders)
    with col5:
        st.metric("Avg Fulfillment", f"{avg_fulfillment:.1f}%")
    
    # Allocation details table
    st.markdown("##### Allocation Details")
    
    # Display table with allocation mode
    display_df = allocation_results[[
        'pt_code', 'product_name', 'customer', 'etd', 
        'requested_qty', 'allocated_qty', 'fulfillment_rate',
        'allocation_mode', 'supply_reference'
    ]].copy()
    
    # Format columns
    display_df['requested_qty'] = display_df['requested_qty'].apply(format_number)
    display_df['allocated_qty'] = display_df['allocated_qty'].apply(format_number)
    display_df['fulfillment_rate'] = display_df['fulfillment_rate'].apply(lambda x: f"{x:.1f}%")
    display_df['allocation_mode'] = display_df['allocation_mode'].apply(
        lambda x: '🔒 HARD' if x == 'HARD' else '🌊 SOFT'
    )
    
    st.dataframe(display_df, use_container_width=True, height=400)
    
    # Visualization
    col1, col2 = st.columns(2)
    
    with col1:
        # Fulfillment by product
        fig1 = AllocationComponents.create_fulfillment_chart_by_product(allocation_results)
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        # Allocation mode distribution
        mode_summary = allocation_results.groupby('allocation_mode').agg({
            'allocated_qty': 'sum'
        }).reset_index()
        
        fig2 = go.Figure(data=[go.Pie(
            labels=mode_summary['allocation_mode'],
            values=mode_summary['allocated_qty'],
            hole=.3,
            marker=dict(colors=['#1f77b4', '#ff7f0e'])
        )])
        fig2.update_layout(title='Allocation by Mode')
        st.plotly_chart(fig2, use_container_width=True)
    
    # Next button
    st.markdown("---")
    col1, col2, col3 = st.columns([2, 1, 2])
    with col2:
        if st.button("Next ➡️", type="primary", use_container_width=True):
            st.session_state['allocation_step'] = 6
            st.rerun()

def show_step5_confirm():
    """Step 5: Confirm and save allocation (for SOFT allocation)"""
    st.markdown("#### ✅ Confirm Allocation Plan")
    
    # Summary
    st.markdown("##### Allocation Summary")
    
    draft = st.session_state.get('draft_allocation', {})
    results = draft.get('results', pd.DataFrame())
    
    if results.empty:
        st.error("No allocation results found")
        return
    
    # Display summary info
    col1, col2 = st.columns(2)
    
    with col1:
        st.write(f"**Method:** {draft.get('method', 'Unknown')}")
        st.write(f"**Type:** {draft.get('allocation_type', 'SOFT')}")
        st.write(f"**Products:** {len(results['pt_code'].unique())}")
        st.write(f"**Orders:** {len(results)}")
    
    with col2:
        st.write(f"**Total Allocated:** {format_number(results['allocated_qty'].sum())}")
        st.write(f"**Avg Fulfillment:** {(results['fulfillment_rate'].mean()):.1f}%")
        if draft.get('parameters', {}).get('notes'):
            st.write(f"**Notes:** {draft['parameters']['notes']}")
    
    # Action buttons
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("💾 Save as Draft", type="secondary", use_container_width=True):
            save_allocation('DRAFT')
    
    with col2:
        if st.button("✅ Save & Approve", type="primary", use_container_width=True):
            save_allocation('APPROVED')
    
    with col3:
        if st.button("📤 Export", use_container_width=True):
            export_allocation_results(results)
    
    with col4:
        if st.button("❌ Cancel", use_container_width=True):
            st.session_state['allocation_mode'] = 'list'
            st.session_state['allocation_step'] = 1
            st.session_state['draft_allocation'] = {}  # Reset to empty dict
            st.session_state['selected_allocation_products'] = []
            st.session_state['supply_mapping'] = {}
            st.session_state['temp_allocation_results'] = None
            st.rerun()

def show_step6_confirm_with_mapping():
    """Step 6: Confirm allocation with supply mapping (for HARD allocation)"""
    show_step5_confirm()  # Reuse existing confirm logic


def save_allocation(status):
    """Save allocation plan to database
    
    Args:
        status: 'DRAFT' or 'APPROVED' - determines initial detail status
        
    Note: Plan table no longer has status field, status is managed at detail level
    """
    try:
        # Helper function to safely convert numeric values
        def safe_float(value):
            """Convert to float safely, handling NaN and inf"""
            try:
                if pd.isna(value) or (isinstance(value, float) and (np.isnan(value) or np.isinf(value))):
                    return 0.0
                return float(value)
            except:
                return 0.0
        
        # Get draft data from session state
        draft = st.session_state.get('draft_allocation', {})
        results = draft.get('results', pd.DataFrame())
        
        if results.empty:
            st.error("No allocation results to save")
            return
        
        # Get supply mapping for HARD allocations
        supply_mapping = st.session_state.get('supply_mapping', {})
        
        # Validate HARD allocation supply mapping if applicable
        allocation_type = draft.get('allocation_type', 'SOFT')
        if allocation_type in ['HARD', 'MIXED'] and supply_mapping:
            logger.info("Validating HARD allocation supply mapping...")
            is_valid, validation_errors = allocation_manager.validate_hard_allocation_supply(
                supply_mapping, results
            )
            
            if not is_valid:
                st.error("❌ Supply validation failed:")
                for error in validation_errors:
                    st.error(f"• {error}")
                
                # Show option to go back and fix
                if st.button("← Go back to fix mappings", key="fix_mappings_btn"):
                    st.session_state['allocation_step'] = 4  # Go back to mapping step
                    st.rerun()
                return
        
        # Get GAP Analysis data for context
        gap_analysis_data = st.session_state.get('gap_analysis_result', pd.DataFrame())
        
        # Build comprehensive allocation context
        gap_context = {
            'snapshot_datetime': datetime.now().isoformat(),
            'gap_analysis': {
                'run_date': st.session_state.get('gap_analysis_run_date', datetime.now()).isoformat() if 'gap_analysis_run_date' in st.session_state else datetime.now().isoformat(),
                'total_products': len(gap_analysis_data['pt_code'].unique()) if not gap_analysis_data.empty else 0,
                'shortage_products': len(gap_analysis_data[gap_analysis_data['gap_quantity'] < 0]) if not gap_analysis_data.empty else 0,
                'surplus_products': len(gap_analysis_data[gap_analysis_data['gap_quantity'] > 0]) if not gap_analysis_data.empty else 0,
                'total_gap_value': safe_float(gap_analysis_data['gap_quantity'].sum()) if not gap_analysis_data.empty else 0,
                'fulfillment_rate': safe_float((gap_analysis_data['total_available'] / gap_analysis_data['total_demand_qty']).mean() * 100) if not gap_analysis_data.empty and (gap_analysis_data['total_demand_qty'] > 0).any() else 0
            },
            'time_adjustments': st.session_state.get('time_adjustments', {
                'etd_offset_days': 0,
                'supply_arrival_offset': 0,
                'wh_transfer_lead_time': 2,
                'transportation_time': 3
            }),
            'filters': {
                'entities': st.session_state.get('selected_entities', []),
                'products': st.session_state.get('selected_products', []),
                'brands': st.session_state.get('selected_brands', []),
                'customers': st.session_state.get('selected_customers', []),
                'date_range': {
                    'start': st.session_state.get('date_from', datetime.now()).strftime('%Y-%m-%d') if 'date_from' in st.session_state else datetime.now().strftime('%Y-%m-%d'),
                    'end': st.session_state.get('date_to', datetime.now() + timedelta(days=30)).strftime('%Y-%m-%d') if 'date_to' in st.session_state else (datetime.now() + timedelta(days=30)).strftime('%Y-%m-%d')
                }
            },
            'data_sources': {
                'demand_sources': st.session_state.get('selected_demand_sources', ['OC']),
                'supply_sources': st.session_state.get('selected_supply_sources', ['Inventory']),
                'include_converted_forecasts': st.session_state.get('include_converted_forecasts', False),
                'exclude_expired_inventory': st.session_state.get('exclude_expired_inventory', True)
            },
            'allocation_info': {
                'method': draft.get('method', 'MANUAL'),
                'type': draft.get('allocation_type', 'SOFT'),
                'parameters': draft.get('parameters', {}),
                'selected_products': st.session_state.get('selected_allocation_products', []),
                'total_orders': len(results),
                'total_allocated': safe_float(results['allocated_qty'].sum()),
                'avg_fulfillment': safe_float(results['fulfillment_rate'].mean()) if 'fulfillment_rate' in results.columns else 0
            }
        }
        
        # Store allocation method and type in context (since not in plan table anymore)
        gap_context['allocation_method'] = draft.get('method', 'MANUAL')
        gap_context['allocation_type'] = draft.get('allocation_type', 'SOFT')
        
        # Add allocation mode breakdown to context
        if draft.get('allocation_type') in ['HARD', 'MIXED']:
            hard_count = len(supply_mapping)
            soft_count = len(results) - hard_count
            allocation_modes = {
                'SOFT': soft_count,
                'HARD': hard_count
            }
            gap_context['allocation_info']['allocation_modes'] = allocation_modes
            
            logger.info(f"Allocation mode breakdown - SOFT: {soft_count}, HARD: {hard_count}")
        
        # Create plan data - use SCM system user (id=1) as default
        plan_data = {
            'creator_id': st.session_state.get('user_id', 1),  # Default to 1 (SCM system user)
            'notes': draft.get('parameters', {}).get('notes', ''),
            'allocation_context': gap_context,
            'status': status  # Used to determine initial detail status (DRAFT or APPROVED)
        }
        
        # Log save action
        logger.info(f"Saving allocation plan with initial status: {status}")
        logger.info(f"Total items: {len(results)}, Total allocated: {safe_float(results['allocated_qty'].sum())}")
        logger.info(f"Creator ID: {plan_data['creator_id']}")
        
        if supply_mapping:
            logger.info(f"HARD allocations: {len(supply_mapping)}")
        
        # Progress indicator
        progress_placeholder = st.empty()
        progress_placeholder.info("💾 Saving allocation plan...")
        
        # Save to database
        allocation_id = allocation_manager.create_allocation_plan(
            plan_data, 
            results, 
            supply_mapping
        )
        
        if allocation_id:
            progress_placeholder.success(f"✅ Allocation plan saved successfully! ID: {allocation_id}")
            logger.info(f"Successfully created allocation plan ID: {allocation_id}")
            
            # If APPROVED, update all details from DRAFT to ALLOCATED
            if status == 'APPROVED':
                with st.spinner("📝 Updating allocation status..."):
                    success = allocation_manager.bulk_update_allocation_status(
                        allocation_id, 'ALLOCATED'
                    )
                    if success:
                        st.info("✅ All items marked as ALLOCATED and ready for delivery")
                        logger.info(f"Updated all details to ALLOCATED status for plan {allocation_id}")
                    else:
                        st.warning("⚠️ Plan created but failed to update status to ALLOCATED")
            
            # Show summary info
            with st.expander("📊 Allocation Summary", expanded=True):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Allocation Number", allocation_id)
                    st.metric("Total Products", len(results['pt_code'].unique()))
                with col2:
                    st.metric("Total Orders", len(results))
                    st.metric("Total Quantity", format_number(results['allocated_qty'].sum()))
                with col3:
                    st.metric("Method", draft.get('method', 'MANUAL'))
                    st.metric("Type", draft.get('allocation_type', 'SOFT'))
                    
                # Show HARD allocation breakdown if applicable
                if supply_mapping:
                    st.markdown("##### 🔒 HARD Allocation Details")
                    hard_summary = results[results['demand_line_id'].astype(str).isin(supply_mapping.keys())]
                    st.write(f"- HARD allocated items: {len(hard_summary)}")
                    st.write(f"- Total HARD quantity: {format_number(hard_summary['allocated_qty'].sum())}")
                    st.write(f"- Products with HARD allocation: {hard_summary['pt_code'].nunique()}")
            
            # Clear ALL allocation-specific session state
            allocation_keys = [
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
                # Smart filter states
                'alloc_smart_filters',
                'alloc_filter_type',
                'alloc_entity_selection',
                'alloc_customer_selection', 
                'alloc_brand_selection',
                'alloc_product_selection',
                # Bulk action states
                'show_bulk_cancel',
                'selected_cancel_ids'
            ]
            
            # Clear all allocation-related session state
            for key in allocation_keys:
                if key in st.session_state:
                    del st.session_state[key]
            
            # Also clear any dynamic keys that might have been created
            keys_to_remove = []
            for key in st.session_state.keys():
                if key.startswith(('select_', 'priority_', 'hard_', 'supply_', 'cancel_', 'reverse_')):
                    keys_to_remove.append(key)
            
            for key in keys_to_remove:
                del st.session_state[key]
            
            # Reset to view mode with new allocation
            st.session_state['allocation_mode'] = 'view'
            st.session_state['allocation_step'] = 1
            st.session_state['draft_allocation'] = {}
            st.session_state['selected_allocation_products'] = []
            st.session_state['selected_allocation_id'] = allocation_id
            
            # Show action buttons
            st.markdown("---")
            st.markdown("### 🎯 What's Next?")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("#### 📋 View Your Plan")
                st.write("Review allocation details and make adjustments if needed")
                if st.button("View Plan", type="primary", use_container_width=True, key="btn_view_plan"):
                    st.rerun()
            
            with col2:
                st.markdown("#### ➕ Create Another")
                st.write("Start a new allocation with different products or methods")
                if st.button("Create Another", use_container_width=True, key="btn_create_another"):
                    st.session_state['allocation_mode'] = 'create'
                    st.session_state['selected_allocation_id'] = None
                    st.rerun()
            
            with col3:
                st.markdown("#### 📊 Back to Analysis")
                st.write("Return to GAP Analysis to review supply-demand balance")
                if st.button("GAP Analysis", use_container_width=True, key="btn_gap_analysis"):
                    st.switch_page("pages/3_📊_GAP_Analysis.py")
            
        else:
            progress_placeholder.error("❌ Failed to save allocation plan")
            st.error("Failed to create allocation plan. Please check the logs for details.")
            logger.error("Failed to create allocation plan - create_allocation_plan returned None")
            
            # Show retry button
            if st.button("🔄 Retry", type="primary"):
                st.rerun()
            
    except Exception as e:
        st.error(f"❌ Error saving allocation: {str(e)}")
        logger.error(f"Error in save_allocation: {str(e)}", exc_info=True)
        
        # Show debug information if debug mode is on
        if st.session_state.get('debug_mode', False):
            with st.expander("🐛 Debug Information", expanded=True):
                st.write("**Error Details:**")
                st.code(str(e))
                
                st.write("**Stack Trace:**")
                import traceback
                st.code(traceback.format_exc())
                
                st.write("**Draft Allocation:**")
                draft_info = st.session_state.get('draft_allocation', {})
                # Remove results DataFrame from display to avoid clutter
                draft_display = {k: v for k, v in draft_info.items() if k != 'results'}
                st.json(draft_display)
                
                st.write("**Results DataFrame Info:**")
                if 'results' in locals() and not results.empty:
                    st.write(f"- Shape: {results.shape}")
                    st.write(f"- Columns: {list(results.columns)}")
                    st.write("- First 5 rows:")
                    st.dataframe(results.head())
                    
                    # Check for problematic values
                    st.write("**Data Quality Check:**")
                    numeric_cols = results.select_dtypes(include=[np.number]).columns
                    for col in numeric_cols:
                        nan_count = results[col].isna().sum()
                        inf_count = np.isinf(results[col].fillna(0)).sum()
                        if nan_count > 0 or inf_count > 0:
                            st.write(f"- {col}: {nan_count} NaN, {inf_count} Inf values")
                
                st.write("**Supply Mapping:**")
                if st.session_state.get('supply_mapping'):
                    st.json(st.session_state.get('supply_mapping', {}))
                else:
                    st.write("No supply mapping (SOFT allocation)")
                
                st.write("**Session State Keys (allocation-related):**")
                alloc_keys = sorted([k for k in st.session_state.keys() if 'alloc' in k.lower()])
                for key in alloc_keys:
                    value = st.session_state[key]
                    if isinstance(value, pd.DataFrame):
                        st.write(f"- {key}: DataFrame with shape {value.shape}")
                    elif isinstance(value, dict):
                        st.write(f"- {key}: Dict with {len(value)} keys")
                    else:
                        st.write(f"- {key}: {type(value).__name__}")
            
            # Show retry button even in debug mode
            if st.button("🔄 Retry Save", type="primary", key="retry_debug"):
                st.rerun()


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
    st.markdown("---")
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
            
            # Bulk actions if plan is active
            if plan.get('display_status') in ['IN_PROGRESS', 'MIXED'] and 'cancellable_qty' in filtered_details.columns:
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
            
            # Display each detail row
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
                        
                        if 'cancellable_qty' in row and row.get('cancellable_qty', 0) > 0:
                            st.caption(f"Cancellable: {format_number(row['cancellable_qty'])}")
                    
                    with col5:
                        # Action buttons based on status
                        if plan.get('display_status') in ['IN_PROGRESS', 'MIXED']:
                            if 'cancellable_qty' in row and row.get('cancellable_qty', 0) > 0:
                                if st.button("🚫", key=f"cancel_{row['id']}", 
                                           help=f"Cancel {row['cancellable_qty']} units"):
                                    st.session_state[f'show_cancel_{row["id"]}'] = True
                            
                            if row.get('delivered_qty', 0) < row.get('effective_allocated_qty', row.get('allocated_qty', 0)):
                                if st.button("🚚", key=f"deliver_{row['id']}", 
                                           help="Mark as delivered"):
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
    st.markdown("---")
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
    'MIXED': '🔀 Mixed - Combination of soft and hard'
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