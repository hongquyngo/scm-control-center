import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
from sqlalchemy import text
import plotly.express as px
import plotly.graph_objects as go

# Import modules
from utils.data_manager import DataManager
from utils.display_components import DisplayComponents
from utils.formatters import format_number, format_currency, format_percentage
from utils.helpers import (
    save_to_session_state, get_from_session_state,
    convert_df_to_excel, export_multiple_sheets
)
from utils.session_state import initialize_session_state
from utils.db import get_db_engine

# Import allocation specific modules
from utils.allocation_manager import AllocationManager
from utils.allocation_methods import AllocationMethods
from utils.allocation_components import AllocationComponents
from utils.allocation_validators import AllocationValidator


# === Functions Implementation ===

def show_allocation_list():
    """Display list of allocation plans with filters"""
    st.markdown("### 📋 Allocation Plans")
    
    # Filters
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        status_filter = st.multiselect(
            "Status", 
            options=['DRAFT', 'APPROVED', 'EXECUTED', 'CANCELLED'],
            default=['DRAFT', 'APPROVED']
        )
    
    with col2:
        date_from = st.date_input("From Date", value=datetime.now() - timedelta(days=30))
    
    with col3:
        date_to = st.date_input("To Date", value=datetime.now())
    
    with col4:
        search_text = st.text_input("Search", placeholder="Allocation number or notes")
    
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
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Plans", len(allocations))
    with col2:
        st.metric("Draft", len(allocations[allocations['status'] == 'DRAFT']))
    with col3:
        st.metric("Approved", len(allocations[allocations['status'] == 'APPROVED']))
    with col4:
        st.metric("Executed", len(allocations[allocations['status'] == 'EXECUTED']))
    
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
                st.caption(f"Items: {row.get('item_count', 0)}")
                if row.get('hard_allocation_count', 0) > 0:
                    st.caption(f"🔒 HARD: {row['hard_allocation_count']}")
            
            with col3:
                status_color = ALLOCATION_STATUS_COLORS.get(row['status'], 'gray')
                st.markdown(f"Status: :{status_color}[{row['status']}]")
                if row['status'] == 'EXECUTED':
                    st.caption(f"Fulfillment: {row.get('fulfillment_rate', 0):.1f}%")
            
            with col4:
                if st.button("👁️ View", key=f"view_{row['id']}"):
                    st.session_state['allocation_mode'] = 'view'
                    st.session_state['selected_allocation_id'] = row['id']
                    st.rerun()
            
            with col5:
                if row['status'] == 'DRAFT':
                    if st.button("✏️ Edit", key=f"edit_{row['id']}"):
                        st.session_state['allocation_mode'] = 'edit'
                        st.session_state['selected_allocation_id'] = row['id']
                        st.rerun()
            
            with col6:
                if row['status'] == 'DRAFT':
                    col6_1, col6_2 = st.columns(2)
                    with col6_1:
                        if st.button("✅ Approve", key=f"approve_{row['id']}", type="primary"):
                            if allocation_manager.approve_allocation(row['id'], st.session_state.get('username', 'System')):
                                st.success("Approved successfully!")
                                st.rerun()
                    with col6_2:
                        if st.button("❌ Cancel", key=f"cancel_{row['id']}"):
                            if allocation_manager.cancel_allocation(row['id']):
                                st.info("Cancelled")
                                st.rerun()
            
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
    """Step 1: Select products for allocation"""
    st.markdown("#### 📦 Select Products for Allocation")
    
    # Get data from GAP analysis or demand/supply data
    gap_data = get_from_session_state('gap_analysis_result', pd.DataFrame())
    demand_data = get_from_session_state('demand_filtered', pd.DataFrame())
    supply_data = get_from_session_state('supply_filtered', pd.DataFrame())
    
    # Check data availability
    if gap_data.empty and demand_data.empty:
        st.error("No data found. Please run GAP Analysis or load demand data first.")
        return
    
    # Prepare product summary
    if not gap_data.empty:
        # Use GAP analysis data if available
        product_summary = gap_data.groupby(['pt_code', 'product_name']).agg({
            'gap_quantity': 'sum',
            'total_demand_qty': 'sum',
            'total_available': 'sum'
        }).reset_index()
        
        product_summary['status'] = product_summary['gap_quantity'].apply(
            lambda x: '🔴 Shortage' if x < 0 else '🟢 Available'
        )
        product_summary['gap_abs'] = product_summary['gap_quantity'].abs()
    else:
        # Use demand data directly
        demand_summary = demand_data.groupby(['pt_code', 'product_name']).agg({
            'demand_quantity': 'sum'
        }).reset_index()
        
        supply_summary = supply_data.groupby(['pt_code']).agg({
            'quantity': 'sum'
        }).reset_index()
        
        product_summary = demand_summary.merge(
            supply_summary, 
            on='pt_code', 
            how='left'
        )
        product_summary['quantity'] = product_summary['quantity'].fillna(0)
        product_summary['gap_quantity'] = product_summary['quantity'] - product_summary['demand_quantity']
        product_summary['total_demand_qty'] = product_summary['demand_quantity']
        product_summary['total_available'] = product_summary['quantity']
        product_summary['status'] = product_summary['gap_quantity'].apply(
            lambda x: '🔴 Shortage' if x < 0 else '🟢 Available'
        )
        product_summary['gap_abs'] = product_summary['gap_quantity'].abs()
    
    # Filter options
    col1, col2, col3 = st.columns(3)
    with col1:
        filter_type = st.radio(
            "Show products:",
            options=['All', 'Shortage Only', 'Available Only'],
            index=0
        )
    
    # Apply filter
    if filter_type == 'Shortage Only':
        product_summary = product_summary[product_summary['gap_quantity'] < 0]
    elif filter_type == 'Available Only':
        product_summary = product_summary[product_summary['gap_quantity'] >= 0]
    
    # Sort by absolute gap
    product_summary = product_summary.sort_values('gap_abs', ascending=False)
    
    if product_summary.empty:
        st.info(f"No products found for filter: {filter_type}")
        return
    
    # Display selection table
    st.markdown("##### Select Products for Allocation")
    
    # Select all checkbox
    select_all = st.checkbox("Select All Products", value=False)
    
    selected_products = []
    
    for idx, row in product_summary.iterrows():
        col1, col2, col3, col4, col5, col6 = st.columns([0.5, 2, 1, 1.5, 1.5, 1.5])
        
        with col1:
            selected = st.checkbox(
                "",
                value=select_all or row['pt_code'] in st.session_state.get('selected_allocation_products', []),
                key=f"select_{row['pt_code']}"
            )
            if selected:
                selected_products.append(row['pt_code'])
        
        with col2:
            st.write(f"**{row['pt_code']}**")
            st.caption(row['product_name'])
        
        with col3:
            st.write(row['status'])
        
        with col4:
            if row['gap_quantity'] < 0:
                st.metric("Shortage", format_number(row['gap_abs']))
            else:
                st.metric("Surplus", format_number(row['gap_abs']))
        
        with col5:
            st.metric("Demand", format_number(row['total_demand_qty']))
        
        with col6:
            st.metric("Supply", format_number(row['total_available']))
    
    # Show summary
    if selected_products:
        st.info(f"Selected {len(selected_products)} products for allocation")
    
    # Next button
    st.markdown("---")
    col1, col2, col3 = st.columns([2, 1, 2])
    with col2:
        if st.button("Next ➡️", type="primary", use_container_width=True, 
                     disabled=len(selected_products) == 0):
            st.session_state['selected_allocation_products'] = selected_products
            st.session_state['allocation_step'] = 2
            st.rerun()
    
    if len(selected_products) == 0:
        st.warning("Please select at least one product to continue")

def show_step2_choose_method():
    """Step 2: Choose allocation method and type"""
    st.markdown("#### 🎯 Choose Allocation Method & Type")
    
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
    
    method = st.session_state['draft_allocation'].get('method', 'FIFO')
    
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
                method = st.session_state['draft_allocation'].get('method', 'FIFO')
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
        # Need to calculate first
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
    
    # Get available supply details
    product_codes = allocation_results['pt_code'].unique().tolist()
    legal_entities = allocation_results['legal_entity'].unique().tolist()
    
    available_supply = allocation_manager.get_available_supply_for_hard_allocation(
        product_codes, legal_entities
    )
    
    if available_supply.empty:
        st.error("No available supply found for HARD allocation")
        return
    
    # Initialize supply mapping in session state
    if 'supply_mapping' not in st.session_state:
        st.session_state['supply_mapping'] = {}
    
    # Group by product for easier mapping
    products = allocation_results['pt_code'].unique()
    
    for product in products:
        st.markdown(f"##### {product}")
        
        # Get demands for this product
        product_demands = allocation_results[
            (allocation_results['pt_code'] == product) & 
            (allocation_results['allocated_qty'] > 0)
        ]
        
        # Get available supply for this product
        product_supply = available_supply[available_supply['pt_code'] == product]
        
        if product_supply.empty:
            st.warning(f"No available supply for {product}")
            continue
        
        # Display mapping interface
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("**Demand Orders**")
            for idx, demand in product_demands.iterrows():
                demand_key = f"{demand.get('demand_line_id', idx)}"
                
                # Check if MIXED type and this product should be HARD
                if allocation_type == 'MIXED':
                    is_hard = st.checkbox(
                        f"🔒 Hard allocation for {demand['customer']}",
                        key=f"hard_{demand_key}",
                        value=demand_key in st.session_state['supply_mapping']
                    )
                    if not is_hard:
                        if demand_key in st.session_state['supply_mapping']:
                            del st.session_state['supply_mapping'][demand_key]
                        continue
                
                st.info(f"""
                **Customer**: {demand['customer']}  
                **ETD**: {demand['etd']}  
                **Quantity**: {demand['allocated_qty']:.0f}
                """)
                
                # Supply selection
                supply_options = []
                for _, supply in product_supply.iterrows():
                    option_text = (
                        f"{supply['source_type']} - {supply['reference']} "
                        f"({supply['available_qty']:.0f} available)"
                    )
                    if supply['origin_country']:
                        option_text += f" - Origin: {supply['origin_country']}"
                    if supply['expected_date']:
                        option_text += f" - ETA: {supply['expected_date']}"
                    supply_options.append({
                        'text': option_text,
                        'value': f"{supply['source_type']}|{supply['source_id']}"
                    })
                
                selected_supply = st.selectbox(
                    "Select supply source:",
                    options=[opt['value'] for opt in supply_options],
                    format_func=lambda x: next(opt['text'] for opt in supply_options if opt['value'] == x),
                    key=f"supply_{demand_key}"
                )
                
                if selected_supply:
                    source_type, source_id = selected_supply.split('|')
                    st.session_state['supply_mapping'][demand_key] = {
                        'source_type': source_type,
                        'source_id': int(source_id),
                        'demand_line_id': demand_key,
                        'product': product,
                        'quantity': demand['allocated_qty']
                    }
        
        with col2:
            st.markdown("**Available Supply**")
            
            # Show supply summary
            supply_summary = product_supply.groupby('source_type').agg({
                'available_qty': 'sum'
            }).reset_index()
            
            for _, row in supply_summary.iterrows():
                st.metric(row['source_type'], f"{row['available_qty']:.0f} units")
            
            # Show supply details
            with st.expander("View Supply Details"):
                display_cols = ['source_type', 'reference', 'available_qty', 
                              'origin_country', 'expected_date']
                st.dataframe(
                    product_supply[display_cols],
                    use_container_width=True
                )
    
    # Validation
    mapped_count = len(st.session_state['supply_mapping'])
    total_hard_demands = len(allocation_results[allocation_results['allocated_qty'] > 0])
    
    if allocation_type == 'HARD' and mapped_count < total_hard_demands:
        st.warning(f"⚠️ Only {mapped_count} of {total_hard_demands} demands have been mapped to supply")
    
    # Next button
    st.markdown("---")
    col1, col2, col3 = st.columns([2, 1, 2])
    with col2:
        can_proceed = (allocation_type == 'MIXED') or (mapped_count == total_hard_demands)
        if st.button("Next ➡️", type="primary", use_container_width=True, disabled=not can_proceed):
            st.session_state['allocation_step'] = 5
            st.rerun()

def show_step4_preview():
    """Step 4: Preview allocation results (for SOFT allocation)"""
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
    
    # Store results
    st.session_state['draft_allocation']['results'] = allocation_results
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    total_demand = allocation_results['requested_qty'].sum()
    total_allocated = allocation_results['allocated_qty'].sum()
    total_orders = len(allocation_results)
    avg_fulfillment = (allocation_results['allocated_qty'] / allocation_results['requested_qty']).mean() * 100
    
    with col1:
        st.metric("Total Demand", format_number(total_demand))
    with col2:
        st.metric("Total Allocated", format_number(total_allocated))
    with col3:
        st.metric("Orders", total_orders)
    with col4:
        st.metric("Avg Fulfillment", f"{avg_fulfillment:.1f}%")
    
    # Allocation details table
    st.markdown("##### Allocation Details")
    
    # For manual method, allow editing
    if method == 'MANUAL':
        edited_df = AllocationComponents.show_editable_allocation_table(allocation_results)
        st.session_state['draft_allocation']['results'] = edited_df
    else:
        # Display read-only table
        display_df = allocation_results[['pt_code', 'product_name', 'customer', 'etd', 
                                       'requested_qty', 'allocated_qty', 'fulfillment_rate']].copy()
        
        # Format columns
        display_df['requested_qty'] = display_df['requested_qty'].apply(format_number)
        display_df['allocated_qty'] = display_df['allocated_qty'].apply(format_number)
        display_df['fulfillment_rate'] = display_df['fulfillment_rate'].apply(lambda x: f"{x:.1f}%")
        
        st.dataframe(display_df, use_container_width=True, height=400)
    
    # Visualization
    col1, col2 = st.columns(2)
    
    with col1:
        # Fulfillment by product
        fig1 = AllocationComponents.create_fulfillment_chart_by_product(allocation_results)
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        # Fulfillment by customer
        fig2 = AllocationComponents.create_fulfillment_chart_by_customer(allocation_results)
        st.plotly_chart(fig2, use_container_width=True)
    
    # Validation warnings
    warnings = AllocationValidator.validate_allocation_results(allocation_results, filtered_supply)
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
    """Save allocation plan to database"""
    try:
        draft = st.session_state.get('draft_allocation', {})
        results = draft.get('results', pd.DataFrame())
        
        # Get time adjustments and filters from session state
        time_adjustments = get_from_session_state('time_adjustments', {})
        filters = {
            'entities': get_from_session_state('selected_entities', []),
            'products': get_from_session_state('selected_products', []),
            'date_range': {
                'start': get_from_session_state('date_from', datetime.now()).strftime('%Y-%m-%d'),
                'end': get_from_session_state('date_to', datetime.now() + timedelta(days=30)).strftime('%Y-%m-%d')
            }
        }
        
        # Create allocation plan
        plan_data = {
            'allocation_method': draft.get('method', 'MANUAL'),
            'allocation_type': draft.get('allocation_type', 'SOFT'),
            'status': status,
            'creator_id': st.session_state.get('user_id', 1),
            'notes': draft.get('parameters', {}).get('notes', ''),
            'approved_by': st.session_state.get('username', 'System') if status == 'APPROVED' else None,
            'approved_date': datetime.now() if status == 'APPROVED' else None,
            'time_adjustments': time_adjustments,
            'filters': filters
        }
        
        # Get supply mapping if exists
        supply_mapping = st.session_state.get('supply_mapping', {})
        
        # Save to database
        allocation_id = allocation_manager.create_allocation_plan(plan_data, results, supply_mapping)
        
        if allocation_id:
            st.success(f"✅ Allocation plan saved successfully! ID: {allocation_id}")
            
            # Clear session state
            st.session_state['allocation_mode'] = 'list'
            st.session_state['allocation_step'] = 1
            st.session_state['draft_allocation'] = {}  # Reset to empty dict instead of None
            st.session_state['selected_allocation_products'] = []
            st.session_state['supply_mapping'] = {}
            st.session_state['temp_allocation_results'] = None
            
            # Redirect to view
            st.session_state['allocation_mode'] = 'view'
            st.session_state['selected_allocation_id'] = allocation_id
            st.rerun()
        else:
            st.error("Failed to save allocation plan")
            
    except Exception as e:
        st.error(f"Error saving allocation: {str(e)}")

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
    """View allocation plan details"""
    allocation_id = st.session_state.get('selected_allocation_id')
    
    if not allocation_id:
        st.error("No allocation selected")
        return
    
    # Load allocation details
    plan, details = allocation_manager.get_allocation_details(allocation_id)
    
    if plan is None:
        st.error("Allocation not found")
        return
    
    # Header
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown(f"### 📋 {plan['allocation_number']}")
    with col2:
        if st.button("← Back to List"):
            st.session_state['allocation_mode'] = 'list'
            st.rerun()
    
    # Plan info with allocation type
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.write("**Date:**", plan['allocation_date'].strftime('%Y-%m-%d %H:%M'))
        st.write("**Method:**", plan['allocation_method'])
    
    with col2:
        status_color = ALLOCATION_STATUS_COLORS.get(plan['status'], 'gray')
        st.markdown(f"**Status:** :{status_color}[{plan['status']}]")
        st.write("**Created by:**", plan.get('creator_name', 'System'))
    
    with col3:
        allocation_type = plan.get('allocation_type', 'SOFT')
        type_icon = {'SOFT': '🌊', 'HARD': '🔒', 'MIXED': '🔀'}.get(allocation_type, '❓')
        st.write(f"**Type:** {type_icon} {allocation_type}")
        if 'allocation_mode' in details.columns:
            hard_count = len(details[details['allocation_mode'] == 'HARD'])
            if hard_count > 0:
                st.write(f"**HARD allocations:** {hard_count}")
    
    with col4:
        if plan['approved_by']:
            st.write("**Approved by:**", plan['approved_by'])
            st.write("**Approved date:**", plan['approved_date'].strftime('%Y-%m-%d'))
    
    with col5:
        total_allocated = details['allocated_qty'].sum()
        total_delivered = details['delivered_qty'].sum()
        st.metric("Total Allocated", format_number(total_allocated))
        if plan['status'] == 'EXECUTED':
            st.metric("Delivered", format_number(total_delivered), 
                     f"{total_delivered/total_allocated*100:.1f}%")
    
    # Show snapshot context if available
    if plan.get('snapshot_context'):
        with st.expander("📸 Snapshot Context"):
            snapshot = plan['snapshot_context']
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Snapshot Time:**", snapshot.get('snapshot_datetime', 'N/A'))
                if snapshot.get('time_adjustments'):
                    st.write("**Time Adjustments:**", snapshot['time_adjustments'].get('mode', 'None'))
                if snapshot.get('filters'):
                    st.write("**Filters Applied:**")
                    filters = snapshot['filters']
                    if filters.get('entities'):
                        st.write(f"- Entities: {', '.join(filters['entities'])}")
                    if filters.get('products'):
                        st.write(f"- Products: {len(filters['products'])} selected")
            
            with col2:
                if snapshot.get('summary'):
                    summary = snapshot['summary']
                    st.write("**Summary:**")
                    st.write(f"- Total Products: {summary.get('total_products', 0)}")
                    st.write(f"- Shortage Products: {summary.get('shortage_products', 0)}")
                    st.write(f"- Fulfillment Rate: {summary.get('fulfillment_rate', 0):.1f}%")
                    if summary.get('allocation_modes'):
                        st.write("**Allocation Modes:**")
                        for mode, count in summary['allocation_modes'].items():
                            st.write(f"- {mode}: {count} orders")
    
    # Allocation details
    st.markdown("---")
    st.markdown("#### Allocation Details")
    
    # Summary by product
    product_summary = details.groupby(['pt_code', 'product_name']).agg({
        'requested_qty': 'sum',
        'allocated_qty': 'sum',
        'delivered_qty': 'sum'
    }).reset_index()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("##### By Product")
        fig1 = AllocationComponents.create_allocation_summary_chart(product_summary, 'product')
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        st.markdown("##### By Customer")
        customer_summary = details.groupby('customer_name').agg({
            'requested_qty': 'sum',
            'allocated_qty': 'sum',
            'delivered_qty': 'sum'
        }).reset_index()
        fig2 = AllocationComponents.create_allocation_summary_chart(customer_summary, 'customer')
        st.plotly_chart(fig2, use_container_width=True)
    
    # Details table with allocation mode
    st.markdown("##### Detail Lines")
    
    # Add fulfillment status and format allocation mode
    details['fulfillment_status'] = details.apply(
        lambda x: '✅ Delivered' if x['delivered_qty'] >= x['allocated_qty'] 
        else '🔄 Partial' if x['delivered_qty'] > 0 
        else '⏳ Pending', axis=1
    )
    
    # Format allocation mode if exists
    if 'allocation_mode' in details.columns:
        details['mode'] = details['allocation_mode'].apply(
            lambda x: '🔒 HARD' if x == 'HARD' else '🌊 SOFT'
        )
        display_columns = ['pt_code', 'product_name', 'customer_name', 'mode', 'allocated_etd',
                          'requested_qty', 'allocated_qty', 'delivered_qty', 'fulfillment_status']
    else:
        display_columns = ['pt_code', 'product_name', 'customer_name', 'allocated_etd',
                          'requested_qty', 'allocated_qty', 'delivered_qty', 'fulfillment_status']
    
    # Add supply reference for HARD allocations
    if 'supply_reference' in details.columns:
        details['supply_ref'] = details['supply_reference'].fillna('')
        display_columns.insert(4, 'supply_ref')
    
    st.dataframe(
        details[display_columns],
        use_container_width=True,
        height=400
    )
    
    # Export button
    if st.button("📤 Export Details"):
        export_allocation_details(plan, details)

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

ALLOCATION_STATUS_COLORS = {
    'DRAFT': 'gray',
    'APPROVED': 'green',
    'EXECUTED': 'blue',
    'CANCELLED': 'red'
}

# === Initialize Components ===
@st.cache_resource
def get_allocation_manager():
    return AllocationManager()

allocation_manager = get_allocation_manager()

# === Header ===
DisplayComponents.show_page_header(
    title="Allocation Plan Management",
    icon="🧩",
    prev_page="pages/3_📊_GAP_Analysis.py",
    next_page="pages/5_📌_PO_Suggestions.py"
)

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