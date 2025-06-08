# utils/allocation_wizard_components.py
"""Components for Allocation Plan Creation Wizard"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Tuple, Dict, List, Any

from utils.helpers import get_from_session_state
from utils.formatters import format_number


# === Data Processing Functions ===

def get_allocation_data():
    """Get and return allocation data from session state"""
    gap_data = get_from_session_state('gap_analysis_result', pd.DataFrame())
    demand_filtered = get_from_session_state('demand_filtered', pd.DataFrame())
    supply_filtered = get_from_session_state('supply_filtered', pd.DataFrame())
    return gap_data, demand_filtered, supply_filtered


def show_no_data_message():
    """Show message when no GAP Analysis data is found"""
    st.error("No GAP Analysis data found. Please run GAP Analysis first to identify products with available supply.")
    if st.button("🔄 Go to GAP Analysis"):
        st.switch_page("pages/3_📊_GAP_Analysis.py")


def prepare_products_data(gap_data, demand_filtered, supply_filtered):
    """Prepare products data with fulfillment calculation and enrichment"""
    # Calculate fulfillment percentage
    import numpy as np
    gap_data['fulfillment_percentage'] = np.where(
        gap_data['total_demand_qty'] > 0,
        (gap_data['total_available'] / gap_data['total_demand_qty']) * 100,
        0
    )
    
    # Filter products with BOTH demand AND available supply
    products_with_supply = gap_data[
        (gap_data['total_demand_qty'] > 0) & 
        (gap_data['total_available'] > 0)
    ].copy()
    
    # Enrich with additional columns
    products_with_supply = enrich_products_data(products_with_supply, demand_filtered, supply_filtered)
    
    return products_with_supply


def enrich_products_data(products_with_supply, demand_filtered, supply_filtered):
    """Enrich products data with additional columns from demand/supply data"""
    # Enrich from demand data
    if not demand_filtered.empty:
        demand_enrichment = demand_filtered.groupby('pt_code').agg({
            'legal_entity': lambda x: list(x.dropna().unique()),
            'customer': lambda x: list(x.dropna().unique()) if 'customer' in demand_filtered.columns else [],
            'brand': lambda x: list(x.dropna().unique()) if 'brand' in demand_filtered.columns else []
        }).reset_index()
        
        products_with_supply = products_with_supply.merge(
            demand_enrichment, 
            on='pt_code', 
            how='left',
            suffixes=('', '_demand')
        )
    
    # Enrich from supply data
    if not supply_filtered.empty:
        supply_enrichment = supply_filtered.groupby('pt_code').agg({
            'legal_entity': lambda x: list(x.dropna().unique()),
            'brand': lambda x: list(x.dropna().unique()) if 'brand' in supply_filtered.columns else []
        }).reset_index()
        
        products_with_supply = products_with_supply.merge(
            supply_enrichment,
            on='pt_code',
            how='left',
            suffixes=('', '_supply')
        )
        
        # Combine legal_entity and brand from both sources
        products_with_supply = combine_enrichment_columns(products_with_supply)
    
    # Show debug info if enabled
    show_debug_info(products_with_supply)
    
    return products_with_supply


def combine_enrichment_columns(products_with_supply):
    """Combine enrichment columns from demand and supply sources"""
    if 'legal_entity_demand' in products_with_supply.columns and 'legal_entity_supply' in products_with_supply.columns:
        products_with_supply['legal_entity'] = products_with_supply.apply(
            lambda row: list(set(
                (row['legal_entity_demand'] if isinstance(row['legal_entity_demand'], list) else []) +
                (row['legal_entity_supply'] if isinstance(row['legal_entity_supply'], list) else [])
            )), axis=1
        )
        products_with_supply.drop(['legal_entity_demand', 'legal_entity_supply'], axis=1, inplace=True)
    
    if 'brand_demand' in products_with_supply.columns and 'brand_supply' in products_with_supply.columns:
        products_with_supply['brand'] = products_with_supply.apply(
            lambda row: list(set(
                (row['brand_demand'] if isinstance(row['brand_demand'], list) else []) +
                (row['brand_supply'] if isinstance(row['brand_supply'], list) else [])
            )), axis=1
        )
        products_with_supply.drop(['brand_demand', 'brand_supply'], axis=1, inplace=True)
    
    return products_with_supply


def show_debug_info(products_with_supply):
    """Show debug information if debug mode is enabled"""
    if st.session_state.get('debug_mode', False):
        with st.expander("🐛 Debug Info"):
            gap_data = get_from_session_state('gap_analysis_result', pd.DataFrame())
            st.write(f"**Total GAP products:** {len(gap_data)}")
            st.write(f"**Products with both demand & supply:** {len(products_with_supply)}")
            st.write(f"**Available columns:** {products_with_supply.columns.tolist()}")
            
            if not products_with_supply.empty:
                sample = products_with_supply.head(3)
                st.write("**Sample enriched data:**")
                for col in ['pt_code', 'legal_entity', 'customer', 'brand']:
                    if col in sample.columns:
                        st.write(f"- {col}: {sample[col].tolist()}")


def show_no_products_message():
    """Show message when no products with supply are found"""
    st.warning("No products with both demand and available supply found in GAP Analysis results.")
    st.info("💡 Products need to have both demand (>0) and available supply (>0) to be allocated.")


def show_filter_options():
    """Show filter options header only"""
    col1, col2 = st.columns([3, 1])
    
    with col1:
        use_smart_filters = st.checkbox(
            "🔧 Enable Smart Filters",
            value=True,
            key="alloc_smart_filters",
            help="Enable advanced filtering options"
        )
    
    with col2:
        items_per_page = st.selectbox(
            "Items per page:",
            options=[10, 20, 50, 100],
            index=1,
            key="alloc_items_per_page"
        )
    
    return use_smart_filters, items_per_page


# Update function signature
def apply_smart_filters(filtered_data: pd.DataFrame, 
                       products_with_supply: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    """
    Apply smart filters to data - all inside expander
    
    Returns:
        Tuple[pd.DataFrame, str]: (filtered_data, filter_type)
    """
    
    
    filter_type = 'All'  # Default
    
    with st.expander("📎 Smart Filters", expanded=True):
        # Product type filter at the top
        # st.markdown("##### Filter Products")
        filter_type = st.radio(
            "Show products:",
            options=['All', 'Shortage Only', 'Available Only'],
            index=0,
            key="alloc_filter_type",
            horizontal=True
        )
        
        # Apply basic filter first
        filtered_data = apply_basic_filter(filtered_data, filter_type)
        
        if filtered_data.empty:
            st.info("No products found for selected filter type.")
            return filtered_data, filter_type
        
        # st.markdown("---")  # Divider
        
        # Advanced filters section
        # st.markdown("##### Advanced Filters")
        
        # Prepare expanded data for filtering
        expanded_data = prepare_expanded_data(filtered_data)
        
        if expanded_data:
            filter_df = pd.DataFrame(expanded_data)
        else:
            filter_df = filtered_data
        
        # Apply cascading filters
        cascade_df, selected_filters = apply_cascading_filters(filter_df)
        
        # Show filter summary
        show_filter_summary(selected_filters)
        
        # Apply filters to get final filtered products
        if any(selected_filters.values()):
            filtered_data = filter_by_selections(products_with_supply, filter_df, selected_filters)
    
    return filtered_data, filter_type


def prepare_expanded_data(filtered_data):
    """Prepare expanded data for smart filtering"""
    expanded_data = []
    
    for _, row in filtered_data.iterrows():
        base_row = {
            'pt_code': row['pt_code'],
            'product_name': row.get('product_name', ''),
            'gap_quantity': row.get('gap_quantity', 0),
            'total_demand_qty': row.get('total_demand_qty', 0),
            'total_available': row.get('total_available', 0),
            'fulfillment_percentage': row.get('fulfillment_percentage', 0)
        }
        
        entities = row.get('legal_entity', []) if isinstance(row.get('legal_entity'), list) else []
        customers = row.get('customer', []) if isinstance(row.get('customer'), list) else []
        brands = row.get('brand', []) if isinstance(row.get('brand'), list) else []
        
        if entities:
            for entity in entities:
                for customer in (customers if customers else [None]):
                    for brand in (brands if brands else [None]):
                        expanded_row = base_row.copy()
                        expanded_row['legal_entity'] = entity
                        expanded_row['customer'] = customer
                        expanded_row['brand'] = brand
                        expanded_data.append(expanded_row)
        else:
            expanded_row = base_row.copy()
            expanded_row['legal_entity'] = None
            expanded_row['customer'] = customers[0] if customers else None
            expanded_row['brand'] = brands[0] if brands else None
            expanded_data.append(expanded_row)
    
    return expanded_data


def apply_cascading_filters(filter_df):
    """Apply cascading filters and return filtered dataframe and selections"""
    cascade_df = filter_df.copy()
    
    # Get current selections from session state
    selected_entities = st.session_state.get('alloc_entity_selection', [])
    selected_customers = st.session_state.get('alloc_customer_selection', [])
    selected_brands = st.session_state.get('alloc_brand_selection', [])
    selected_products = st.session_state.get('alloc_product_selection', [])
    
    # Row 1: Entity, Customer, Brand
    filter_col1, filter_col2, filter_col3 = st.columns(3)
    
    # Entity Filter
    with filter_col1:
        selected_entities = show_entity_filter(cascade_df, selected_entities)
        if selected_entities:
            cascade_df = cascade_df[cascade_df['legal_entity'].isin(selected_entities)]
    
    # Customer Filter
    with filter_col2:
        selected_customers = show_customer_filter(cascade_df, selected_customers)
        if selected_customers:
            cascade_df = cascade_df[cascade_df['customer'].isin(selected_customers)]
    
    # Brand Filter
    with filter_col3:
        selected_brands = show_brand_filter(cascade_df, selected_brands)
        if selected_brands:
            cascade_df = cascade_df[cascade_df['brand'].isin(selected_brands)]
    
    # Row 2: Product search
    st.markdown("")  # Add spacing
    selected_products = show_product_filter(cascade_df, selected_products)
    
    return cascade_df, {
        'entities': selected_entities,
        'customers': selected_customers,
        'brands': selected_brands,
        'products': selected_products
    }


def show_entity_filter(cascade_df, selected_entities):
    """Show entity filter and return selection"""
    if 'legal_entity' in cascade_df.columns:
        entity_options = sorted(cascade_df['legal_entity'].dropna().unique())
        if entity_options:
            return st.multiselect(
                "Legal Entity",
                options=entity_options,
                default=selected_entities if selected_entities else [],
                key="alloc_entity_selection",
                placeholder="Choose legal entities..."
            )
    return []


def show_customer_filter(cascade_df, selected_customers):
    """Show customer filter and return selection"""
    if 'customer' in cascade_df.columns:
        customer_options = sorted(cascade_df['customer'].dropna().unique())
        valid_customers = [c for c in selected_customers if c in customer_options]
        
        if customer_options:
            return st.multiselect(
                "Customer",
                options=customer_options,
                default=valid_customers,
                key="alloc_customer_selection",
                placeholder="Choose customers..."
            )
    return []


def show_brand_filter(cascade_df, selected_brands):
    """Show brand filter and return selection"""
    if 'brand' in cascade_df.columns:
        brand_options = sorted(cascade_df['brand'].dropna().unique())
        valid_brands = [b for b in selected_brands if b in brand_options]
        
        if brand_options:
            return st.multiselect(
                "Brand",
                options=brand_options,
                default=valid_brands,
                key="alloc_brand_selection",
                placeholder="Choose brands..."
            )
    return []


def show_product_filter(cascade_df, selected_products):
    """Show product filter and return selection"""
    if 'pt_code' in cascade_df.columns:
        product_df = cascade_df[['pt_code', 'product_name']].drop_duplicates()
        product_options = []
        for _, row in product_df.iterrows():
            pt_code = str(row['pt_code'])
            product_name = str(row['product_name'])[:50] if pd.notna(row['product_name']) else ""
            product_options.append(f"{pt_code} - {product_name}")
        
        product_options = sorted(product_options)
        valid_products = [p for p in selected_products if p in product_options]
        
        return st.multiselect(
            "Product (PT Code - Name)",
            options=product_options,
            default=valid_products,
            key="alloc_product_selection",
            placeholder="Type to search products...",
            help="Search by PT Code or Product Name"
        )
    return []


def show_filter_summary(selected_filters):
    """Show summary of active filters"""
    active_filters = []
    if selected_filters['entities']:
        active_filters.append(f"{len(selected_filters['entities'])} entities")
    if selected_filters['customers']:
        active_filters.append(f"{len(selected_filters['customers'])} customers")
    if selected_filters['brands']:
        active_filters.append(f"{len(selected_filters['brands'])} brands")
    if selected_filters['products']:
        active_filters.append(f"{len(selected_filters['products'])} products")
    
    if active_filters:
        col_summary1, col_summary2 = st.columns([3, 1])
        with col_summary1:
            st.caption(f"🔍 Active filters: {', '.join(active_filters)}")
        with col_summary2:
            if st.button("🔄 Clear All", key="clear_smart_filters"):
                st.session_state['alloc_entity_selection'] = []
                st.session_state['alloc_customer_selection'] = []
                st.session_state['alloc_brand_selection'] = []
                st.session_state['alloc_product_selection'] = []
                st.rerun()


def filter_by_selections(products_with_supply, filter_df, selected_filters):
    """Filter products by selections"""
    temp_df = filter_df.copy()
    
    if selected_filters['entities']:
        temp_df = temp_df[temp_df['legal_entity'].isin(selected_filters['entities'])]
    if selected_filters['customers']:
        temp_df = temp_df[temp_df['customer'].isin(selected_filters['customers'])]
    if selected_filters['brands']:
        temp_df = temp_df[temp_df['brand'].isin(selected_filters['brands'])]
    if selected_filters['products']:
        selected_pt_codes = [p.split(' - ')[0] for p in selected_filters['products']]
        temp_df = temp_df[temp_df['pt_code'].isin(selected_pt_codes)]
    
    filtered_pt_codes = set(temp_df['pt_code'].unique())
    return products_with_supply[products_with_supply['pt_code'].isin(filtered_pt_codes)]


def apply_basic_filter(filtered_data, filter_type):
    """Apply basic filter (Shortage/Available)"""
    if filter_type == 'Shortage Only':
        return filtered_data[filtered_data['gap_quantity'] < 0]
    elif filter_type == 'Available Only':
        return filtered_data[filtered_data['gap_quantity'] >= 0]
    return filtered_data


def show_no_filtered_data_message(filter_type, use_smart_filters):
    """Show message when no data matches filters"""
    st.info("No products found matching the selected filters.")
    
    # More specific hints based on filter type
    if filter_type == 'Shortage Only':
        st.caption("💡 Try selecting 'All' or 'Available Only' - there might not be any shortage products.")
    elif filter_type == 'Available Only':
        st.caption("💡 Try selecting 'All' or 'Shortage Only' - there might not be any available products.")
    
    # Check if smart filters are too restrictive
    if use_smart_filters and any([
        st.session_state.get('alloc_entity_selection', []),
        st.session_state.get('alloc_customer_selection', []),
        st.session_state.get('alloc_brand_selection', []),
        st.session_state.get('alloc_product_selection', [])
    ]):
        st.caption("💡 Try clearing some advanced filters to see more products.")


def prepare_product_summary(filtered_data):
    """Prepare product summary data"""
    product_summary = filtered_data.groupby(['pt_code', 'product_name']).agg({
        'gap_quantity': 'sum',
        'total_demand_qty': 'sum',
        'total_available': 'sum',
        'fulfillment_percentage': 'mean'
    }).reset_index()
    
    # Add status column
    product_summary['status'] = product_summary['gap_quantity'].apply(
        lambda x: '🔴 Shortage' if x < 0 else '🟢 Available'
    )
    product_summary['gap_abs'] = product_summary['gap_quantity'].abs()
    
    # Sort by absolute gap
    product_summary = product_summary.sort_values('gap_abs', ascending=False)
    
    return product_summary


def show_summary_metrics(product_summary):
    """Show summary metrics"""
    st.markdown("---")
    col_sum1, col_sum2, col_sum3 = st.columns(3)
    
    with col_sum1:
        st.metric("Total Products", len(product_summary))
    with col_sum2:
        shortage_count = len(product_summary[product_summary['gap_quantity'] < 0])
        st.metric("Shortage Products", shortage_count)
    with col_sum3:
        available_count = len(product_summary[product_summary['gap_quantity'] >= 0])
        st.metric("Available Products", available_count)


def ensure_valid_current_page(total_pages):
    """Ensure current page number is valid"""
    if 'alloc_current_page' not in st.session_state:
        st.session_state['alloc_current_page'] = 1
    
    st.session_state['alloc_current_page'] = min(st.session_state['alloc_current_page'], total_pages)
    st.session_state['alloc_current_page'] = max(st.session_state['alloc_current_page'], 1)


def show_pagination_controls(total_pages, total_products):
    """Show pagination controls"""
    st.markdown("---")
    page_col1, page_col2, page_col3, page_col4, page_col5 = st.columns([1, 1, 2, 1, 1])
    
    with page_col1:
        if st.button("⏮️ First", disabled=st.session_state['alloc_current_page'] == 1):
            st.session_state['alloc_current_page'] = 1
            st.rerun()
    
    with page_col2:
        if st.button("◀️ Previous", disabled=st.session_state['alloc_current_page'] == 1):
            st.session_state['alloc_current_page'] -= 1
            st.rerun()
    
    with page_col3:
        st.markdown(f"<div style='text-align: center'><b>Page {st.session_state['alloc_current_page']} of {total_pages}</b></div>", unsafe_allow_html=True)
        st.caption(f"<div style='text-align: center'>Total: {total_products} products</div>", unsafe_allow_html=True)
    
    with page_col4:
        if st.button("Next ▶️", disabled=st.session_state['alloc_current_page'] >= total_pages):
            st.session_state['alloc_current_page'] += 1
            st.rerun()
    
    with page_col5:
        if st.button("Last ⏭️", disabled=st.session_state['alloc_current_page'] >= total_pages):
            st.session_state['alloc_current_page'] = total_pages
            st.rerun()


def show_product_selection(page_products, filtered_data):
    """Show product selection interface and return selected products"""
    st.markdown("##### Select Products for Allocation")
    
    # Select all on current page
    select_all_page = st.checkbox(
        f"Select All on Current Page ({len(page_products)} items)", 
        value=False,
        key=f"select_all_page_{st.session_state['alloc_current_page']}"
    )
    
    # Get previously selected products
    if 'selected_allocation_products' not in st.session_state:
        st.session_state['selected_allocation_products'] = []
    
    selected_products = st.session_state['selected_allocation_products'].copy()
    
    # Display header
    show_selection_header()
    
    # Display products
    for idx, row in page_products.iterrows():
        show_product_row(row, selected_products, select_all_page)
    
    # Update session state
    st.session_state['selected_allocation_products'] = selected_products
    
    # Show selection summary
    show_selection_summary(selected_products, filtered_data)
    
    return selected_products


def show_selection_header():
    """Show selection table header"""
    st.markdown("---")
    header_col1, header_col2, header_col3, header_col4, header_col5, header_col6 = st.columns([0.5, 2, 1, 1.5, 1.5, 1.5])
    with header_col1:
        st.markdown("**Select**")
    with header_col2:
        st.markdown("**Product**")
    with header_col3:
        st.markdown("**Status**")
    with header_col4:
        st.markdown("**Gap**")
    with header_col5:
        st.markdown("**Demand**")
    with header_col6:
        st.markdown("**Supply**")
    st.markdown("---")


def show_product_row(row, selected_products, select_all_page):
    """Show individual product row for selection"""
    col1, col2, col3, col4, col5, col6 = st.columns([0.5, 2, 1, 1.5, 1.5, 1.5])
    
    with col1:
        is_selected = row['pt_code'] in selected_products
        
        selected = st.checkbox(
            "",
            value=select_all_page or is_selected,
            key=f"select_{row['pt_code']}_{st.session_state['alloc_current_page']}"
        )
        
        if selected and row['pt_code'] not in selected_products:
            selected_products.append(row['pt_code'])
        elif not selected and row['pt_code'] in selected_products:
            selected_products.remove(row['pt_code'])
    
    with col2:
        st.write(f"**{row['pt_code']}**")
        st.caption(row.get('product_name', '')[:50] + ('...' if len(row.get('product_name', '')) > 50 else ''))
    
    with col3:
        st.write(row['status'])
    
    with col4:
        if row['gap_quantity'] < 0:
            st.metric("Shortage", format_number(row['gap_abs']), label_visibility="collapsed")
        else:
            st.metric("Surplus", format_number(row['gap_abs']), label_visibility="collapsed")
    
    with col5:
        st.metric("Demand", format_number(row.get('total_demand_qty', 0)), label_visibility="collapsed")
    
    with col6:
        st.metric("Supply", format_number(row.get('total_available', 0)), label_visibility="collapsed")


def show_selection_summary(selected_products, filtered_data):
    """Show summary of selected products"""
    st.markdown("---")
    if selected_products:
        summary_col1, summary_col2 = st.columns([3, 1])
        with summary_col1:
            st.success(f"✅ Selected {len(selected_products)} products for allocation")
            
            # Calculate totals
            selected_data = filtered_data[filtered_data['pt_code'].isin(selected_products)]
            if not selected_data.empty:
                total_demand = selected_data['total_demand_qty'].sum()
                total_supply = selected_data['total_available'].sum()
                total_gap = selected_data['gap_quantity'].sum()
                
                import numpy as np
                avg_fulfillment = np.where(
                    total_demand > 0,
                    (total_supply / total_demand) * 100,
                    0
                )
                
                st.caption(
                    f"Total Demand: {format_number(total_demand)} | "
                    f"Total Supply: {format_number(total_supply)} | "
                    f"Net Gap: {format_number(total_gap)} | "
                    f"Avg Fulfillment: {avg_fulfillment:.1f}%"
                )
        
        with summary_col2:
            if st.button("🗑️ Clear Selection"):
                st.session_state['selected_allocation_products'] = []
                st.rerun()
    else:
        st.warning("⚠️ Please select at least one product to continue")


def show_step1_next_button(selected_products):
    """Show next button for step 1"""
    st.markdown("---")
    next_col1, next_col2, next_col3 = st.columns([2, 1, 2])
    with next_col2:
        if st.button("Next ➡️", type="primary", use_container_width=True, 
                     disabled=len(selected_products) == 0):
            st.session_state['allocation_step'] = 2
            # Clear pagination state when moving to next step
            if 'alloc_current_page' in st.session_state:
                del st.session_state['alloc_current_page']
            st.rerun()