# utils/allocation_wizard_components.py
"""Components for Allocation Plan Creation Wizard"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Tuple

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

# File: utils/allocation_wizard_components.py
# Function: prepare_products_data()

def prepare_products_data(gap_data, demand_filtered, supply_filtered):
    """Prepare products data with fulfillment calculation and enrichment
    
    Logic mới: Include tất cả periods của products có supply, không chỉ periods có supply > 0
    """
    # Debug input data
    if st.session_state.get('debug_mode', False):
        st.write("🐛 prepare_products_data - Input data:")
        st.write(f"- gap_data shape: {gap_data.shape}")
        st.write(f"- demand_filtered shape: {demand_filtered.shape}")
        st.write(f"- supply_filtered shape: {supply_filtered.shape}")
    
    # Step 1: Identify products that have SOME supply across ANY period
    products_supply_summary = gap_data.groupby('pt_code').agg({
        'total_available': 'sum',
        'total_demand_qty': 'sum'
    }).reset_index()
    
    # Filter products với total supply > 0 VÀ total demand > 0
    valid_products = products_supply_summary[
        (products_supply_summary['total_available'] > 0) & 
        (products_supply_summary['total_demand_qty'] > 0)
    ]['pt_code'].tolist()
    
    if st.session_state.get('debug_mode', False):
        st.write(f"🐛 Found {len(valid_products)} products with both demand and supply")
        st.write(f"🐛 Valid products sample: {valid_products[:5]}")
    
    # Step 2: Get ALL periods for these valid products (không filter by period supply)
    products_with_supply = gap_data[
        gap_data['pt_code'].isin(valid_products)
    ].copy()
    
    # Debug specific product
    if st.session_state.get('debug_mode', False) and 'P007001219' in valid_products:
        p007_data = products_with_supply[products_with_supply['pt_code'] == 'P007001219']
        st.write(f"🐛 P007001219 periods after filter: {len(p007_data)}")
        st.dataframe(p007_data[['period', 'total_demand_qty', 'total_available']])
    
    # Step 3: Add fulfillment percentage if not exists
    if 'fulfillment_percentage' not in products_with_supply.columns:
        # Use existing fulfillment_rate_percent if available
        if 'fulfillment_rate_percent' in products_with_supply.columns:
            products_with_supply['fulfillment_percentage'] = products_with_supply['fulfillment_rate_percent']
        else:
            # Calculate if needed
            products_with_supply['fulfillment_percentage'] = np.where(
                products_with_supply['total_demand_qty'] > 0,
                (products_with_supply['total_available'] / products_with_supply['total_demand_qty']) * 100,
                0
            )
    
    # Step 4: Enrich with additional columns from demand/supply data
    products_with_supply = enrich_products_data(products_with_supply, demand_filtered, supply_filtered)
    
    # Debug final output
    if st.session_state.get('debug_mode', False):
        st.write("🐛 prepare_products_data - Output:")
        st.write(f"- Final shape: {products_with_supply.shape}")
        st.write(f"- Unique products: {products_with_supply['pt_code'].nunique()}")
        st.write(f"- Total periods: {len(products_with_supply)}")
        
        # Check columns
        important_cols = ['pt_code', 'period', 'original_demand_qty', 'total_demand_qty', 
                         'total_available', 'fulfillment_rate_percent']
        available_cols = [col for col in important_cols if col in products_with_supply.columns]
        st.write(f"🐛 Available important columns: {available_cols}")
    
    return products_with_supply


def enrich_products_data(products_with_supply, demand_filtered, supply_filtered):
    """Enrich products data with additional columns from demand/supply data
    
    Không thay đổi logic, chỉ add thêm debug info
    """
    if st.session_state.get('debug_mode', False):
        st.write("🐛 enrich_products_data - Starting enrichment")
    
    # Enrich from demand data
    if not demand_filtered.empty:
        # Get unique attributes per product from demand
        demand_enrichment = demand_filtered.groupby('pt_code').agg({
            'legal_entity': lambda x: list(x.dropna().unique()) if 'legal_entity' in demand_filtered.columns else [],
            'customer': lambda x: list(x.dropna().unique()) if 'customer' in demand_filtered.columns else [],
            'brand': lambda x: list(x.dropna().unique()) if 'brand' in demand_filtered.columns else []
        }).reset_index()
        
        # Handle the aggregation based on what columns exist
        agg_dict = {}
        if 'legal_entity' in demand_filtered.columns:
            agg_dict['legal_entity'] = lambda x: list(x.dropna().unique())
        if 'customer' in demand_filtered.columns:
            agg_dict['customer'] = lambda x: list(x.dropna().unique())
        if 'brand' in demand_filtered.columns:
            agg_dict['brand'] = lambda x: list(x.dropna().unique())
        
        if agg_dict:
            demand_enrichment = demand_filtered.groupby('pt_code').agg(agg_dict).reset_index()
            products_with_supply = products_with_supply.merge(
                demand_enrichment, 
                on='pt_code', 
                how='left',
                suffixes=('', '_demand')
            )
    
    # Enrich from supply data
    if not supply_filtered.empty:
        # Similar for supply
        agg_dict = {}
        if 'legal_entity' in supply_filtered.columns:
            agg_dict['legal_entity'] = lambda x: list(x.dropna().unique())
        if 'brand' in supply_filtered.columns:
            agg_dict['brand'] = lambda x: list(x.dropna().unique())
        
        if agg_dict:
            supply_enrichment = supply_filtered.groupby('pt_code').agg(agg_dict).reset_index()
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
    """Combine enrichment columns from demand and supply sources
    
    Helper function - no changes needed
    """
    # Combine legal_entity lists
    if 'legal_entity_demand' in products_with_supply.columns and 'legal_entity_supply' in products_with_supply.columns:
        products_with_supply['legal_entity'] = products_with_supply.apply(
            lambda row: list(set(
                (row['legal_entity_demand'] if isinstance(row['legal_entity_demand'], list) else []) +
                (row['legal_entity_supply'] if isinstance(row['legal_entity_supply'], list) else [])
            )), axis=1
        )
        products_with_supply.drop(['legal_entity_demand', 'legal_entity_supply'], axis=1, inplace=True)
    
    # Combine brand lists
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
    """Show debug information if debug mode is enabled
    
    Helper function - enhanced với more info
    """
    if st.session_state.get('debug_mode', False):
        with st.expander("🐛 Enrichment Debug Info"):
            gap_data = get_from_session_state('gap_analysis_result', pd.DataFrame())
            st.write(f"**Total GAP products:** {len(gap_data)}")
            st.write(f"**Products with both demand & supply:** {products_with_supply['pt_code'].nunique()}")
            st.write(f"**Total periods included:** {len(products_with_supply)}")
            st.write(f"**Available columns:** {products_with_supply.columns.tolist()}")
            
            if not products_with_supply.empty:
                # Show sample data
                sample = products_with_supply.head(3)
                st.write("**Sample enriched data:**")
                for col in ['pt_code', 'legal_entity', 'customer', 'brand']:
                    if col in sample.columns:
                        st.write(f"- {col}: {sample[col].tolist()}")
                
                # Check specific product if exists
                if 'P007001219' in products_with_supply['pt_code'].values:
                    st.write("\n**P007001219 specific data:**")
                    p007_data = products_with_supply[products_with_supply['pt_code'] == 'P007001219']
                    st.write(f"- Periods: {len(p007_data)}")
                    st.write(f"- Total demand: {p007_data.get('total_demand_qty', pd.Series()).sum()}")
                    st.write(f"- Total supply: {p007_data.get('total_available', pd.Series()).sum()}")

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
    """Prepare product summary data với logic mới và error handling tốt hơn"""
    # Initialize empty DataFrame để tránh UnboundLocalError
    product_summary = pd.DataFrame()
    
    try:
        # Check if filtered_data is empty
        if filtered_data.empty:
            return product_summary
        
        # Debug: Print available columns
        if st.session_state.get('debug_mode', False):
            st.write("🐛 Debug - Available columns in filtered_data:")
            st.write(list(filtered_data.columns))
            st.write(f"🐛 Number of rows: {len(filtered_data)}")
        
        # Prepare demand column - check multiple possible column names
        demand_col = None
        possible_demand_cols = ['original_demand_qty', 'total_demand_qty', 'demand_quantity']
        
        for col in possible_demand_cols:
            if col in filtered_data.columns:
                demand_col = col
                break
        
        # If no demand column found, create one
        if demand_col is None:
            st.warning("⚠️ No demand column found, using total_demand_qty as fallback")
            filtered_data['original_demand_qty'] = filtered_data.get('total_demand_qty', 0)
            demand_col = 'original_demand_qty'
        else:
            # Ensure we have original_demand_qty for consistency
            if demand_col != 'original_demand_qty':
                filtered_data['original_demand_qty'] = filtered_data[demand_col]
        
        # Debug: Check demand values
        if st.session_state.get('debug_mode', False):
            st.write(f"🐛 Using demand column: {demand_col}")
            st.write(f"🐛 Sample demand values: {filtered_data['original_demand_qty'].head()}")
        
        # Group by product
        agg_dict = {
            'original_demand_qty': 'sum',
            'total_available': 'sum',
            'gap_quantity': 'sum',
            'fulfillment_rate_percent': 'mean'
        }
        
        # Remove columns that don't exist
        agg_dict = {k: v for k, v in agg_dict.items() if k in filtered_data.columns}
        
        product_summary = filtered_data.groupby(['pt_code', 'product_name']).agg(agg_dict).reset_index()
        
        # Add period count
        if 'period' in filtered_data.columns:
            period_counts = filtered_data.groupby(['pt_code'])['period'].nunique().reset_index()
            period_counts.columns = ['pt_code', 'period_count']
            product_summary = product_summary.merge(period_counts, on='pt_code', how='left')
        else:
            product_summary['period_count'] = 1
        
        # Rename columns
        product_summary = product_summary.rename(columns={
            'original_demand_qty': 'period_demand_sum',
            'total_available': 'available_supply_sum'
        })
        
        # Ensure required columns exist with defaults
        if 'period_demand_sum' not in product_summary.columns:
            product_summary['period_demand_sum'] = 0
        if 'available_supply_sum' not in product_summary.columns:
            product_summary['available_supply_sum'] = 0
        if 'fulfillment_rate_percent' not in product_summary.columns:
            product_summary['fulfillment_rate_percent'] = 0
        
        # Recalculate net GAP
        product_summary['net_gap'] = (
            product_summary['available_supply_sum'] - 
            product_summary['period_demand_sum']
        )
        
        # Add status
        def get_status(row):
            avg_fulfillment = row.get('fulfillment_rate_percent', 0)
            net_gap = row.get('net_gap', 0)
            demand = row.get('period_demand_sum', 0)
            
            # Special case: No demand
            if demand == 0:
                if net_gap > 0:
                    return '📦 Supply Only'
                else:
                    return '❓ No Activity'
            
            # Timing issue: supply exists but low fulfillment
            if net_gap > 0 and avg_fulfillment < 50:
                return '🟡 Timing Issue'
            elif avg_fulfillment >= 80:
                return '🟢 Available'
            elif avg_fulfillment >= 50:
                return '🟡 Partial'
            else:
                return '🔴 Shortage'
        
        product_summary['status'] = product_summary.apply(get_status, axis=1)
        
        # Add absolute gap for sorting
        product_summary['gap_abs'] = product_summary['net_gap'].abs()
        
        # Sort by absolute gap
        product_summary = product_summary.sort_values('gap_abs', ascending=False)
        
        # Debug final summary
        if st.session_state.get('debug_mode', False):
            st.write("🐛 Product summary sample:")
            st.dataframe(product_summary.head())
        
        return product_summary
        
    except Exception as e:
        st.error(f"Error in prepare_product_summary: {str(e)}")
        if st.session_state.get('debug_mode', False):
            import traceback
            st.code(traceback.format_exc())
        return pd.DataFrame()

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
    # Initialize if not exists
    if 'alloc_current_page' not in st.session_state:
        st.session_state['alloc_current_page'] = 1
    
    # Ensure it's within valid range
    if total_pages > 0:
        st.session_state['alloc_current_page'] = min(
            max(1, st.session_state['alloc_current_page']), 
            total_pages
        )
    else:
        st.session_state['alloc_current_page'] = 1

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


# File: utils/allocation_wizard_components.py
# Function: show_selection_header()

def show_selection_header():
    """Show selection table header với updated columns"""
    st.markdown("---")
    # Updated với 7 columns matching show_product_row
    header_col1, header_col2, header_col3, header_col4, header_col5, header_col6, header_col7 = st.columns([0.5, 2, 1.2, 1.2, 1.2, 1.2, 1])
    
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
    with header_col7:
        st.markdown("**Avg Fill %**")
        
    st.markdown("---")

# File: utils/allocation_wizard_components.py
# Function: show_product_row()

def show_product_row(row, selected_products, select_all_page):
    """Show individual product row with better error handling"""
    # 7 columns
    col1, col2, col3, col4, col5, col6, col7 = st.columns([0.5, 2, 1.2, 1.2, 1.2, 1.2, 1])
    
    with col1:
        # Checkbox
        is_selected = row['pt_code'] in selected_products
        
        selected = st.checkbox(
            "Select",  # Add label to fix warning
            value=select_all_page or is_selected,
            key=f"select_{row['pt_code']}_{st.session_state.get('alloc_current_page', 1)}",
            label_visibility="collapsed"  # Hide label but satisfy requirement
        )
        
        if selected and row['pt_code'] not in selected_products:
            selected_products.append(row['pt_code'])
        elif not selected and row['pt_code'] in selected_products:
            selected_products.remove(row['pt_code'])
    
    with col2:
        # Product info
        st.write(f"**{row['pt_code']}**")
        product_name = str(row.get('product_name', ''))
        display_name = product_name[:50] + ('...' if len(product_name) > 50 else '')
        st.caption(display_name)
        
        # Show period count
        period_count = int(row.get('period_count', 1))
        if period_count > 0:
            st.caption(f"📅 {period_count} period{'s' if period_count > 1 else ''}")
    
    with col3:
        # Status
        status = row.get('status', '❓ Unknown')
        st.write(status)
        
        # Help for special statuses
        if 'Supply Only' in status:
            st.caption("No demand", help="Product has supply but no demand")
        elif 'Timing Issue' in status:
            st.caption("Wrong timing", help="Supply available but not when needed")
    
    with col4:
        # Gap with safe handling
        gap = float(row.get('net_gap', row.get('gap_quantity', 0)))
        
        if gap < 0:
            st.metric(
                "Gap",
                f"-{format_number(abs(gap))}",
                label_visibility="collapsed",
                help="Shortage: Demand exceeds supply"
            )
        else:
            st.metric(
                "Gap",
                f"+{format_number(gap)}",
                label_visibility="collapsed",
                help="Surplus: Supply exceeds demand"
            )
    
    with col5:
        # Demand with zero handling
        demand = float(row.get('period_demand_sum', 0))
        st.metric(
            "Demand",
            format_number(demand),
            label_visibility="collapsed",
            help="Total period demand (excluding backlog)"
        )
    
    with col6:
        # Supply
        supply = float(row.get('available_supply_sum', 0))
        st.metric(
            "Supply",
            format_number(supply),
            label_visibility="collapsed",
            help="Total available supply"
        )
    
    with col7:
        # Fulfillment with special handling for no demand
        fulfillment = float(row.get('fulfillment_rate_percent', 0))
        demand = float(row.get('period_demand_sum', 0))
        period_count = int(row.get('period_count', 1))
        
        # Color coding
        if demand == 0:
            delta_color = "off"  # Gray for no demand
            display_text = "N/A"
            tooltip = "No demand to fulfill"
        else:
            if fulfillment >= 80:
                delta_color = "normal"
            elif fulfillment >= 50:
                delta_color = "off"
            else:
                delta_color = "inverse"
            
            display_text = f"{fulfillment:.0f}%"
            
            # Tooltip
            tooltip_parts = [
                f"Average fulfillment across {period_count} period{'s' if period_count > 1 else ''}: {fulfillment:.1f}%"
            ]
            
            if fulfillment < 50:
                tooltip_parts.append("⚠️ Critical shortage")
            elif fulfillment < 80:
                tooltip_parts.append("⚡ Partial shortage")
            else:
                tooltip_parts.append("✅ Good coverage")
            
            tooltip = "\n".join(tooltip_parts)
        
        st.metric(
            "Avg Fill %",
            display_text,
            label_visibility="collapsed",
            delta_color=delta_color,
            help=tooltip
        )

def show_selection_summary(selected_products, filtered_data):
    """Show summary of selected products với updated metrics"""
    st.markdown("---")
    
    if selected_products:
        summary_col1, summary_col2 = st.columns([3, 1])
        
        with summary_col1:
            st.success(f"✅ Selected {len(selected_products)} products for allocation")
            
            # Get selected data
            selected_data = filtered_data[filtered_data['pt_code'].isin(selected_products)]
            
            if not selected_data.empty:
                # Use correct column names
                demand_col = 'original_demand_qty' if 'original_demand_qty' in selected_data.columns else 'total_demand_qty'
                
                # Calculate totals
                total_period_demand = selected_data[demand_col].sum()
                total_supply = selected_data['total_available'].sum()
                net_gap = total_supply - total_period_demand
                avg_fulfillment = selected_data['fulfillment_rate_percent'].mean()
                
                # Display metrics
                metric_cols = st.columns(4)
                
                with metric_cols[0]:
                    st.metric(
                        "Total Period Demand",
                        format_number(total_period_demand),
                        help="Sum of period demand (no backlog)"
                    )
                
                with metric_cols[1]:
                    st.metric(
                        "Total Supply",
                        format_number(total_supply),
                        help="Sum of available supply"
                    )
                
                with metric_cols[2]:
                    delta_color = "normal" if net_gap >= 0 else "inverse"
                    st.metric(
                        "Net Gap",
                        format_number(abs(net_gap)),
                        delta="Surplus" if net_gap >= 0 else "Shortage",
                        delta_color=delta_color
                    )
                
                with metric_cols[3]:
                    st.metric(
                        "Avg Fulfillment",
                        f"{avg_fulfillment:.1f}%",
                        help="Average fulfillment rate"
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

    # File: utils/allocation_wizard_components.py
    # Thêm debug function để verify data structure

def debug_gap_data_structure(gap_data):
    """Debug function to verify GAP data columns"""
    if st.session_state.get('debug_mode', False):
        st.write("🐛 GAP Data Columns:", gap_data.columns.tolist())
        
        # Check for demand columns
        demand_cols = [col for col in gap_data.columns if 'demand' in col.lower()]
        st.write("🐛 Demand columns found:", demand_cols)
        
        # Sample data
        if not gap_data.empty:
            st.write("🐛 Sample GAP data:")
            st.dataframe(gap_data.head(3))