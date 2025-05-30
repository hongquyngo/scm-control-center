import streamlit as st
import pandas as pd
from datetime import datetime
import numpy as np
from sqlalchemy import text
from utils.db import get_db_engine
from utils.data_loader import (
    load_outbound_demand_data, 
    load_customer_forecast_data,
    load_inventory_data,
    load_pending_can_data,
    load_pending_po_data,
    load_pending_wh_transfer_data,  # New import
    load_active_allocations
)
from utils.helpers import (
    convert_df_to_excel,
    export_multiple_sheets,
    convert_to_period,
    sort_period_columns,
    format_number,
    format_currency,
    format_percentage,
    check_missing_dates,
    save_to_session_state,
    parse_week_period,
    parse_month_period,
    is_past_period  
)

# === Page Config ===
st.set_page_config(
    page_title="GAP Analysis - SCM",
    page_icon="📊",
    layout="wide"
)

# === Constants ===
PERIOD_TYPES = ["Daily", "Weekly", "Monthly"]
STYLE_MODES = ["None", "🔴 Highlight Shortage", "🌈 Heatmap"]

# === Header with Navigation ===
col1, col2, col3 = st.columns([1, 4, 1])
with col1:
    if st.button("← Supply"):
        st.switch_page("pages/2_📥_Supply_Analysis.py")
with col2:
    st.title("📊 Supply-Demand GAP Analysis")
with col3:
    if st.button("🏠 Dashboard"):
        st.switch_page("main.py")

st.markdown("---")

# === Data Loading Functions ===
def load_and_prepare_demand_data(selected_demand_sources, include_converted):
    """Load and standardize demand data based on source selection"""
    df_parts = []
    
    # Debug mode
    debug_mode = st.session_state.get('debug_mode', False)
    
    if "OC" in selected_demand_sources:
        df_oc = load_outbound_demand_data()
        if not df_oc.empty:
            if debug_mode:
                st.write(f"Debug - OC data shape: {df_oc.shape}")
                st.write(f"Debug - OC columns: {df_oc.columns.tolist()}")
            
            df_oc["source_type"] = "OC"
            standardized_oc = standardize_demand_df(df_oc, is_forecast=False)
            
            if debug_mode:
                st.write(f"Debug - Standardized OC shape: {standardized_oc.shape}")
                # Check for potential duplicates
                dup_check = standardized_oc.groupby(['pt_code', 'etd']).size()
                dups = dup_check[dup_check > 1]
                if not dups.empty:
                    st.warning(f"Debug - Found products with multiple OC lines on same date:")
                    st.write(dups)
            
            df_parts.append(standardized_oc)
    
    if "Forecast" in selected_demand_sources:
        df_fc = load_customer_forecast_data()
        if not df_fc.empty:
            df_fc["source_type"] = "Forecast"
            standardized_fc = standardize_demand_df(df_fc, is_forecast=True)
            
            # Handle converted forecasts
            if not include_converted and 'is_converted_to_oc' in standardized_fc.columns:
                standardized_fc = standardized_fc[standardized_fc["is_converted_to_oc"] != "Yes"]
            
            df_parts.append(standardized_fc)
    
    if df_parts:
        final_df = pd.concat(df_parts, ignore_index=True)
        
        if debug_mode:
            st.write(f"Debug - Final demand shape: {final_df.shape}")
            st.write(f"Debug - Products: {final_df['pt_code'].nunique()}")
        
        return final_df
    
    return pd.DataFrame()

def standardize_demand_df(df, is_forecast):
    """Standardize column names and data types for demand dataframes"""
    df = df.copy()
    
    # Date columns
    df["etd"] = pd.to_datetime(df["etd"], errors="coerce")
    df["oc_date"] = pd.to_datetime(df.get("oc_date", pd.NaT), errors="coerce")
    
    # Quantity and value columns
    if is_forecast:
        df['demand_quantity'] = pd.to_numeric(df['standard_quantity'], errors='coerce').fillna(0)
        df['value_in_usd'] = pd.to_numeric(df.get('total_amount_usd', 0), errors='coerce').fillna(0)
        df['demand_number'] = df.get('forecast_number', '')
        df['is_converted_to_oc'] = df.get('is_converted_to_oc', 'No')
        df['demand_line_id'] = df['forecast_line_id'].astype(str) + '_FC'
    else:
        df['demand_quantity'] = pd.to_numeric(df['pending_standard_delivery_quantity'], errors='coerce').fillna(0)
        df['value_in_usd'] = pd.to_numeric(df.get('outstanding_amount_usd', 0), errors='coerce').fillna(0)
        df['demand_number'] = df.get('oc_number', '')
        df['is_converted_to_oc'] = 'N/A'
        df['demand_line_id'] = df['ocd_id'].astype(str) + '_OC'
    
    # Clean string columns - IMPORTANT for proper grouping
    string_cols = ['product_name', 'pt_code', 'brand', 'legal_entity', 'customer']
    for col in string_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
            # Remove any double spaces
            df[col] = df[col].str.replace(r'\s+', ' ', regex=True)
    
    # Clean UOM and package size
    df['standard_uom'] = df.get('standard_uom', '').astype(str).str.strip().str.upper()
    df['package_size'] = df.get('package_size', '').astype(str).str.strip()
    
    return df

def enhance_demand_with_allocation_info(df_demand):
    """Add allocation information to demand dataframe"""
    
    # Load allocation data
    engine = get_db_engine()
    
    try:
        oc_allocations = pd.read_sql(text("""
            SELECT 
                demand_reference_id,
                SUM(allocated_qty) as total_allocated,
                SUM(delivered_qty) as total_delivered,
                SUM(allocated_qty - delivered_qty) as undelivered_allocated
            FROM allocation_details ad
            JOIN allocation_plans ap ON ad.allocation_plan_id = ap.id
            WHERE ap.status IN ('APPROVED', 'EXECUTED')
              AND ad.demand_type = 'OC'
              AND ad.status NOT IN ('CANCELLED')
            GROUP BY demand_reference_id
        """), engine)
        
        if not oc_allocations.empty and 'demand_line_id' in df_demand.columns:
            # Extract OC IDs for merging
            df_demand['merge_id'] = df_demand['demand_line_id'].str.extract(r'(\d+)_OC')
            df_demand['merge_id'] = pd.to_numeric(df_demand['merge_id'], errors='coerce')
            
            # Merge with allocation data
            df_demand = df_demand.merge(
                oc_allocations,
                left_on='merge_id',
                right_on='demand_reference_id',
                how='left',
                suffixes=('', '_alloc')
            )
            
            # Clean up
            df_demand.drop(columns=['merge_id', 'demand_reference_id'], inplace=True, errors='ignore')
    except:
        # If allocation query fails, continue without allocation data
        pass
    
    # Fill allocation columns
    allocation_cols = ['total_allocated', 'total_delivered', 'undelivered_allocated']
    for col in allocation_cols:
        if col not in df_demand.columns:
            df_demand[col] = 0
        else:
            df_demand[col] = df_demand[col].fillna(0)
    
    # Calculate unallocated demand
    df_demand['unallocated_demand'] = df_demand['demand_quantity'] - df_demand['total_allocated']
    df_demand['unallocated_demand'] = df_demand['unallocated_demand'].clip(lower=0)
    
    # Add allocation status
    df_demand['allocation_status'] = df_demand.apply(
        lambda x: 'Fully Allocated' if x['unallocated_demand'] <= 0 
        else 'Partial' if x['total_allocated'] > 0 
        else 'Not Allocated', axis=1
    )
    
    return df_demand

def load_and_prepare_supply_data(selected_supply_sources, exclude_expired=True):
    """Load and standardize supply data based on source selection"""
    today = pd.to_datetime("today").normalize()
    df_parts = []
    
    debug_mode = st.session_state.get('debug_mode', False)
    
    # Load Inventory
    if "Inventory" in selected_supply_sources:
        inv_df = load_inventory_data()
        if not inv_df.empty:
            inv_df = prepare_inventory_data(inv_df, today, exclude_expired)
            df_parts.append(inv_df)
    
    # Load Pending CAN
    if "Pending CAN" in selected_supply_sources:
        can_df = load_pending_can_data()
        if not can_df.empty:
            can_df = prepare_can_data(can_df)
            df_parts.append(can_df)
    
    # Load Pending PO
    if "Pending PO" in selected_supply_sources:
        po_df = load_pending_po_data()
        if not po_df.empty:
            po_df = prepare_po_data(po_df)
            df_parts.append(po_df)
    
    # Load Pending WH Transfer - NEW
    if "Pending WH Transfer" in selected_supply_sources:
        wht_df = load_pending_wh_transfer_data()
        if not wht_df.empty:
            wht_df = prepare_wh_transfer_data(wht_df, today, exclude_expired)
            df_parts.append(wht_df)
    
    if not df_parts:
        return pd.DataFrame()
    
    # Standardize all parts
    standardized_parts = [standardize_supply_df(df) for df in df_parts]
    final_df = pd.concat(standardized_parts, ignore_index=True)
    
    if debug_mode:
        st.write(f"Debug - Supply data shape: {final_df.shape}")
        st.write(f"Debug - Supply sources: {final_df['source_type'].value_counts().to_dict()}")
    
    return final_df

def prepare_inventory_data(inv_df, today, exclude_expired):
    """Prepare inventory data"""
    inv_df = inv_df.copy()
    inv_df["source_type"] = "Inventory"
    inv_df["date_ref"] = today
    inv_df["quantity"] = pd.to_numeric(inv_df["remaining_quantity"], errors="coerce").fillna(0)
    inv_df["legal_entity"] = inv_df["owning_company_name"]
    inv_df["expiry_date"] = pd.to_datetime(inv_df["expiry_date"], errors="coerce")
    
    if exclude_expired:
        inv_df = inv_df[(inv_df["expiry_date"].isna()) | (inv_df["expiry_date"] >= today)]
    
    return inv_df

def prepare_can_data(can_df):
    """Prepare CAN data"""
    can_df = can_df.copy()
    can_df["source_type"] = "Pending CAN"
    can_df["date_ref"] = pd.to_datetime(can_df["arrival_date"], errors="coerce")
    can_df["quantity"] = pd.to_numeric(can_df["pending_quantity"], errors="coerce").fillna(0)
    can_df["legal_entity"] = can_df["consignee"]
    return can_df

def prepare_po_data(po_df):
    """Prepare PO data"""
    po_df = po_df.copy()
    po_df["source_type"] = "Pending PO"
    po_df["date_ref"] = pd.to_datetime(po_df["cargo_ready_date"], errors="coerce")
    po_df["quantity"] = pd.to_numeric(po_df["pending_standard_arrival_quantity"], errors="coerce").fillna(0)
    po_df["legal_entity"] = po_df["legal_entity"]
    return po_df

def prepare_wh_transfer_data(wht_df, today, exclude_expired):
    """Prepare warehouse transfer data - NEW"""
    wht_df = wht_df.copy()
    
    wht_df["source_type"] = "Pending WH Transfer"
    # Assuming transfer will complete in 2-3 days, we estimate arrival date
    wht_df["transfer_date"] = pd.to_datetime(wht_df["transfer_date"], errors="coerce")
    wht_df["date_ref"] = wht_df["transfer_date"] + pd.Timedelta(days=5)  # Estimate 2 days for transfer
    wht_df["quantity"] = pd.to_numeric(wht_df["transfer_quantity"], errors="coerce").fillna(0)
    wht_df["legal_entity"] = wht_df["owning_company_name"]
    wht_df["expiry_date"] = pd.to_datetime(wht_df["expiry_date"], errors="coerce")
    
    if exclude_expired:
        wht_df = wht_df[(wht_df["expiry_date"].isna()) | (wht_df["expiry_date"] >= today)]
    
    return wht_df

def standardize_supply_df(df):
    """Standardize supply dataframe columns"""
    df = df.copy()
    
    # Clean string columns
    string_cols = ["pt_code", "product_name", "brand"]
    for col in string_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
            df[col] = df[col].str.replace(r'\s+', ' ', regex=True)
    
    # Clean UOM and package size
    df["standard_uom"] = df.get("standard_uom", "").astype(str).str.strip().str.upper()
    df["package_size"] = df.get("package_size", "").astype(str).str.strip()
    
    # Select standard columns
    return df[[
        "source_type", "pt_code", "product_name", "brand", 
        "package_size", "standard_uom", "legal_entity", "date_ref", 
        "quantity"
    ]]

def adjust_supply_for_allocations(df_supply):
    """Adjust supply quantities based on active allocations"""
    
    # Load active allocations
    df_allocations = load_active_allocations()
    
    if not df_allocations.empty and not df_supply.empty:
        # Adjust available supply by subtracting undelivered allocations
        for _, alloc in df_allocations.iterrows():
            # Match by product and entity
            mask = (
                (df_supply['pt_code'] == alloc['pt_code']) & 
                (df_supply['legal_entity'] == alloc['legal_entity_name'])
            )
            
            # Subtract undelivered allocation from available supply
            if mask.any():
                df_supply.loc[mask, 'quantity'] = df_supply.loc[mask, 'quantity'].apply(
                    lambda x: max(0, x - alloc['undelivered_qty'])
                )
    
    # Remove rows with zero quantity
    df_supply = df_supply[df_supply['quantity'] > 0].copy()
    
    return df_supply

# === Source Selection ===
def select_gap_sources():
    """Select demand and supply sources for GAP analysis"""
    st.markdown("### 📊 Data Source Selection")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📤 Demand Sources")
        
        col1_1, col1_2 = st.columns(2)
        with col1_1:
            demand_oc = st.checkbox("OC", value=True, key="demand_oc")
        with col1_2:
            demand_forecast = st.checkbox("Forecast", value=True, key="demand_forecast")
        
        selected_demand_sources = []
        if demand_oc:
            selected_demand_sources.append("OC")
        if demand_forecast:
            selected_demand_sources.append("Forecast")
        
        if demand_forecast:
            include_converted = st.checkbox(
                "Include Converted Forecasts", 
                value=False,
                help="⚠️ May cause double counting if OC is also selected",
                key="include_converted_forecasts"
            )
        else:
            include_converted = False
        
        # Customer filter - only for demand
        st.markdown("##### Customer Filter")
        
        # Get customers from session state or empty list
        all_customers = st.session_state.get('filter_customers', [])
        
        selected_customers = st.multiselect(
            "Select Customers", 
            options=all_customers,
            key="gap_customer",
            help="Filter demand by specific customers",
            placeholder="Choose an option" if all_customers else "No customers available"
        )
    
    with col2:
        st.markdown("#### 📥 Supply Sources")
        
        col2_1, col2_2 = st.columns(2)
        with col2_1:
            supply_inv = st.checkbox("Inventory", value=True, key="supply_inv")
            supply_can = st.checkbox("Pending CAN", value=True, key="supply_can")
        with col2_2:
            supply_po = st.checkbox("Pending PO", value=True, key="supply_po")
            supply_wht = st.checkbox("Pending WH Transfer", value=True, key="supply_wht")  # NEW
        
        exclude_expired = st.checkbox(
            "Exclude Expired", 
            value=True,
            help="Exclude expired inventory items",
            key="exclude_expired_gap"
        )
        
        selected_supply_sources = []
        if supply_inv:
            selected_supply_sources.append("Inventory")
        if supply_can:
            selected_supply_sources.append("Pending CAN")
        if supply_po:
            selected_supply_sources.append("Pending PO")
        if supply_wht:
            selected_supply_sources.append("Pending WH Transfer")  # NEW
    
    return {
        "demand": selected_demand_sources,
        "supply": selected_supply_sources,
        "include_converted": include_converted,
        "exclude_expired": exclude_expired,
        "selected_customers": selected_customers
    }

# === Filtering Functions ===
# === Updated Filtering Functions for GAP Analysis ===
def apply_gap_filters(df_demand=None, df_supply=None):
    """Apply filters for GAP analysis with enhanced product search"""
    with st.expander("📎 Advanced Filters", expanded=True):
        # Get available options from data if provided
        if df_demand is not None and df_supply is not None and not df_demand.empty and not df_supply.empty:
            # Entities
            all_entities = sorted(set(
                df_demand["legal_entity"].dropna().unique().tolist() + 
                df_supply["legal_entity"].dropna().unique().tolist()
            ))
            
            # Products - Enhanced with product names
            demand_products = df_demand[['pt_code', 'product_name']].drop_duplicates()
            supply_products = df_supply[['pt_code', 'product_name']].drop_duplicates()
            all_products_df = pd.concat([demand_products, supply_products]).drop_duplicates()
            
            # Filter out invalid PT codes
            all_products_df = all_products_df[
                (all_products_df['pt_code'].notna()) & 
                (all_products_df['pt_code'] != '') &
                (all_products_df['pt_code'] != 'nan')
            ]
            
            product_options = []
            for _, row in all_products_df.iterrows():
                pt_code = str(row['pt_code'])
                product_name = str(row['product_name'])[:50] if pd.notna(row['product_name']) else ""
                option = f"{pt_code} - {product_name}"
                product_options.append(option)
            
            product_options = sorted(list(set(product_options)))
            
            # Brands
            all_brands = sorted(set(
                df_demand["brand"].dropna().unique().tolist() + 
                df_supply["brand"].dropna().unique().tolist()
            ))
            
            # Get date ranges
            demand_dates = df_demand["etd"].dropna()
            supply_dates = df_supply["date_ref"].dropna()
            
            if len(demand_dates) > 0 and len(supply_dates) > 0:
                default_start = min(demand_dates.min(), supply_dates.min()).date()
                default_end = max(demand_dates.max(), supply_dates.max()).date()
            elif len(demand_dates) > 0:
                default_start = demand_dates.min().date()
                default_end = demand_dates.max().date()
            elif len(supply_dates) > 0:
                default_start = supply_dates.min().date()
                default_end = supply_dates.max().date()
            else:
                default_start = default_end = datetime.today().date()
        else:
            # Use session state if available
            all_entities = st.session_state.get('filter_entities', [])
            product_options = st.session_state.get('filter_product_options', [])
            all_brands = st.session_state.get('filter_brands', [])
            default_start = default_end = datetime.today().date()
        
        # Filter controls
        col1, col2, col3 = st.columns(3)
        
        with col1:
            selected_entity = st.multiselect(
                "Legal Entity", 
                options=all_entities,
                key="gap_entity"
            )
        
        with col2:
            selected_products = st.multiselect(
                "Product (PT Code - Name)", 
                options=product_options,
                key="gap_product",
                help="Search by PT Code or Product Name"
            )
            
            # Extract selected PT codes
            selected_pt_codes = [p.split(' - ')[0] for p in selected_products]
        
        with col3:
            selected_brand = st.multiselect(
                "Brand", 
                options=all_brands,
                key="gap_brand"
            )
        
        # Date range
        col5, col6 = st.columns(2)
        
        with col5:
            start_date = st.date_input(
                "From Date", 
                value=default_start,
                key="gap_start_date"
            )
        
        with col6:
            end_date = st.date_input(
                "To Date", 
                value=default_end,
                key="gap_end_date"
            )
    
    return {
        "entity": selected_entity,
        "product": selected_pt_codes,  # Return PT codes for filtering
        "brand": selected_brand,
        "start_date": start_date,
        "end_date": end_date
    }

def update_filter_options(df_demand, df_supply):
    """Update filter options based on loaded data with enhanced product search"""
    # Get unique values
    all_entities = sorted(set(
        df_demand["legal_entity"].dropna().astype(str).unique().tolist() + 
        df_supply["legal_entity"].dropna().astype(str).unique().tolist()
    ))
    all_entities = [e for e in all_entities if e and e != 'nan']
    
    # Enhanced product options with names
    demand_products = df_demand[['pt_code', 'product_name']].drop_duplicates()
    supply_products = df_supply[['pt_code', 'product_name']].drop_duplicates()
    all_products_df = pd.concat([demand_products, supply_products]).drop_duplicates()
    
    # Filter out invalid PT codes
    all_products_df = all_products_df[
        (all_products_df['pt_code'].notna()) & 
        (all_products_df['pt_code'] != '') &
        (all_products_df['pt_code'] != 'nan')
    ]
    
    product_options = []
    all_pt_codes = []
    for _, row in all_products_df.iterrows():
        pt_code = str(row['pt_code'])
        product_name = str(row['product_name'])[:50] if pd.notna(row['product_name']) else ""
        option = f"{pt_code} - {product_name}"
        product_options.append(option)
        all_pt_codes.append(pt_code)
    
    product_options = sorted(list(set(product_options)))
    all_pt_codes = sorted(list(set(all_pt_codes)))
    
    all_brands = sorted(set(
        df_demand["brand"].dropna().astype(str).unique().tolist() + 
        df_supply["brand"].dropna().astype(str).unique().tolist()
    ))
    all_brands = [b for b in all_brands if b and b != 'nan']
    
    # Get customers from demand only
    all_customers = []
    if 'customer' in df_demand.columns:
        all_customers = sorted(df_demand["customer"].dropna().astype(str).unique().tolist())
        all_customers = [c for c in all_customers if c and c != 'nan']
    
    # Store in session state
    st.session_state['filter_entities'] = all_entities
    st.session_state['filter_products'] = all_pt_codes  # Keep for backward compatibility
    st.session_state['filter_product_options'] = product_options  # New: product with names
    st.session_state['filter_brands'] = all_brands
    st.session_state['filter_customers'] = all_customers
    
    # Debug info
    if st.session_state.get('debug_mode', False):
        st.write(f"Debug - Filter options updated:")
        st.write(f"- Entities: {len(all_entities)}")
        st.write(f"- Products: {len(all_pt_codes)}")
        st.write(f"- Product Options: {len(product_options)}")
        st.write(f"- Brands: {len(all_brands)}")
        st.write(f"- Customers: {len(all_customers)}")
        if len(all_customers) == 0:
            st.warning("Debug - No customers found in demand data!")


def apply_filters_to_data(df_demand, df_supply, filters, selected_customers=None):
    """Apply filters to demand and supply dataframes"""
    # Apply filters to demand
    filtered_demand = df_demand.copy()
    if filters["entity"]:
        filtered_demand = filtered_demand[filtered_demand["legal_entity"].isin(filters["entity"])]
    if selected_customers:  # Customer filter from demand source selection
        filtered_demand = filtered_demand[filtered_demand["customer"].isin(selected_customers)]
    if filters["product"]:
        filtered_demand = filtered_demand[filtered_demand["pt_code"].isin(filters["product"])]
    if filters["brand"]:
        filtered_demand = filtered_demand[filtered_demand["brand"].isin(filters["brand"])]
    
    # Date filter for demand
    start_ts = pd.to_datetime(filters["start_date"])
    end_ts = pd.to_datetime(filters["end_date"])
    filtered_demand = filtered_demand[
        filtered_demand["etd"].isna() | 
        ((filtered_demand["etd"] >= start_ts) & (filtered_demand["etd"] <= end_ts))
    ]
    
    # Apply filters to supply
    filtered_supply = df_supply.copy()
    if filters["entity"]:
        filtered_supply = filtered_supply[filtered_supply["legal_entity"].isin(filters["entity"])]
    if filters["product"]:
        filtered_supply = filtered_supply[filtered_supply["pt_code"].isin(filters["product"])]
    if filters["brand"]:
        filtered_supply = filtered_supply[filtered_supply["brand"].isin(filters["brand"])]
    
    # Date filter for supply
    filtered_supply = filtered_supply[
        (filtered_supply["date_ref"] >= start_ts) & 
        (filtered_supply["date_ref"] <= end_ts)
    ]
    
    return filtered_demand, filtered_supply

# === GAP Calculation ===
def calculate_gap_with_carry_forward(df_demand, df_supply, period_type="Weekly"):
    """Calculate supply-demand gap with allocation awareness"""
    
    debug_mode = st.session_state.get('debug_mode', False)
    
    # Enhance demand with allocation info
    df_demand_enhanced = enhance_demand_with_allocation_info(df_demand)
    
    df_d = df_demand_enhanced.copy()
    df_s = df_supply.copy()
    
    # Convert to periods
    df_d["period"] = convert_to_period(df_d["etd"], period_type)
    df_s["period"] = convert_to_period(df_s["date_ref"], period_type)
    
    # Remove rows with invalid periods
    df_d = df_d[df_d["period"].notna() & (df_d["period"] != "")]
    df_s = df_s[df_s["period"].notna() & (df_s["period"] != "")]
    
    if debug_mode:
        st.write("Debug - Before grouping:")
        st.write(f"Unique products in demand: {df_d['pt_code'].nunique()}")
        st.write(f"Unique products in supply: {df_s['pt_code'].nunique()}")
    
    # Group demand by product and period
    demand_grouped = df_d.groupby(
        ["pt_code", "product_name", "package_size", "standard_uom", "period"],
        as_index=False,
        dropna=False
    ).agg({
        "demand_quantity": "sum",
        "unallocated_demand": "sum"
    })
    
    # Use unallocated demand for GAP calculation
    demand_grouped["total_demand_qty"] = demand_grouped["unallocated_demand"]
    
    # Group supply
    supply_grouped = df_s.groupby(
        ["pt_code", "product_name", "package_size", "standard_uom", "period"],
        as_index=False,
        dropna=False
    ).agg({
        "quantity": "sum"
    })
    supply_grouped.rename(columns={"quantity": "total_supply_qty"}, inplace=True)
    
    if debug_mode:
        st.write("Debug - After grouping:")
        st.write(f"Unique products in demand grouped: {demand_grouped['pt_code'].nunique()}")
        st.write(f"Unique products in supply grouped: {supply_grouped['pt_code'].nunique()}")
    
    # Get all periods and products
    all_periods = get_all_periods(demand_grouped, supply_grouped, period_type)
    all_products = get_all_products(demand_grouped, supply_grouped)
    
    if debug_mode:
        st.write(f"Debug - Total unique products for GAP: {len(all_products)}")
        st.write(f"Debug - All products pt_codes: {all_products['pt_code'].nunique()}")
    
    # Calculate gap for each product
    results = []
    for _, product in all_products.iterrows():
        product_gap = calculate_product_gap(
            product, 
            all_periods, 
            demand_grouped, 
            supply_grouped
        )
        results.extend(product_gap)
    
    gap_df = pd.DataFrame(results)
    
    # Final count
    if debug_mode and not gap_df.empty:
        st.write(f"Debug - Final GAP analysis unique products: {gap_df['pt_code'].nunique()}")
        st.write(f"Debug - Total GAP rows: {len(gap_df)}")
        st.write(f"Debug - Rows with zero demand: {len(gap_df[gap_df['total_demand_qty'] == 0])}")
    
    return gap_df

def calculate_product_gap_smart(product, all_periods, demand_grouped, supply_grouped):
    """Calculate gap for a single product across all periods - smart version"""
    pt_code = product["pt_code"]
    product_name = product["product_name"]
    package_size = product["package_size"]
    uom = product["standard_uom"]
    
    carry_forward_qty = 0
    results = []
    
    # Check if this product has any demand or supply across all periods
    product_has_demand = not demand_grouped[demand_grouped["pt_code"] == pt_code].empty
    product_has_supply = not supply_grouped[supply_grouped["pt_code"] == pt_code].empty
    
    for period in all_periods:
        # Get demand for this period
        demand_mask = (
            (demand_grouped["pt_code"] == pt_code) & 
            (demand_grouped["period"] == period)
        )
        demand = demand_grouped.loc[demand_mask, "total_demand_qty"].sum()
        
        # Get supply for this period
        supply_mask = (
            (supply_grouped["pt_code"] == pt_code) & 
            (supply_grouped["period"] == period)
        )
        supply = supply_grouped.loc[supply_mask, "total_supply_qty"].sum()
        
        # Only create a row if:
        # 1. There is demand in this period, OR
        # 2. There is supply in this period, OR
        # 3. There is carry forward inventory AND (product has future demand)
        has_activity = demand > 0 or supply > 0 or (carry_forward_qty > 0 and product_has_demand)
        
        if has_activity:
            # Calculate gap
            total_available = carry_forward_qty + supply
            gap = total_available - demand
            
            # Calculate fulfillment rate
            if demand > 0:
                fulfill_rate = min(100, (total_available / demand * 100))
            else:
                # If no demand, fulfillment is 100%
                fulfill_rate = 100
                
            status = "✅ Fulfilled" if gap >= 0 else "❌ Shortage"
            
            results.append({
                "pt_code": pt_code,
                "product_name": product_name,
                "package_size": package_size,
                "standard_uom": uom,
                "period": period,
                "begin_inventory": carry_forward_qty,
                "supply_in_period": supply,
                "total_available": total_available,
                "total_demand_qty": demand,
                "gap_quantity": gap,
                "fulfillment_rate_percent": fulfill_rate,
                "fulfillment_status": status,
            })
        
        # Update carry forward
        if has_activity:
            carry_forward_qty = max(0, total_available - demand)
        # If no activity but has carry forward, keep it for next period
    
    return results

def get_all_periods(demand_grouped, supply_grouped, period_type):
    """Get all unique periods sorted chronologically"""
    all_periods_raw = list(
        set(demand_grouped["period"]).union(set(supply_grouped["period"]))
    )
    
    # Filter out invalid periods
    all_periods_raw = [
        p for p in all_periods_raw 
        if pd.notna(p) and str(p).strip() != "" and str(p) != "nan"
    ]
    
    # Sort based on period type
    if period_type == "Weekly":
        return sorted(all_periods_raw, key=parse_week_period)
    elif period_type == "Monthly":
        return sorted(all_periods_raw, key=parse_month_period)
    else:  # Daily
        return sorted(all_periods_raw)

def get_all_products(demand_grouped, supply_grouped):
    """Get all unique products - only by pt_code"""
    # Get unique products by pt_code only
    demand_products = demand_grouped[["pt_code"]].drop_duplicates()
    supply_products = supply_grouped[["pt_code"]].drop_duplicates()
    
    # Combine and get unique pt_codes
    all_pt_codes = pd.concat([demand_products, supply_products]).drop_duplicates()
    
    # Now get full product info for each unique pt_code
    # Prefer demand info, fallback to supply info
    products_list = []
    
    for pt_code in all_pt_codes['pt_code'].unique():
        # Try to get from demand first
        demand_match = demand_grouped[demand_grouped['pt_code'] == pt_code].head(1)
        if not demand_match.empty:
            products_list.append(demand_match[["pt_code", "product_name", "package_size", "standard_uom"]])
        else:
            # Get from supply
            supply_match = supply_grouped[supply_grouped['pt_code'] == pt_code].head(1)
            if not supply_match.empty:
                products_list.append(supply_match[["pt_code", "product_name", "package_size", "standard_uom"]])
    
    if products_list:
        products_df = pd.concat(products_list, ignore_index=True)
        # Clean product data
        for col in ["pt_code", "product_name", "package_size", "standard_uom"]:
            products_df[col] = products_df[col].astype(str).str.strip()
        return products_df
    else:
        return pd.DataFrame(columns=["pt_code", "product_name", "package_size", "standard_uom"])

def calculate_product_gap(product, all_periods, demand_grouped, supply_grouped):
    """Calculate gap for a single product across all periods - only create rows when needed"""
    pt_code = product["pt_code"]
    product_name = product["product_name"]
    package_size = product["package_size"]
    uom = product["standard_uom"]
    
    carry_forward_qty = 0
    results = []
    has_created_row = False
    
    # Get all demand data for this product
    product_demand_data = demand_grouped[demand_grouped["pt_code"] == pt_code]
    
    # Get all supply data for this product  
    product_supply_data = supply_grouped[supply_grouped["pt_code"] == pt_code]
    
    for period in all_periods:
        # Get demand for this period
        demand_in_period = product_demand_data[product_demand_data["period"] == period]
        demand = demand_in_period["total_demand_qty"].sum() if not demand_in_period.empty else 0
        
        # Get supply for this period
        supply_in_period = product_supply_data[product_supply_data["period"] == period]
        supply = supply_in_period["total_supply_qty"].sum() if not supply_in_period.empty else 0
        
        # Calculate total available
        total_available = carry_forward_qty + supply
        
        # Decision logic: Should we create a row?
        should_create_row = False
        
        if demand > 0:
            # Always create row if there's demand
            should_create_row = True
        elif supply > 0 and has_created_row:
            # Create row if there's supply AND we've already created at least one row for this product
            should_create_row = True
        elif carry_forward_qty > 0 and not product_demand_data.empty:
            # Create row if there's carry forward AND product has demand in some period
            should_create_row = True
        
        if should_create_row:
            # Calculate gap
            gap = total_available - demand
            
            # Calculate fulfillment rate
            if demand > 0:
                fulfill_rate = min(100, (total_available / demand * 100))
            else:
                fulfill_rate = 100
                
            status = "✅ Fulfilled" if gap >= 0 else "❌ Shortage"
            
            results.append({
                "pt_code": pt_code,
                "product_name": product_name,
                "package_size": package_size,
                "standard_uom": uom,
                "period": period,
                "begin_inventory": carry_forward_qty,
                "supply_in_period": supply,
                "total_available": total_available,
                "total_demand_qty": demand,
                "gap_quantity": gap,
                "fulfillment_rate_percent": fulfill_rate,
                "fulfillment_status": status,
            })
            
            has_created_row = True
            
            # Update carry forward
            carry_forward_qty = max(0, gap)
        else:
            # No row created, but still update carry forward if there was supply
            if supply > 0:
                carry_forward_qty += supply
    
    return results

# === Display Options ===
# Replace the get_gap_display_options function with this fixed version:

def get_gap_display_options():
    """Get display options for GAP analysis"""
    st.markdown("### ⚙️ Display Options")
    
    col1, col2, col3, col4 = st.columns(4)
    
    # Initialize default values in session state if not exists
    if 'gap_period_type' not in st.session_state:
        st.session_state['gap_period_type'] = "Weekly"
    if 'gap_shortage_checkbox' not in st.session_state:
        st.session_state['gap_shortage_checkbox'] = False
    if 'gap_exclude_zero_demand' not in st.session_state:
        st.session_state['gap_exclude_zero_demand'] = True
    if 'gap_exclude_missing_dates' not in st.session_state:
        st.session_state['gap_exclude_missing_dates'] = True
    
    period_type = col1.selectbox(
        "Group By Period", 
        PERIOD_TYPES, 
        index=PERIOD_TYPES.index(st.session_state['gap_period_type']),
        key="gap_period_type"
    )
    
    show_shortage_only = col2.checkbox(
        "🔎 Show only shortages", 
        value=st.session_state['gap_shortage_checkbox'],
        key="gap_shortage_checkbox"
    )
    
    exclude_zero_demand = col3.checkbox(
        "🚫 Exclude zero demand", 
        value=st.session_state['gap_exclude_zero_demand'],
        key="gap_exclude_zero_demand",
        help="Hide products that have supply but no demand"
    )
    
    exclude_missing_dates = col4.checkbox(
        "📅 Exclude missing dates",
        value=st.session_state['gap_exclude_missing_dates'],
        key="gap_exclude_missing_dates",
        help="Exclude records with missing ETD or reference dates"
    )
    
    return {
        "period_type": period_type,
        "show_shortage_only": show_shortage_only,
        "exclude_zero_demand": exclude_zero_demand,
        "exclude_missing_dates": exclude_missing_dates
    }



# === Display Functions ===

def show_gap_summary(gap_df, display_options):
    """Show GAP analysis summary with past period warning"""
    st.markdown("### 📊 GAP Analysis Summary")
    
    # Check for past periods in the data
    today = pd.Timestamp.now().normalize()
    past_periods_count = 0
    
    # Count unique periods that are in the past
    for period in gap_df['period'].unique():
        if pd.notna(period) and is_past_period(str(period), display_options.get("period_type", "Weekly")):
            # Count records with this past period
            past_periods_count += len(gap_df[gap_df['period'] == period])
    
    if past_periods_count > 0:
        st.warning(f"🔴 Found {past_periods_count} records with past ETD")
    
    # Calculate metrics on ORIGINAL gap_df (before display filters)
    total_products = gap_df["pt_code"].nunique()
    shortage_products = len(gap_df[gap_df["gap_quantity"] < 0]["pt_code"].unique())
    total_shortage = gap_df["gap_quantity"].where(gap_df["gap_quantity"] < 0, 0).abs().sum()
    
    # Calculate average fulfillment rate correctly
    fulfillment_rates = gap_df[gap_df["total_demand_qty"] > 0]["fulfillment_rate_percent"].copy()
    fulfillment_rates = fulfillment_rates.clip(upper=100)
    avg_fulfillment = fulfillment_rates.mean() if len(fulfillment_rates) > 0 else 0
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Products", f"{total_products:,}")
    
    with col2:
        st.metric(
            "Products with Shortage", 
            f"{shortage_products:,}",
            delta=f"{shortage_products/total_products*100:.1f}%" if total_products > 0 else "0%"
        )
    
    with col3:
        st.metric("Total Shortage Qty", format_number(total_shortage))
    
    with col4:
        st.metric("Avg Fulfillment Rate", format_percentage(avg_fulfillment))
    
    # Show info about filtered view
    if display_options["show_shortage_only"] or display_options["exclude_zero_demand"]:
        # Count items after applying filters
        filtered_df = gap_df.copy()
        
        # Debug: Show counts before filtering
        if st.session_state.get('debug_mode', False):
            st.write(f"Debug - Total rows before filtering: {len(filtered_df)}")
            st.write(f"Debug - Rows with zero demand: {len(filtered_df[filtered_df['total_demand_qty'] == 0])}")
            st.write(f"Debug - Rows with positive demand: {len(filtered_df[filtered_df['total_demand_qty'] > 0])}")
        
        if display_options["show_shortage_only"]:
            filtered_df = filtered_df[filtered_df["gap_quantity"] < 0]
        if display_options["exclude_zero_demand"]:
            filtered_df = filtered_df[filtered_df["total_demand_qty"] > 0]
        
        st.info(f"🔍 Showing {len(filtered_df)} items out of {len(gap_df)} total items (filters applied)")


def show_allocation_impact_summary(df_demand_enhanced):
    """Show how allocation affects the GAP analysis"""
    
    if 'allocation_status' in df_demand_enhanced.columns:
        st.markdown("#### 📦 Allocation Impact on Demand")
        
        total_demand_original = df_demand_enhanced['demand_quantity'].sum()
        total_allocated = df_demand_enhanced['total_allocated'].sum()
        total_delivered = df_demand_enhanced['total_delivered'].sum()
        total_unallocated = df_demand_enhanced['unallocated_demand'].sum()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Original Demand", format_number(total_demand_original))
        
        with col2:
            st.metric("Already Allocated", format_number(total_allocated))
            
        with col3:
            st.metric("Already Delivered", format_number(total_delivered))
            
        with col4:
            st.metric("Net Unallocated", format_number(total_unallocated))

def show_gap_detail_table(gap_df, display_options):
    """Show detailed GAP analysis table"""
    st.markdown("### 📄 GAP Details by Product & Period")
    
    # Apply display filters
    display_df = gap_df.copy()
    
    if display_options["show_shortage_only"]:
        display_df = display_df[display_df["gap_quantity"] < 0]
    
    if display_options["exclude_zero_demand"]:
        # Filter out rows where total_demand_qty is 0
        display_df = display_df[display_df["total_demand_qty"] > 0]
    
    if display_df.empty:
        st.info("No data to display with current filter settings.")
        return
    
    # Format display - work on filtered data
    display_df_formatted = display_df.copy()
    
    # Format numeric columns
    numeric_cols = [
        "begin_inventory", "supply_in_period", "total_available", 
        "total_demand_qty", "gap_quantity"
    ]
    
    for col in numeric_cols:
        display_df_formatted[col] = display_df_formatted[col].apply(lambda x: format_number(x))
    
    display_df_formatted["fulfillment_rate_percent"] = display_df_formatted["fulfillment_rate_percent"].apply(
        lambda x: format_percentage(x)
    )
    
    # Add a status column for color indication
    display_df_formatted['Status'] = display_df_formatted['fulfillment_status']
    
    st.dataframe(display_df_formatted, use_container_width=True, height=400)


def show_gap_pivot_view(gap_df, display_options):
    """Show GAP pivot view with past period indicators and styling options"""
    st.markdown("### 📊 Pivot View - GAP by Period")
    
    # Apply filters for display
    display_df = gap_df.copy()
    
    if display_options.get("show_shortage_only", False):
        display_df = display_df[display_df["gap_quantity"] < 0]
    
    if display_options.get("exclude_zero_demand", False):
        display_df = display_df[display_df["total_demand_qty"] > 0]
    
    if display_df.empty:
        st.info("No data to display in pivot view with current filters.")
        return
    
    # Style selection
    style_mode = st.radio(
        "🎨 Styling Mode",
        options=STYLE_MODES,
        horizontal=True,
        key="gap_style_mode"
    )
    
    # Create pivot - aggregate by product across all periods
    pivot_gap = display_df.pivot_table(
        index=["product_name", "pt_code"], 
        columns="period", 
        values="gap_quantity",
        aggfunc="sum",
        fill_value=0
    ).reset_index()
    
    # Sort columns
    pivot_gap = sort_period_columns(pivot_gap, display_options["period_type"], ["product_name", "pt_code"])
    
    # Create display version with past period indicators
    display_pivot = pivot_gap.copy()
    
    # Rename columns with indicators for past periods
    renamed_columns = {}
    for col in pivot_gap.columns:
        if col not in ["product_name", "pt_code"]:
            if is_past_period(str(col), display_options["period_type"]):
                renamed_columns[col] = f"🔴 {col}"
    
    if renamed_columns:
        display_pivot = display_pivot.rename(columns=renamed_columns)
        st.info("🔴 = Past period (already occurred)")
    
    # Apply styling based on mode
    if style_mode == "🔴 Highlight Shortage":
        # Create a styled dataframe with custom formatting
        def style_negative(val):
            try:
                # Remove any formatting first
                val_str = str(val).replace(',', '')
                if float(val_str) < 0:
                    return 'background-color: #ffcccc; color: red; font-weight: bold;'
            except:
                pass
            return ''
        
        # Format values first
        for col in display_pivot.columns[2:]:
            display_pivot[col] = display_pivot[col].apply(lambda x: format_number(x))
        
        styled_df = display_pivot.style.applymap(
            style_negative,
            subset=display_pivot.columns[2:]
        )
        st.dataframe(styled_df, use_container_width=True)
    
    elif style_mode == "🌈 Heatmap":
        # For heatmap, we need numeric values, so use original pivot_gap
        # But still apply the renamed columns
        heatmap_pivot = pivot_gap.copy()
        if renamed_columns:
            heatmap_pivot = heatmap_pivot.rename(columns=renamed_columns)
        
        styled_df = heatmap_pivot.style.background_gradient(
            cmap='RdYlGn', 
            subset=heatmap_pivot.columns[2:], 
            axis=1
        ).format("{:,.0f}", subset=heatmap_pivot.columns[2:])
        st.dataframe(styled_df, use_container_width=True)
    
    else:
        # Format for display
        for col in display_pivot.columns[2:]:
            display_pivot[col] = display_pivot[col].apply(lambda x: format_number(x))
        st.dataframe(display_pivot, use_container_width=True)
    



# === Export Functions ===
def show_export_section(gap_df):
    """Show export options"""
    st.markdown("### 📤 Export Options")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Export detailed GAP
        excel_data = convert_df_to_excel(gap_df, "GAP Analysis")
        st.download_button(
            "📊 Export GAP Details",
            data=excel_data,
            file_name=f"gap_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    
    with col2:
        # Export shortage summary
        shortage_df = gap_df[gap_df["gap_quantity"] < 0]
        if not shortage_df.empty:
            shortage_summary = shortage_df.groupby(['pt_code', 'product_name']).agg({
                'gap_quantity': 'sum',
                'total_demand_qty': 'sum',
                'total_available': 'sum'
            }).reset_index()
            
            excel_data = convert_df_to_excel(shortage_summary, "Shortage Summary")
            st.download_button(
                "🚨 Export Shortage Summary",
                data=excel_data,
                file_name=f"shortage_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
    
    with col3:
        # Export multiple sheets
        if st.button("📑 Export Complete Report"):
            # Prepare multiple dataframes
            sheets_dict = {
                "GAP Details": gap_df,
                "Shortage Summary": shortage_df.groupby(['pt_code', 'product_name']).agg({
                    'gap_quantity': 'sum',
                    'total_demand_qty': 'sum'
                }).reset_index() if not shortage_df.empty else pd.DataFrame(),
                "Product Summary": gap_df.groupby(['pt_code', 'product_name']).agg({
                    'total_demand_qty': 'sum',
                    'total_available': 'sum',
                    'gap_quantity': 'sum'
                }).reset_index()
            }
            
            excel_data = export_multiple_sheets(sheets_dict)
            st.download_button(
                "Download Complete Report",
                data=excel_data,
                file_name=f"gap_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

# === Action Buttons Section ===
def show_action_buttons(gap_df):
    """Show action buttons based on GAP analysis results"""
    st.markdown("---")
    st.header("🎯 Next Actions")
    
    shortage_exists = not gap_df[gap_df['gap_quantity'] < 0].empty
    surplus_exists = not gap_df[gap_df['gap_quantity'] > 0].empty
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if shortage_exists:
            st.markdown("### 🚨 Shortage Detected")
            shortage_count = len(gap_df[gap_df['gap_quantity'] < 0]['pt_code'].unique())
            st.info(f"Found {shortage_count} products with shortage")
            
            if st.button("🧩 Create Allocation Plan", type="primary", use_container_width=True):
                st.switch_page("pages/4_🧩_Allocation_Plan.py")
        else:
            st.success("✅ No shortage detected!")
    
    with col2:
        if shortage_exists:
            st.markdown("### 📦 Replenishment Needed")
            total_shortage = gap_df[gap_df['gap_quantity'] < 0]['gap_quantity'].abs().sum()
            st.info(f"Total shortage: {format_number(total_shortage)} units")
            
            if st.button("📌 Generate PO Suggestions", type="secondary", use_container_width=True):
                st.switch_page("pages/5_📌_PO_Suggestions.py")
    
    with col3:
        if surplus_exists:
            st.markdown("### 📈 Surplus Available")
            surplus_count = len(gap_df[gap_df['gap_quantity'] > 0]['pt_code'].unique())
            st.info(f"Found {surplus_count} products with surplus")
            
            if st.button("🔄 Reallocation Options", use_container_width=True):
                save_to_session_state('show_reallocation', True)
                st.switch_page("pages/5_📌_PO_Suggestions.py")

# Thay thế phần Main Page Logic với code sau:

# === Main Page Logic ===

# Initialize session state
if 'gap_analysis_ran' not in st.session_state:
    st.session_state['gap_analysis_ran'] = False
if 'gap_analysis_data' not in st.session_state:
    st.session_state['gap_analysis_data'] = None

# Debug mode toggle
col1, col2 = st.columns([6, 1])
with col2:
    st.session_state['debug_mode'] = st.checkbox("Debug Mode", value=False)

# Check if we have saved analysis from session
if st.session_state.get('gap_analysis_result') is not None and not st.session_state['gap_analysis_ran']:
    st.info(f"📅 Using previous analysis from: {st.session_state.get('last_analysis_time', 'Unknown')}")
    if st.button("🔄 Run New Analysis"):
        # Clear previous results
        for key in ['gap_analysis_result', 'demand_filtered', 'supply_filtered']:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()

# Pre-load data on first run to populate filters
if 'initial_data_loaded' not in st.session_state:
    with st.spinner("Initializing data..."):
        # Load with default sources - UPDATED
        default_demand_sources = ["OC", "Forecast"]
        default_supply_sources = ["Inventory", "Pending CAN", "Pending PO", "Pending WH Transfer"]  # NEW
        
        # Load demand data
        df_demand_temp = load_and_prepare_demand_data(
            default_demand_sources,
            False  # include_converted = False
        )
        
        # Load supply data
        df_supply_temp = load_and_prepare_supply_data(
            default_supply_sources, 
            True  # exclude_expired = True
        )
        
        if not df_demand_temp.empty and not df_supply_temp.empty:
            # Store in session state
            st.session_state['temp_demand_data'] = df_demand_temp
            st.session_state['temp_supply_data'] = df_supply_temp
            # Update filter options
            update_filter_options(df_demand_temp, df_supply_temp)
        
        st.session_state['initial_data_loaded'] = True
        st.session_state['last_source_key'] = "OC-Forecast_Inventory-Pending CAN-Pending PO-Pending WH Transfer"  # UPDATED

# Now show source selection with populated filters
selected_sources = select_gap_sources()

# Display options - ALWAYS SHOW but with safe handling
display_options = get_gap_display_options()

# Check if sources changed and reload if needed
source_key = f"{'-'.join(selected_sources['demand'])}_{'-'.join(selected_sources['supply'])}"

if st.session_state.get('last_source_key') != source_key:
    st.session_state['last_source_key'] = source_key
    # Clear cached data when sources change
    if 'temp_demand_data' in st.session_state:
        del st.session_state['temp_demand_data']
    if 'temp_supply_data' in st.session_state:
        del st.session_state['temp_supply_data']
    
    # Reload data with new sources
    if selected_sources["demand"] and selected_sources["supply"]:
        with st.spinner("Reloading data..."):
            # Load demand data
            df_demand_temp = load_and_prepare_demand_data(
                selected_sources["demand"],
                selected_sources["include_converted"]
            )
            
            # Load supply data
            df_supply_temp = load_and_prepare_supply_data(
                selected_sources["supply"], 
                selected_sources["exclude_expired"]
            )
            
            if not df_demand_temp.empty and not df_supply_temp.empty:
                # Store in session state
                st.session_state['temp_demand_data'] = df_demand_temp
                st.session_state['temp_supply_data'] = df_supply_temp
                # Update filter options
                update_filter_options(df_demand_temp, df_supply_temp)

# Get data from session state for filters
df_for_filters_demand = st.session_state.get('temp_demand_data', pd.DataFrame())
df_for_filters_supply = st.session_state.get('temp_supply_data', pd.DataFrame())

# Filters setup with loaded data
filters = apply_gap_filters(df_for_filters_demand, df_for_filters_supply)

# Load data button
if st.button("🚀 Run GAP Analysis", type="primary", use_container_width=True):
    
    if not selected_sources["demand"] or not selected_sources["supply"]:
        st.error("Please select at least one demand source and one supply source.")
    else:
        # Use cached data if available
        if 'temp_demand_data' in st.session_state and 'temp_supply_data' in st.session_state:
            df_demand_all = st.session_state['temp_demand_data']
            df_supply_all = st.session_state['temp_supply_data']
        else:
            # Load demand data
            with st.spinner("Loading demand data..."):
                df_demand_all = load_and_prepare_demand_data(
                    selected_sources["demand"],
                    selected_sources["include_converted"]
                )
            
            # Load supply data
            with st.spinner("Loading supply data..."):
                df_supply_all = load_and_prepare_supply_data(
                    selected_sources["supply"], 
                    selected_sources["exclude_expired"]
                )
        
        # Adjust supply for allocations
        with st.spinner("Adjusting supply for allocations..."):
            df_supply_adjusted = adjust_supply_for_allocations(df_supply_all.copy())
        
        # Apply filters
        df_demand_filtered, df_supply_filtered = apply_filters_to_data(
            df_demand_all, df_supply_adjusted, filters, selected_sources.get("selected_customers", [])
        )
        
        # Store filtered data
        st.session_state['gap_analysis_data'] = {
            'demand': df_demand_filtered,
            'supply': df_supply_filtered
        }
        st.session_state['gap_analysis_ran'] = True
        st.session_state['gap_display_options'] = display_options
        
        # Clear cached GAP results to force recalculation
        if 'gap_df_cached' in st.session_state:
            del st.session_state['gap_df_cached']
        if 'gap_period_type_cache' in st.session_state:
            del st.session_state['gap_period_type_cache']
        
        # Check data quality
        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            demand_missing = check_missing_dates(df_demand_filtered, "etd", show_warning=False)
            if demand_missing > 0:
                st.warning(f"⚠️ Demand: {demand_missing} records with missing ETD")
        
        with col2:
            supply_missing = check_missing_dates(df_supply_filtered, "date_ref", show_warning=False)
            if supply_missing > 0:
                st.warning(f"⚠️ Supply: {supply_missing} records with missing dates")

# Display results if analysis has been run
if st.session_state.get('gap_analysis_ran', False) and st.session_state.get('gap_analysis_data') is not None:
    
    # Get data from session state
    df_demand_filtered = st.session_state['gap_analysis_data']['demand']
    df_supply_filtered = st.session_state['gap_analysis_data']['supply']
    
    # Get display options from session state or use current
    stored_display_options = st.session_state.get('gap_display_options', display_options)
    
    # Apply date exclusion if requested
    if stored_display_options["exclude_missing_dates"]:
        df_demand_filtered_display = df_demand_filtered[df_demand_filtered["etd"].notna()]
        df_supply_filtered_display = df_supply_filtered[df_supply_filtered["date_ref"].notna()]
    else:
        df_demand_filtered_display = df_demand_filtered
        df_supply_filtered_display = df_supply_filtered
    
    # Show allocation impact
    df_demand_enhanced = enhance_demand_with_allocation_info(df_demand_filtered_display)
    show_allocation_impact_summary(df_demand_enhanced)
    
    # Check if we need to recalculate GAP based on period change
    current_period = display_options["period_type"]  # From current UI
    cached_period = st.session_state.get('gap_period_type_cache', None)
    
    if 'gap_df_cached' not in st.session_state or cached_period != current_period:
        # Calculate GAP
        with st.spinner("Calculating supply-demand gaps..."):
            gap_df = calculate_gap_with_carry_forward(
                df_demand_filtered_display, 
                df_supply_filtered_display, 
                current_period
            )
            # Cache the results
            st.session_state['gap_df_cached'] = gap_df
            st.session_state['gap_period_type_cache'] = current_period
    else:
        gap_df = st.session_state['gap_df_cached']
    
    # Save to session state for other pages
    save_to_session_state('gap_analysis_result', gap_df)
    save_to_session_state('demand_filtered', df_demand_filtered_display)
    save_to_session_state('supply_filtered', df_supply_filtered_display)
    save_to_session_state('last_gap_analysis', gap_df)
    save_to_session_state('last_analysis_time', datetime.now().strftime('%Y-%m-%d %H:%M'))
    
    if not gap_df.empty:
        # Display results with current display options (not stored ones)
        show_gap_summary(gap_df, display_options)
        show_gap_detail_table(gap_df, display_options)
        show_gap_pivot_view(gap_df, display_options)
        
        # Export section
        st.markdown("---")
        show_export_section(gap_df)
        
        # Action buttons
        show_action_buttons(gap_df)
    else:
        st.warning("No data available for the selected filters and sources.")

# Help section
with st.expander("ℹ️ Understanding GAP Analysis", expanded=False):
    st.markdown("""
    ### How GAP Analysis Works
    
    **Allocation Awareness:**
    - Supply is adjusted for allocated but undelivered quantities
    - Demand shows both original and unallocated amounts
    - GAP calculation uses net unallocated demand
    
    **Carry-Forward Logic:**
    - Excess inventory from one period is carried to the next
    - Only positive inventory is carried forward (no negative carry)
    - This reflects real warehouse operations
    
    **Key Metrics:**
    - **Begin Inventory**: Carried from previous period
    - **Supply in Period**: New arrivals in this period
    - **Total Available**: Begin + Supply
    - **GAP**: Available - Demand (negative = shortage)
    - **Fulfillment Rate**: % of demand that can be met
    
    **Data Sources:**
    - **Demand**: OC (confirmed orders) + Forecast (predictions)
    - **Supply**: 
      - Inventory (current stock)
      - Pending CAN (arrived but not stocked)
      - Pending PO (in transit)
      - Pending WH Transfer (between warehouses)
    - **Allocations**: Committed supply not yet delivered
    
    **Common Issues:**
    - **Missing ETD**: Demand without delivery dates
    - **Converted Forecasts**: May cause double counting with OC
    - **Expired Inventory**: Excluded by default
    - **Transfer Delays**: WH transfers estimated at 2 days
    
    **Next Steps:**
    1. **Shortage → Allocation Plan**: Distribute limited supply
    2. **Shortage → PO Suggestions**: Order more inventory
    3. **Surplus → Reallocation**: Move excess to other locations
    """)

# Footer
st.markdown("---")
st.caption(f"Analysis completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")