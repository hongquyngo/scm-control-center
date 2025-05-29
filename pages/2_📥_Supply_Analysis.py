import streamlit as st
import pandas as pd
from datetime import datetime
from utils.data_loader import (
    load_inventory_data,
    load_pending_can_data,
    load_pending_po_data,
    load_pending_wh_transfer_data  # New import
)
from utils.helpers import (
    convert_df_to_excel,
    convert_to_period,
    sort_period_columns,
    format_number,
    format_currency,
    check_missing_dates,
    save_to_session_state
)

# === Page Config ===
st.set_page_config(
    page_title="Supply Analysis - SCM",
    page_icon="📥",
    layout="wide"
)

# === Constants ===
SUPPLY_SOURCES = ["Inventory Only", "Pending CAN Only", "Pending PO Only", "Pending WH Transfer Only", "All"]
PERIOD_TYPES = ["Daily", "Weekly", "Monthly"]

# === Header with Navigation ===
col1, col2, col3 = st.columns([1, 4, 1])
with col1:
    if st.button("← Demand"):
        st.switch_page("pages/1_📤_Demand_Analysis.py")
with col2:
    st.title("📥 Inbound Supply Analysis")
with col3:
    if st.button("GAP Analysis →"):
        st.switch_page("pages/3_📊_GAP_Analysis.py")

# Dashboard button
if st.button("🏠 Dashboard", use_container_width=False):
    st.switch_page("main.py")

st.markdown("---")

# === Data Source Selection ===
def select_supply_source():
    """Allow user to choose supply data source"""
    col1, col2 = st.columns([3, 1])
    with col1:
        source = st.radio(
            "Select Supply Source:",
            SUPPLY_SOURCES,
            index=4,  # Default to "All" (now index 4)
            horizontal=True,
            key="supply_source_radio"
        )
    with col2:
        exclude_expired = st.checkbox(
            "📅 Exclude expired inventory", 
            value=True, 
            key="supply_expired_checkbox",
            help="Hide inventory items that have passed their expiry date"
        )
    return source, exclude_expired

# === Data Loading Functions ===
def load_and_prepare_supply_data(supply_source, exclude_expired=True):
    """Load and standardize supply data based on source selection"""
    today = pd.to_datetime("today").normalize()
    df_parts = []
    
    # Load Inventory
    if supply_source in ["Inventory Only", "All"]:
        with st.spinner("Loading inventory data..."):
            inv_df = load_inventory_data()
            if not inv_df.empty:
                inv_df = prepare_inventory_data(inv_df, today, exclude_expired)
                df_parts.append(inv_df)
    
    # Load Pending CAN
    if supply_source in ["Pending CAN Only", "All"]:
        with st.spinner("Loading pending CAN data..."):
            can_df = load_pending_can_data()
            if not can_df.empty:
                can_df = prepare_can_data(can_df)
                df_parts.append(can_df)
    
    # Load Pending PO
    if supply_source in ["Pending PO Only", "All"]:
        with st.spinner("Loading pending PO data..."):
            po_df = load_pending_po_data()
            if not po_df.empty:
                po_df = prepare_po_data(po_df)
                df_parts.append(po_df)
    
    # Load Pending WH Transfer
    if supply_source in ["Pending WH Transfer Only", "All"]:
        with st.spinner("Loading pending warehouse transfer data..."):
            wht_df = load_pending_wh_transfer_data()
            if not wht_df.empty:
                wht_df = prepare_wh_transfer_data(wht_df, exclude_expired)
                df_parts.append(wht_df)
    
    if not df_parts:
        return pd.DataFrame()
    
    # Standardize all parts
    standardized_parts = [standardize_supply_df(df) for df in df_parts]
    return pd.concat(standardized_parts, ignore_index=True)

def prepare_inventory_data(inv_df, today, exclude_expired):
    """Prepare inventory data"""
    inv_df = inv_df.copy()
    inv_df["source_type"] = "Inventory"
    inv_df["supply_number"] = inv_df["inventory_history_id"].astype(str)
    inv_df["date_ref"] = today
    inv_df["quantity"] = pd.to_numeric(inv_df["remaining_quantity"], errors="coerce").fillna(0)
    inv_df["value_in_usd"] = pd.to_numeric(inv_df["inventory_value_usd"], errors="coerce").fillna(0)
    inv_df["legal_entity"] = inv_df["owning_company_name"]
    inv_df["expiry_date"] = pd.to_datetime(inv_df["expiry_date"], errors="coerce")
    
    # Add days until expiry
    inv_df["days_until_expiry"] = (inv_df["expiry_date"] - today).dt.days
    
    if exclude_expired:
        inv_df = inv_df[(inv_df["expiry_date"].isna()) | (inv_df["expiry_date"] >= today)]
    
    return inv_df

def prepare_can_data(can_df):
    """Prepare CAN data"""
    can_df = can_df.copy()
    can_df["source_type"] = "Pending CAN"
    can_df["supply_number"] = can_df["arrival_note_number"].astype(str)
    can_df["date_ref"] = pd.to_datetime(can_df["arrival_date"], errors="coerce")
    can_df["quantity"] = pd.to_numeric(can_df["pending_quantity"], errors="coerce").fillna(0)
    can_df["value_in_usd"] = pd.to_numeric(can_df["pending_value_usd"], errors="coerce").fillna(0)
    can_df["legal_entity"] = can_df["consignee"]
    
    # Add days since arrival
    can_df["days_since_arrival"] = can_df.get("days_since_arrival", 0)
    
    return can_df

def prepare_po_data(po_df):
    """Prepare PO data"""
    po_df = po_df.copy()
    po_df["source_type"] = "Pending PO"
    po_df["supply_number"] = po_df["po_number"].astype(str)
    po_df["date_ref"] = pd.to_datetime(po_df["cargo_ready_date"], errors="coerce")
    po_df["quantity"] = pd.to_numeric(po_df["pending_standard_arrival_quantity"], errors="coerce").fillna(0)
    po_df["value_in_usd"] = pd.to_numeric(po_df["outstanding_arrival_amount_usd"], errors="coerce").fillna(0)
    po_df["legal_entity"] = po_df["legal_entity"]
    
    # Add vendor info
    po_df["vendor"] = po_df.get("vendor_name", "")
    
    return po_df

def prepare_wh_transfer_data(wht_df, exclude_expired):
    """Prepare warehouse transfer data"""
    today = pd.to_datetime("today").normalize()
    wht_df = wht_df.copy()
    
    wht_df["source_type"] = "Pending WH Transfer"
    wht_df["supply_number"] = wht_df["warehouse_transfer_line_id"].astype(str)
    wht_df["date_ref"] = pd.to_datetime(wht_df["transfer_date"], errors="coerce")
    wht_df["quantity"] = pd.to_numeric(wht_df["transfer_quantity"], errors="coerce").fillna(0)
    wht_df["value_in_usd"] = pd.to_numeric(wht_df["warehouse_transfer_value_usd"], errors="coerce").fillna(0)
    wht_df["legal_entity"] = wht_df["owning_company_name"]
    wht_df["expiry_date"] = pd.to_datetime(wht_df["expiry_date"], errors="coerce")
    
    # Add transfer route info
    wht_df["transfer_route"] = wht_df["from_warehouse"] + " → " + wht_df["to_warehouse"]
    
    # Add days since transfer started
    wht_df["days_in_transfer"] = (today - wht_df["date_ref"]).dt.days
    
    # Add days until expiry
    wht_df["days_until_expiry"] = (wht_df["expiry_date"] - today).dt.days
    
    if exclude_expired:
        wht_df = wht_df[(wht_df["expiry_date"].isna()) | (wht_df["expiry_date"] >= today)]
    
    return wht_df

def standardize_supply_df(df):
    """Standardize supply dataframe columns"""
    df = df.copy()
    
    # String columns
    string_cols = ["pt_code", "product_name", "brand"]
    for col in string_cols:
        df[col] = df.get(col, "").astype(str)
    
    df["standard_uom"] = df.get("standard_uom", "")
    df["package_size"] = df.get("package_size", "")
    
    # Select standard columns
    standard_cols = [
        "source_type", "supply_number", "pt_code", "product_name", "brand", 
        "package_size", "standard_uom", "legal_entity", "date_ref", 
        "quantity", "value_in_usd"
    ]
    
    # Add optional columns if they exist
    optional_cols = ["expiry_date", "days_until_expiry", "days_since_arrival", 
                     "vendor", "transfer_route", "days_in_transfer"]
    for col in optional_cols:
        if col in df.columns:
            standard_cols.append(col)
    
    return df[standard_cols]

# === Filtering Functions ===
def apply_supply_filters(df):
    """Apply filters to supply dataframe"""
    with st.expander("📎 Filters", expanded=True):
        # Row 1: Entity, Brand, Product
        col1, col2, col3 = st.columns(3)
        with col1:
            selected_entity = st.multiselect(
                "Legal Entity", 
                sorted(df["legal_entity"].dropna().unique()), 
                key="supply_legal_entity"
            )
        with col2:
            selected_brand = st.multiselect(
                "Brand", 
                sorted(df["brand"].dropna().unique()), 
                key="supply_brand"
            )
        with col3:
            selected_pt = st.multiselect(
                "PT Code", 
                sorted(df["pt_code"].dropna().unique()), 
                key="supply_pt_code"
            )
        
        # Row 2: Source type, Date range
        col4, col5, col6 = st.columns(3)
        with col4:
            selected_source_type = st.multiselect(
                "Source Type",
                df["source_type"].unique(),
                default=list(df["source_type"].unique()),
                key="supply_source_type_filter"
            )
        with col5:
            default_start = df["date_ref"].min().date() if pd.notnull(df["date_ref"].min()) else datetime.today().date()
            start_date = st.date_input("From Date (Reference)", default_start, key="supply_start_date")
        with col6:
            default_end = df["date_ref"].max().date() if pd.notnull(df["date_ref"].max()) else datetime.today().date()
            end_date = st.date_input("To Date (Reference)", default_end, key="supply_end_date")
        
        # Row 3: Additional filters based on source
        if "days_until_expiry" in df.columns:
            col7, col8, col9 = st.columns(3)
            with col7:
                expiry_warning_days = st.number_input(
                    "Show items expiring within (days)",
                    min_value=0,
                    max_value=365,
                    value=30,
                    key="expiry_warning_days"
                )
    
    # Apply filters
    filtered_df = df.copy()
    
    if selected_entity:
        filtered_df = filtered_df[filtered_df["legal_entity"].isin(selected_entity)]
    if selected_brand:
        filtered_df = filtered_df[filtered_df["brand"].isin(selected_brand)]
    if selected_pt:
        filtered_df = filtered_df[filtered_df["pt_code"].isin(selected_pt)]
    if selected_source_type:
        filtered_df = filtered_df[filtered_df["source_type"].isin(selected_source_type)]
    
    # Date filter
    start_ts = pd.to_datetime(start_date)
    end_ts = pd.to_datetime(end_date)
    filtered_df = filtered_df[
        (filtered_df["date_ref"] >= start_ts) & 
        (filtered_df["date_ref"] <= end_ts)
    ]
    
    # Return additional filter values if needed
    filter_params = {
        'expiry_warning_days': expiry_warning_days if 'expiry_warning_days' in locals() else None
    }
    
    return filtered_df, start_date, end_date, filter_params

# === Display Functions ===
def show_supply_summary(filtered_df, filter_params):
    """Show supply summary metrics"""
    st.markdown("### 📊 Supply Summary")
    
    # Calculate metrics by source type
    source_summary = filtered_df.groupby('source_type').agg({
        'quantity': 'sum',
        'value_in_usd': 'sum',
        'pt_code': 'nunique'
    }).reset_index()
    
    # Display overall metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        total_products = filtered_df["pt_code"].nunique()
        st.metric("Total Unique Products", f"{total_products:,}")
    with col2:
        total_quantity = filtered_df["quantity"].sum()
        st.metric("Total Quantity", format_number(total_quantity))
    with col3:
        total_value = filtered_df["value_in_usd"].sum()
        st.metric("Total Value", format_currency(total_value, "USD"))
    with col4:
        missing_dates = filtered_df["date_ref"].isna().sum()
        if missing_dates > 0:
            st.metric("⚠️ Missing Dates", f"{missing_dates} records", delta_color="inverse")
    
    # Source breakdown
    st.markdown("#### 📦 Supply by Source")
    
    # Create columns for each source type
    source_cols = st.columns(len(source_summary))
    for idx, (col, row) in enumerate(zip(source_cols, source_summary.itertuples())):
        with col:
            st.markdown(f"**{row.source_type}**")
            st.metric("Products", f"{row.pt_code:,}", label_visibility="collapsed")
            st.metric("Quantity", format_number(row.quantity), label_visibility="collapsed")
            st.metric("Value", format_currency(row.value_in_usd, "USD", 0), label_visibility="collapsed")
    
    # Special warnings
    show_supply_warnings(filtered_df, filter_params)

def show_supply_warnings(filtered_df, filter_params):
    """Show supply-specific warnings"""
    warning_shown = False
    
    # Expiry warning for inventory and WH transfer
    if "days_until_expiry" in filtered_df.columns and filter_params.get('expiry_warning_days'):
        expiry_warning_days = filter_params['expiry_warning_days']
        expiring_soon = filtered_df[
            (filtered_df["source_type"].isin(["Inventory", "Pending WH Transfer"])) & 
            (filtered_df["days_until_expiry"] <= expiry_warning_days) &
            (filtered_df["days_until_expiry"] >= 0)
        ]
        
        if not expiring_soon.empty:
            st.warning(f"⚠️ {len(expiring_soon)} items expiring within {expiry_warning_days} days!")
            warning_shown = True
    
    # Delayed CAN warning
    if "days_since_arrival" in filtered_df.columns:
        delayed_cans = filtered_df[
            (filtered_df["source_type"] == "Pending CAN") & 
            (filtered_df["days_since_arrival"] > 7)
        ]
        
        if not delayed_cans.empty:
            st.warning(f"⚠️ {len(delayed_cans)} CAN items pending stock-in for more than 7 days!")
            warning_shown = True
    
    # Long transfer warning
    if "days_in_transfer" in filtered_df.columns:
        long_transfers = filtered_df[
            (filtered_df["source_type"] == "Pending WH Transfer") & 
            (filtered_df["days_in_transfer"] > 3)
        ]
        
        if not long_transfers.empty:
            st.warning(f"⚠️ {len(long_transfers)} warehouse transfers taking more than 3 days!")
            warning_shown = True
    
    if not warning_shown:
        st.success("✅ No critical supply issues detected")

def show_supply_detail_table(filtered_df):
    """Show detailed supply table"""
    st.markdown("### 🔍 Supply Details")
    
    # Tab view for different source types
    source_types = filtered_df["source_type"].unique()
    if len(source_types) > 1:
        tabs = st.tabs(list(source_types) + ["All"])
        
        for idx, source in enumerate(source_types):
            with tabs[idx]:
                source_df = filtered_df[filtered_df["source_type"] == source]
                display_source_table(source_df, source)
        
        # All tab
        with tabs[-1]:
            display_source_table(filtered_df, "All")
    else:
        display_source_table(filtered_df, source_types[0] if len(source_types) > 0 else "All")

def display_source_table(df, source_type):
    """Display table for specific source type"""
    if df.empty:
        st.info(f"No {source_type} data available")
        return
    
    # Prepare display columns based on source type
    base_columns = [
        "pt_code", "product_name", "brand", "package_size", "standard_uom",
        "date_ref", "quantity", "value_in_usd", "supply_number", "legal_entity"
    ]
    
    # Add source-specific columns
    if source_type == "Inventory" and "days_until_expiry" in df.columns:
        display_columns = base_columns + ["expiry_date", "days_until_expiry"]
    elif source_type == "Pending CAN" and "days_since_arrival" in df.columns:
        display_columns = base_columns + ["days_since_arrival"]
    elif source_type == "Pending PO" and "vendor" in df.columns:
        display_columns = base_columns + ["vendor"]
    elif source_type == "Pending WH Transfer":
        display_columns = base_columns + ["transfer_route", "days_in_transfer"]
        if "expiry_date" in df.columns:
            display_columns += ["expiry_date", "days_until_expiry"]
    elif source_type == "All":
        display_columns = base_columns + ["source_type"]
        # Add all optional columns if they exist
        for col in ["expiry_date", "days_until_expiry", "days_since_arrival", 
                   "vendor", "transfer_route", "days_in_transfer"]:
            if col in df.columns and col not in display_columns:
                display_columns.append(col)
    else:
        display_columns = base_columns
    
    # Filter columns that actually exist
    display_columns = [col for col in display_columns if col in df.columns]
    
    display_df = df[display_columns].copy()
    
    # Sort by date ref
    display_df = display_df.sort_values("date_ref", ascending=True)
    
    # Format columns
    display_df = format_supply_display_df(display_df)
    
    # Apply conditional formatting
    if "days_until_expiry" in display_df.columns:
        styled_df = display_df.style.apply(highlight_expiry_rows, axis=1)
        st.dataframe(styled_df, use_container_width=True)
    else:
        st.dataframe(display_df, use_container_width=True)

def format_supply_display_df(df):
    """Format dataframe columns for display"""
    df = df.copy()
    
    # Format quantity and value
    df["quantity"] = df["quantity"].apply(lambda x: format_number(x))
    df["value_in_usd"] = df["value_in_usd"].apply(lambda x: format_currency(x, "USD"))
    
    # Format dates
    df["date_ref"] = df["date_ref"].apply(
        lambda x: "❌ Missing" if pd.isna(x) else x.strftime("%Y-%m-%d")
    )
    
    if "expiry_date" in df.columns:
        df["expiry_date"] = df["expiry_date"].apply(
            lambda x: "" if pd.isna(x) else x.strftime("%Y-%m-%d")
        )
    
    # Format days columns
    if "days_until_expiry" in df.columns:
        df["days_until_expiry"] = df["days_until_expiry"].apply(
            lambda x: f"{int(x)} days" if pd.notna(x) else ""
        )
    
    if "days_since_arrival" in df.columns:
        df["days_since_arrival"] = df["days_since_arrival"].apply(
            lambda x: f"{int(x)} days" if pd.notna(x) else ""
        )
        
    if "days_in_transfer" in df.columns:
        df["days_in_transfer"] = df["days_in_transfer"].apply(
            lambda x: f"{int(x)} days" if pd.notna(x) else ""
        )
    
    return df

def highlight_expiry_rows(row):
    """Highlight rows based on expiry status"""
    if "days_until_expiry" in row:
        days_str = row["days_until_expiry"]
        if days_str and "days" in days_str:
            days = int(days_str.split()[0])
            if days <= 7:
                return ["background-color: #ffcccc"] * len(row)  # Red for very urgent
            elif days <= 30:
                return ["background-color: #ffe6cc"] * len(row)  # Orange for urgent
    return [""] * len(row)

def show_supply_grouped_view(filtered_df, start_date, end_date):
    """Show grouped supply by period"""
    st.markdown("### 📊 Grouped Supply by Product")
    st.markdown(f"📅 Period: **{start_date}** to **{end_date}**")
    
    # Controls
    col1, col2, col3 = st.columns(3)
    with col1:
        period = st.selectbox("Group by Period", PERIOD_TYPES, index=1)
    with col2:
        show_only_nonzero = st.checkbox("Show only products with quantity > 0", value=True, key="supply_show_nonzero")
    with col3:
        group_by_source = st.checkbox("Separate by source type", value=False, key="supply_group_by_source")
    
    # Filter out missing dates for grouping
    df_summary = filtered_df[filtered_df["date_ref"].notna()].copy()
    
    if df_summary.empty:
        st.info("No data with valid dates for grouping")
        return
    
    # Create period column
    df_summary["period"] = convert_to_period(df_summary["date_ref"], period)
    
    if group_by_source:
        # Create tabs for each source
        source_types = df_summary["source_type"].unique()
        tabs = st.tabs(list(source_types))
        
        for idx, source in enumerate(source_types):
            with tabs[idx]:
                source_df = df_summary[df_summary["source_type"] == source]
                display_supply_pivot(source_df, period, show_only_nonzero, source)
    else:
        # Combined view
        display_supply_pivot(df_summary, period, show_only_nonzero, "All Sources")

def display_supply_pivot(df_summary, period, show_only_nonzero, title):
    """Display supply pivot table"""
    # Create pivot table
    pivot_df = create_supply_pivot(df_summary, show_only_nonzero)
    pivot_df = sort_period_columns(pivot_df, period, ["product_name", "pt_code"])
    
    # Display pivot
    display_pivot = format_pivot_for_display(pivot_df)
    st.dataframe(display_pivot, use_container_width=True)
    
    # Show totals
    show_supply_totals(df_summary, period)
    
    # Export button
    excel_data = convert_df_to_excel(display_pivot, f"Supply_{title}")
    st.download_button(
        label=f"📤 Export {title} to Excel",
        data=excel_data,
        file_name=f"supply_{title.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        key=f"export_{title}"
    )

def create_supply_pivot(df_summary, show_only_nonzero):
    """Create pivot table for supply"""
    pivot_df = (
        df_summary
        .groupby(["product_name", "pt_code", "period"])
        .agg(total_quantity=("quantity", "sum"))
        .reset_index()
        .pivot(index=["product_name", "pt_code"], columns="period", values="total_quantity")
        .fillna(0)
        .reset_index()
    )
    
    if show_only_nonzero:
        pivot_df = pivot_df[pivot_df.iloc[:, 2:].sum(axis=1) > 0]
    
    return pivot_df

def format_pivot_for_display(pivot_df):
    """Format pivot table for display"""
    display_pivot = pivot_df.copy()
    for col in display_pivot.columns[2:]:  # Skip product_name and pt_code
        display_pivot[col] = display_pivot[col].apply(lambda x: format_number(x))
    return display_pivot

def show_supply_totals(df_summary, period):
    """Show period totals for supply"""
    # Calculate aggregates
    qty_by_period = df_summary.groupby("period")["quantity"].sum()
    val_by_period = df_summary.groupby("period")["value_in_usd"].sum()
    
    # Create summary DataFrame
    summary_data = {"Metric": ["🔢 TOTAL QUANTITY", "💰 TOTAL VALUE (USD)"]}
    
    for period_val in qty_by_period.index:
        summary_data[period_val] = [
            format_number(qty_by_period[period_val]),
            format_currency(val_by_period[period_val], "USD")
        ]
    
    display_final = pd.DataFrame(summary_data)
    display_final = sort_period_columns(display_final, period, ["Metric"])
    
    st.markdown("#### 🔢 Column Totals")
    st.dataframe(display_final, use_container_width=True)

# === Main Page Logic ===
st.subheader("📥 Inbound Supply Capability")

# Source selection and options
supply_source, exclude_expired = select_supply_source()

# Load and prepare data
df_all = load_and_prepare_supply_data(supply_source, exclude_expired)

if df_all.empty:
    st.info("No supply data available for the selected source.")
    st.stop()

# Apply filters
filtered_df, start_date, end_date, filter_params = apply_supply_filters(df_all)

# Save to session state for other pages
save_to_session_state('supply_analysis_data', filtered_df)
save_to_session_state('supply_analysis_filters', {
    'source': supply_source,
    'exclude_expired': exclude_expired,
    'start_date': start_date,
    'end_date': end_date
})

# Display sections
show_supply_summary(filtered_df, filter_params)
show_supply_detail_table(filtered_df)
show_supply_grouped_view(filtered_df, start_date, end_date)

# === Additional Page Features ===
st.markdown("---")

# Quick Actions
col1, col2, col3, col4 = st.columns(4)
with col1:
    if st.button("📊 Go to GAP Analysis", type="primary", use_container_width=True):
        st.switch_page("pages/3_📊_GAP_Analysis.py")

with col2:
    if st.button("📤 View Demand", use_container_width=True):
        st.switch_page("pages/1_📤_Demand_Analysis.py")

with col3:
    if st.button("📈 View Dashboard", use_container_width=True):
        st.switch_page("main.py")

with col4:
    if st.button("🔄 Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

# Help section
with st.expander("ℹ️ How to use Supply Analysis", expanded=False):
    st.markdown("""
    ### Understanding Supply Data
    
    **Data Sources:**
    - **Inventory**: Current stock on hand with expiry tracking
    - **Pending CAN**: Goods arrived but not yet stocked in
    - **Pending PO**: Purchase orders in transit
    - **Pending WH Transfer**: Goods being transferred between warehouses
    - **All**: Combined view of total supply pipeline
    
    **Key Metrics:**
    - **Date Reference**: 
      - Inventory: Today's date
      - CAN: Arrival date
      - PO: Cargo ready date
      - WH Transfer: Transfer start date
    - **Days Until Expiry**: For inventory and transfer items (red < 7 days, orange < 30 days)
    - **Days Since Arrival**: For CAN items pending stock-in
    - **Days in Transfer**: For warehouse transfers in progress
    
    **Important Features:**
    - **Exclude Expired**: Toggle to hide/show expired inventory
    - **Source Breakdown**: See quantity and value by each supply type
    - **Period Grouping**: Analyze supply availability over time
    
    **Common Actions:**
    1. Check inventory items nearing expiry
    2. Follow up on delayed CAN stock-ins
    3. Monitor incoming PO pipeline
    4. Track warehouse transfers in progress
    5. Compare supply timing with demand (go to GAP Analysis)
    
    **Tips:**
    - Use "Separate by source type" in grouped view for detailed analysis
    - Export period data for supply planning meetings
    - Set expiry warning days based on your product shelf life
    - Monitor long warehouse transfers (>3 days) for potential delays
    """)

# Footer
st.markdown("---")
st.caption(f"Last data refresh: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")