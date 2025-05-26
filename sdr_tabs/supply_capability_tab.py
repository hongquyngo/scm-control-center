import streamlit as st
import pandas as pd
from datetime import datetime
from io import BytesIO
from data_loader import (
    load_inventory_data,
    load_pending_can_data,
    load_pending_po_data
)

# === Constants ===
SUPPLY_SOURCES = ["Inventory Only", "Pending CAN Only", "Pending PO Only", "All"]
PERIOD_TYPES = ["Daily", "Weekly", "Monthly"]

# === Main Entry Point ===
def show_inbound_supply_tab():
    """Main entry point for Inbound Supply tab"""
    st.subheader("📥 Inbound Supply Capability")
    
    # Source selection and options
    supply_source = select_supply_source()
    exclude_expired = st.checkbox("📅 Exclude expired inventory", value=True, key="supply_expired_checkbox")
    
    # Load and prepare data
    df_all = load_and_prepare_supply_data(supply_source, exclude_expired)
    
    if df_all.empty:
        st.info("No supply data available.")
        return
    
    # Apply filters
    filtered_df, start_date, end_date = apply_supply_filters(df_all)
    
    # Display sections
    show_supply_summary(filtered_df)
    show_supply_detail_table(filtered_df)
    show_supply_grouped_view(filtered_df, start_date, end_date)

# === Data Source Selection ===
def select_supply_source():
    """Allow user to choose supply data source"""
    return st.radio(
        "Select Supply Source:",
        SUPPLY_SOURCES,
        index=3,  # Default to "All"
        horizontal=True,
        key="supply_source_radio"
    )

# === Data Loading ===
def load_and_prepare_supply_data(supply_source, exclude_expired=True):
    """Load and standardize supply data based on source selection"""
    today = pd.to_datetime("today").normalize()
    df_parts = []
    
    # Load Inventory
    if supply_source in ["Inventory Only", "All"]:
        inv_df = load_inventory_data()
        if not inv_df.empty:
            inv_df = prepare_inventory_data(inv_df, today, exclude_expired)
            df_parts.append(inv_df)
    
    # Load Pending CAN
    if supply_source in ["Pending CAN Only", "All"]:
        can_df = load_pending_can_data()
        if not can_df.empty:
            can_df = prepare_can_data(can_df)
            df_parts.append(can_df)
    
    # Load Pending PO
    if supply_source in ["Pending PO Only", "All"]:
        po_df = load_pending_po_data()
        if not po_df.empty:
            po_df = prepare_po_data(po_df)
            df_parts.append(po_df)
    
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
    return po_df

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
    return df[[
        "source_type", "supply_number", "pt_code", "product_name", "brand", 
        "package_size", "standard_uom", "legal_entity", "date_ref", 
        "quantity", "value_in_usd"
    ]]

# === Filtering ===
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
        
        # Row 2: Date range
        col4, col5 = st.columns(2)
        with col4:
            default_start = df["date_ref"].min().date() if pd.notnull(df["date_ref"].min()) else datetime.today().date()
            start_date = st.date_input("From Date (Reference)", default_start, key="supply_start_date")
        with col5:
            default_end = df["date_ref"].max().date() if pd.notnull(df["date_ref"].max()) else datetime.today().date()
            end_date = st.date_input("To Date (Reference)", default_end, key="supply_end_date")
    
    # Apply filters
    filtered_df = df.copy()
    
    if selected_entity:
        filtered_df = filtered_df[filtered_df["legal_entity"].isin(selected_entity)]
    if selected_brand:
        filtered_df = filtered_df[filtered_df["brand"].isin(selected_brand)]
    if selected_pt:
        filtered_df = filtered_df[filtered_df["pt_code"].isin(selected_pt)]
    
    # Date filter
    start_ts = pd.to_datetime(start_date)
    end_ts = pd.to_datetime(end_date)
    filtered_df = filtered_df[
        (filtered_df["date_ref"] >= start_ts) & 
        (filtered_df["date_ref"] <= end_ts)
    ]
    
    return filtered_df, start_date, end_date

# === Display Functions ===
def show_supply_summary(filtered_df):
    """Show supply summary metrics"""
    st.markdown("### 📊 Supply Summary")
    
    # Calculate metrics
    total_products = filtered_df["pt_code"].nunique()
    total_value = filtered_df["value_in_usd"].sum()
    missing_dates = filtered_df["date_ref"].isna().sum()
    
    # Display metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Unique Products", f"{total_products:,}")
    with col2:
        st.metric("Total Value (USD)", f"${total_value:,.2f}")
    with col3:
        if missing_dates > 0:
            st.metric("⚠️ Missing Dates", f"{missing_dates} records", delta_color="inverse")

def show_supply_detail_table(filtered_df):
    """Show detailed supply table with highlighting"""
    st.markdown("### 🔍 Supply Details")
    
    # Warning for missing dates
    missing_date_count = filtered_df["date_ref"].isna().sum()
    if missing_date_count > 0:
        st.warning(f"⚠️ Found {missing_date_count} records with missing reference dates")
    
    # Prepare display
    display_df = filtered_df.copy()
    
    # Sort with missing dates first
    display_df["date_is_null"] = display_df["date_ref"].isna()
    display_df = display_df.sort_values(["date_is_null", "date_ref"], ascending=[False, True])
    display_df = display_df.drop("date_is_null", axis=1)
    
    # Format columns
    display_df = format_supply_display_df(display_df)
    
    # Rename for clarity
    display_df.rename(columns={"quantity": "supply_quantity"}, inplace=True)
    
    # Select display columns
    display_columns = [
        "pt_code", "product_name", "brand", "package_size", "standard_uom",
        "date_ref", "supply_quantity", "value_in_usd", "source_type", 
        "supply_number", "legal_entity"
    ]
    
    # Apply styling
    styled_df = display_df[display_columns].style.apply(highlight_missing_dates, axis=1)
    st.dataframe(styled_df, use_container_width=True)

def show_supply_grouped_view(filtered_df, start_date, end_date):
    """Show grouped supply by period"""
    st.markdown("### 📦 Grouped Supply by Product")
    st.markdown(f"📅 Period: **{start_date}** to **{end_date}**")
    
    # Controls
    col1, col2 = st.columns(2)
    with col1:
        period = st.selectbox("Group by Period", PERIOD_TYPES, index=1)
    with col2:
        show_only_nonzero = st.checkbox("Show only products with quantity > 0", value=True, key="supply_show_nonzero")
    
    # Filter out missing dates for grouping
    df_summary = filtered_df[filtered_df["date_ref"].notna()].copy()
    
    if df_summary.empty:
        st.info("No data with valid dates for grouping")
        return
    
    # Create period column
    df_summary["period"] = create_period_column(df_summary["date_ref"], period)
    
    # Create pivot table
    pivot_df = create_supply_pivot(df_summary, show_only_nonzero)
    pivot_df = sort_period_columns(pivot_df, period)
    
    # Display pivot
    display_pivot = format_pivot_for_display(pivot_df)
    st.dataframe(display_pivot, use_container_width=True)
    
    # Show totals
    show_supply_totals(df_summary, period)
    
    # Export button
    excel_data = convert_df_to_excel(display_pivot)
    st.download_button(
        label="📤 Export Grouped Supply to Excel",
        data=excel_data,
        file_name="grouped_supply_by_period.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

# === Helper Functions ===
def format_supply_display_df(df):
    """Format dataframe columns for display"""
    df = df.copy()
    df["quantity"] = df["quantity"].apply(lambda x: f"{x:,.0f}")
    df["value_in_usd"] = df["value_in_usd"].apply(lambda x: f"${x:,.2f}")
    df["date_ref"] = df["date_ref"].apply(
        lambda x: "❌ Missing" if pd.isna(x) else x.strftime("%Y-%m-%d")
    )
    return df

def highlight_missing_dates(row):
    """Highlight rows with missing dates"""
    if row["date_ref"] == "❌ Missing":
        return ["background-color: #ffcccc"] * len(row)
    return [""] * len(row)

def create_period_column(date_series, period_type):
    """Create period column based on period type"""
    if period_type == "Daily":
        return date_series.dt.strftime("%Y-%m-%d")
    elif period_type == "Weekly":
        year = date_series.dt.isocalendar().year
        week = date_series.dt.isocalendar().week
        return "Week " + week.astype(str).str.zfill(2) + " - " + year.astype(str)
    else:  # Monthly
        return date_series.dt.to_period("M").dt.strftime("%b %Y")

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
        display_pivot[col] = display_pivot[col].apply(lambda x: f"{x:,.0f}")
    return display_pivot

def show_supply_totals(df_summary, period):
    """Show period totals for supply"""
    # Prepare data
    df_grouped = df_summary.copy()
    df_grouped["quantity"] = pd.to_numeric(df_grouped["quantity"], errors="coerce").fillna(0)
    df_grouped["value_in_usd"] = pd.to_numeric(df_grouped["value_in_usd"], errors="coerce").fillna(0)
    
    # Calculate aggregates
    qty_by_period = df_grouped.groupby("period")["quantity"].sum()
    val_by_period = df_grouped.groupby("period")["value_in_usd"].sum()
    
    # Create summary DataFrame
    summary_data = {"Metric": ["🔢 TOTAL QUANTITY", "💰 TOTAL VALUE (USD)"]}
    
    for period in qty_by_period.index:
        summary_data[period] = [
            f"{qty_by_period[period]:,.0f}",
            f"${val_by_period[period]:,.2f}"
        ]
    
    display_final = pd.DataFrame(summary_data)
    display_final = sort_period_columns(display_final, period)
    
    st.markdown("#### 🔢 Column Totals")
    st.dataframe(display_final, use_container_width=True)

# === Shared helper functions (same as demand_tab) ===
def sort_period_columns(df, period_type):
    """Sort dataframe columns by period"""
    # Identify info columns
    info_cols = ["Metric"] if "Metric" in df.columns else ["product_name", "pt_code"]
    
    # Get period columns
    period_cols = [col for col in df.columns if col not in info_cols]
    period_cols = [p for p in period_cols if pd.notna(p) and str(p).strip() != "" and str(p) != "nan"]
    
    # Sort based on period type
    if period_type == "Weekly":
        def parse_week_key(x):
            try:
                parts = str(x).split(" - ")
                if len(parts) == 2:
                    week = int(parts[0].replace("Week", "").strip())
                    year = int(parts[1].strip())
                    return (year, week)
            except:
                pass
            return (9999, 99)
        sorted_periods = sorted(period_cols, key=parse_week_key)
    
    elif period_type == "Monthly":
        def parse_month_key(x):
            try:
                return pd.to_datetime("01 " + str(x), format="%d %b %Y")
            except:
                return pd.Timestamp.max
        sorted_periods = sorted(period_cols, key=parse_month_key)
    
    else:  # Daily
        sorted_periods = sorted(period_cols)
    
    return df[info_cols + sorted_periods]


def convert_df_to_excel(df):
    """Convert dataframe to Excel bytes"""
    output = BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="Data")
        
        # Auto-adjust column widths
        worksheet = writer.sheets["Data"]
        for i, col in enumerate(df.columns):
            max_len = max(
                df[col].astype(str).map(len).max(),
                len(str(col))
            ) + 2
            worksheet.set_column(i, i, max_len)
    
    return output.getvalue()