import streamlit as st
import pandas as pd
from io import BytesIO
from datetime import datetime

# === Constants ===
PERIOD_TYPES = ["Daily", "Weekly", "Monthly"]
STYLE_MODES = ["None", "🔴 Highlight Shortage", "🌈 Heatmap"]

# === Main Entry Point ===
def show_gap_analysis_tab(df_demand_all_sources, df_supply_all_sources):
    """Main entry point for GAP Analysis tab"""
    st.subheader("📊 Inventory GAP Analysis (Carry-Forward Logic)")
    
    # Source selection
    selected_sources = select_gap_sources(df_demand_all_sources, df_supply_all_sources)
    
    # Filter data by source
    df_demand, df_supply = filter_by_sources(
        df_demand_all_sources, 
        df_supply_all_sources, 
        selected_sources
    )
    
    # Apply filters
    df_demand_filtered = apply_gap_demand_filters(df_demand)
    df_supply_filtered = apply_gap_supply_filters(df_supply)
    
    # Check and show warnings
    show_data_quality_warnings(df_demand_filtered, df_supply_filtered, selected_sources)
    
    # Show conversion statistics
    if "Forecast" in selected_sources["demand"]:
        show_conversion_statistics(df_demand_filtered)
    
    # Period and display options
    display_options = get_gap_display_options()
    
    # Apply date exclusion if requested
    if display_options["exclude_missing_dates"]:
        df_demand_filtered = df_demand_filtered[df_demand_filtered["etd"].notna()]
        df_supply_filtered = df_supply_filtered[df_supply_filtered["date_ref"].notna()]
    
    # Calculate GAP
    gap_df = calculate_gap_with_carry_forward(
        df_demand_filtered, 
        df_supply_filtered, 
        display_options["period_type"]
    )
    
    # Apply display filters
    display_gap_df = apply_gap_display_filters(gap_df, display_options)
    
    # Display sections
    show_gap_summary(display_gap_df, gap_df, display_options)
    show_gap_detail_table(display_gap_df)
    show_gap_pivot_view(display_gap_df, display_options)

# === Source Selection ===
def select_gap_sources(df_demand_all_sources, df_supply_all_sources):
    """Select demand and supply sources for GAP analysis"""
    col1, col2 = st.columns(2)
    
    with col1:
        selected_demand_sources = st.multiselect(
            "Select Demand Sources", 
            df_demand_all_sources["source_type"].unique(), 
            default=list(df_demand_all_sources["source_type"].unique()), 
            key="gap_demand_sources"
        )
    
    with col2:
        include_converted_forecasts = st.checkbox(
            "Include Converted Forecasts in GAP Analysis", 
            value=False,
            help="Uncheck to exclude forecasts that have already been converted to OC (avoid double counting)"
        )
    
    selected_supply_sources = st.multiselect(
        "Select Supply Sources", 
        df_supply_all_sources["source_type"].unique(), 
        default=list(df_supply_all_sources["source_type"].unique()), 
        key="gap_supply_sources"
    )
    
    return {
        "demand": selected_demand_sources,
        "supply": selected_supply_sources,
        "include_converted": include_converted_forecasts
    }

def filter_by_sources(df_demand_all, df_supply_all, selected_sources):
    """Filter dataframes by selected sources"""
    # Filter demand by source type
    df_demand = df_demand_all[df_demand_all["source_type"].isin(selected_sources["demand"])]
    
    # Filter out converted forecasts if needed
    if not selected_sources["include_converted"] and 'is_converted_to_oc' in df_demand.columns:
        df_demand = df_demand[
            (df_demand["source_type"] != "Forecast") | 
            (df_demand["is_converted_to_oc"] == "No")
        ]
    
    # Filter supply by source type
    df_supply = df_supply_all[df_supply_all["source_type"].isin(selected_sources["supply"])]
    
    return df_demand, df_supply

# === Filtering ===
def apply_gap_demand_filters(df):
    """Apply demand-specific filters for GAP analysis"""
    with st.expander("📎 Demand Filters", expanded=False):
        col1, col2, col3 = st.columns(3)
        
        selected_entity = col1.multiselect(
            "Legal Entity", 
            df["legal_entity"].dropna().unique(), 
            key="gap_demand_entity"
        )
        
        selected_customer = col2.multiselect(
            "Customer", 
            df["customer"].dropna().unique(), 
            key="gap_demand_customer"
        )
        
        selected_pt = col3.multiselect(
            "PT Code", 
            df["pt_code"].dropna().unique(), 
            key="gap_demand_pt"
        )
        
        col4, col5 = st.columns(2)
        default_start = df["etd"].min().date() if pd.notnull(df["etd"].min()) else datetime.today().date()
        default_end = df["etd"].max().date() if pd.notnull(df["etd"].max()) else datetime.today().date()
        
        start_date = col4.date_input(
            "From Date (ETD)", 
            default_start, 
            key="gap_demand_start_date"
        )
        
        end_date = col5.date_input(
            "To Date (ETD)", 
            default_end, 
            key="gap_demand_end_date"
        )
        
        # Conversion status filter
        if 'is_converted_to_oc' in df.columns:
            conversion_options = df["is_converted_to_oc"].dropna().unique().tolist()
            selected_conversion = st.multiselect(
                "Conversion Status", 
                sorted(conversion_options), 
                key="gap_demand_conversion"
            )
    
    # Apply filters
    filtered_df = df.copy()
    
    if selected_entity:
        filtered_df = filtered_df[filtered_df["legal_entity"].isin(selected_entity)]
    if selected_customer:
        filtered_df = filtered_df[filtered_df["customer"].isin(selected_customer)]
    if selected_pt:
        filtered_df = filtered_df[filtered_df["pt_code"].isin(selected_pt)]
    
    if 'is_converted_to_oc' in df.columns and 'selected_conversion' in locals() and selected_conversion:
        filtered_df = filtered_df[filtered_df["is_converted_to_oc"].isin(selected_conversion)]
    
    # Date filter (include null ETD)
    start_ts = pd.to_datetime(start_date)
    end_ts = pd.to_datetime(end_date)
    filtered_df = filtered_df[
        filtered_df["etd"].isna() | 
        ((filtered_df["etd"] >= start_ts) & (filtered_df["etd"] <= end_ts))
    ]
    
    return filtered_df

def apply_gap_supply_filters(df):
    """Apply supply-specific filters for GAP analysis"""
    with st.expander("📎 Supply Filters", expanded=False):
        col1, col2, col3 = st.columns(3)
        
        selected_entity = col1.multiselect(
            "Legal Entity", 
            df["legal_entity"].dropna().unique(), 
            key="gap_supply_entity"
        )
        
        selected_brand = col2.multiselect(
            "Brand", 
            df["brand"].dropna().unique(), 
            key="gap_supply_brand"
        )
        
        selected_pt = col3.multiselect(
            "PT Code", 
            df["pt_code"].dropna().unique(), 
            key="gap_supply_pt"
        )
        
        col4, col5 = st.columns(2)
        default_start = df["date_ref"].min().date() if pd.notnull(df["date_ref"].min()) else datetime.today().date()
        default_end = df["date_ref"].max().date() if pd.notnull(df["date_ref"].max()) else datetime.today().date()
        
        start_date = col4.date_input(
            "From Date (Ref)", 
            default_start, 
            key="gap_supply_start_date"
        )
        
        end_date = col5.date_input(
            "To Date (Ref)", 
            default_end, 
            key="gap_supply_end_date"
        )
        
        exclude_expired = st.checkbox(
            "📅 Exclude expired inventory", 
            value=True, 
            key="gap_exclude_expired"
        )
    
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
    
    # Expired filter
    if exclude_expired and "expiry_date" in filtered_df.columns:
        today = pd.to_datetime("today").normalize()
        filtered_df = filtered_df[
            (filtered_df["expiry_date"].isna()) | 
            (filtered_df["expiry_date"] >= today)
        ]
    
    return filtered_df

# === Display Options ===
def get_gap_display_options():
    """Get display options for GAP analysis"""
    col1, col2, col3, col4 = st.columns(4)
    
    period_type = col1.selectbox(
        "Group By Period", 
        PERIOD_TYPES, 
        index=2,  # Default to Monthly
        key="gap_period_type"
    )
    
    show_shortage_only = col2.checkbox(
        "🔎 Show only shortages", 
        value=True, 
        key="gap_shortage_checkbox"
    )
    
    exclude_zero_demand = col3.checkbox(
        "🚫 Exclude zero demand", 
        value=True, 
        key="gap_exclude_zero_demand",
        help="Hide products that have supply but no demand"
    )
    
    exclude_missing_dates = col4.checkbox(
        "📅 Exclude missing dates",
        value=True,
        key="gap_exclude_missing_dates",
        help="Exclude records with missing ETD or reference dates from GAP calculation"
    )
    
    return {
        "period_type": period_type,
        "show_shortage_only": show_shortage_only,
        "exclude_zero_demand": exclude_zero_demand,
        "exclude_missing_dates": exclude_missing_dates
    }

def apply_gap_display_filters(gap_df, display_options):
    """Apply display filters to GAP dataframe"""
    display_df = gap_df.copy()
    
    # Filter 1: Show only shortages
    if display_options["show_shortage_only"]:
        display_df = display_df[display_df["gap_quantity"] < 0]
    
    # Filter 2: Exclude products with zero demand
    if display_options["exclude_zero_demand"]:
        products_with_demand = (
            gap_df[gap_df["total_demand_qty"] > 0][["pt_code", "product_name"]]
            .drop_duplicates()
        )
        
        if len(products_with_demand) > 0:
            display_df = display_df.merge(
                products_with_demand[["pt_code"]], 
                on="pt_code", 
                how="inner"
            )
    
    return display_df

# === Warnings and Statistics ===
def show_data_quality_warnings(df_demand, df_supply, selected_sources):
    """Show warnings about data quality issues"""
    demand_missing_dates = df_demand["etd"].isna().sum()
    supply_missing_dates = df_supply["date_ref"].isna().sum()
    
    if demand_missing_dates > 0 or supply_missing_dates > 0:
        col1, col2 = st.columns(2)
        with col1:
            if demand_missing_dates > 0:
                st.warning(f"⚠️ Demand: {demand_missing_dates} records with missing ETD")
        with col2:
            if supply_missing_dates > 0:
                st.warning(f"⚠️ Supply: {supply_missing_dates} records with missing dates")
    
    # Double-counting warning
    if "Forecast" in selected_sources["demand"] and "OC" in selected_sources["demand"]:
        if selected_sources["include_converted"] and 'is_converted_to_oc' in df_demand.columns:
            converted_count = len(df_demand[
                (df_demand["source_type"] == "Forecast") & 
                (df_demand["is_converted_to_oc"] == "Yes")
            ])
            
            if converted_count > 0:
                st.warning(
                    f"⚠️ **Double-counting Risk**: You have included {converted_count} converted "
                    f"forecast records along with OC records. This may result in double-counting "
                    f"of demand. Consider unchecking 'Include Converted Forecasts'."
                )

def show_conversion_statistics(df_demand_filtered):
    """Show forecast conversion statistics"""
    forecast_df = df_demand_filtered[df_demand_filtered["source_type"] == "Forecast"]
    
    if len(forecast_df) > 0 and 'is_converted_to_oc' in forecast_df.columns:
        converted = len(forecast_df[forecast_df["is_converted_to_oc"] == "Yes"])
        not_converted = len(forecast_df[forecast_df["is_converted_to_oc"] == "No"])
        
        st.markdown("#### 📈 Forecast Conversion Status")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Forecasts", len(forecast_df))
        with col2:
            st.metric("Converted to OC", converted)
        with col3:
            st.metric("Not Converted", not_converted)

# === Display Functions ===
def show_gap_summary(display_gap_df, gap_df, display_options):
    """Show GAP analysis summary"""
    st.markdown("### 📊 GAP Analysis Summary")
    
    # Calculate metrics
    total_unique_products = display_gap_df["pt_code"].nunique()
    total_shortage = display_gap_df["gap_quantity"].where(
        display_gap_df["gap_quantity"] < 0, 0
    ).abs().sum()
    
    # Display metrics
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Unique Products", f"{int(total_unique_products):,}")
    with col2:
        st.metric("Total Shortage Quantity", f"{total_shortage:,.0f}")
    
    # Info about excluded products
    if display_options["exclude_zero_demand"]:
        total_products_before = gap_df["pt_code"].nunique()
        excluded_products = total_products_before - total_unique_products
        if excluded_products > 0:
            st.info(f"ℹ️ Excluded {excluded_products} products with zero demand across all periods")

def show_gap_detail_table(display_gap_df):
    """Show detailed GAP analysis table"""
    st.markdown("### 📄 GAP Details by Product & Period")
    
    # Format display
    display_df = format_gap_display_df(display_gap_df)
    st.dataframe(display_df, use_container_width=True)

def show_gap_pivot_view(display_gap_df, display_options):
    """Show GAP pivot view with styling options"""
    st.markdown("### 📊 Pivot View by GAP Quantity")
    
    # Style selection
    style_mode = st.radio(
        "🎨 Styling Mode for Pivot Table",
        options=STYLE_MODES,
        horizontal=True,
        key="gap_style_mode"
    )
    
    # Create pivot
    pivot_gap = create_gap_pivot(display_gap_df)
    pivot_gap = sort_period_columns(pivot_gap, display_options["period_type"])
    
    # Store numeric values
    numeric_pivot = pivot_gap.copy()
    
    # Create display version
    display_pivot = format_pivot_for_display(pivot_gap)
    
    # Apply styling based on mode
    if style_mode == "🔴 Highlight Shortage":
        styled_df = apply_shortage_highlighting(numeric_pivot)
        st.dataframe(styled_df, use_container_width=True)
    
    elif style_mode == "🌈 Heatmap":
        styled_df = apply_heatmap_styling(numeric_pivot)
        st.dataframe(styled_df, use_container_width=True)
    
    else:
        st.dataframe(display_pivot, use_container_width=True)
    
    # Export buttons
    show_gap_export_buttons(display_pivot, display_gap_df)

# === GAP Calculation ===
def calculate_gap_with_carry_forward(df_demand, df_supply, period_type="Weekly"):
    """Calculate supply-demand gap with inventory carry-forward logic"""
    df_d = df_demand.copy()
    df_s = df_supply.copy()
    
    # Convert to periods
    df_d["period"] = convert_to_period(df_d["etd"], period_type)
    df_s["period"] = convert_to_period(df_s["date_ref"], period_type)
    
    # Group by product and period
    demand_grouped = group_demand_by_period(df_d)
    supply_grouped = group_supply_by_period(df_s)
    
    # Get all periods and products
    all_periods = get_all_periods(demand_grouped, supply_grouped, period_type)
    all_products = get_all_products(demand_grouped, supply_grouped)
    
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
    
    return pd.DataFrame(results)

def group_demand_by_period(df_demand):
    """Group demand by product and period"""
    return df_demand.groupby([
        "pt_code", "product_name", "package_size", "standard_uom", "period"
    ]).agg(
        total_demand_qty=("demand_quantity", "sum")
    ).reset_index()

def group_supply_by_period(df_supply):
    """Group supply by product and period"""
    return df_supply.groupby([
        "pt_code", "product_name", "package_size", "standard_uom", "period"
    ]).agg(
        total_supply_qty=("quantity", "sum")
    ).reset_index()

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
    """Get all unique products"""
    return pd.concat([
        demand_grouped[["pt_code", "product_name", "package_size", "standard_uom"]],
        supply_grouped[["pt_code", "product_name", "package_size", "standard_uom"]]
    ]).drop_duplicates()

def calculate_product_gap(product, all_periods, demand_grouped, supply_grouped):
    """Calculate gap for a single product across all periods"""
    pt_code = product["pt_code"]
    product_name = product["product_name"]
    package_size = product["package_size"]
    uom = product["standard_uom"]
    
    carry_forward_qty = 0
    results = []
    
    for period in all_periods:
        # Get demand for this period
        demand = demand_grouped[
            (demand_grouped["pt_code"] == pt_code) & 
            (demand_grouped["period"] == period)
        ]["total_demand_qty"].sum()
        
        # Get supply for this period
        supply = supply_grouped[
            (supply_grouped["pt_code"] == pt_code) & 
            (supply_grouped["period"] == period)
        ]["total_supply_qty"].sum()
        
        # Calculate gap
        total_available = carry_forward_qty + supply
        gap = total_available - demand
        fulfill_rate = (total_available / demand * 100) if demand > 0 else 100
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
        
        # Carry forward only positive inventory
        carry_forward_qty = max(0, gap)
    
    return results

# === Helper Functions ===
def convert_to_period(series, period_type):
    """Convert datetime series to period strings"""
    if period_type == "Daily":
        return series.dt.strftime("%Y-%m-%d")
    elif period_type == "Weekly":
        week_series = series.dt.isocalendar().week
        year_series = series.dt.isocalendar().year
        return "Week " + week_series.astype(str).str.zfill(2) + " - " + year_series.astype(str)
    elif period_type == "Monthly":
        return series.dt.to_period("M").dt.strftime("%b %Y")
    else:
        return series.astype(str)

def parse_week_period(period_str):
    """Parse week period for sorting"""
    try:
        parts = str(period_str).split(" - ")
        if len(parts) == 2:
            week_part = parts[0].replace("Week", "").strip()
            year_part = parts[1].strip()
            return (int(year_part), int(week_part))
    except:
        pass
    return (9999, 99)

def parse_month_period(period_str):
    """Parse month period for sorting"""
    try:
        return pd.to_datetime("01 " + str(period_str), format="%d %b %Y")
    except:
        return pd.Timestamp.max

def format_gap_display_df(gap_df):
    """Format GAP dataframe for display"""
    display_df = gap_df.copy()
    
    # Format numeric columns
    numeric_cols = [
        "begin_inventory", "supply_in_period", "total_available", 
        "total_demand_qty", "gap_quantity"
    ]
    
    for col in numeric_cols:
        display_df[col] = display_df[col].apply(lambda x: f"{x:,.0f}")
    
    display_df["fulfillment_rate_percent"] = display_df["fulfillment_rate_percent"].apply(
        lambda x: f"{x:,.1f}%"
    )
    
    return display_df

def create_gap_pivot(gap_df):
    """Create pivot table for GAP analysis"""
    return (
        gap_df.groupby(["product_name", "pt_code", "period"])
        .agg(gap_quantity=("gap_quantity", "sum"))
        .reset_index()
        .pivot(index=["product_name", "pt_code"], columns="period", values="gap_quantity")
        .fillna(0)
        .reset_index()
    )

def format_pivot_for_display(pivot_df):
    """Format pivot table for display"""
    display_pivot = pivot_df.copy()
    for col in display_pivot.columns[2:]:  # Skip product_name and pt_code
        display_pivot[col] = display_pivot[col].apply(lambda x: f"{x:,.0f}")
    return display_pivot

def apply_shortage_highlighting(numeric_pivot):
    """Apply red highlighting to shortage values"""
    return numeric_pivot.style.applymap(
        lambda x: "background-color: #fdd; color: red; font-weight: bold;" if x < 0 else "",
        subset=numeric_pivot.columns[2:]
    ).format("{:,.0f}", subset=numeric_pivot.columns[2:])

def apply_heatmap_styling(numeric_pivot):
    """Apply heatmap gradient styling"""
    return numeric_pivot.style.background_gradient(
        cmap='RdYlGn_r', 
        subset=numeric_pivot.columns[2:], 
        axis=1
    ).format(
        "{:,.0f}", 
        subset=numeric_pivot.columns[2:]
    )

def show_gap_export_buttons(display_pivot, display_gap_df):
    """Show export buttons for GAP analysis"""
    col1, col2 = st.columns(2)
    
    with col1:
        st.download_button(
            "📊 Export GAP Pivot to Excel", 
            convert_df_to_excel(display_pivot),
            "gap_analysis_pivot.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    
    with col2:
        st.download_button(
            "📤 Export GAP Details to Excel", 
            convert_df_to_excel(display_gap_df),
            "gap_analysis_details.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

def sort_period_columns(df, period_type):
    """Sort dataframe columns by period"""
    # Identify info columns
    info_cols = ["product_name", "pt_code"]
    
    # Get period columns
    period_cols = [col for col in df.columns if col not in info_cols]
    period_cols = [p for p in period_cols if pd.notna(p) and str(p).strip() != "" and str(p) != "nan"]
    
    # Sort based on period type
    if period_type == "Weekly":
        sorted_periods = sorted(period_cols, key=parse_week_period)
    elif period_type == "Monthly":
        sorted_periods = sorted(period_cols, key=parse_month_period)
    else:  # Daily
        sorted_periods = sorted(period_cols)
    
    return df[info_cols + sorted_periods]

def convert_df_to_excel(df):
    """Convert dataframe to Excel bytes"""
    output = BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="GAP Analysis")
        
        # Auto-adjust column widths
        worksheet = writer.sheets["GAP Analysis"]
        for i, col in enumerate(df.columns):
            max_len = max(
                df[col].astype(str).map(len).max(),
                len(str(col))
            ) + 2
            worksheet.set_column(i, i, max_len)
    
    return output.getvalue()