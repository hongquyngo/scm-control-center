import streamlit as st
import pandas as pd
from io import BytesIO
from datetime import datetime


# =========================
# 📊 Main GAP Calculation with Carry-Forward Logic
# =========================
def calculate_gap_with_carry_forward(df_demand, df_supply, period_type="Weekly"):
    df_d = df_demand.copy()
    df_s = df_supply.copy()

    # Convert period
    df_d["period"] = convert_to_period(df_d["etd"], period_type)
    df_s["period"] = convert_to_period(df_s["date_ref"], period_type)

    # Group demand and supply
    demand_grouped = df_d.groupby(["pt_code", "product_name", "package_size", "standard_uom", "period"]).agg(
        total_demand_qty=("demand_quantity", "sum")
    ).reset_index()

    supply_grouped = df_s.groupby(["pt_code", "product_name", "package_size", "standard_uom", "period"]).agg(
        total_supply_qty=("quantity", "sum")
    ).reset_index()

    # Get all unique periods and sort them chronologically
    all_periods_raw = list(set(demand_grouped["period"]).union(set(supply_grouped["period"])))
    
    # Filter out invalid periods
    all_periods_raw = [p for p in all_periods_raw if pd.notna(p) and str(p).strip() != "" and str(p) != "nan"]
    
    # Sort periods properly
    if period_type == "Weekly":
        all_periods = sorted(all_periods_raw, key=lambda x: parse_week_period(x))
    elif period_type == "Monthly":
        all_periods = sorted(all_periods_raw, key=lambda x: parse_month_period(x))
    else:  # Daily
        all_periods = sorted(all_periods_raw)

    # Get all unique products
    all_keys = pd.concat([
        demand_grouped[["pt_code", "product_name", "package_size", "standard_uom"]],
        supply_grouped[["pt_code", "product_name", "package_size", "standard_uom"]]
    ]).drop_duplicates()

    results = []
    for _, row in all_keys.iterrows():
        pt_code = row["pt_code"]
        product_name = row["product_name"]
        package_size = row["package_size"]
        uom = row["standard_uom"]
        carry_forward_qty = 0

        for period in all_periods:
            demand = demand_grouped[
                (demand_grouped["pt_code"] == pt_code) & 
                (demand_grouped["period"] == period)
            ]["total_demand_qty"].sum()

            supply = supply_grouped[
                (supply_grouped["pt_code"] == pt_code) & 
                (supply_grouped["period"] == period)
            ]["total_supply_qty"].sum()

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

            carry_forward_qty = max(0, gap)  # only forward surplus

    return pd.DataFrame(results)


# =========================
# 🔧 Helper functions for period parsing
# =========================
def parse_week_period(period_str):
    """Parse 'Week 01 - 2025' format and return sortable tuple"""
    try:
        parts = str(period_str).split(" - ")
        if len(parts) == 2:
            week_part = parts[0].replace("Week", "").strip()
            year_part = parts[1].strip()
            week = int(week_part)
            year = int(year_part)
            return (year, week)
    except:
        pass
    return (9999, 99)  # Sort invalid periods at the end


def parse_month_period(period_str):
    """Parse 'Jan 2025' format and return sortable datetime"""
    try:
        return pd.to_datetime("01 " + str(period_str), format="%d %b %Y")
    except:
        return pd.Timestamp.max


# =========================
# 🔎 Demand & Supply Filters
# =========================
# === Trong function apply_demand_filters() ===
# Thêm filter cho conversion status

def apply_demand_filters(df):
    with st.expander("📎 Demand Filters", expanded=False):
        col1, col2, col3 = st.columns(3)
        selected_entity = col1.multiselect("Legal Entity", df["legal_entity"].dropna().unique(), key="gap_demand_entity")
        selected_customer = col2.multiselect("Customer", df["customer"].dropna().unique(), key="gap_demand_customer")
        selected_pt = col3.multiselect("PT Code", df["pt_code"].dropna().unique(), key="gap_demand_pt")

        col4, col5 = st.columns(2)
        default_start = df["etd"].min().date() if pd.notnull(df["etd"].min()) else datetime.today().date()
        default_end = df["etd"].max().date() if pd.notnull(df["etd"].max()) else datetime.today().date()
        start_date = col4.date_input("From Date (ETD)", default_start, key="gap_demand_start_date")
        end_date = col5.date_input("To Date (ETD)", default_end, key="gap_demand_end_date")

        # Add conversion status filter
        if 'is_converted_to_oc' in df.columns:
            conversion_options = df["is_converted_to_oc"].dropna().unique().tolist()
            selected_conversion = st.multiselect(
                "Conversion Status", 
                sorted(conversion_options), 
                key="gap_demand_conversion"
            )

    df = df.copy()
    if selected_entity:
        df = df[df["legal_entity"].isin(selected_entity)]
    if selected_customer:
        df = df[df["customer"].isin(selected_customer)]
    if selected_pt:
        df = df[df["pt_code"].isin(selected_pt)]
    
    # Apply conversion filter
    if 'is_converted_to_oc' in df.columns and 'selected_conversion' in locals() and selected_conversion:
        df = df[df["is_converted_to_oc"].isin(selected_conversion)]

    # Convert dates to pandas Timestamp for comparison
    start_ts = pd.to_datetime(start_date)
    end_ts = pd.to_datetime(end_date)
    
    # Include rows with null ETD to avoid filtering them out
    df = df[
        df["etd"].isna() | 
        ((df["etd"] >= start_ts) & (df["etd"] <= end_ts))
    ]

    return df

def apply_supply_filters(df):
    with st.expander("📎 Supply Filters", expanded=False):
        col1, col2, col3 = st.columns(3)
        selected_entity = col1.multiselect("Legal Entity", df["legal_entity"].dropna().unique(), key="gap_supply_entity")
        selected_brand = col2.multiselect("Brand", df["brand"].dropna().unique(), key="gap_supply_brand")
        selected_pt = col3.multiselect("PT Code", df["pt_code"].dropna().unique(), key="gap_supply_pt")

        col4, col5 = st.columns(2)
        default_start = df["date_ref"].min().date() if pd.notnull(df["date_ref"].min()) else datetime.today().date()
        default_end = df["date_ref"].max().date() if pd.notnull(df["date_ref"].max()) else datetime.today().date()
        start_date = col4.date_input("From Date (Ref)", default_start, key="gap_supply_start_date")
        end_date = col5.date_input("To Date (Ref)", default_end, key="gap_supply_end_date")

        exclude_expired = st.checkbox("📅 Exclude expired inventory", value=True, key="gap_exclude_expired")

    df = df.copy()
    if selected_entity:
        df = df[df["legal_entity"].isin(selected_entity)]
    if selected_brand:
        df = df[df["brand"].isin(selected_brand)]
    if selected_pt:
        df = df[df["pt_code"].isin(selected_pt)]
    
    # Convert dates for comparison
    start_ts = pd.to_datetime(start_date)
    end_ts = pd.to_datetime(end_date)
    df = df[(df["date_ref"] >= start_ts) & (df["date_ref"] <= end_ts)]
    
    if exclude_expired and "expiry_date" in df.columns:
        today = pd.to_datetime("today").normalize()
        df = df[(df["expiry_date"].isna()) | (df["expiry_date"] >= today)]
    
    return df


# =========================
# 📊 Streamlit Tab Entry
# =========================
def show_gap_analysis_tab(df_demand_all_sources, df_supply_all_sources):
    st.subheader("📊 Inventory GAP Analysis (Carry-Forward Logic)")

    col1, col2 = st.columns(2)
    selected_demand_sources = col1.multiselect(
        "Select Demand Sources", 
        df_demand_all_sources["source_type"].unique(), 
        default=list(df_demand_all_sources["source_type"].unique()), 
        key="gap_demand_sources"
    )
    
    # Option to include/exclude converted forecasts
    include_converted_forecasts = col2.checkbox(
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

    # STEP 1: Filter by source type
    df_demand = df_demand_all_sources[df_demand_all_sources["source_type"].isin(selected_demand_sources)]
    
    # STEP 2: Filter out converted forecasts if needed
    if not include_converted_forecasts and 'is_converted_to_oc' in df_demand.columns:
        # Keep all OC records and only non-converted forecasts
        df_demand = df_demand[
            (df_demand["source_type"] != "Forecast") | 
            (df_demand["is_converted_to_oc"] == "No")
        ]
    
    df_supply = df_supply_all_sources[df_supply_all_sources["source_type"].isin(selected_supply_sources)]

    # STEP 3: Apply additional filters (entity, customer, PT code, dates)
    df_demand_filtered = apply_demand_filters(df_demand)
    df_supply_filtered = apply_supply_filters(df_supply)
    
    # Check for missing dates
    demand_missing_dates = df_demand_filtered["etd"].isna().sum()
    supply_missing_dates = df_supply_filtered["date_ref"].isna().sum()
    
    if demand_missing_dates > 0 or supply_missing_dates > 0:
        col1, col2 = st.columns(2)
        with col1:
            if demand_missing_dates > 0:
                st.warning(f"⚠️ Demand: {demand_missing_dates} records with missing ETD")
        with col2:
            if supply_missing_dates > 0:
                st.warning(f"⚠️ Supply: {supply_missing_dates} records with missing dates")

    # ✅ Show conversion statistics based on FILTERED data
    if "Forecast" in selected_demand_sources and len(df_demand_filtered) > 0:
        forecast_df_filtered = df_demand_filtered[df_demand_filtered["source_type"] == "Forecast"]
        
        if len(forecast_df_filtered) > 0 and 'is_converted_to_oc' in forecast_df_filtered.columns:
            # Count from FILTERED data
            converted = len(forecast_df_filtered[forecast_df_filtered["is_converted_to_oc"] == "Yes"])
            not_converted = len(forecast_df_filtered[forecast_df_filtered["is_converted_to_oc"] == "No"])
            
            # Show warning only if relevant
            if include_converted_forecasts and converted > 0 and "OC" in selected_demand_sources:
                st.warning(
                    f"⚠️ **Double-counting Risk**: You have included {converted} converted "
                    f"forecast records along with OC records. This may result in double-counting "
                    f"of demand. Consider unchecking 'Include Converted Forecasts'."
                )
            
            # Show conversion statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Forecasts", len(forecast_df_filtered))
            with col2:
                st.metric("Converted to OC", converted)
            with col3:
                st.metric("Not Converted", not_converted)

    # Period selection and filter options
    col1, col2, col3, col4 = st.columns(4)
    period_type = col1.selectbox("Group By Period", ["Daily", "Weekly", "Monthly"], index=2, key="gap_period_type")
    show_shortage_only = col2.checkbox("🔎 Show only shortages", value=True, key="gap_shortage_checkbox")
    exclude_zero_demand = col3.checkbox(
        "🚫 Exclude zero demand", 
        value=True, 
        key="gap_exclude_zero_demand",
        help="Hide products that have supply but no demand"
    )
    # NEW: Add checkbox to exclude records with missing dates
    exclude_missing_dates = col4.checkbox(
        "📅 Exclude missing dates",
        value=True,
        key="gap_exclude_missing_dates",
        help="Exclude records with missing ETD or reference dates from GAP calculation"
    )
    
    # Filter out missing dates if requested
    if exclude_missing_dates:
        df_demand_filtered = df_demand_filtered[df_demand_filtered["etd"].notna()]
        df_supply_filtered = df_supply_filtered[df_supply_filtered["date_ref"].notna()]

    # Calculate GAP with filtered data
    gap_df = calculate_gap_with_carry_forward(df_demand_filtered, df_supply_filtered, period_type)
    
    # Apply filters based on checkboxes
    display_gap_df = gap_df.copy()
    
    # Filter 1: Show only shortages
    if show_shortage_only:
        display_gap_df = display_gap_df[display_gap_df["gap_quantity"] < 0]
    
    # Filter 2: Exclude products with zero demand
    if exclude_zero_demand:
        # Group by product to check if ANY period has demand > 0
        products_with_demand = (
            gap_df[gap_df["total_demand_qty"] > 0][["pt_code", "product_name"]]
            .drop_duplicates()
        )
        
        # Keep only products that have demand in at least one period
        if len(products_with_demand) > 0:
            display_gap_df = display_gap_df.merge(
                products_with_demand[["pt_code"]], 
                on="pt_code", 
                how="inner"
            )

    st.markdown("### 📄 GAP Details by Product & Period")
    
    # Calculate statistics from display_gap_df (after all filters)
    total_unique_products = display_gap_df["pt_code"].nunique()
    total_gap = display_gap_df["gap_quantity"].where(display_gap_df["gap_quantity"] < 0, 0).abs().sum()
    
    # Add info about filtered products
    if exclude_zero_demand:
        total_products_before = gap_df["pt_code"].nunique()
        excluded_products = total_products_before - total_unique_products
        if excluded_products > 0:
            st.info(f"ℹ️ Excluded {excluded_products} products with zero demand across all periods")
    
    st.markdown(f"🔢 Total Unique Products: **{int(total_unique_products):,}**  💵 Total Shortage Quantity: **{total_gap:,.0f}**")

    # Format display columns
    display_df = display_gap_df.copy()
    display_df["begin_inventory"] = display_df["begin_inventory"].apply(lambda x: f"{x:,.0f}")
    display_df["supply_in_period"] = display_df["supply_in_period"].apply(lambda x: f"{x:,.0f}")
    display_df["total_available"] = display_df["total_available"].apply(lambda x: f"{x:,.0f}")
    display_df["total_demand_qty"] = display_df["total_demand_qty"].apply(lambda x: f"{x:,.0f}")
    display_df["gap_quantity"] = display_df["gap_quantity"].apply(lambda x: f"{x:,.0f}")
    display_df["fulfillment_rate_percent"] = display_df["fulfillment_rate_percent"].apply(lambda x: f"{x:,.1f}%")
    
    st.dataframe(display_df, use_container_width=True)

    # === PIVOT VIEW ===
    st.markdown("### 📊 Pivot View by GAP Quantity")
    style_mode = st.radio(
        "🎨 Styling Mode for Pivot Table",
        options=["None", "🔴 Highlight Shortage", "🌈 Heatmap"],
        horizontal=True,
        key="gap_style_mode"
    )

    # Use display_gap_df for pivot (respects all filters)
    pivot_gap = (
        display_gap_df.groupby(["product_name", "pt_code", "period"])
        .agg(gap_quantity=("gap_quantity", "sum"))
        .reset_index()
        .pivot(index=["product_name", "pt_code"], columns="period", values="gap_quantity")
        .fillna(0)
        .reset_index()
    )
    
    pivot_gap = sort_period_columns(pivot_gap, period_type)

    # Store numeric values for styling
    numeric_pivot = pivot_gap.copy()
    
    # Create display version with formatted values
    display_pivot = pivot_gap.copy()
    for col in display_pivot.columns[2:]:  # Skip product_name and pt_code columns
        display_pivot[col] = display_pivot[col].apply(lambda x: f"{x:,.0f}")

    if style_mode == "🔴 Highlight Shortage":
        # Apply styling to numeric data, then format
        styled_df = numeric_pivot.style.applymap(
            lambda x: "background-color: #fdd; color: red; font-weight: bold;" if x < 0 else "",
            subset=numeric_pivot.columns[2:]
        ).format("{:,.0f}", subset=numeric_pivot.columns[2:])
        
        st.dataframe(styled_df, use_container_width=True)

    elif style_mode == "🌈 Heatmap":
        # Use numeric values for heatmap
        st.dataframe(
            numeric_pivot.style.background_gradient(
                cmap='RdYlGn_r', 
                subset=numeric_pivot.columns[2:], 
                axis=1
            ).format(
                "{:,.0f}", 
                subset=numeric_pivot.columns[2:]
            ), 
            use_container_width=True
        )

    else:
        # Display formatted version without styling
        st.dataframe(display_pivot, use_container_width=True)

    # === Export buttons ===
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
# =========================
# 🔧 Utility: Convert period format
# =========================
def convert_to_period(series, period_type):
    """Convert datetime series to period strings"""
    if period_type == "Daily":
        return series.dt.strftime("%Y-%m-%d")
    elif period_type == "Weekly":
        # Get ISO week number with proper formatting
        week_series = series.dt.isocalendar().week
        year_series = series.dt.isocalendar().year
        return "Week " + week_series.astype(str).str.zfill(2) + " - " + year_series.astype(str)
    elif period_type == "Monthly":
        return series.dt.to_period("M").dt.strftime("%b %Y")
    else:
        return series.astype(str)


# =========================
# 📋 Helper: Sort period columns
# =========================
def sort_period_columns(df, period_type):
    """Sort dataframe columns by period"""
    # Identify non-period columns (product info)
    if "Metric" in df.columns:
        info_cols = ["Metric"]
    else:
        info_cols = ["product_name", "pt_code"]
    
    # Get period columns
    period_cols = [col for col in df.columns if col not in info_cols]
    
    # Filter out invalid column names
    period_cols = [p for p in period_cols if pd.notna(p) and str(p).strip() != "" and str(p) != "nan"]
    
    if period_type == "Weekly":
        sorted_periods = sorted(period_cols, key=lambda x: parse_week_period(x))
    elif period_type == "Monthly":
        sorted_periods = sorted(period_cols, key=lambda x: parse_month_period(x))
    else:  # Daily
        sorted_periods = sorted(period_cols)
    
    return df[info_cols + sorted_periods]


# =========================
# 📋 Helper: Export to Excel
# =========================
def convert_df_to_excel(df):
    """Convert dataframe to Excel file bytes"""
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