import pandas as pd
import streamlit as st
from io import BytesIO
from datetime import datetime, timedelta
import numpy as np

# === EXCEL EXPORT FUNCTIONS ===

def convert_df_to_excel(df, sheet_name="Data"):
    """
    Convert dataframe to Excel bytes with auto-formatting
    
    Args:
        df: pandas DataFrame
        sheet_name: Name for the Excel sheet
    
    Returns:
        BytesIO object containing Excel file
    """
    output = BytesIO()
    
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name=sheet_name)
        
        # Get workbook and worksheet
        workbook = writer.book
        worksheet = writer.sheets[sheet_name]
        
        # Add header format
        header_format = workbook.add_format({
            'bold': True,
            'text_wrap': True,
            'valign': 'top',
            'fg_color': '#D7E4BD',
            'border': 1
        })
        
        # Write headers with format
        for col_num, value in enumerate(df.columns.values):
            worksheet.write(0, col_num, value, header_format)
        
        # Auto-adjust column widths
        for i, col in enumerate(df.columns):
            # Calculate max length in column
            max_len = df[col].astype(str).map(len).max()
            # Compare with header length
            max_len = max(max_len, len(str(col))) + 2
            # Set column width (max 50)
            worksheet.set_column(i, i, min(max_len, 50))
    
    return output.getvalue()


def export_multiple_sheets(dataframes_dict, filename_prefix="export"):
    """
    Export multiple dataframes to different sheets in one Excel file
    
    Args:
        dataframes_dict: Dictionary of {sheet_name: dataframe}
        filename_prefix: Prefix for the filename
    
    Returns:
        BytesIO object containing Excel file
    """
    output = BytesIO()
    
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        for sheet_name, df in dataframes_dict.items():
            df.to_excel(writer, index=False, sheet_name=sheet_name[:31])  # Excel sheet name limit
            
            # Format each sheet
            workbook = writer.book
            worksheet = writer.sheets[sheet_name[:31]]
            
            # Header format
            header_format = workbook.add_format({
                'bold': True,
                'bg_color': '#D7E4BD',
                'border': 1
            })
            
            # Apply header format
            for col_num, value in enumerate(df.columns.values):
                worksheet.write(0, col_num, value, header_format)
    
    return output.getvalue()


# === PERIOD CONVERSION FUNCTIONS ===

def convert_to_period(date_series, period_type="Weekly"):
    """
    Convert datetime series to period strings
    
    Args:
        date_series: pandas Series with datetime values
        period_type: "Daily", "Weekly", or "Monthly"
    
    Returns:
        pandas Series with period strings
    """
    if period_type == "Daily":
        return date_series.dt.strftime("%Y-%m-%d")
    elif period_type == "Weekly":
        week = date_series.dt.isocalendar().week
        year = date_series.dt.isocalendar().year
        return "Week " + week.astype(str).str.zfill(2) + " - " + year.astype(str)
    elif period_type == "Monthly":
        return date_series.dt.to_period("M").dt.strftime("%b %Y")
    else:
        return date_series.astype(str)


def parse_week_period(period_str):
    """Parse week period string for sorting"""
    try:
        # Handle None or NaN values
        if pd.isna(period_str) or not period_str:
            return (9999, 99)
            
        # Convert to string and strip whitespace
        period_str = str(period_str).strip()
        
        # Split by " - " to get week part and year part
        parts = period_str.split(" - ")
        if len(parts) == 2:
            # Extract week number from "Week XX"
            week_part = parts[0].strip()  # "Week 01"
            year_part = parts[1].strip()  # "2025"
            
            # Remove "Week " prefix and get the number
            if week_part.startswith("Week "):
                week_str = week_part[5:].strip()  # Remove "Week " (5 characters)
                week = int(week_str)
                year = int(year_part)
                
                return (year, week)
    except Exception:
        pass
    
    # Return high values for invalid formats to sort them to the end
    return (9999, 99)

def parse_month_period(period_str):
    """Parse month period string for sorting"""
    try:
        return pd.to_datetime("01 " + str(period_str), format="%d %b %Y")
    except:
        return pd.Timestamp.max


def sort_period_columns(df, period_type, info_cols=None):
    """
    Sort dataframe columns by period
    
    Args:
        df: DataFrame with period columns
        period_type: "Daily", "Weekly", or "Monthly"
        info_cols: List of non-period columns to keep first
    
    Returns:
        DataFrame with sorted columns
    """
    if info_cols is None:
        # Auto-detect info columns
        info_cols = [col for col in df.columns if not any(
            x in str(col) for x in ['Week', '2024', '2025', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                                    'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        )]
    
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


# === FORMATTING FUNCTIONS ===

def format_number(value, decimal_places=0, prefix="", suffix=""):
    """Format number with thousands separator"""
    if pd.isna(value):
        return ""
    formatted = f"{value:,.{decimal_places}f}"
    return f"{prefix}{formatted}{suffix}"


def format_currency(value, currency="USD", decimal_places=2):
    """Format currency value"""
    if currency == "USD":
        return format_number(value, decimal_places, prefix="$")
    elif currency == "VND":
        return format_number(value, 0, suffix=" VND")
    else:
        return format_number(value, decimal_places, suffix=f" {currency}")


def format_percentage(value, decimal_places=1):
    """Format percentage value"""
    if pd.isna(value):
        return ""
    return f"{value:.{decimal_places}f}%"


# === DATA VALIDATION FUNCTIONS ===

def check_missing_dates(df, date_column, show_warning=True):
    """
    Check for missing dates in dataframe
    
    Args:
        df: DataFrame to check
        date_column: Name of date column
        show_warning: Whether to show Streamlit warning
    
    Returns:
        Number of missing dates
    """
    missing_count = df[date_column].isna().sum()
    
    if show_warning and missing_count > 0:
        st.warning(f"⚠️ Found {missing_count} records with missing {date_column}")
    
    return missing_count


def check_past_dates(df, date_column, show_warning=True):
    """
    Check for past dates in dataframe
    
    Args:
        df: DataFrame to check
        date_column: Name of date column
        show_warning: Whether to show Streamlit warning
    
    Returns:
        Number of past dates
    """
    today = pd.Timestamp.now().normalize()
    past_count = df[df[date_column] < today].shape[0]
    
    if show_warning and past_count > 0:
        st.warning(f"🔴 Found {past_count} records with past {date_column}")
    
    return past_count

def validate_quantity_columns(df, quantity_columns):
    """Validate and clean quantity columns"""
    for col in quantity_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    return df


def is_past_period(period_str, period_type):
    """Check if a period string represents a past period"""
    today = datetime.now()
    
    try:
        if period_type == "Daily":
            # Format: YYYY-MM-DD
            period_date = pd.to_datetime(period_str)
            return period_date.date() < today.date()
            
        elif period_type == "Weekly":
            # Format: Week XX - YYYY
            if "Week" in str(period_str):
                parts = str(period_str).split(" - ")
                if len(parts) == 2:
                    week_num = int(parts[0].replace("Week ", "").strip())
                    year = int(parts[1].strip())
                    
                    # Create date for the last day of that week (Sunday)
                    # First, get January 1st of that year
                    jan1 = datetime(year, 1, 1)
                    
                    # Find the first Sunday of the year
                    days_to_sunday = (6 - jan1.weekday()) % 7
                    first_sunday = jan1 + timedelta(days=days_to_sunday)
                    
                    # Calculate the target week's Sunday
                    if jan1.weekday() <= 3:  # Thursday or earlier
                        # Week 1 contains January 1st
                        target_sunday = first_sunday + timedelta(weeks=week_num - 1)
                    else:
                        # Week 1 starts after January 1st
                        target_sunday = first_sunday + timedelta(weeks=week_num - 2)
                    
                    return target_sunday.date() < today.date()
                    
        elif period_type == "Monthly":
            # Format: Mon YYYY (e.g., "Jan 2025", "Feb 2025")
            # Convert month name to datetime
            period_date = pd.to_datetime(period_str, format='%b %Y')
            # Check if the entire month has passed
            next_month = period_date + pd.DateOffset(months=1)
            return next_month.date() <= today.date()
            
    except Exception as e:
        # If debug mode is available, log the error
        print(f"Error parsing period '{period_str}': {e}")
        pass
    
    return False

# === STYLING FUNCTIONS ===

def highlight_negative_values(val):
    """Highlight negative values in red"""
    if isinstance(val, (int, float)) and val < 0:
        return 'color: red; font-weight: bold'
    return ''


def highlight_shortage_rows(row, gap_column='gap_quantity'):
    """Highlight rows with shortage"""
    if row[gap_column] < 0:
        return ['background-color: #ffcccc'] * len(row)
    return [''] * len(row)


def apply_alternating_row_colors(df):
    """Apply alternating row colors to dataframe"""
    styles = []
    for i in range(len(df)):
        if i % 2 == 0:
            styles.append(['background-color: #f9f9f9'] * len(df.columns))
        else:
            styles.append([''] * len(df.columns))
    return styles


# === FILTER FUNCTIONS ===

def create_date_filter(col, label, key, default_start=None, default_end=None):
    """Create date input filters"""
    if default_start is None:
        default_start = datetime.today().date()
    if default_end is None:
        default_end = datetime.today().date()
    
    return col.date_input(label, value=(default_start, default_end), key=key)


def create_multiselect_filter(col, label, options, key):
    """Create multiselect filter with 'Select All' option"""
    all_selected = col.checkbox(f"Select all {label}", key=f"{key}_all")
    
    if all_selected:
        selected = col.multiselect(label, options, default=options, key=key)
    else:
        selected = col.multiselect(label, options, key=key)
    
    return selected


# === METRIC CALCULATION FUNCTIONS ===

def calculate_fulfillment_rate(available, demand):
    """Calculate fulfillment rate percentage"""
    if demand == 0:
        return 100.0
    return min(100.0, (available / demand) * 100)


def calculate_days_of_supply(inventory, daily_demand):
    """Calculate days of supply"""
    if daily_demand == 0:
        return float('inf')
    return inventory / daily_demand


# === SESSION STATE HELPERS ===

def save_to_session_state(key, value):
    """Save value to session state with timestamp"""
    st.session_state[key] = value
    st.session_state[f"{key}_timestamp"] = datetime.now()


def get_from_session_state(key, default=None):
    """Get value from session state"""
    return st.session_state.get(key, default)


def clear_session_state_pattern(pattern):
    """Clear session state keys matching pattern"""
    keys_to_clear = [key for key in st.session_state.keys() if pattern in key]
    for key in keys_to_clear:
        del st.session_state[key]


# === NOTIFICATION HELPERS ===

def show_success_message(message, duration=3):
    """Show success message that auto-disappears"""
    placeholder = st.empty()
    placeholder.success(message)
    import time
    time.sleep(duration)
    placeholder.empty()


def show_data_quality_score(df, required_columns):
    """Calculate and display data quality score"""
    total_records = len(df)
    missing_data = 0
    
    for col in required_columns:
        if col in df.columns:
            missing_data += df[col].isna().sum()
    
    quality_score = 100 * (1 - missing_data / (total_records * len(required_columns)))
    
    if quality_score >= 95:
        st.success(f"✅ Data Quality Score: {quality_score:.1f}%")
    elif quality_score >= 80:
        st.warning(f"⚠️ Data Quality Score: {quality_score:.1f}%")
    else:
        st.error(f"❌ Data Quality Score: {quality_score:.1f}%")
    
    return quality_score


# === NEW FUNCTIONS FOR ENHANCED FEATURES ===

def format_alert_message(icon, message, value=None, action=None):
    """Format alert message for display"""
    parts = [f"{icon} {message}"]
    if value:
        parts.append(f": {value}")
    if action:
        parts.append(f" ({action})")
    return " ".join(parts)


def calculate_working_days(start_date, end_date, working_days_per_week=5):
    """Calculate number of working days between two dates"""
    if pd.isna(start_date) or pd.isna(end_date):
        return 0
        
    days = (end_date - start_date).days
    if working_days_per_week == 7:
        return days
    
    # Simple calculation (can be enhanced)
    weeks = days // 7
    remaining_days = days % 7
    
    working_days = weeks * working_days_per_week
    for i in range(remaining_days):
        day = start_date + timedelta(days=i)
        if day.weekday() < working_days_per_week:
            working_days += 1
    
    return max(0, working_days)


def format_metric_card(title, value, subtitle=None, delta=None, delta_color="normal"):
    """Format a metric card with HTML"""
    html = f"""
    <div style="
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    ">
        <h4 style="margin: 0; color: #666;">{title}</h4>
        <h2 style="margin: 10px 0; color: #333;">{value}</h2>
    """
    
    if subtitle:
        html += f'<p style="margin: 0; color: #888; font-size: 0.9em;">{subtitle}</p>'
    
    if delta:
        color = "#28a745" if delta_color == "normal" else "#dc3545"
        html += f'<p style="margin: 5px 0; color: {color}; font-weight: bold;">{delta}</p>'
    
    html += "</div>"
    return html


def create_download_button(df, filename, button_label="📥 Download Excel"):
    """Create a download button for dataframe"""
    excel_data = convert_df_to_excel(df)
    
    return st.download_button(
        label=button_label,
        data=excel_data,
        file_name=f"{filename}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )


def aggregate_by_period(df, date_column, value_columns, period_type="Weekly", agg_func="sum"):
    """Aggregate dataframe by period"""
    df = df.copy()
    df[date_column] = pd.to_datetime(df[date_column])
    
    # Convert to period
    df['period'] = convert_to_period(df[date_column], period_type)
    
    # Group and aggregate
    if isinstance(value_columns, str):
        value_columns = [value_columns]
    
    agg_dict = {col: agg_func for col in value_columns}
    result = df.groupby('period').agg(agg_dict).reset_index()
    
    return result


def calculate_safety_metrics(inventory_df, demand_df, product_col='pt_code', 
                           inventory_col='remaining_quantity', 
                           demand_col='pending_standard_delivery_quantity'):
    """Calculate safety stock metrics"""
    # Merge inventory and demand
    merged = pd.merge(
        inventory_df.groupby(product_col)[inventory_col].sum().reset_index(),
        demand_df.groupby(product_col)[demand_col].sum().reset_index(),
        on=product_col,
        how='outer'
    ).fillna(0)
    
    # Calculate metrics
    merged['coverage_days'] = merged.apply(
        lambda x: x[inventory_col] / x[demand_col] * 30 if x[demand_col] > 0 else float('inf'),
        axis=1
    )
    
    merged['stock_status'] = merged['coverage_days'].apply(
        lambda x: 'Critical' if x < 7 else 'Low' if x < 14 else 'Normal' if x < 90 else 'Excess'
    )
    
    return merged


def create_period_comparison(current_period_df, previous_period_df, metric_columns):
    """Create period-over-period comparison"""
    comparison = {}
    
    for col in metric_columns:
        current_val = current_period_df[col].sum() if col in current_period_df else 0
        previous_val = previous_period_df[col].sum() if col in previous_period_df else 0
        
        if previous_val > 0:
            change_pct = ((current_val - previous_val) / previous_val) * 100
        else:
            change_pct = 100 if current_val > 0 else 0
        
        comparison[col] = {
            'current': current_val,
            'previous': previous_val,
            'change': current_val - previous_val,
            'change_pct': change_pct
        }
    
    return comparison


def validate_settings_import(settings_dict, required_keys):
    """Validate imported settings structure"""
    def check_keys(d, keys, parent=''):
        missing = []
        for key in keys:
            if isinstance(key, dict):
                for k, sub_keys in key.items():
                    if k not in d:
                        missing.append(f"{parent}.{k}" if parent else k)
                    else:
                        missing.extend(check_keys(d[k], sub_keys, f"{parent}.{k}" if parent else k))
            else:
                if key not in d:
                    missing.append(f"{parent}.{key}" if parent else key)
        return missing
    
    missing_keys = check_keys(settings_dict, required_keys)
    return len(missing_keys) == 0, missing_keys


def format_timestamp(timestamp, format_str="%Y-%m-%d %H:%M:%S"):
    """Format timestamp for display"""
    if isinstance(timestamp, str):
        return timestamp
    elif isinstance(timestamp, datetime):
        return timestamp.strftime(format_str)
    elif pd.notna(timestamp):
        return pd.to_datetime(timestamp).strftime(format_str)
    else:
        return "N/A"


def get_color_scale(value, thresholds, colors):
    """Get color based on value and thresholds"""
    for i, threshold in enumerate(thresholds):
        if value <= threshold:
            return colors[i]
    return colors[-1]


def create_summary_stats(df, numeric_columns):
    """Create summary statistics for numeric columns"""
    stats = {}
    
    for col in numeric_columns:
        if col in df.columns:
            stats[col] = {
                'count': df[col].count(),
                'mean': df[col].mean(),
                'std': df[col].std(),
                'min': df[col].min(),
                '25%': df[col].quantile(0.25),
                '50%': df[col].quantile(0.50),
                '75%': df[col].quantile(0.75),
                'max': df[col].max(),
                'sum': df[col].sum()
            }
    
    return pd.DataFrame(stats).T


def detect_anomalies(df, value_column, method='iqr', threshold=1.5):
    """Detect anomalies in data using IQR method"""
    if method == 'iqr':
        Q1 = df[value_column].quantile(0.25)
        Q3 = df[value_column].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        
        df['is_anomaly'] = (df[value_column] < lower_bound) | (df[value_column] > upper_bound)
    
    return df


def create_alert_summary(alerts_list):
    """Create a summary of alerts by category"""
    summary = {
        'critical': len([a for a in alerts_list if a.get('level') == 'critical']),
        'warning': len([a for a in alerts_list if a.get('level') == 'warning']),
        'info': len([a for a in alerts_list if a.get('level') == 'info'])
    }
    
    summary['total'] = sum(summary.values())
    return summary