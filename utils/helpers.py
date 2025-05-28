import pandas as pd
import streamlit as st
from io import BytesIO
from datetime import datetime
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
        parts = str(period_str).split(" - ")
        if len(parts) == 2:
            week = int(parts[0].replace("Week", "").strip())
            year = int(parts[1].strip())
            return (year, week)
    except:
        pass
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


def validate_quantity_columns(df, quantity_columns):
    """Validate and clean quantity columns"""
    for col in quantity_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    return df


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