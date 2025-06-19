# utils/helpers.py - Refactored Helper Functions

import pandas as pd
import streamlit as st
from io import BytesIO
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
import logging

# Configure logging
logger = logging.getLogger(__name__)

# === CONSTANTS ===
EXCEL_SHEET_NAME_LIMIT = 31
DEFAULT_EXCEL_ENGINE = "xlsxwriter"
DEFAULT_DATE_FORMAT = "%Y-%m-%d"
WEEK_FORMAT = "Week {week} - {year}"
MONTH_FORMAT = "%b %Y"

# Header formatting for Excel
EXCEL_HEADER_FORMAT = {
    'bold': True,
    'text_wrap': True,
    'valign': 'top',
    'fg_color': '#D7E4BD',
    'border': 1
}

# === EXCEL EXPORT FUNCTIONS ===

def convert_df_to_excel(df: pd.DataFrame, sheet_name: str = "Data") -> bytes:
    """
    Convert dataframe to Excel bytes with auto-formatting
    
    Args:
        df: DataFrame to convert
        sheet_name: Name of the Excel sheet
        
    Returns:
        Excel file as bytes
    """
    if df.empty:
        logger.warning("Attempting to convert empty DataFrame to Excel")
        return BytesIO().getvalue()
    
    output = BytesIO()
    
    try:
        with pd.ExcelWriter(output, engine=DEFAULT_EXCEL_ENGINE) as writer:
            # Truncate sheet name if too long
            sheet_name = sheet_name[:EXCEL_SHEET_NAME_LIMIT]
            df.to_excel(writer, index=False, sheet_name=sheet_name)
            
            workbook = writer.book
            worksheet = writer.sheets[sheet_name]
            
            # Add header format
            header_format = workbook.add_format(EXCEL_HEADER_FORMAT)
            
            # Write headers with format
            for col_num, value in enumerate(df.columns.values):
                worksheet.write(0, col_num, value, header_format)
            
            # Auto-adjust column widths
            for i, col in enumerate(df.columns):
                # Calculate column width based on content
                try:
                    max_len = df[col].astype(str).map(len).max()
                    max_len = max(max_len, len(str(col))) + 2
                    worksheet.set_column(i, i, min(max_len, 50))
                except Exception as e:
                    logger.debug(f"Could not calculate width for column {col}: {e}")
                    worksheet.set_column(i, i, 15)  # Default width
        
        return output.getvalue()
        
    except Exception as e:
        logger.error(f"Error converting DataFrame to Excel: {e}")
        raise

def export_multiple_sheets(dataframes_dict: Dict[str, pd.DataFrame]) -> bytes:
    """
    Export multiple dataframes to different sheets in one Excel file
    
    Args:
        dataframes_dict: Dictionary mapping sheet names to DataFrames
        
    Returns:
        Excel file as bytes
    """
    if not dataframes_dict:
        logger.warning("No DataFrames provided for multi-sheet export")
        return BytesIO().getvalue()
    
    output = BytesIO()
    
    try:
        with pd.ExcelWriter(output, engine=DEFAULT_EXCEL_ENGINE) as writer:
            for sheet_name, df in dataframes_dict.items():
                if df is None or df.empty:
                    logger.debug(f"Skipping empty sheet: {sheet_name}")
                    continue
                    
                # Truncate sheet name if too long
                truncated_name = sheet_name[:EXCEL_SHEET_NAME_LIMIT]
                df.to_excel(writer, index=False, sheet_name=truncated_name)
                
                # Format each sheet
                workbook = writer.book
                worksheet = writer.sheets[truncated_name]
                
                # Header format
                header_format = workbook.add_format(EXCEL_HEADER_FORMAT)
                
                # Apply header format
                for col_num, value in enumerate(df.columns.values):
                    worksheet.write(0, col_num, value, header_format)
                    
                # Auto-adjust column widths
                for i, col in enumerate(df.columns):
                    try:
                        max_len = df[col].astype(str).map(len).max()
                        max_len = max(max_len, len(str(col))) + 2
                        worksheet.set_column(i, i, min(max_len, 50))
                    except:
                        worksheet.set_column(i, i, 15)
        
        return output.getvalue()
        
    except Exception as e:
        logger.error(f"Error exporting multiple sheets: {e}")
        raise

# === PERIOD CONVERSION FUNCTIONS ===

def convert_to_period(date_series: Union[pd.Series, Any], period_type: str) -> Union[pd.Series, str, None]:
    """
    Convert date series to period string with proper weekly format
    
    Args:
        date_series: Series of dates or single date value
        period_type: Type of period ('Daily', 'Weekly', 'Monthly')
        
    Returns:
        Converted period string(s) or None if conversion fails
    """
    if isinstance(date_series, pd.Series):
        # Ensure datetime type
        date_series = pd.to_datetime(date_series, errors='coerce')
        
        if period_type == "Daily":
            return date_series.dt.strftime(DEFAULT_DATE_FORMAT)
        elif period_type == "Weekly":
            # Create Week X - YYYY format
            week_numbers = date_series.dt.isocalendar().week
            years = date_series.dt.year
            return pd.Series([
                WEEK_FORMAT.format(week=w, year=y) 
                for w, y in zip(week_numbers, years)
            ], index=date_series.index)
        elif period_type == "Monthly":
            return date_series.dt.strftime(MONTH_FORMAT)
        else:
            logger.warning(f"Unknown period type: {period_type}")
            return date_series
    else:
        # Handle single value
        try:
            date_val = pd.to_datetime(date_series, errors='coerce')
            if pd.isna(date_val):
                return None
                
            if period_type == "Daily":
                return date_val.strftime(DEFAULT_DATE_FORMAT)
            elif period_type == "Weekly":
                week_num = date_val.isocalendar().week
                year = date_val.year
                return WEEK_FORMAT.format(week=week_num, year=year)
            elif period_type == "Monthly":
                return date_val.strftime(MONTH_FORMAT)
            else:
                logger.warning(f"Unknown period type: {period_type}")
                return str(date_val)
        except Exception as e:
            logger.debug(f"Error converting date to period: {e}")
            return None

def parse_week_period(period_str: str) -> Tuple[int, int]:
    """
    Parse week period string for sorting
    
    Args:
        period_str: Week period string (e.g., "Week 5 - 2024")
        
    Returns:
        Tuple of (year, week) for sorting
    """
    try:
        if pd.isna(period_str) or not period_str:
            return (9999, 99)
            
        period_str = str(period_str).strip()
        
        # Handle standard format "Week X - YYYY"
        if " - " in period_str:
            parts = period_str.split(" - ")
            if len(parts) == 2 and parts[0].startswith("Week "):
                week_str = parts[0].replace("Week ", "").strip()
                year_str = parts[1].strip()
                
                week = int(week_str)
                year = int(year_str)
                
                # Validate week number
                if 1 <= week <= 53:
                    return (year, week)
                    
    except (ValueError, AttributeError) as e:
        logger.debug(f"Error parsing week period '{period_str}': {e}")
    
    return (9999, 99)

def parse_month_period(period_str: str) -> pd.Timestamp:
    """
    Parse month period string for sorting
    
    Args:
        period_str: Month period string (e.g., "Jan 2024")
        
    Returns:
        Timestamp for sorting
    """
    try:
        if pd.isna(period_str) or not period_str:
            return pd.Timestamp.max
            
        # Try parsing with day prefix
        return pd.to_datetime(f"01 {str(period_str)}", format=f"%d {MONTH_FORMAT}")
    except Exception as e:
        logger.debug(f"Error parsing month period '{period_str}': {e}")
        return pd.Timestamp.max

def sort_period_columns(df: pd.DataFrame, period_type: str, 
                       info_cols: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Sort dataframe columns by period
    
    Args:
        df: DataFrame with period columns
        period_type: Type of period for sorting
        info_cols: Non-period columns to keep at the beginning
        
    Returns:
        DataFrame with sorted columns
    """
    if df.empty:
        return df
    
    if info_cols is None:
        # Auto-detect info columns by excluding common period patterns
        period_patterns = ['Week', '20', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        info_cols = []
        for col in df.columns:
            col_str = str(col)
            if not any(pattern in col_str for pattern in period_patterns):
                info_cols.append(col)
    
    # Get period columns
    period_cols = [col for col in df.columns if col not in info_cols]
    
    # Remove invalid period columns
    valid_period_cols = []
    for col in period_cols:
        if pd.notna(col) and str(col).strip() != "" and str(col) != "nan":
            valid_period_cols.append(col)
    
    # Sort based on period type
    try:
        if period_type == "Weekly":
            sorted_periods = sorted(valid_period_cols, key=parse_week_period)
        elif period_type == "Monthly":
            sorted_periods = sorted(valid_period_cols, key=parse_month_period)
        else:  # Daily
            sorted_periods = sorted(valid_period_cols)
    except Exception as e:
        logger.error(f"Error sorting period columns: {e}")
        sorted_periods = valid_period_cols
    
    # Return dataframe with info columns first, then sorted period columns
    return df[info_cols + sorted_periods]

def is_past_period(period_str: str, period_type: str, 
                  reference_date: Optional[datetime] = None) -> bool:
    """
    Check if a period string represents a past period
    
    Args:
        period_str: Period string to check
        period_type: Type of period
        reference_date: Reference date for comparison (default: today)
        
    Returns:
        True if period is in the past
    """
    if reference_date is None:
        reference_date = datetime.now()
    
    try:
        if pd.isna(period_str) or not period_str:
            return False
            
        if period_type == "Daily":
            period_date = pd.to_datetime(period_str, errors='coerce')
            if pd.notna(period_date):
                return period_date.date() < reference_date.date()
                
        elif period_type == "Weekly":
            year, week = parse_week_period(str(period_str))
            if year < 9999:  # Valid parse
                # Get the last day of the ISO week
                jan4 = datetime(year, 1, 4)
                week_start = jan4 - timedelta(days=jan4.isoweekday() - 1)
                target_week_start = week_start + timedelta(weeks=week - 1)
                target_week_end = target_week_start + timedelta(days=6)
                return target_week_end.date() < reference_date.date()
                
        elif period_type == "Monthly":
            period_date = parse_month_period(str(period_str))
            if period_date != pd.Timestamp.max:
                # Check if the entire month has passed
                next_month = period_date + pd.DateOffset(months=1)
                return next_month.date() <= reference_date.date()
                
    except Exception as e:
        logger.debug(f"Error checking if period is past: {e}")
    
    return False

# === SESSION STATE HELPERS ===

def save_to_session_state(key: str, value: Any, add_timestamp: bool = True):
    """
    Save value to session state with optional timestamp
    
    Args:
        key: Session state key
        value: Value to save
        add_timestamp: Whether to add timestamp
    """
    st.session_state[key] = value
    if add_timestamp:
        st.session_state[f"{key}_timestamp"] = datetime.now()

def get_from_session_state(key: str, default: Any = None) -> Any:
    """
    Get value from session state
    
    Args:
        key: Session state key
        default: Default value if key not found
        
    Returns:
        Value from session state or default
    """
    return st.session_state.get(key, default)

def clear_session_state_pattern(pattern: str):
    """
    Clear session state keys matching pattern
    
    Args:
        pattern: Pattern to match in key names
    """
    keys_to_clear = [key for key in st.session_state.keys() if pattern in key]
    for key in keys_to_clear:
        del st.session_state[key]
    
    if keys_to_clear:
        logger.debug(f"Cleared {len(keys_to_clear)} session state keys matching '{pattern}'")

# === METRIC CALCULATION FUNCTIONS ===

def calculate_fulfillment_rate(available: float, demand: float) -> float:
    """
    Calculate fulfillment rate percentage
    
    Args:
        available: Available quantity
        demand: Demand quantity
        
    Returns:
        Fulfillment rate as percentage (0-100)
    """
    if demand <= 0:
        return 100.0 if available >= 0 else 0.0
    return min(100.0, max(0.0, (available / demand) * 100))

def calculate_days_of_supply(inventory: float, daily_demand: float) -> float:
    """
    Calculate days of supply
    
    Args:
        inventory: Current inventory level
        daily_demand: Average daily demand
        
    Returns:
        Days of supply (inf if no demand)
    """
    if daily_demand <= 0:
        return float('inf') if inventory > 0 else 0.0
    return max(0.0, inventory / daily_demand)

def calculate_working_days(start_date: datetime, end_date: datetime, 
                         working_days_per_week: int = 5) -> int:
    """
    Calculate number of working days between two dates
    
    Args:
        start_date: Start date
        end_date: End date
        working_days_per_week: Number of working days per week (1-7)
        
    Returns:
        Number of working days
    """
    if pd.isna(start_date) or pd.isna(end_date):
        return 0
    
    # Ensure start_date is before end_date
    if start_date > end_date:
        start_date, end_date = end_date, start_date
    
    # Validate working days per week
    working_days_per_week = max(1, min(7, working_days_per_week))
    
    total_days = (end_date - start_date).days + 1
    
    if working_days_per_week == 7:
        return total_days
    
    # Calculate full weeks and remaining days
    full_weeks = total_days // 7
    remaining_days = total_days % 7
    
    working_days = full_weeks * working_days_per_week
    
    # Count working days in remaining days
    current_date = start_date + timedelta(days=full_weeks * 7)
    for _ in range(remaining_days):
        if current_date.weekday() < working_days_per_week:
            working_days += 1
        current_date += timedelta(days=1)
    
    return max(0, working_days)

# === NOTIFICATION HELPERS ===

def show_success_message(message: str, duration: int = 3):
    """
    Show success message that auto-disappears
    
    Args:
        message: Message to display
        duration: Duration in seconds
    """
    placeholder = st.empty()
    placeholder.success(message)
    
    # Note: In production, consider using st.toast() instead
    import time
    time.sleep(duration)
    placeholder.empty()

def create_download_button(df: pd.DataFrame, filename: str, 
                         button_label: str = "📥 Download Excel",
                         key: Optional[str] = None) -> None:
    """
    Create a download button for dataframe
    
    Args:
        df: DataFrame to download
        filename: Base filename (without extension)
        button_label: Label for the download button
        key: Unique key for the button
    """
    if df.empty:
        st.warning("No data available for download")
        return
        
    try:
        excel_data = convert_df_to_excel(df)
        
        st.download_button(
            label=button_label,
            data=excel_data,
            file_name=f"{filename}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key=key
        )
    except Exception as e:
        st.error(f"Error creating download: {str(e)}")

# === ANALYSIS FUNCTIONS ===

def aggregate_by_period(df: pd.DataFrame, date_column: str, 
                       value_columns: Union[str, List[str]], 
                       period_type: str = "Weekly", 
                       agg_func: Union[str, Dict[str, str]] = "sum") -> pd.DataFrame:
    """
    Aggregate dataframe by period
    
    Args:
        df: Source DataFrame
        date_column: Column containing dates
        value_columns: Column(s) to aggregate
        period_type: Type of period for aggregation
        agg_func: Aggregation function(s)
        
    Returns:
        Aggregated DataFrame
    """
    if df.empty or date_column not in df.columns:
        return pd.DataFrame()
    
    df = df.copy()
    
    # Ensure date column is datetime
    df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
    
    # Remove rows with invalid dates
    df = df[df[date_column].notna()]
    
    if df.empty:
        return pd.DataFrame()
    
    # Convert to period
    df['period'] = convert_to_period(df[date_column], period_type)
    
    # Ensure value_columns is a list
    if isinstance(value_columns, str):
        value_columns = [value_columns]
    
    # Filter to existing columns
    value_columns = [col for col in value_columns if col in df.columns]
    
    if not value_columns:
        return pd.DataFrame()
    
    # Create aggregation dictionary
    if isinstance(agg_func, str):
        agg_dict = {col: agg_func for col in value_columns}
    else:
        agg_dict = agg_func
    
    # Group and aggregate
    try:
        result = df.groupby('period').agg(agg_dict).reset_index()
        return result
    except Exception as e:
        logger.error(f"Error aggregating by period: {e}")
        return pd.DataFrame()

def calculate_safety_metrics(inventory_df: pd.DataFrame, 
                           demand_df: pd.DataFrame, 
                           product_col: str = 'pt_code', 
                           inventory_col: str = 'remaining_quantity', 
                           demand_col: str = 'pending_standard_delivery_quantity',
                           days_forward: int = 30) -> pd.DataFrame:
    """
    Calculate safety stock metrics
    
    Args:
        inventory_df: DataFrame with inventory data
        demand_df: DataFrame with demand data
        product_col: Column name for product identifier
        inventory_col: Column name for inventory quantity
        demand_col: Column name for demand quantity
        days_forward: Number of days to calculate coverage
        
    Returns:
        DataFrame with safety metrics
    """
    if inventory_df.empty and demand_df.empty:
        return pd.DataFrame()
    
    # Aggregate by product
    inventory_agg = pd.DataFrame()
    demand_agg = pd.DataFrame()
    
    if not inventory_df.empty and product_col in inventory_df.columns and inventory_col in inventory_df.columns:
        inventory_agg = inventory_df.groupby(product_col)[inventory_col].sum().reset_index()
    
    if not demand_df.empty and product_col in demand_df.columns and demand_col in demand_df.columns:
        demand_agg = demand_df.groupby(product_col)[demand_col].sum().reset_index()
    
    # Merge inventory and demand
    if not inventory_agg.empty and not demand_agg.empty:
        merged = pd.merge(
            inventory_agg,
            demand_agg,
            on=product_col,
            how='outer'
        ).fillna(0)
    elif not inventory_agg.empty:
        merged = inventory_agg.copy()
        merged[demand_col] = 0
    elif not demand_agg.empty:
        merged = demand_agg.copy()
        merged[inventory_col] = 0
    else:
        return pd.DataFrame()
    
    # Calculate metrics
    merged['daily_demand'] = merged[demand_col] / max(days_forward, 1)
    merged['coverage_days'] = merged.apply(
        lambda x: calculate_days_of_supply(x[inventory_col], x['daily_demand']),
        axis=1
    )
    
    # Categorize stock status
    def categorize_stock(days):
        if days == float('inf'):
            return 'No Demand'
        elif days < 7:
            return 'Critical'
        elif days < 14:
            return 'Low'
        elif days < 90:
            return 'Normal'
        else:
            return 'Excess'
    
    merged['stock_status'] = merged['coverage_days'].apply(categorize_stock)
    
    return merged

def create_period_comparison(current_period_df: pd.DataFrame, 
                           previous_period_df: pd.DataFrame, 
                           metric_columns: List[str]) -> Dict[str, Dict[str, Any]]:
    """
    Create period-over-period comparison
    
    Args:
        current_period_df: Current period data
        previous_period_df: Previous period data
        metric_columns: Columns to compare
        
    Returns:
        Dictionary with comparison metrics
    """
    comparison = {}
    
    for col in metric_columns:
        current_val = 0
        previous_val = 0
        
        if not current_period_df.empty and col in current_period_df.columns:
            current_val = current_period_df[col].sum()
            
        if not previous_period_df.empty and col in previous_period_df.columns:
            previous_val = previous_period_df[col].sum()
        
        # Calculate change
        change = current_val - previous_val
        
        # Calculate percentage change
        if previous_val != 0:
            change_pct = (change / abs(previous_val)) * 100
        else:
            change_pct = 100 if current_val > 0 else 0
        
        comparison[col] = {
            'current': current_val,
            'previous': previous_val,
            'change': change,
            'change_pct': change_pct
        }
    
    return comparison

def create_summary_stats(df: pd.DataFrame, 
                        numeric_columns: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Create summary statistics for numeric columns
    
    Args:
        df: Source DataFrame
        numeric_columns: List of numeric columns (auto-detect if None)
        
    Returns:
        DataFrame with summary statistics
    """
    if df.empty:
        return pd.DataFrame()
    
    # Auto-detect numeric columns if not provided
    if numeric_columns is None:
        numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    else:
        # Filter to existing numeric columns
        numeric_columns = [col for col in numeric_columns 
                          if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]
    
    if not numeric_columns:
        return pd.DataFrame()
    
    stats = {}
    
    for col in numeric_columns:
        try:
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
        except Exception as e:
            logger.debug(f"Error calculating stats for column {col}: {e}")
            continue
    
    return pd.DataFrame(stats).T

def create_alert_summary(alerts_list: List[Dict[str, Any]]) -> Dict[str, int]:
    """
    Create a summary of alerts by category
    
    Args:
        alerts_list: List of alert dictionaries with 'level' key
        
    Returns:
        Dictionary with alert counts by level
    """
    summary = {
        'critical': 0,
        'warning': 0,
        'info': 0,
        'total': 0
    }
    
    for alert in alerts_list:
        level = alert.get('level', 'info').lower()
        if level in summary:
            summary[level] += 1
        else:
            summary['info'] += 1  # Default to info for unknown levels
    
    summary['total'] = sum(v for k, v in summary.items() if k != 'total')
    
    return summary

# === STANDARDIZED PERIOD HANDLING ===

def create_period_pivot(
    df: pd.DataFrame,
    group_cols: List[str],
    period_col: str,
    value_col: str,
    agg_func: Union[str, Callable] = "sum",
    period_type: str = "Weekly",
    show_only_nonzero: bool = True,
    fill_value: Any = 0
) -> pd.DataFrame:
    """
    Create standardized pivot table for any analysis page
    
    Args:
        df: Source dataframe
        group_cols: Columns to group by
        period_col: Column containing period values
        value_col: Column to aggregate
        agg_func: Aggregation function
        period_type: Type of period for sorting
        show_only_nonzero: Whether to filter out zero rows
        fill_value: Value to fill NaN
    
    Returns:
        Pivot dataframe with sorted columns
    """
    if df.empty:
        return pd.DataFrame()
    
    # Validate inputs
    missing_cols = [col for col in group_cols + [period_col, value_col] if col not in df.columns]
    if missing_cols:
        logger.error(f"Missing columns in dataframe: {missing_cols}")
        return pd.DataFrame()
    
    try:
        # Create pivot
        pivot_df = df.pivot_table(
            index=group_cols,
            columns=period_col,
            values=value_col,
            aggfunc=agg_func,
            fill_value=fill_value
        ).reset_index()
        
        # Filter non-zero rows if requested
        if show_only_nonzero and len(pivot_df.columns) > len(group_cols):
            numeric_cols = [col for col in pivot_df.columns if col not in group_cols]
            if numeric_cols:
                row_sums = pivot_df[numeric_cols].sum(axis=1)
                pivot_df = pivot_df[row_sums > 0]
        
        # Sort columns by period
        pivot_df = sort_period_columns(pivot_df, period_type, group_cols)
        
        return pivot_df
        
    except Exception as e:
        logger.error(f"Error creating pivot: {str(e)}")
        return pd.DataFrame()

def apply_period_indicators(
    df: pd.DataFrame,
    period_type: str,
    exclude_cols: List[str],
    indicator: str = "🔴",
    reference_date: Optional[datetime] = None
) -> pd.DataFrame:
    """
    Add indicators to past period columns
    
    Args:
        df: Dataframe with period columns
        period_type: Type of period
        exclude_cols: Columns to exclude from indicators
        indicator: Indicator symbol for past periods
        reference_date: Reference date for comparison
    
    Returns:
        DataFrame with renamed columns
    """
    if df.empty:
        return df
    
    display_df = df.copy()
    renamed_columns = {}
    
    for col in display_df.columns:
        if col not in exclude_cols:
            if is_past_period(str(col), period_type, reference_date):
                renamed_columns[col] = f"{indicator} {col}"
    
    if renamed_columns:
        display_df = display_df.rename(columns=renamed_columns)
    
    return display_df



def format_pivot_with_periods(
    pivot_df: pd.DataFrame,
    period_type: str,
    info_cols: List[str],
    value_formatter: Optional[Callable] = None,
    add_indicators: bool = True
) -> pd.DataFrame:
    """
    Format pivot table with period indicators and value formatting
    
    Args:
        pivot_df: Pivot dataframe
        period_type: Type of period
        info_cols: Non-period columns
        value_formatter: Function to format values
        add_indicators: Whether to add past period indicators
    
    Returns:
        Formatted dataframe ready for display
    """
    if pivot_df.empty:
        return pivot_df
        
    display_df = pivot_df.copy()
    
    # Add period indicators if requested
    if add_indicators:
        display_df = apply_period_indicators(display_df, period_type, info_cols)
    
    # Format numeric columns if formatter provided
    if value_formatter:
        for col in display_df.columns:
            if col not in info_cols:
                try:
                    display_df[col] = display_df[col].apply(value_formatter)
                except Exception as e:
                    logger.debug(f"Could not format column {col}: {e}")
    
    return display_df

# === STANDARDIZED FILTERING FUNCTIONS ===

def apply_standard_filter_option(
    df: pd.DataFrame,
    filter_option: str,
    date_columns: List[str],
    today: Optional[pd.Timestamp] = None
) -> pd.DataFrame:
    """
    Apply standard filter options used across all detail tables
    
    Args:
        df: Source dataframe
        filter_option: Filter option
        date_columns: List of date columns to check
        today: Reference date for "past" comparison
    
    Returns:
        Filtered dataframe
    """
    if df.empty or filter_option == "Show All":
        return df
    
    if today is None:
        today = pd.Timestamp.now().normalize()
    
    display_df = df.copy()
    
    # Determine filter type and column suffix
    filter_missing = "Missing" in filter_option
    filter_past = "Past" in filter_option
    is_original = "Original" in filter_option
    is_adjusted = "Adjusted" in filter_option
    
    # Find the appropriate column
    target_col = None
    for col in date_columns:
        if col not in display_df.columns:
            continue
            
        if is_original and col.endswith("_original"):
            target_col = col
            break
        elif is_adjusted and col.endswith("_adjusted"):
            target_col = col
            break
        elif not is_original and not is_adjusted:
            target_col = col
            break
    
    if target_col is None:
        logger.warning(f"No suitable date column found for filter: {filter_option}")
        return display_df
    
    # Apply filter
    if filter_missing:
        display_df = display_df[display_df[target_col].isna()]
    elif filter_past:
        display_df[target_col] = pd.to_datetime(display_df[target_col], errors='coerce')
        display_df = display_df[display_df[target_col].notna() & (display_df[target_col] < today)]
    
    return display_df

# === PERIOD AGGREGATION FUNCTIONS ===

def create_period_summary(
    df: pd.DataFrame,
    period_col: str,
    period_type: str,
    agg_configs: List[Dict[str, Any]],
    add_indicators: bool = True,
    reference_date: Optional[datetime] = None
) -> pd.DataFrame:
    """
    Create period summary with multiple metrics
    
    Args:
        df: Source dataframe
        period_col: Column containing periods
        period_type: Type of period
        agg_configs: List of dicts with 'column', 'agg_func', 'label', 'formatter'
        add_indicators: Whether to add past period indicators
        reference_date: Reference date for indicators
    
    Returns:
        Summary dataframe with formatted values
    """
    if df.empty or period_col not in df.columns:
        return pd.DataFrame()
    
    # Validate agg_configs
    valid_configs = []
    for config in agg_configs:
        if 'column' in config and config['column'] in df.columns:
            config.setdefault('agg_func', 'sum')
            config.setdefault('label', config['column'])
            config.setdefault('formatter', str)
            valid_configs.append(config)
    
    if not valid_configs:
        return pd.DataFrame()
    
    # Group by period
    grouped = df.groupby(period_col)
    
    # Create summary data
    summary_data = {"Metric": []}
    periods = sorted(grouped.groups.keys(), key=lambda x: parse_period_key(x, period_type))
    
    # Add metrics
    for config in valid_configs:
        summary_data["Metric"].append(config['label'])
        
        for period in periods:
            try:
                period_data = grouped.get_group(period)
                value = period_data[config['column']].agg(config['agg_func'])
                
                # Format value
                formatted_value = config['formatter'](value)
            except Exception as e:
                logger.debug(f"Error aggregating {config['column']} for period {period}: {e}")
                formatted_value = "N/A"
            
            # Add period indicator if needed
            col_name = str(period)
            if add_indicators and is_past_period(col_name, period_type, reference_date):
                col_name = f"🔴 {col_name}"
            
            if col_name not in summary_data:
                summary_data[col_name] = []
            
            summary_data[col_name].append(formatted_value)
    
    return pd.DataFrame(summary_data)

def parse_period_key(period: Any, period_type: str) -> Any:
    """
    Parse period for sorting based on type
    
    Args:
        period: Period value to parse
        period_type: Type of period
        
    Returns:
        Sortable representation of the period
    """
    if pd.isna(period) or str(period).strip() == "" or str(period) == "nan":
        return (9999, 99)  # Sort invalid periods last
    
    if period_type == "Weekly":
        return parse_week_period(str(period))
    elif period_type == "Monthly":
        return parse_month_period(str(period))
    else:  # Daily
        try:
            return pd.to_datetime(period)
        except:
            return pd.Timestamp.max

# === DATA QUALITY HELPERS ===

def check_data_quality_summary(
    df: pd.DataFrame,
    date_columns: List[str],
    required_columns: Optional[List[str]] = None,
    reference_date: Optional[datetime] = None
) -> Dict[str, Any]:
    """
    Comprehensive data quality check
    
    Args:
        df: DataFrame to check
        date_columns: Date columns to validate
        required_columns: Required columns to check for completeness
        reference_date: Reference date for "past" checks
    
    Returns:
        Dict with quality metrics
    """
    quality = {
        'total_records': len(df),
        'missing_dates': {},
        'past_dates': {},
        'missing_required': {},
        'quality_score': 100.0,
        'issues': []
    }
    
    if df.empty:
        quality['quality_score'] = 0.0
        quality['issues'].append("No data available")
        return quality
    
    if reference_date is None:
        reference_date = pd.Timestamp.now().normalize()
    
    total_issues = 0
    total_checks = 0
    
    # Check date columns
    for col in date_columns:
        if col in df.columns:
            total_checks += len(df)
            dates = pd.to_datetime(df[col], errors='coerce')
            
            missing_count = dates.isna().sum()
            quality['missing_dates'][col] = missing_count
            total_issues += missing_count
            
            if missing_count > 0:
                quality['issues'].append(f"{missing_count} missing values in {col}")
            
            past_count = (dates < reference_date).sum()
            quality['past_dates'][col] = past_count
            
            if past_count > 0:
                quality['issues'].append(f"{past_count} past dates in {col}")
    
    # Check required columns
    if required_columns:
        for col in required_columns:
            if col in df.columns:
                total_checks += len(df)
                missing_count = df[col].isna().sum()
                quality['missing_required'][col] = missing_count
                total_issues += missing_count
                
                if missing_count > 0:
                    quality['issues'].append(f"{missing_count} missing values in required field {col}")
    
    # Calculate overall score
    if total_checks > 0:
        quality['quality_score'] = max(0, 100 * (1 - total_issues / total_checks))
    
    return quality

# === SESSION STATE HELPERS (ENHANCED) ===

def save_analysis_state(
    page_name: str,
    data: Dict[str, Any],
    ttl_minutes: int = 5
):
    """
    Save analysis state with TTL
    
    Args:
        page_name: Name of the page
        data: Data to save
        ttl_minutes: Time to live in minutes
    """
    save_to_session_state(f'{page_name}_analysis_data', data)
    save_to_session_state(f'{page_name}_analysis_ttl', ttl_minutes, add_timestamp=False)

def get_analysis_state(page_name: str) -> Optional[Dict[str, Any]]:
    """
    Get analysis state if not expired
    
    Returns:
        Saved data or None if expired/not found
    """
    data = get_from_session_state(f'{page_name}_analysis_data')
    timestamp = get_from_session_state(f'{page_name}_analysis_data_timestamp')
    ttl = get_from_session_state(f'{page_name}_analysis_ttl', 5)
    
    if data and timestamp:
        elapsed_minutes = (datetime.now() - timestamp).total_seconds() / 60
        if elapsed_minutes <= ttl:
            return data
        else:
            # Clean up expired data
            clear_analysis_state(page_name)
    
    return None

def clear_analysis_state(page_name: str):
    """Clear analysis state for a page"""
    keys_to_clear = [
        f'{page_name}_analysis_data',
        f'{page_name}_analysis_data_timestamp',
        f'{page_name}_analysis_ttl'
    ]
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]

# === EXPORT HELPERS ===

def create_multi_sheet_export(
    sheets_config: List[Dict[str, Any]],
    filename_prefix: str
) -> Tuple[Optional[bytes], Optional[str]]:
    """
    Create multi-sheet Excel export
    
    Args:
        sheets_config: List of dicts with 'name', 'data', 'formatter' (optional)
        filename_prefix: Prefix for filename
    
    Returns:
        Tuple of (excel_bytes, filename) or (None, None) if no data
    """
    sheets_dict = {}
    
    for config in sheets_config:
        if 'name' not in config or 'data' not in config:
            logger.warning(f"Invalid sheet config: {config}")
            continue
            
        df = config['data']
        if df is not None and not df.empty:
            # Apply formatter if provided
            if 'formatter' in config and callable(config['formatter']):
                try:
                    df = config['formatter'](df)
                except Exception as e:
                    logger.error(f"Error applying formatter to sheet '{config['name']}': {e}")
            
            # Truncate sheet name to Excel limit
            sheet_name = str(config['name'])[:EXCEL_SHEET_NAME_LIMIT]
            sheets_dict[sheet_name] = df
    
    if sheets_dict:
        try:
            excel_data = export_multiple_sheets(sheets_dict)
            filename = f"{filename_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
            return excel_data, filename
        except Exception as e:
            logger.error(f"Error creating multi-sheet export: {e}")
            return None, None
    
    return None, None

# === UTILITY FUNCTIONS ===

def format_number(value: Any, decimals: int = 0) -> str:
    """
    Format number with thousand separators
    
    Args:
        value: Number to format
        decimals: Number of decimal places
        
    Returns:
        Formatted string
    """
    try:
        if pd.isna(value) or value is None:
            return ""
        
        if isinstance(value, (int, float)):
            if decimals > 0:
                return f"{value:,.{decimals}f}"
            else:
                return f"{int(value):,}"
        
        return str(value)
    except:
        return str(value)

def format_percentage(value: Any, decimals: int = 1) -> str:
    """
    Format value as percentage
    
    Args:
        value: Number to format (0-100 scale)
        decimals: Number of decimal places
        
    Returns:
        Formatted percentage string
    """
    try:
        if pd.isna(value) or value is None:
            return ""
        
        if isinstance(value, (int, float)):
            return f"{value:.{decimals}f}%"
        
        return str(value)
    except:
        return str(value)

def format_currency(value: Any, currency_symbol: str = "$", decimals: int = 2) -> str:
    """
    Format value as currency
    
    Args:
        value: Number to format
        currency_symbol: Currency symbol
        decimals: Number of decimal places
        
    Returns:
        Formatted currency string
    """
    try:
        if pd.isna(value) or value is None:
            return ""
        
        if isinstance(value, (int, float)):
            return f"{currency_symbol}{value:,.{decimals}f}"
        
        return str(value)
    except:
        return str(value)