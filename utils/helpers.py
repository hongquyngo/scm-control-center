# utils/helpers.py - Simplified Helper Functions

import pandas as pd
import streamlit as st
from io import BytesIO
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple

# === EXCEL EXPORT FUNCTIONS ===

def convert_df_to_excel(df: pd.DataFrame, sheet_name: str = "Data") -> bytes:
    """
    Convert dataframe to Excel bytes with auto-formatting
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
            max_len = df[col].astype(str).map(len).max()
            max_len = max(max_len, len(str(col))) + 2
            worksheet.set_column(i, i, min(max_len, 50))
    
    return output.getvalue()

def export_multiple_sheets(dataframes_dict: Dict[str, pd.DataFrame]) -> bytes:
    """
    Export multiple dataframes to different sheets in one Excel file
    """
    output = BytesIO()
    
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        for sheet_name, df in dataframes_dict.items():
            df.to_excel(writer, index=False, sheet_name=sheet_name[:31])
            
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

def convert_to_period(date_series: pd.Series, period_type: str = "Weekly") -> pd.Series:
    """
    Convert datetime series to period strings
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

def parse_week_period(period_str: str) -> Tuple[int, int]:
    """Parse week period string for sorting"""
    try:
        if pd.isna(period_str) or not period_str:
            return (9999, 99)
            
        period_str = str(period_str).strip()
        parts = period_str.split(" - ")
        
        if len(parts) == 2:
            week_part = parts[0].strip()
            year_part = parts[1].strip()
            
            if week_part.startswith("Week "):
                week_str = week_part[5:].strip()
                week = int(week_str)
                year = int(year_part)
                return (year, week)
    except Exception:
        pass
    
    return (9999, 99)

def parse_month_period(period_str: str) -> pd.Timestamp:
    """Parse month period string for sorting"""
    try:
        return pd.to_datetime("01 " + str(period_str), format="%d %b %Y")
    except:
        return pd.Timestamp.max

def sort_period_columns(df: pd.DataFrame, period_type: str, 
                       info_cols: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Sort dataframe columns by period
    """
    if info_cols is None:
        # Auto-detect info columns
        info_cols = [col for col in df.columns if not any(
            x in str(col) for x in ['Week', '2024', '2025', 'Jan', 'Feb', 'Mar', 
                                    'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 
                                    'Oct', 'Nov', 'Dec']
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

def is_past_period(period_str: str, period_type: str) -> bool:
    """Check if a period string represents a past period"""
    today = datetime.now()
    
    try:
        if period_type == "Daily":
            period_date = pd.to_datetime(period_str)
            return period_date.date() < today.date()
            
        elif period_type == "Weekly":
            if "Week" in str(period_str):
                parts = str(period_str).split(" - ")
                if len(parts) == 2:
                    week_num = int(parts[0].replace("Week ", "").strip())
                    year = int(parts[1].strip())
                    
                    # Calculate the last day of that week
                    jan1 = datetime(year, 1, 1)
                    days_to_sunday = (6 - jan1.weekday()) % 7
                    first_sunday = jan1 + timedelta(days=days_to_sunday)
                    
                    if jan1.weekday() <= 3:
                        target_sunday = first_sunday + timedelta(weeks=week_num - 1)
                    else:
                        target_sunday = first_sunday + timedelta(weeks=week_num - 2)
                    
                    return target_sunday.date() < today.date()
                    
        elif period_type == "Monthly":
            period_date = pd.to_datetime(period_str, format='%b %Y')
            next_month = period_date + pd.DateOffset(months=1)
            return next_month.date() <= today.date()
            
    except Exception:
        pass
    
    return False

# === SESSION STATE HELPERS ===

def save_to_session_state(key: str, value: Any):
    """Save value to session state with timestamp"""
    st.session_state[key] = value
    st.session_state[f"{key}_timestamp"] = datetime.now()

def get_from_session_state(key: str, default: Any = None) -> Any:
    """Get value from session state"""
    return st.session_state.get(key, default)

def clear_session_state_pattern(pattern: str):
    """Clear session state keys matching pattern"""
    keys_to_clear = [key for key in st.session_state.keys() if pattern in key]
    for key in keys_to_clear:
        del st.session_state[key]

# === METRIC CALCULATION FUNCTIONS ===

def calculate_fulfillment_rate(available: float, demand: float) -> float:
    """Calculate fulfillment rate percentage"""
    if demand == 0:
        return 100.0
    return min(100.0, (available / demand) * 100)

def calculate_days_of_supply(inventory: float, daily_demand: float) -> float:
    """Calculate days of supply"""
    if daily_demand == 0:
        return float('inf')
    return inventory / daily_demand

def calculate_working_days(start_date: datetime, end_date: datetime, 
                         working_days_per_week: int = 5) -> int:
    """Calculate number of working days between two dates"""
    if pd.isna(start_date) or pd.isna(end_date):
        return 0
        
    days = (end_date - start_date).days
    if working_days_per_week == 7:
        return days
    
    # Simple calculation
    weeks = days // 7
    remaining_days = days % 7
    
    working_days = weeks * working_days_per_week
    for i in range(remaining_days):
        day = start_date + timedelta(days=i)
        if day.weekday() < working_days_per_week:
            working_days += 1
    
    return max(0, working_days)

# === NOTIFICATION HELPERS ===

def show_success_message(message: str, duration: int = 3):
    """Show success message that auto-disappears"""
    placeholder = st.empty()
    placeholder.success(message)
    import time
    time.sleep(duration)
    placeholder.empty()

def create_download_button(df: pd.DataFrame, filename: str, 
                         button_label: str = "📥 Download Excel") -> None:
    """Create a download button for dataframe"""
    excel_data = convert_df_to_excel(df)
    
    st.download_button(
        label=button_label,
        data=excel_data,
        file_name=f"{filename}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

# === ANALYSIS FUNCTIONS ===

def aggregate_by_period(df: pd.DataFrame, date_column: str, 
                       value_columns: List[str], period_type: str = "Weekly", 
                       agg_func: str = "sum") -> pd.DataFrame:
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

def calculate_safety_metrics(inventory_df: pd.DataFrame, demand_df: pd.DataFrame, 
                           product_col: str = 'pt_code', 
                           inventory_col: str = 'remaining_quantity', 
                           demand_col: str = 'pending_standard_delivery_quantity') -> pd.DataFrame:
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

def create_period_comparison(current_period_df: pd.DataFrame, 
                           previous_period_df: pd.DataFrame, 
                           metric_columns: List[str]) -> Dict[str, Dict[str, Any]]:
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

def create_summary_stats(df: pd.DataFrame, numeric_columns: List[str]) -> pd.DataFrame:
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

def create_alert_summary(alerts_list: List[Dict[str, Any]]) -> Dict[str, int]:
    """Create a summary of alerts by category"""
    summary = {
        'critical': len([a for a in alerts_list if a.get('level') == 'critical']),
        'warning': len([a for a in alerts_list if a.get('level') == 'warning']),
        'info': len([a for a in alerts_list if a.get('level') == 'info'])
    }
    
    summary['total'] = sum(summary.values())
    return summary