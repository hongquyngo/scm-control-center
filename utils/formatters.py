# utils/formatters.py - Consolidated Formatting & Validation

import pandas as pd
from datetime import datetime
from typing import Any, Union, List, Tuple, Optional
import random

# === FORMATTING FUNCTIONS ===

def format_number(value: Union[int, float], decimal_places: int = 0, 
                 prefix: str = "", suffix: str = "") -> str:
    """Format number with thousands separator"""
    if pd.isna(value):
        return ""
    formatted = f"{value:,.{decimal_places}f}"
    return f"{prefix}{formatted}{suffix}"

def format_currency(value: Union[int, float], currency: str = "USD", 
                   decimal_places: int = 2) -> str:
    """Format currency value"""
    if pd.isna(value):
        return ""
    
    currency_formats = {
        "USD": lambda v: format_number(v, decimal_places, prefix="$"),
        "VND": lambda v: format_number(v, 0, suffix=" VND"),
    }
    
    formatter = currency_formats.get(currency, 
                                   lambda v: format_number(v, decimal_places, suffix=f" {currency}"))
    return formatter(value)

def format_percentage(value: Union[int, float], decimal_places: int = 1) -> str:
    """Format percentage value"""
    if pd.isna(value):
        return ""
    return f"{value:.{decimal_places}f}%"

def format_date(date_value: Any, format_str: str = "%Y-%m-%d") -> str:
    """Format date value"""
    if pd.isna(date_value):
        return ""
    if isinstance(date_value, str):
        return date_value
    return pd.to_datetime(date_value).strftime(format_str)

def format_timestamp(timestamp: Any, format_str: str = "%Y-%m-%d %H:%M:%S") -> str:
    """Format timestamp for display"""
    if isinstance(timestamp, str):
        return timestamp
    elif isinstance(timestamp, datetime):
        return timestamp.strftime(format_str)
    elif pd.notna(timestamp):
        return pd.to_datetime(timestamp).strftime(format_str)
    else:
        return "N/A"

def format_quantity_with_uom(quantity: Union[int, float], uom: str = "") -> str:
    """Format quantity with unit of measure"""
    formatted_qty = format_number(quantity)
    return f"{formatted_qty} {uom}".strip() if uom else formatted_qty

def format_days(days: Union[int, float]) -> str:
    """Format days with appropriate label"""
    if pd.isna(days):
        return ""
    days_int = int(days)
    return f"{days_int} day{'s' if days_int != 1 else ''}"

# === VALIDATION FUNCTIONS ===

def validate_dataframe_columns(df: pd.DataFrame, 
                             required_columns: List[str]) -> Tuple[bool, List[str]]:
    """Validate required columns exist in dataframe"""
    missing_columns = [col for col in required_columns if col not in df.columns]
    is_valid = len(missing_columns) == 0
    return is_valid, missing_columns

def validate_date_range(start_date: datetime, end_date: datetime) -> bool:
    """Validate date range"""
    return start_date <= end_date

def validate_numeric_input(value: Any, min_value: float = None, 
                         max_value: float = None) -> Tuple[bool, Optional[str]]:
    """Validate numeric input"""
    try:
        numeric_value = float(value)
        
        if min_value is not None and numeric_value < min_value:
            return False, f"Value must be >= {min_value}"
        
        if max_value is not None and numeric_value > max_value:
            return False, f"Value must be <= {max_value}"
        
        return True, None
    except (ValueError, TypeError):
        return False, "Invalid numeric value"

def validate_product_code(pt_code: str) -> bool:
    """Validate product code format"""
    if not pt_code or pt_code.strip() == "" or pt_code.lower() == "nan":
        return False
    return True

def validate_quantity_columns(df: pd.DataFrame, 
                            quantity_columns: List[str]) -> pd.DataFrame:
    """Validate and clean quantity columns"""
    df = df.copy()
    for col in quantity_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    return df

# === DATA QUALITY FUNCTIONS ===

def check_missing_dates(df: pd.DataFrame, date_column: str) -> int:
    """Check for missing dates in dataframe"""
    if date_column not in df.columns:
        return 0
    return df[date_column].isna().sum()


def check_past_dates(df: pd.DataFrame, date_column: str) -> int:
    """Check for past dates in dataframe"""
    if date_column not in df.columns:
        return 0
    
    # Ensure date column is datetime
    df_copy = df.copy()
    df_copy[date_column] = pd.to_datetime(df_copy[date_column], errors='coerce')
    
    # Use normalized datetime for consistent comparison
    today = pd.Timestamp.now().normalize()
    
    # Count records with dates strictly before today
    past_mask = (df_copy[date_column] < today) & df_copy[date_column].notna()
    return past_mask.sum()



def check_data_quality(df: pd.DataFrame, 
                      required_columns: List[str]) -> float:
    """Calculate data quality score"""
    if df.empty or not required_columns:
        return 0.0
    
    total_records = len(df)
    missing_data = 0
    
    for col in required_columns:
        if col in df.columns:
            missing_data += df[col].isna().sum()
    
    quality_score = 100 * (1 - missing_data / (total_records * len(required_columns)))
    return quality_score

def detect_anomalies(df: pd.DataFrame, value_column: str, 
                    method: str = 'iqr', threshold: float = 1.5) -> pd.DataFrame:
    """Detect anomalies in data using IQR method"""
    df = df.copy()
    
    if method == 'iqr' and value_column in df.columns:
        Q1 = df[value_column].quantile(0.25)
        Q3 = df[value_column].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        
        df['is_anomaly'] = (
            (df[value_column] < lower_bound) | 
            (df[value_column] > upper_bound)
        )
    
    return df

# === STYLING FUNCTIONS ===

def highlight_negative_values(val: Any) -> str:
    """Style function for negative values"""
    if isinstance(val, (int, float)) and val < 0:
        return 'color: red; font-weight: bold'
    return ''

def highlight_shortage_rows(row: pd.Series, gap_column: str = 'gap_quantity') -> list:
    """Style function for shortage rows"""
    if row.get(gap_column, 0) < 0:
        return ['background-color: #ffcccc'] * len(row)
    return [''] * len(row)

def highlight_expiry_rows(row: pd.Series) -> list:
    """Style function for expiry status"""
    if "days_until_expiry" in row:
        days_str = str(row["days_until_expiry"])
        if days_str and "days" in days_str:
            try:
                days = int(days_str.split()[0])
                if days <= 7:
                    return ["background-color: #ffcccc"] * len(row)  # Red
                elif days <= 30:
                    return ["background-color: #ffe6cc"] * len(row)  # Orange
            except:
                pass
    return [""] * len(row)

def highlight_etd_issues(row: pd.Series) -> list:
    """Style function for ETD issues"""
    etd_value = str(row.get("etd", ""))
    
    if "❌ Missing" in etd_value:
        return ["background-color: #fff3cd"] * len(row)  # Yellow
    elif "🔴" in etd_value:
        return ["background-color: #f8d7da"] * len(row)  # Light red
    
    return [""] * len(row)

def apply_gradient_style(df: pd.DataFrame, columns: List[str], 
                        cmap: str = 'RdYlGn', axis: int = 1):
    """Apply gradient coloring to dataframe columns"""
    return df.style.background_gradient(cmap=cmap, subset=columns, axis=axis)

# Thêm vào file utils/formatters.py
def generate_allocation_number():
    """Generate unique allocation number with format: ALLOC-YYYYMMDD-XXXX"""

    date_part = datetime.now().strftime('%Y%m%d')
    random_part = str(random.randint(1000, 9999))
    return f"ALLOC-{date_part}-{random_part}"