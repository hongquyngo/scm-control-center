# utils/date_mode_component.py

import streamlit as st
import pandas as pd
from typing import Dict, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class DateModeComponent:
    """Shared component for date mode selection across pages"""
    
    @staticmethod
    def render_date_mode_selector(key_prefix: str = "") -> bool:
        """
        Render date mode selector and return whether to use adjusted dates
        
        Args:
            key_prefix: Prefix for session state keys to avoid conflicts
            
        Returns:
            bool: True if using adjusted dates, False for original dates
        """
        try:
            # Initialize session state with consistent key
            session_key = f'{key_prefix}date_mode'
            if session_key not in st.session_state:
                st.session_state[session_key] = "Adjusted Dates"
            
            # Create columns for layout
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                # Ensure consistent values
                current_value = st.session_state.get(session_key, "Adjusted Dates")
                if current_value not in ["Adjusted Dates", "Original Dates"]:
                    current_value = "Adjusted Dates"
                    st.session_state[session_key] = current_value
                
                date_mode = st.radio(
                    "📅 Date Analysis Mode",
                    ["Adjusted Dates", "Original Dates"],
                    index=0 if current_value == "Adjusted Dates" else 1,
                    horizontal=True,
                    key=f"{key_prefix}date_mode_radio",
                    help="Choose whether to analyze using time-adjusted dates or original dates"
                )
                
                # Update session state
                st.session_state[session_key] = date_mode
            
            with col2:
                # Get adjustment status with error handling
                try:
                    from utils.adjustments.time_adjustment_integration import TimeAdjustmentIntegration
                    integration = TimeAdjustmentIntegration()
                    status = integration.get_adjustment_status()
                except ImportError:
                    logger.warning("TimeAdjustmentIntegration not available")
                    status = {'mode': 'none', 'rules_count': 0}
                except Exception as e:
                    logger.error(f"Error getting adjustment status: {str(e)}")
                    status = {'mode': 'none', 'rules_count': 0}
                
                if date_mode == "Adjusted Dates":
                    if status.get('mode') == 'advanced' and status.get('rules_count', 0) > 0:
                        st.success(f"✅ {status['rules_count']} rules active")
                        
                        # Tooltip with details
                        if status.get('affected_records', 0) > 0:
                            st.caption(f"{status['affected_records']:,} records adjusted")
                    elif status.get('mode') == 'simple':
                        st.info("📌 Simple adjustments")
                    else:
                        st.warning("⚠️ No adjustments configured")
                else:
                    st.info("📊 Using original dates")
            
            with col3:
                # Link to adjustment settings
                if st.button("⚙️ Configure", key=f"{key_prefix}config_btn"):
                    st.switch_page("pages/6_⚙️_Data_Adjustment_Settings.py")
            
            return date_mode == "Adjusted Dates"
            
        except Exception as e:
            logger.error(f"Error in date mode selector: {str(e)}")
            # Return default value on error
            return True
    
    @staticmethod
    def get_date_column_for_display(df: pd.DataFrame, base_column: str, use_adjusted: bool) -> str:
        """
        Get the appropriate date column based on mode
        
        Args:
            df: DataFrame to check
            base_column: Base date column name (e.g., 'etd')
            use_adjusted: Whether to use adjusted dates
            
        Returns:
            str: Column name to use
        """
        if df.empty:
            return base_column
        
        if use_adjusted:
            # Check if adjusted column exists
            adjusted_col = f"{base_column}_adjusted"
            if adjusted_col in df.columns:
                logger.debug(f"Using adjusted column: {adjusted_col}")
                return adjusted_col
        else:
            # Check if original column exists
            original_col = f"{base_column}_original"
            if original_col in df.columns:
                logger.debug(f"Using original column: {original_col}")
                return original_col
        
        # Fallback to base column
        logger.debug(f"Using base column: {base_column}")
        return base_column
    

    @staticmethod
    def show_adjustment_summary(df: pd.DataFrame, date_columns: list, page_name: str):
        """Show summary of adjustments applied to the data"""
        
        if df.empty:
            return
        
        adjustments_found = False
        adjustment_info = []
        
        for base_col in date_columns:
            try:
                original_col = f"{base_col}_original"
                adjusted_col = f"{base_col}_adjusted"
                
                if original_col in df.columns and adjusted_col in df.columns:
                    # Create working copy to avoid modifying original
                    work_df = df[[original_col, adjusted_col]].copy()
                    
                    # Ensure datetime conversion with error handling
                    work_df[original_col] = pd.to_datetime(work_df[original_col], errors='coerce')
                    work_df[adjusted_col] = pd.to_datetime(work_df[adjusted_col], errors='coerce')
                    
                    # Identify different types of adjustments
                    original_missing = work_df[original_col].isna()
                    adjusted_missing = work_df[adjusted_col].isna()
                    
                    # Case 1: Missing → Date (Absolute date applied to missing)
                    missing_to_date = original_missing & ~adjusted_missing
                    
                    # Case 2: Date → Missing (shouldn't happen, but check)
                    date_to_missing = ~original_missing & adjusted_missing
                    
                    # Case 3: Date → Different Date (normal adjustment)
                    both_have_dates = ~original_missing & ~adjusted_missing
                    work_df['_temp_diff'] = (work_df[adjusted_col] - work_df[original_col]).dt.days
                    date_adjusted = both_have_dates & (work_df['_temp_diff'] != 0)
                    
                    # Combine all adjustment cases
                    all_adjusted = missing_to_date | date_to_missing | date_adjusted
                    adjusted_records = all_adjusted.sum()
                    
                    if adjusted_records > 0:
                        adjustments_found = True
                        
                        # Calculate statistics for different cases
                        stats = {
                            'column': base_col,
                            'records_adjusted': adjusted_records,
                            'missing_to_date': missing_to_date.sum(),
                            'date_to_missing': date_to_missing.sum(),
                            'date_adjusted': date_adjusted.sum()
                        }
                        
                        # For date adjustments, calculate min/max/avg
                        if date_adjusted.sum() > 0:
                            adjusted_diffs = work_df.loc[date_adjusted, '_temp_diff']
                            stats.update({
                                'min_adjustment': int(adjusted_diffs.min()),
                                'max_adjustment': int(adjusted_diffs.max()),
                                'avg_adjustment': float(adjusted_diffs.mean())
                            })
                        else:
                            stats.update({
                                'min_adjustment': None,
                                'max_adjustment': None,
                                'avg_adjustment': None
                            })
                        
                        adjustment_info.append(stats)
                        
            except Exception as e:
                logger.error(f"Error calculating adjustment summary for {base_col}: {str(e)}")
                continue
        
        # Display enhanced summary
        if adjustments_found:
            with st.expander("📊 Adjustment Summary", expanded=False):
                for info in adjustment_info:
                    st.markdown(f"**{info['column'].upper()} Adjustments:**")
                    
                    # Overall stats
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Total Adjusted", f"{info['records_adjusted']:,}")
                    
                    with col2:
                        if info['missing_to_date'] > 0:
                            st.metric("Missing → Date", f"{info['missing_to_date']:,}")
                    
                    with col3:
                        if info['date_to_missing'] > 0:
                            st.metric("Date → Missing", f"{info['date_to_missing']:,}")
                    
                    with col4:
                        if info['date_adjusted'] > 0:
                            st.metric("Date Shifted", f"{info['date_adjusted']:,}")
                    
                    # Detailed stats for date adjustments
                    if info['min_adjustment'] is not None:
                        st.markdown("**Date Shift Details:**")
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Min Shift", f"{info['min_adjustment']:+d} days")
                        
                        with col2:
                            st.metric("Max Shift", f"{info['max_adjustment']:+d} days")
                        
                        with col3:
                            st.metric("Avg Shift", f"{info['avg_adjustment']:+.1f} days")
                    
                    # Special note for absolute date adjustments
                    if info['missing_to_date'] > 0:
                        st.info(f"ℹ️ {info['missing_to_date']} missing dates were set to specific dates (absolute date adjustment)")

    @staticmethod
    def update_date_range_filter(df: pd.DataFrame, date_column: str, 
                               use_adjusted: bool, key_prefix: str) -> Tuple[Any, Any]:
        """
        Update date range filter based on date mode
        
        Returns:
            tuple: (start_date, end_date)
        """
        if df.empty:
            return pd.Timestamp.now().date(), pd.Timestamp.now().date()
        
        try:
            # Get the appropriate column
            display_column = DateModeComponent.get_date_column_for_display(
                df, date_column, use_adjusted
            )
            
            # Validate column exists
            if display_column not in df.columns:
                logger.warning(f"Date column '{display_column}' not found")
                return pd.Timestamp.now().date(), pd.Timestamp.now().date()
            
            # Get date range with safe conversion
            dates = pd.to_datetime(df[display_column], errors='coerce').dropna()
            
            if len(dates) > 0:
                min_date = dates.min()
                max_date = dates.max()
            else:
                logger.warning(f"No valid dates found in column '{display_column}'")
                min_date = max_date = pd.Timestamp.now()
            
            # Create date inputs
            col1, col2 = st.columns(2)
            
            with col1:
                start_date = st.date_input(
                    f"From Date ({display_column.replace('_', ' ').title()})",
                    value=min_date.date(),
                    key=f"{key_prefix}start_date_{display_column}"
                )
            
            with col2:
                end_date = st.date_input(
                    f"To Date ({display_column.replace('_', ' ').title()})",
                    value=max_date.date(),
                    key=f"{key_prefix}end_date_{display_column}"
                )
            
            # Validate date range
            if start_date > end_date:
                st.error("Start date cannot be after end date!")
                # Swap dates
                start_date, end_date = end_date, start_date
            
            return start_date, end_date
            
        except Exception as e:
            logger.error(f"Error in date range filter: {str(e)}")
            return pd.Timestamp.now().date(), pd.Timestamp.now().date()
    
    @staticmethod
    def validate_date_columns(df: pd.DataFrame, base_columns: list) -> Dict[str, bool]:
        """
        Validate which date columns exist in the dataframe
        
        Args:
            df: DataFrame to check
            base_columns: List of base date column names
            
        Returns:
            Dict mapping column types to existence status
        """
        validation = {}
        
        for base_col in base_columns:
            validation[f"{base_col}"] = base_col in df.columns
            validation[f"{base_col}_original"] = f"{base_col}_original" in df.columns
            validation[f"{base_col}_adjusted"] = f"{base_col}_adjusted" in df.columns
        
        return validation
    
    @staticmethod
    def get_date_columns_info(df: pd.DataFrame, base_column: str) -> Dict[str, Any]:
        """
        Get information about date columns availability and statistics
        
        Args:
            df: DataFrame to analyze
            base_column: Base date column name
            
        Returns:
            Dict with column information
        """
        info = {
            'base_exists': base_column in df.columns,
            'original_exists': f"{base_column}_original" in df.columns,
            'adjusted_exists': f"{base_column}_adjusted" in df.columns,
            'has_adjustments': False,
            'adjustment_stats': None
        }
        
        # Check if adjustments exist
        if info['original_exists'] and info['adjusted_exists']:
            try:
                original = pd.to_datetime(df[f"{base_column}_original"], errors='coerce')
                adjusted = pd.to_datetime(df[f"{base_column}_adjusted"], errors='coerce')
                
                # Calculate differences
                diff = (adjusted - original).dt.days
                adjusted_mask = diff.notna() & (diff != 0)
                
                if adjusted_mask.any():
                    info['has_adjustments'] = True
                    info['adjustment_stats'] = {
                        'count': adjusted_mask.sum(),
                        'min_days': int(diff[adjusted_mask].min()),
                        'max_days': int(diff[adjusted_mask].max()),
                        'avg_days': float(diff[adjusted_mask].mean())
                    }
                    
            except Exception as e:
                logger.error(f"Error calculating adjustment stats: {str(e)}")
        
        return info