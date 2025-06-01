# utils/display_components.py - Reusable Display Components

import streamlit as st
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any, Optional, Callable, Tuple
from .formatters import (
    format_number, format_currency, format_percentage, 
    check_missing_dates, check_past_dates, check_data_quality
)
from .helpers import convert_df_to_excel

class DisplayComponents:
    """Reusable display components for all pages"""
    
    @staticmethod
    def show_page_header(title: str, icon: str, 
                        prev_page: Optional[str] = None, 
                        next_page: Optional[str] = None):
        """Show standardized page header with navigation"""
        col1, col2, col3 = st.columns([1, 4, 1])
        
        with col1:
            if prev_page:
                page_name = prev_page.split('_')[-1].replace('.py', '')
                if st.button(f"← {page_name}"):
                    st.switch_page(prev_page)
        
        with col2:
            st.title(f"{icon} {title}")
        
        with col3:
            if next_page:
                page_name = next_page.split('_')[-1].replace('.py', '')
                if st.button(f"{page_name} →"):
                    st.switch_page(next_page)
        
        # Dashboard button
        if st.button("🏠 Dashboard", use_container_width=False):
            st.switch_page("main.py")
        
        st.markdown("---")
    
    @staticmethod
    def show_data_quality_warnings(df: pd.DataFrame, date_column: str, 
                                 data_type: str = "") -> Tuple[int, int]:
        """Show data quality warnings"""
        col1, col2, col3 = st.columns(3)
        
        with col1:
            missing_dates = check_missing_dates(df, date_column)
            if missing_dates > 0:
                st.warning(f"⚠️ {data_type}: {missing_dates} records with missing {date_column}")
        
        with col2:
            past_dates = check_past_dates(df, date_column)
            if past_dates > 0:
                st.error(f"🔴 {data_type}: {past_dates} records with past {date_column}")
        
        with col3:
            # Data quality score
            required_cols = ['pt_code', 'product_name', date_column]
            quality_score = check_data_quality(df, required_cols)
            
            if quality_score >= 95:
                st.success(f"✅ Data Quality: {quality_score:.1f}%")
            elif quality_score >= 80:
                st.warning(f"⚠️ Data Quality: {quality_score:.1f}%")
            else:
                st.error(f"❌ Data Quality: {quality_score:.1f}%")
        
        return missing_dates, past_dates
    
    @staticmethod
    def show_metric_card(title: str, value: Any, delta: Any = None, 
                        help_text: Optional[str] = None, 
                        format_type: str = "number",
                        delta_color: str = "normal"):
        """Show formatted metric card"""
        format_functions = {
            "currency": lambda v: format_currency(v, "USD"),
            "percentage": format_percentage,
            "number": format_number
        }
        
        formatter = format_functions.get(format_type, format_number)
        display_value = formatter(value)
        
        st.metric(
            label=title,
            value=display_value,
            delta=delta,
            delta_color=delta_color,
            help=help_text
        )
    
    @staticmethod
    def show_summary_metrics(metrics: List[Dict[str, Any]], cols: int = 4):
        """Show summary metrics in columns"""
        columns = st.columns(cols)
        
        for idx, metric in enumerate(metrics):
            col_idx = idx % cols
            with columns[col_idx]:
                DisplayComponents.show_metric_card(**metric)
    
    @staticmethod
    def show_dataframe_with_styling(df: pd.DataFrame, 
                                  style_function: Optional[Callable] = None,
                                  height: int = 400,
                                  use_container_width: bool = True):
        """Show dataframe with optional styling"""
        if style_function and not df.empty:
            styled_df = df.style.apply(style_function, axis=1)
            st.dataframe(styled_df, use_container_width=use_container_width, height=height)
        else:
            st.dataframe(df, use_container_width=use_container_width, height=height)
    
    @staticmethod
    def show_alerts_panel(alerts: List[Dict[str, Any]], 
                         warnings: List[Dict[str, Any]]):
        """Show alerts and warnings panel"""
        if alerts:
            st.markdown("### 🚨 Critical Alerts")
            for alert in alerts:
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.error(f"{alert['icon']} {alert['message']}")
                with col2:
                    if alert.get('value'):
                        st.metric("Impact", alert['value'], label_visibility="collapsed")
        
        if warnings:
            st.markdown("### ⚠️ Warnings")
            for warning in warnings:
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.warning(f"{warning['icon']} {warning['message']}")
                with col2:
                    if warning.get('value'):
                        st.metric("Impact", warning['value'], label_visibility="collapsed")
    
    @staticmethod
    def show_export_button(df: pd.DataFrame, filename: str, 
                         button_label: str = "📥 Download Excel"):
        """Show export button for dataframe"""
        excel_data = convert_df_to_excel(df)
        st.download_button(
            label=button_label,
            data=excel_data,
            file_name=f"{filename}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    
    @staticmethod
    def show_period_selector(key: str = "period", 
                           default_index: int = 1) -> str:
        """Show period type selector"""
        PERIOD_TYPES = ["Daily", "Weekly", "Monthly"]
        return st.selectbox(
            "Group By Period", 
            PERIOD_TYPES, 
            index=default_index,
            key=key
        )
    
    @staticmethod
    def show_help_section(title: str, content: str):
        """Show expandable help section"""
        with st.expander(f"ℹ️ {title}", expanded=False):
            st.markdown(content)
    
    @staticmethod
    def show_action_buttons(actions: List[Dict[str, Any]]):
        """Show action buttons"""
        cols = st.columns(len(actions))
        
        for idx, action in enumerate(actions):
            with cols[idx]:
                button_type = action.get("type", "secondary")
                if st.button(
                    action["label"], 
                    type=button_type, 
                    use_container_width=True,
                    key=action.get("key")
                ):
                    if action.get("callback"):
                        action["callback"]()
                    elif action.get("page"):
                        st.switch_page(action["page"])
    
    @staticmethod
    def show_tabs_with_data(tabs_data: Dict[str, pd.DataFrame], 
                          display_function: Callable):
        """Show tabs with different dataframes"""
        if len(tabs_data) > 1:
            tabs = st.tabs(list(tabs_data.keys()))
            
            for idx, (tab_name, df) in enumerate(tabs_data.items()):
                with tabs[idx]:
                    display_function(df, tab_name)
        else:
            # Single tab, show directly
            for tab_name, df in tabs_data.items():
                display_function(df, tab_name)
    
    @staticmethod
    def show_debug_info(info: Dict[str, Any]):
        """Show debug information panel"""
        if st.session_state.get('debug_mode', False):
            with st.expander("🐛 Debug Information", expanded=True):
                for key, value in info.items():
                    st.write(f"**{key}:** {value}")
    
    @staticmethod
    def show_demand_summary_box(df: pd.DataFrame):
        """Show demand summary metrics box"""
        metrics = [
            {
                "title": "Total Products",
                "value": df["pt_code"].nunique(),
                "format_type": "number"
            },
            {
                "title": "Total Value",
                "value": df["value_in_usd"].sum(),
                "format_type": "currency"
            },
            {
                "title": "Missing ETD",
                "value": df["etd"].isna().sum(),
                "format_type": "number",
                "delta": "Records" if df["etd"].isna().sum() > 0 else None
            },
            {
                "title": "Past ETD",
                "value": len(df[df["etd"] < pd.Timestamp.now()]),
                "format_type": "number",
                "delta": "Overdue" if len(df[df["etd"] < pd.Timestamp.now()]) > 0 else None
            }
        ]
        
        DisplayComponents.show_summary_metrics(metrics)
    
    @staticmethod
    def show_supply_summary_box(df: pd.DataFrame):
        """Show supply summary metrics box"""
        # Group by source type
        source_summary = df.groupby('source_type').agg({
            'quantity': 'sum',
            'value_in_usd': 'sum',
            'pt_code': 'nunique'
        }).reset_index()
        
        # Overall metrics
        st.markdown("#### 📊 Overall Supply")
        overall_metrics = [
            {
                "title": "Total Products",
                "value": df["pt_code"].nunique(),
                "format_type": "number"
            },
            {
                "title": "Total Quantity",
                "value": df["quantity"].sum(),
                "format_type": "number"
            },
            {
                "title": "Total Value",
                "value": df["value_in_usd"].sum(),
                "format_type": "currency"
            },
            {
                "title": "Missing Dates",
                "value": df["date_ref"].isna().sum(),
                "format_type": "number"
            }
        ]
        DisplayComponents.show_summary_metrics(overall_metrics)
        
        # Source breakdown
        st.markdown("#### 📦 Supply by Source")
        source_cols = st.columns(len(source_summary))
        
        for idx, (col, row) in enumerate(zip(source_cols, source_summary.itertuples())):
            with col:
                st.markdown(f"**{row.source_type}**")
                st.metric("Products", f"{row.pt_code:,}", label_visibility="collapsed")
                st.metric("Quantity", format_number(row.quantity), label_visibility="collapsed")
                st.metric("Value", format_currency(row.value_in_usd, "USD", 0), label_visibility="collapsed")