# utils/display_components.py - Reusable Display Components

import streamlit as st
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any, Optional, Callable, Tuple, Union  
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
    

    # Update trong utils/display_components.py

    @staticmethod
    def show_data_quality_warnings(df: pd.DataFrame, date_columns: Union[str, List[str]], 
                                data_type: str = "") -> Tuple[int, int]:
        """Show data quality warnings for one or multiple date columns"""
        # Convert single column to list
        if isinstance(date_columns, str):
            date_columns = [date_columns]
        
        total_missing = 0
        total_past = 0
        
        # Calculate for all date columns
        for date_column in date_columns:
            if date_column in df.columns:
                missing = check_missing_dates(df, date_column)
                past = check_past_dates(df, date_column)
                total_missing += missing
                total_past += past
        
        # Display warnings
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if total_missing > 0:
                st.warning(f"⚠️ {data_type}: {total_missing} records with missing dates")
        
        with col2:
            if total_past > 0:
                st.error(f"🔴 {data_type}: {total_past} records with past dates")
        
        with col3:
            # Data quality score across all date columns
            required_cols = ['pt_code', 'product_name'] + date_columns
            quality_score = check_data_quality(df, required_cols)
            
            if quality_score >= 95:
                st.success(f"✅ Data Quality: {quality_score:.1f}%")
            elif quality_score >= 80:
                st.warning(f"⚠️ Data Quality: {quality_score:.1f}%")
            else:
                st.error(f"❌ Data Quality: {quality_score:.1f}%")
        
        return total_missing, total_past

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


# Phần thêm vào cuối file display_components.py hiện tại

    # === NEW ENHANCED METHODS ===
    
    @staticmethod
    def render_page_layout(
        page_config: Dict[str, Any],
        content_renderer: Callable,
        **kwargs
    ):
        """
        Standardized page layout template for all analysis pages
        
        Args:
            page_config: Dict with title, icon, prev_page, next_page
            content_renderer: Function to render main content
            **kwargs: Additional args to pass to content_renderer
        """
        # Page setup
        st.set_page_config(
            page_title=f"{page_config['title']} - SCM",
            page_icon=page_config['icon'],
            layout="wide"
        )
        
        # Debug mode toggle
        col_debug1, col_debug2 = st.columns([6, 1])
        with col_debug2:
            debug_mode = st.checkbox(
                "🐛 Debug Mode", 
                value=False, 
                key=f"{page_config.get('key_prefix', 'page')}_debug_mode"
            )
        
        if debug_mode:
            st.info("🐛 Debug Mode is ON - Additional information will be displayed")
        
        # Page header with navigation
        DisplayComponents.show_page_header(
            title=page_config['title'],
            icon=page_config['icon'],
            prev_page=page_config.get('prev_page'),
            next_page=page_config.get('next_page')
        )
        
        # Render main content
        content_renderer(debug_mode=debug_mode, **kwargs)
        
        # Footer
        st.markdown("---")
        st.caption(f"Last data refresh: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    @staticmethod
    def render_source_selector(
        options: List[str],
        default_index: int = 0,
        radio_label: str = "Select Source:",
        key: str = "source_selector",
        horizontal: bool = True,
        additional_options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Standardized source selector for all pages
        
        Returns:
            Dict with 'source' and any additional options
        """
        result = {}
        
        # Main source selection
        if len(options) > 1:
            source = st.radio(
                radio_label,
                options,
                index=default_index,
                horizontal=horizontal,
                key=key
            )
            result['source'] = source
        else:
            result['source'] = options[0]
            st.info(f"Source: {options[0]}")
        
        # Additional options (e.g., exclude_expired, include_converted)
        if additional_options:
            for opt_key, opt_config in additional_options.items():
                if opt_config['type'] == 'checkbox':
                    result[opt_key] = st.checkbox(
                        opt_config['label'],
                        value=opt_config.get('default', False),
                        key=opt_config.get('key', opt_key),
                        help=opt_config.get('help')
                    )
                elif opt_config['type'] == 'multiselect':
                    result[opt_key] = st.multiselect(
                        opt_config['label'],
                        options=opt_config['options'],
                        default=opt_config.get('default', []),
                        key=opt_config.get('key', opt_key),
                        help=opt_config.get('help')
                    )
        
        return result
    
    @staticmethod
    def render_display_options(
        page_name: str,
        show_period: bool = True,
        show_filters: List[str] = None,
        additional_options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Standardized display options section
        
        Args:
            page_name: For unique keys
            show_period: Whether to show period selector
            show_filters: List of filter checkboxes to show
            additional_options: Any page-specific options
        """
        st.markdown("### ⚙️ Display Options")
        
        options = {}
        cols = st.columns(4)
        col_idx = 0
        
        # Period selector
        if show_period:
            with cols[col_idx % 4]:
                options['period_type'] = st.selectbox(
                    "Group By Period",
                    ["Daily", "Weekly", "Monthly"],
                    index=1,
                    key=f"{page_name}_period_select"
                )
            col_idx += 1
        
        # Common filter checkboxes
        if show_filters:
            for filter_name in show_filters:
                with cols[col_idx % 4]:
                    if filter_name == "shortage_only":
                        options['show_shortage_only'] = st.checkbox(
                            "🔴 Show only shortages",
                            value=False,
                            key=f"{page_name}_shortage_only"
                        )
                    elif filter_name == "zero_demand":
                        options['exclude_zero_demand'] = st.checkbox(
                            "🚫 Exclude zero demand",
                            value=True,
                            key=f"{page_name}_exclude_zero"
                        )
                    elif filter_name == "missing_dates":
                        options['exclude_missing_dates'] = st.checkbox(
                            "📅 Exclude missing dates",
                            value=True,
                            key=f"{page_name}_exclude_missing"
                        )
                    elif filter_name == "nonzero":
                        options['show_only_nonzero'] = st.checkbox(
                            "Show only products with quantity > 0",
                            value=True,
                            key=f"{page_name}_show_nonzero"
                        )
                col_idx += 1
        
        # Additional page-specific options
        if additional_options:
            for opt_key, opt_config in additional_options.items():
                with cols[col_idx % 4]:
                    if opt_config['type'] == 'checkbox':
                        options[opt_key] = st.checkbox(
                            opt_config['label'],
                            value=opt_config.get('default', False),
                            key=f"{page_name}_{opt_key}"
                        )
                    elif opt_config['type'] == 'number_input':
                        options[opt_key] = st.number_input(
                            opt_config['label'],
                            min_value=opt_config.get('min', 0),
                            max_value=opt_config.get('max', 100),
                            value=opt_config.get('default', 0),
                            key=f"{page_name}_{opt_key}"
                        )
                col_idx += 1
        
        return options
    
    @staticmethod
    def render_data_loading_section(
        button_label: str = "🚀 Load Data",
        button_key: str = "load_data",
        show_spinner: bool = True,
        spinner_text: str = "Loading data..."
    ) -> bool:
        """
        Standardized data loading button
        
        Returns:
            bool: True if button clicked
        """
        return st.button(
            button_label,
            type="primary",
            use_container_width=True,
            key=button_key
        )
    
    @staticmethod
    def render_filter_option_radio(
        options: List[str],
        default: str = "Show All",
        key: str = "filter_option",
        help_text: Optional[str] = None
    ) -> str:
        """
        Standardized filter option radio buttons
        Used in detail tables for all pages
        """
        st.markdown("**Filter Options:**")
        return st.radio(
            "Select filter:",
            options=options,
            index=options.index(default) if default in options else 0,
            horizontal=True,
            key=key,
            help=help_text
        )
    
    @staticmethod
    def render_summary_section(
        metrics: List[Dict[str, Any]],
        title: str = "### 📊 Summary",
        cols_per_row: int = 3,
        additional_content: Optional[Callable] = None
    ):
        """
        Standardized summary section with metrics
        
        Args:
            metrics: List of metric configs for show_metric_card
            title: Section title
            cols_per_row: Number of columns per row
            additional_content: Optional function to render additional content
        """
        st.markdown(title)
        
        # Render metrics in rows
        for i in range(0, len(metrics), cols_per_row):
            cols = st.columns(cols_per_row)
            for j, metric in enumerate(metrics[i:i+cols_per_row]):
                if j < len(cols):
                    with cols[j]:
                        DisplayComponents.show_metric_card(**metric)
        
        # Additional content (e.g., warnings, status comparisons)
        if additional_content:
            additional_content()
    
    @staticmethod
    def render_detail_table_with_filter(
        df: pd.DataFrame,
        filter_options: List[str],
        filter_apply_func: Callable,
        format_func: Callable,
        style_func: Optional[Callable] = None,
        height: int = 600,
        key_prefix: str = "detail"
    ):
        """
        Standardized detail table with filter options
        
        Args:
            df: DataFrame to display
            filter_options: List of filter options for radio
            filter_apply_func: Function to apply selected filter
            format_func: Function to format dataframe
            style_func: Optional function to style rows
            height: Table height
            key_prefix: For unique keys
        """
        if df.empty:
            st.info("No data to display")
            return
        
        # Filter options
        filter_option = DisplayComponents.render_filter_option_radio(
            options=filter_options,
            key=f"{key_prefix}_filter_option"
        )
        
        # Apply filter
        display_df = filter_apply_func(df, filter_option)
        
        # Show filtered count
        if filter_option != "Show All":
            st.info(f"Showing {len(display_df):,} records matching filter: {filter_option}")
        
        if display_df.empty:
            return
        
        # Format dataframe
        display_df_formatted = format_func(display_df)
        
        # Apply styling if provided
        if style_func:
            styled_df = display_df_formatted.style.apply(style_func, axis=1)
            st.dataframe(styled_df, use_container_width=True, height=height)
        else:
            st.dataframe(display_df_formatted, use_container_width=True, height=height)
    
    @staticmethod
    def render_pivot_section(
        df: pd.DataFrame,
        create_pivot_func: Callable,
        period_type: str,
        show_totals: bool = True,
        title: str = "### 📊 Pivot View",
        key_prefix: str = "pivot"
    ):
        """
        Standardized pivot view section
        
        Args:
            df: Source dataframe
            create_pivot_func: Function to create pivot
            period_type: Period type for grouping
            show_totals: Whether to show totals
            title: Section title
            key_prefix: For unique keys
        """
        st.markdown(title)
        
        if df.empty:
            st.info("No data to display")
            return
        
        # Create pivot
        pivot_df = create_pivot_func(df, period_type)
        
        if pivot_df.empty:
            st.info("No data to display after grouping")
            return
        
        # Show legend for past periods
        st.info("🔴 = Past period (already occurred)")
        
        # Display pivot
        st.dataframe(pivot_df, use_container_width=True, height=400)
        
        # Show totals if requested
        if show_totals:
            DisplayComponents.show_period_totals(df, period_type, key_prefix)
    
    @staticmethod
    def show_period_totals(
        df: pd.DataFrame,
        period_type: str,
        key_prefix: str,
        value_columns: List[str] = None
    ):
        """
        Show period totals with indicators
        Standardized for all pages
        """
        st.markdown("#### 🔢 Column Totals")
        
        # Implementation based on common pattern from all pages
        # This would need the actual implementation from the pages
        st.info("Period totals implementation needed")
    
    @staticmethod
    def render_export_section(
        export_configs: List[Dict[str, Any]],
        title: str = "### 📤 Export Options"
    ):
        """
        Standardized export section
        
        Args:
            export_configs: List of dicts with 'data', 'filename', 'label'
        """
        st.markdown(title)
        
        cols = st.columns(len(export_configs))
        for idx, (col, config) in enumerate(zip(cols, export_configs)):
            with col:
                if config.get('data') is not None and not config['data'].empty:
                    DisplayComponents.show_export_button(
                        df=config['data'],
                        filename=config['filename'],
                        button_label=config.get('label', '📥 Download')
                    )
    
    @staticmethod
    def render_action_section(
        actions: List[Dict[str, Any]],
        title: str = "### 🎯 Actions"
    ):
        """
        Standardized action buttons section
        
        Args:
            actions: List of action configs for show_action_buttons
        """
        st.markdown("---")
        st.markdown(title)
        DisplayComponents.show_action_buttons(actions)
    
    @staticmethod
    def show_no_data_message(
        message: str = "No data available.",
        suggestion: Optional[str] = None
    ):
        """Show standardized no data message"""
        st.info(f"ℹ️ {message}")
        if suggestion:
            st.caption(f"💡 {suggestion}")
    
    @staticmethod
    def show_data_loading_spinner(
        func: Callable,  # Function first (no default)
        message: str = "Loading data...",
        *args,
        **kwargs
    ) -> Any:
        """
        Show loading spinner while executing function
        
        Args:
            func: Function to execute
            message: Loading message to display
            *args: Positional arguments for func
            **kwargs: Keyword arguments for func
        
        Returns:
            Result from func
        """
        with st.spinner(message):
            return func(*args, **kwargs)
    
    @staticmethod
    def create_consistent_tabs(
        tab_configs: List[Dict[str, Any]],
        key_prefix: str = "tabs"
    ) -> List[Any]:
        """
        Create consistent tabs across pages
        
        Args:
            tab_configs: List of dicts with 'label', 'data', 'renderer'
            key_prefix: Prefix for tab keys
            
        Returns:
            List of tab objects (empty if single tab)
        """
        if not tab_configs:
            return []
            
        if len(tab_configs) == 1:
            # Single tab, no need for tabs
            config = tab_configs[0]
            if 'renderer' in config and callable(config['renderer']):
                config['renderer'](config.get('data'))
            return []
        
        # Multiple tabs
        tab_labels = [config['label'] for config in tab_configs]
        tabs = st.tabs(tab_labels)
        
        for tab, config in zip(tabs, tab_configs):
            with tab:
                if 'renderer' in config and callable(config['renderer']):
                    config['renderer'](config.get('data'))
        
        return tabs