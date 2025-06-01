import streamlit as st
import pandas as pd
from typing import Dict, List, Any, Optional
from utils.data_manager import DataManager
from datetime import datetime, date
import hashlib
import json


class PreviewManager:
    """Manage preview and filter data for adjustments"""
    
    def __init__(self):
        self.data_manager = DataManager()
        self._filter_cache = {}
        
        # Column mappings for different data sources
        self.entity_columns = {
            "OC": "legal_entity",
            "Forecast": "legal_entity", 
            "Inventory": "legal_entity",
            "Pending CAN": "consignee",
            "Pending PO": "legal_entity",
            "Pending WH Transfer": "owning_company_name"
        }
        
        self.number_columns = {
            "OC": "oc_number",
            "Forecast": "forecast_number", 
            "Inventory": "inventory_history_id",
            "Pending CAN": "arrival_note_number",
            "Pending PO": "po_number",
            "Pending WH Transfer": "warehouse_transfer_line_id"
        }
        
        self.date_columns = {
            "OC": "etd",
            "Forecast": "etd",
            "Inventory": None,
            "Pending CAN": "arrival_date", 
            "Pending PO": ["cargo_ready_date", "crd"],
            "Pending WH Transfer": "transfer_date"
        }
    
    # === Data Loading ===
    
    def _load_data_by_source(self, data_source: str) -> pd.DataFrame:
        """Load data based on source type"""
        try:
            # Use DataManager's existing cache
            if data_source == "OC":
                return self.data_manager.load_demand_oc()
            elif data_source == "Forecast":
                return self.data_manager.load_demand_forecast()
            elif data_source == "Inventory":
                return self.data_manager.load_inventory()
            elif data_source == "Pending CAN":
                return self.data_manager.load_pending_can()
            elif data_source == "Pending PO":
                return self.data_manager.load_pending_po()
            elif data_source == "Pending WH Transfer":
                return self.data_manager.load_pending_wh_transfer()
            else:
                return pd.DataFrame()
        except Exception as e:
            st.error(f"Error loading {data_source} data: {str(e)}")
            return pd.DataFrame()
    
    # === Filter Options ===
    
    def _create_cache_key(self, data_source: str, selections: Dict[str, List[str]] = None) -> str:
        """Create a unique cache key based on data source and selections"""
        key_data = {
            'data_source': data_source,
            'selections': selections or {}
        }
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    @st.cache_data(ttl=300, show_spinner=False)
    def get_filter_options(_self, data_source: str) -> Dict[str, List[str]]:
        """Get basic filter options for a data source"""
        df = _self._load_data_by_source(data_source)
        
        if df.empty:
            return {'entities': [], 'customers': [], 'products': [], 'numbers': [], 'brands': []}
        
        return _self._extract_options_from_df(df, data_source)
    
    def get_interactive_filter_options(self, data_source: str, 
                                     selected_entities: List[str] = None,
                                     selected_customers: List[str] = None, 
                                     selected_products: List[str] = None,
                                     selected_numbers: List[str] = None,
                                     selected_brands: List[str] = None) -> Dict[str, List[str]]:
        """Get filter options based on current selections (interactive filtering)"""
        try:
            # Create a proper cache key based on actual selections
            selections = {
                'entities': sorted(selected_entities or []),
                'customers': sorted(selected_customers or []),
                'products': sorted(selected_products or []),
                'numbers': sorted(selected_numbers or []),
                'brands': sorted(selected_brands or [])
            }
            
            cache_key = self._create_cache_key(data_source, selections)
            
            # Check in-memory cache first
            if cache_key in self._filter_cache:
                return self._filter_cache[cache_key]
            
            # If no filters selected, return all options
            if not any([selected_entities, selected_customers, selected_products, selected_numbers, selected_brands]):
                result = self.get_filter_options(data_source)
                self._filter_cache[cache_key] = result
                return result
            
            # Otherwise, compute filtered options
            df = self._load_data_by_source(data_source)
            
            if df.empty:
                return {'entities': [], 'customers': [], 'products': [], 'numbers': [], 'brands': []}
            
            # Apply current selections to filter dataframe
            filtered_df = self._apply_filters_to_df(df, data_source, {
                'entities': selected_entities,
                'customers': selected_customers,
                'products': selected_products,
                'numbers': selected_numbers,
                'brands': selected_brands
            })
            
            # Extract available options from filtered data
            result = self._extract_options_from_df(filtered_df, data_source)
            
            # Cache the result
            self._filter_cache[cache_key] = result
            
            # Limit cache size
            if len(self._filter_cache) > 100:
                # Remove oldest entries
                keys_to_remove = list(self._filter_cache.keys())[:50]
                for key in keys_to_remove:
                    del self._filter_cache[key]
            
            return result
            
        except Exception as e:
            st.error(f"Error getting interactive filter options: {str(e)}")
            return {'entities': [], 'customers': [], 'products': [], 'numbers': [], 'brands': []}
    
    def _apply_filters_to_df(self, df: pd.DataFrame, data_source: str, selections: Dict[str, List[str]]) -> pd.DataFrame:
        """Apply filter selections to dataframe"""
        filtered_df = df.copy()
        
        # Apply entity filter
        if selections.get('entities'):
            entity_col = self.entity_columns.get(data_source, "legal_entity")
            if entity_col in filtered_df.columns:
                filtered_df = filtered_df[filtered_df[entity_col].isin(selections['entities'])]
        
        # Apply customer filter (only for demand sources)
        if selections.get('customers') and data_source in ["OC", "Forecast"]:
            if 'customer' in filtered_df.columns:
                filtered_df = filtered_df[filtered_df['customer'].isin(selections['customers'])]
        
        # Apply product filter
        if selections.get('products'):
            pt_codes = [p.split(' - ')[0] for p in selections['products'] if ' - ' in p]
            if pt_codes and 'pt_code' in filtered_df.columns:
                filtered_df = filtered_df[filtered_df['pt_code'].isin(pt_codes)]
        
        # Apply number filter
        if selections.get('numbers'):
            number_col = self.number_columns.get(data_source, "id")
            if number_col in filtered_df.columns:
                filtered_df = filtered_df[filtered_df[number_col].astype(str).isin(selections['numbers'])]
        
        # Apply brand filter
        if selections.get('brands'):
            if 'brand' in filtered_df.columns:
                filtered_df = filtered_df[filtered_df['brand'].isin(selections['brands'])]
        
        return filtered_df
    
    def _extract_options_from_df(self, df: pd.DataFrame, data_source: str) -> Dict[str, List[str]]:
        """Extract filter options from dataframe - NO LIMITS"""
        options = {'entities': [], 'customers': [], 'products': [], 'numbers': [], 'brands': []}
        
        # Entities
        entity_col = self.entity_columns.get(data_source, "legal_entity")
        if entity_col in df.columns:
            entities = df[entity_col].dropna().unique()
            options['entities'] = sorted([str(e) for e in entities if str(e) != 'nan'])
        
        # Customers (only for demand sources)
        if data_source in ["OC", "Forecast"] and 'customer' in df.columns:
            customers = df['customer'].dropna().unique()
            options['customers'] = sorted([str(c) for c in customers if str(c) != 'nan'])
        
        # Products - NO LIMIT
        if 'pt_code' in df.columns and 'product_name' in df.columns:
            # Get unique products
            products_df = df[['pt_code', 'product_name']].drop_duplicates()
            products_df = products_df[
                products_df['pt_code'].notna() & 
                (products_df['pt_code'] != '') & 
                (products_df['pt_code'] != 'nan')
            ]
            
            # Sort by pt_code to get consistent ordering
            products_df = products_df.sort_values('pt_code')
            
            # No limit - get all products
            options['products'] = [
                f"{row['pt_code']} - {str(row['product_name'])[:50]}" 
                for _, row in products_df.iterrows()
            ]
        
        # Numbers - NO LIMIT
        number_col = self.number_columns.get(data_source, "id")
        if number_col in df.columns:
            numbers = df[number_col].dropna().unique()
            numbers = [str(n) for n in numbers if str(n) != 'nan']
            
            # Sort numbers naturally
            try:
                # Try to sort as integers if possible
                numbers = sorted(numbers, key=lambda x: (len(x), int(x) if x.isdigit() else x))
            except:
                numbers = sorted(numbers)
            
            # No limit - get all numbers
            options['numbers'] = numbers
        
        # Brands
        if 'brand' in df.columns:
            brands = df['brand'].dropna().unique()
            options['brands'] = sorted([str(b) for b in brands if str(b) != 'nan'])
        
        return options
    
    # === Rule Preview ===
    
    def preview_rule_impact(self, rule: Dict[str, Any]) -> Optional[pd.DataFrame]:
        """Preview the impact of a specific rule"""
        try:
            df = self._load_data_by_source(rule['data_source'])
            
            if df.empty:
                return None
            
            # Apply rule filters
            filtered_df = self._apply_rule_filters(df, rule)
            
            if filtered_df.empty:
                return None
            
            # Create preview summary
            adjustment_type = rule.get('adjustment_type', 'Relative (Days)')
            
            if adjustment_type == 'Absolute (Date)':
                absolute_date = rule.get('absolute_date', 'Unknown')
                preview_data = {
                    'Data Source': [rule['data_source']],
                    'Records Affected': [len(filtered_df)],
                    'Adjustment Type': ['Absolute Date'],
                    'Target Date': [absolute_date],
                    'Filter Summary': [self._get_filter_summary(rule['filters'])]
                }
            else:
                preview_data = {
                    'Data Source': [rule['data_source']],
                    'Records Affected': [len(filtered_df)],
                    'Adjustment Type': ['Relative Days'],
                    'Offset Days': [rule.get('offset_days', 0)],
                    'Filter Summary': [self._get_filter_summary(rule['filters'])]
                }
            
            # Add date impact if applicable
            self._add_date_impact(preview_data, filtered_df, rule)
            
            return pd.DataFrame(preview_data)
            
        except Exception as e:
            st.error(f"Error previewing rule impact: {str(e)}")
            return None
    
    def _apply_rule_filters(self, df: pd.DataFrame, rule: Dict[str, Any]) -> pd.DataFrame:
        """Apply rule filters to dataframe"""
        filters = rule.get('filters', {})
        
        # Convert rule filters to selections format
        selections = {}
        for filter_type, filter_values in filters.items():
            if filter_values != ['All']:
                selections[filter_type + 's'] = filter_values  # Add 's' for consistency
        
        return self._apply_filters_to_df(df, rule['data_source'], selections)
    
    def _add_date_impact(self, preview_data: Dict, filtered_df: pd.DataFrame, rule: Dict[str, Any]):
        """Add date impact information to preview"""
        date_col = self._get_date_column(rule['data_source'], filtered_df)
        
        if date_col and date_col in filtered_df.columns:
            dates = pd.to_datetime(filtered_df[date_col], errors='coerce').dropna()
            if len(dates) > 0:
                adjustment_type = rule.get('adjustment_type', 'Relative (Days)')
                
                if adjustment_type == 'Absolute (Date)':
                    # For absolute date, show original date range and target date
                    preview_data['Original Date Range'] = [
                        f"{dates.min().strftime('%Y-%m-%d')} to {dates.max().strftime('%Y-%m-%d')}"
                    ]
                    preview_data['New Date'] = [rule.get('absolute_date', 'Unknown')]
                else:
                    # For relative adjustment
                    preview_data['Original Date Range'] = [
                        f"{dates.min().strftime('%Y-%m-%d')} to {dates.max().strftime('%Y-%m-%d')}"
                    ]
                    
                    offset_days = rule.get('offset_days', 0)
                    if offset_days != 0:
                        new_dates = dates + pd.Timedelta(days=offset_days)
                        preview_data['New Date Range'] = [
                            f"{new_dates.min().strftime('%Y-%m-%d')} to {new_dates.max().strftime('%Y-%m-%d')}"
                        ]
        else:
            preview_data['Note'] = [f"No date adjustment applicable for {rule['data_source']}"]
    
    def _get_date_column(self, data_source: str, df: pd.DataFrame) -> Optional[str]:
        """Get the appropriate date column for a data source"""
        date_col = self.date_columns.get(data_source)
        
        if isinstance(date_col, list):
            # Try multiple column options (e.g., for PO)
            for col in date_col:
                if col in df.columns:
                    return col
            return None
        
        return date_col if date_col and date_col in df.columns else None
    
    def _get_filter_summary(self, filters: Dict[str, List[str]]) -> str:
        """Get human readable filter summary"""
        summary_parts = []
        
        for filter_type, filter_values in filters.items():
            if filter_values != ['All'] and filter_values:
                count = len(filter_values)
                display_name = {
                    'entity': 'Entity',
                    'customer': 'Customer',
                    'product': 'Product', 
                    'number': 'Number',
                    'brand': 'Brand'
                }.get(filter_type, filter_type.title())
                
                if count == 1:
                    value = filter_values[0]
                    if filter_type == 'product' and len(value) > 30:
                        value = value[:30] + "..."
                    summary_parts.append(f"{display_name}: {value}")
                else:
                    summary_parts.append(f"{display_name}: {count} selected")
        
        return "; ".join(summary_parts) if summary_parts else "All records"
    
    # === Sample Records ===
    
    def get_sample_affected_records(self, rule: Dict[str, Any], limit: int = 5) -> Optional[pd.DataFrame]:
        """Get sample records that would be affected by the rule"""
        try:
            df = self._load_data_by_source(rule['data_source'])
            
            if df.empty:
                return None
            
            filtered_df = self._apply_rule_filters(df, rule)
            
            if filtered_df.empty:
                return None
            
            # Select relevant columns for display
            display_columns = self._get_display_columns(rule['data_source'], filtered_df)
            
            # Get sample records
            sample_df = filtered_df[display_columns].head(limit).copy()
            
            # Add new date column if there's a date column
            date_col = self._get_date_column(rule['data_source'], sample_df)
            adjustment_type = rule.get('adjustment_type', 'Relative (Days)')
            
            if date_col and date_col in sample_df.columns:
                if adjustment_type == 'Absolute (Date)':
                    # For absolute date adjustment
                    old_date_col = f"Old {date_col.replace('_', ' ').title()}"
                    sample_df[old_date_col] = pd.to_datetime(sample_df[date_col], errors='coerce').dt.strftime('%Y-%m-%d')
                    
                    # All dates become the target date
                    new_date_col = f"New {date_col.replace('_', ' ').title()}"
                    target_date = rule.get('absolute_date', datetime.now().date().isoformat())
                    sample_df[new_date_col] = target_date
                    
                    # Remove original date column
                    sample_df = sample_df.drop(columns=[date_col])
                    
                elif adjustment_type == 'Relative (Days)' and rule.get('offset_days', 0) != 0:
                    # For relative adjustment
                    old_date_col = f"Old {date_col.replace('_', ' ').title()}"
                    sample_df[old_date_col] = pd.to_datetime(sample_df[date_col], errors='coerce').dt.strftime('%Y-%m-%d')
                    
                    # Calculate new date
                    new_date_col = f"New {date_col.replace('_', ' ').title()}"
                    sample_df[new_date_col] = (pd.to_datetime(sample_df[date_col], errors='coerce') + 
                                               pd.Timedelta(days=rule['offset_days'])).dt.strftime('%Y-%m-%d')
                    
                    # Remove original date column
                    sample_df = sample_df.drop(columns=[date_col])
            
            return sample_df
            
        except Exception as e:
            st.error(f"Error getting sample records: {str(e)}")
            return None
    
    def _get_display_columns(self, data_source: str, df: pd.DataFrame) -> List[str]:
        """Get relevant columns for displaying sample records"""
        display_columns = []
        
        # Define column order preference
        preferred_order = ['pt_code', 'product_name', 'brand']
        
        # Add preferred columns first
        for col in preferred_order:
            if col in df.columns:
                display_columns.append(col)
        
        # Entity
        entity_col = self.entity_columns.get(data_source, "legal_entity")
        if entity_col in df.columns:
            display_columns.append(entity_col)
        
        # Customer (for demand sources)
        if data_source in ["OC", "Forecast"] and 'customer' in df.columns:
            display_columns.append('customer')
        
        # Number
        number_col = self.number_columns.get(data_source, "id")
        if number_col in df.columns:
            display_columns.append(number_col)
        
        # Date
        date_col = self._get_date_column(data_source, df)
        if date_col:
            display_columns.append(date_col)
        
        # Remove duplicates while preserving order
        return list(dict.fromkeys(display_columns))
    
    # === Preview UI ===
    
    def show_rule_preview_ui(self, rule: Dict[str, Any]):
        """Show formatted rule preview UI with styling"""
        try:
            # Get preview data
            preview_df = self.preview_rule_impact(rule)
            sample_df = self.get_sample_affected_records(rule, limit=5)
            
            if preview_df is None:
                st.warning("⚠️ No data matches the filter criteria.")
                return
            
            # Create a preview container for better spacing
            with st.container():
                # Show impact summary
                st.markdown("#### 📊 Impact Summary")
                
                # Create columns for better layout
                adjustment_type = rule.get('adjustment_type', 'Relative (Days)')
                
                if adjustment_type == 'Absolute (Date)':
                    col1, col2, col3 = st.columns([2, 2, 1])
                    
                    for _, row in preview_df.iterrows():
                        with col1:
                            # Data source and records affected
                            st.metric(
                                "Data Source",
                                row['Data Source'],
                                f"{row['Records Affected']:,} records",
                                delta_color="off"
                            )
                        
                        with col2:
                            # Absolute date adjustment
                            st.metric(
                                "Date Adjustment",
                                "Set to specific date",
                                f"→ {row.get('Target Date', 'Unknown')}",
                                delta_color="normal"
                            )
                        
                        with col3:
                            # Filter summary
                            st.markdown("**Filters Applied:**")
                            st.caption(row['Filter Summary'])
                else:
                    # Original relative adjustment display
                    col1, col2, col3 = st.columns([2, 2, 1])
                    
                    for _, row in preview_df.iterrows():
                        with col1:
                            # Data source and records affected
                            st.metric(
                                "Data Source",
                                row['Data Source'],
                                f"{row['Records Affected']:,} records",
                                delta_color="off"
                            )
                        
                        with col2:
                            # Offset impact
                            offset_days = rule.get('offset_days', 0)
                            if offset_days > 0:
                                st.metric(
                                    "Date Adjustment",
                                    f"{offset_days} days later",
                                    "Dates moved forward →",
                                    delta_color="normal"
                                )
                            elif offset_days < 0:
                                st.metric(
                                    "Date Adjustment", 
                                    f"{abs(offset_days)} days earlier",
                                    "← Dates moved backward",
                                    delta_color="inverse"
                                )
                            else:
                                st.metric(
                                    "Date Adjustment",
                                    "No change",
                                    "Dates unchanged",
                                    delta_color="off"
                                )
                        
                        with col3:
                            # Filter summary
                            st.markdown("**Filters Applied:**")
                            st.caption(row['Filter Summary'])
                
                # Date range impact
                if 'Original Date Range' in preview_df.columns and pd.notna(preview_df.iloc[0]['Original Date Range']):
                    st.markdown("---")
                    
                    if adjustment_type == 'Absolute (Date)':
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("**📅 Original Date Range**")
                            st.info(preview_df.iloc[0]['Original Date Range'])
                        with col2:
                            st.markdown("**📅 New Date (All Records)**")
                            st.success(preview_df.iloc[0].get('New Date', 'Unknown'))
                    else:
                        if 'New Date Range' in preview_df.columns:
                            col1, col2 = st.columns(2)
                            with col1:
                                st.markdown("**📅 Original Date Range**")
                                st.info(preview_df.iloc[0]['Original Date Range'])
                            with col2:
                                st.markdown("**📅 New Date Range**")
                                if rule.get('offset_days', 0) > 0:
                                    st.success(preview_df.iloc[0]['New Date Range'])
                                else:
                                    st.warning(preview_df.iloc[0]['New Date Range'])
            
            # Show sample records if available
            if sample_df is not None and not sample_df.empty:
                st.markdown("---")
                st.markdown("#### 📋 Sample Affected Records")
                
                # Add some styling info
                st.caption("Showing how dates will change for sample records")
                
                # Style the dataframe
                styled_df = sample_df.copy()
                
                # Truncate long text fields (except date columns)
                for col in styled_df.columns:
                    if styled_df[col].dtype == 'object' and 'date' not in col.lower():
                        styled_df[col] = styled_df[col].astype(str).apply(
                            lambda x: x[:40] + "..." if len(str(x)) > 40 else x
                        )
                
                # Display with highlighting for date columns
                st.dataframe(
                    styled_df,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        col: st.column_config.TextColumn(
                            col,
                            width="medium" if 'date' in col.lower() else None
                        )
                        for col in styled_df.columns if 'date' in col.lower()
                    }
                )
                
                if len(sample_df) >= 5:
                    records_affected = preview_df.iloc[0]['Records Affected']
                    if records_affected > 5:
                        st.caption(f"📝 Showing first 5 of {records_affected:,} affected records")
            
        except Exception as e:
            st.error(f"Error showing preview: {str(e)}")
            if st.session_state.get('debug_mode', False):
                st.exception(e)
    
    # === Cache Management ===
    
    def clear_cache(self):
        """Clear filter cache"""
        self._filter_cache.clear()
        st.cache_data.clear()