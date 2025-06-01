# utils/filters.py - Unified Filter Management

import streamlit as st
import pandas as pd
from datetime import datetime, date
from typing import Dict, List, Any, Optional, Tuple

class FilterManager:
    """Centralized filter management across all pages"""
    
    @staticmethod
    def create_product_filter(df: pd.DataFrame, key_prefix: str = "") -> List[str]:
        """Create enhanced product filter with PT Code and Name"""
        if df.empty:
            return []
        
        # Get unique products
        unique_products = df[['pt_code', 'product_name']].drop_duplicates()
        unique_products = unique_products[
            (unique_products['pt_code'].notna()) & 
            (unique_products['pt_code'] != '') &
            (unique_products['pt_code'] != 'nan')
        ]
        
        # Create display options
        product_options = []
        for _, row in unique_products.iterrows():
            pt_code = str(row['pt_code'])
            product_name = str(row['product_name'])[:50] if pd.notna(row['product_name']) else ""
            option = f"{pt_code} - {product_name}"
            product_options.append(option)
        
        # Show multiselect
        selected_products = st.multiselect(
            "Product (PT Code - Name)", 
            sorted(product_options),
            key=f"{key_prefix}product_filter",
            help="Search by PT Code or Product Name"
        )
        
        # Extract PT codes
        return [p.split(' - ')[0] for p in selected_products]
    
    @staticmethod
    def create_standard_filters(df: pd.DataFrame, key_prefix: str = "", 
                              show_customer: bool = False) -> Dict[str, Any]:
        """Create standard set of filters"""
        filters = {}
        
        # Entity filter
        if 'legal_entity' in df.columns:
            entities = df["legal_entity"].dropna().unique().tolist()
            filters['entity'] = st.multiselect(
                "Legal Entity", 
                sorted(entities),
                key=f"{key_prefix}entity_filter"
            )
        
        # Customer filter (optional)
        if show_customer and 'customer' in df.columns:
            customers = df["customer"].dropna().unique().tolist()
            filters['customer'] = st.multiselect(
                "Customer", 
                sorted(customers),
                key=f"{key_prefix}customer_filter"
            )
        
        # Product filter
        filters['product'] = FilterManager.create_product_filter(df, key_prefix)
        
        # Brand filter
        if 'brand' in df.columns:
            brands = df["brand"].dropna().unique().tolist()
            filters['brand'] = st.multiselect(
                "Brand", 
                sorted(brands),
                key=f"{key_prefix}brand_filter"
            )
        
        return filters
    
    @staticmethod
    def create_date_range_filter(df: pd.DataFrame, date_column: str,
                               key_prefix: str = "") -> Tuple[date, date]:
        """Create date range filter"""
        col1, col2 = st.columns(2)
        
        # Get default dates from data
        if date_column in df.columns:
            dates = pd.to_datetime(df[date_column], errors='coerce').dropna()
            if len(dates) > 0:
                # Convert to date objects properly
                default_start = dates.min().date() if hasattr(dates.min(), 'date') else dates.min()
                default_end = dates.max().date() if hasattr(dates.max(), 'date') else dates.max()
            else:
                default_start = default_end = datetime.today().date()
        else:
            default_start = default_end = datetime.today().date()
        
        with col1:
            start_date = st.date_input(
                "From Date", 
                value=default_start,
                key=f"{key_prefix}start_date"
            )
        
        with col2:
            end_date = st.date_input(
                "To Date", 
                value=default_end,
                key=f"{key_prefix}end_date"
            )
        
        return start_date, end_date
    
    @staticmethod
    def apply_filters(df: pd.DataFrame, filters: Dict[str, Any], 
                     date_column: Optional[str] = None) -> pd.DataFrame:
        """Apply filters to dataframe"""
        filtered_df = df.copy()
        
        # Entity filter
        if filters.get("entity"):
            filtered_df = filtered_df[filtered_df["legal_entity"].isin(filters["entity"])]
        
        # Customer filter
        if filters.get("customer") and "customer" in filtered_df.columns:
            filtered_df = filtered_df[filtered_df["customer"].isin(filters["customer"])]
        
        # Product filter
        if filters.get("product"):
            filtered_df = filtered_df[filtered_df["pt_code"].isin(filters["product"])]
        
        # Brand filter
        if filters.get("brand"):
            filtered_df = filtered_df[filtered_df["brand"].isin(filters["brand"])]
        
        # Date filter
        if date_column and filters.get("start_date") and filters.get("end_date"):
            # Convert to pandas timestamps for comparison
            start_ts = pd.to_datetime(filters["start_date"])
            end_ts = pd.to_datetime(filters["end_date"])
            
            # Ensure date column is datetime
            filtered_df[date_column] = pd.to_datetime(filtered_df[date_column], errors='coerce')
            
            # Handle null dates
            filtered_df = filtered_df[
                filtered_df[date_column].isna() |
                ((filtered_df[date_column] >= start_ts) & (filtered_df[date_column] <= end_ts))
            ]
        
        return filtered_df
    
    @staticmethod
    def create_demand_filters(df_demand: pd.DataFrame) -> Dict[str, Any]:
        """Create demand-specific filters"""
        with st.expander("📎 Filters", expanded=True):
            col1, col2, col3 = st.columns(3)
            
            # Standard filters
            with col1:
                filters = FilterManager.create_standard_filters(
                    df_demand, "demand_", show_customer=True
                )
            
            # Date range
            with col2:
                start_date, end_date = FilterManager.create_date_range_filter(
                    df_demand, "etd", "demand_"
                )
                filters['start_date'] = start_date
                filters['end_date'] = end_date
            
            # Additional demand filters
            with col3:
                if 'is_converted_to_oc' in df_demand.columns:
                    conversion_options = df_demand["is_converted_to_oc"].dropna().unique().tolist()
                    filters['conversion_status'] = st.multiselect(
                        "Conversion Status", 
                        sorted(conversion_options),
                        key="demand_conversion_filter"
                    )
        
        return filters
    
    @staticmethod
    def create_supply_filters(df_supply: pd.DataFrame) -> Dict[str, Any]:
        """Create supply-specific filters"""
        with st.expander("📎 Filters", expanded=True):
            # Basic filters
            filters = FilterManager.create_standard_filters(df_supply, "supply_")
            
            # Source type filter
            if 'source_type' in df_supply.columns:
                source_types = df_supply["source_type"].unique().tolist()
                filters['source_type'] = st.multiselect(
                    "Source Type",
                    source_types,
                    default=source_types,
                    key="supply_source_type_filter"
                )
            
            # Date range
            start_date, end_date = FilterManager.create_date_range_filter(
                df_supply, "date_ref", "supply_"
            )
            filters['start_date'] = start_date
            filters['end_date'] = end_date
            
            # Expiry filter
            if "days_until_expiry" in df_supply.columns:
                filters['expiry_warning_days'] = st.number_input(
                    "Show items expiring within (days)",
                    min_value=0,
                    max_value=365,
                    value=30,
                    key="expiry_warning_days"
                )
        
        return filters
    
    @staticmethod
    def create_gap_filters(df_demand: Optional[pd.DataFrame] = None, 
                          df_supply: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """Create GAP analysis filters"""
        with st.expander("📎 Advanced Filters", expanded=True):
            # Combine data for filter options
            if df_demand is not None and df_supply is not None:
                # Combine unique values
                all_df = pd.concat([
                    df_demand[['legal_entity', 'pt_code', 'product_name', 'brand']],
                    df_supply[['legal_entity', 'pt_code', 'product_name', 'brand']]
                ]).drop_duplicates()
                
                filters = FilterManager.create_standard_filters(all_df, "gap_")
                
                # Date range from both sources
                demand_dates = df_demand["etd"].dropna() if "etd" in df_demand.columns else pd.Series()
                supply_dates = df_supply["date_ref"].dropna() if "date_ref" in df_supply.columns else pd.Series()
                
                if len(demand_dates) > 0 and len(supply_dates) > 0:
                    min_date = min(demand_dates.min().date() if hasattr(demand_dates.min(), 'date') else demand_dates.min(), 
                                   supply_dates.min().date() if hasattr(supply_dates.min(), 'date') else supply_dates.min())
                    max_date = max(demand_dates.max().date() if hasattr(demand_dates.max(), 'date') else demand_dates.max(), 
                                   supply_dates.max().date() if hasattr(supply_dates.max(), 'date') else supply_dates.max())
                elif len(demand_dates) > 0:
                    min_date = demand_dates.min().date() if hasattr(demand_dates.min(), 'date') else demand_dates.min()
                    max_date = demand_dates.max().date() if hasattr(demand_dates.max(), 'date') else demand_dates.max()
                elif len(supply_dates) > 0:
                    min_date = supply_dates.min().date() if hasattr(supply_dates.min(), 'date') else supply_dates.min()
                    max_date = supply_dates.max().date() if hasattr(supply_dates.max(), 'date') else supply_dates.max()
                else:
                    min_date = max_date = datetime.today().date()
                
                # Create dummy dataframe for date filter
                dummy_df = pd.DataFrame({'date': [min_date, max_date]})
                start_date, end_date = FilterManager.create_date_range_filter(
                    dummy_df, 'date', "gap_"
                )
                filters['start_date'] = start_date
                filters['end_date'] = end_date
            else:
                # Fallback to empty filters
                filters = {
                    'entity': [],
                    'product': [],
                    'brand': [],
                    'start_date': datetime.today().date(),
                    'end_date': datetime.today().date()
                }
        
        return filters