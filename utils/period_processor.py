# === COMPLETE NEW VERSION - SIMPLIFIED ===
# utils/period_processor.py
import pandas as pd
import logging
from typing import Dict, Optional, Tuple, List
from datetime import datetime
from utils.helpers import convert_to_period

logger = logging.getLogger(__name__)

class PeriodBasedGAPProcessor:
    """Process all data by period for GAP calculation - SIMPLIFIED"""
    
    def __init__(self, period_type: str = 'Weekly'):
        self.period_type = period_type
    
    def process_for_gap(self, 
                       demand_df: pd.DataFrame, 
                       supply_df: pd.DataFrame,
                       use_adjusted_demand: bool = True,
                       use_adjusted_supply: bool = True) -> pd.DataFrame:
        """
        Process all data by period for GAP calculation - NO ALLOCATION NEEDED
        """
        
        # Step 1: Add period column to all dataframes
        demand_with_period = self._add_period_column(
            demand_df, 
            date_col=self._get_demand_date_column(demand_df, use_adjusted_demand),
            df_type='demand'
        )
        
        supply_with_period = self._add_period_column(
            supply_df,
            date_col=None,  # Will be determined by source_type
            df_type='supply',
            use_adjusted=use_adjusted_supply
        )
        
        # Step 2: Group by product + period
        demand_grouped = self._group_demand_by_period(demand_with_period)
        supply_grouped = self._group_supply_by_period(supply_with_period)
        
        # Step 3: Merge data (NO ALLOCATION)
        period_data = self._merge_period_data(demand_grouped, supply_grouped)
        
        # Step 4: Calculate net values (SIMPLIFIED)
        period_data = self._calculate_net_values(period_data)
        
        return period_data
    
    def _get_demand_date_column(self, df: pd.DataFrame, use_adjusted: bool) -> str:
        """Get appropriate demand date column"""
        if use_adjusted and 'etd_adjusted' in df.columns:
            return 'etd_adjusted'
        return 'etd'
    
    def _get_supply_date_column(self, row: pd.Series, use_adjusted: bool) -> str:
        """Get appropriate supply date column based on source type"""
        date_mapping = {
            'Inventory': 'date_ref',
            'Pending CAN': 'arrival_date',
            'Pending PO': 'eta',
            'Pending WH Transfer': 'transfer_date'
        }
        
        base_col = date_mapping.get(row['source_type'], 'date_ref')
        
        if use_adjusted:
            adjusted_col = f"{base_col}_adjusted"
            if adjusted_col in row.index:
                return adjusted_col
        
        return base_col
    
    def _add_period_column(self, df: pd.DataFrame, date_col: Optional[str], 
                          df_type: str, use_adjusted: bool = False) -> pd.DataFrame:
        """Add period column based on date column and type"""
        if df.empty:
            return df
            
        df = df.copy()
        
        # Handle different date columns for supply types
        if df_type == 'supply' and 'source_type' in df.columns:
            df['period'] = df.apply(
                lambda row: self._get_supply_period(row, use_adjusted),
                axis=1
            )
        else:
            if date_col and date_col in df.columns:
                df['period'] = df[date_col].apply(
                    lambda x: convert_to_period(x, self.period_type)
                )
            else:
                logger.warning(f"Date column {date_col} not found in {df_type} dataframe")
                df['period'] = None
        
        # Remove invalid periods
        df = df[df['period'].notna() & (df['period'] != 'nan')]
        
        return df
    
    def _get_supply_period(self, row: pd.Series, use_adjusted: bool) -> Optional[str]:
        """Get period for supply based on source type"""
        date_col = self._get_supply_date_column(row, use_adjusted)
        
        if date_col in row and pd.notna(row[date_col]):
            return convert_to_period(row[date_col], self.period_type)
        
        return None
    
    def _group_demand_by_period(self, demand_df: pd.DataFrame) -> pd.DataFrame:
        """Group demand by product + period - SIMPLIFIED"""
        if demand_df.empty:
            return pd.DataFrame()
            
        # Simple aggregation - demand_quantity is already net of delivered
        agg_dict = {
            'demand_quantity': 'sum',
            'product_name': 'first',
            'package_size': 'first',
            'standard_uom': 'first'
        }
        
        return demand_df.groupby(['pt_code', 'period']).agg(agg_dict).reset_index()
    
    def _group_supply_by_period(self, supply_df: pd.DataFrame) -> pd.DataFrame:
        """Group supply by product + period"""
        if supply_df.empty:
            return pd.DataFrame()
        
        agg_dict = {
            'quantity': 'sum',
            'product_name': 'first',
            'package_size': 'first',
            'standard_uom': 'first'
        }
        
        result_df = supply_df.groupby(['pt_code', 'period']).agg(agg_dict).reset_index()
        result_df = result_df.rename(columns={'quantity': 'supply_quantity'})
        
        return result_df

    def _merge_period_data(self, demand_df: pd.DataFrame, 
                          supply_df: pd.DataFrame) -> pd.DataFrame:
        """Merge period data - SIMPLIFIED without allocation"""
        # Get all unique product-period combinations
        all_keys = set()
        
        for df in [demand_df, supply_df]:
            if not df.empty and 'pt_code' in df.columns and 'period' in df.columns:
                keys = df[['pt_code', 'period']].apply(tuple, axis=1)
                all_keys.update(keys)
        
        if not all_keys:
            return pd.DataFrame()
        
        # Create base dataframe
        base_data = pd.DataFrame(list(all_keys), columns=['pt_code', 'period'])
        
        # Get product info from BOTH demand and supply
        product_info_list = []
        
        # Get from demand first
        if not demand_df.empty and 'product_name' in demand_df.columns:
            demand_product_info = demand_df[['pt_code', 'product_name', 'package_size', 'standard_uom']].drop_duplicates()
            product_info_list.append(demand_product_info)
        
        # Get from supply (for supply-only products)
        if not supply_df.empty:
            supply_info_cols = ['pt_code']
            if 'product_name' in supply_df.columns:
                supply_info_cols.append('product_name')
            if 'package_size' in supply_df.columns:
                supply_info_cols.append('package_size')
            if 'standard_uom' in supply_df.columns:
                supply_info_cols.append('standard_uom')
            
            if len(supply_info_cols) > 1:
                supply_product_info = supply_df[supply_info_cols].drop_duplicates()
                product_info_list.append(supply_product_info)
        
        # Combine product info
        if product_info_list:
            all_product_info = pd.concat(product_info_list, ignore_index=True)
            all_product_info = all_product_info.drop_duplicates(subset=['pt_code'], keep='first')
            base_data = base_data.merge(all_product_info, on='pt_code', how='left')
        
        # Merge demand data
        if not demand_df.empty:
            demand_cols = [col for col in demand_df.columns 
                          if col not in ['product_name', 'package_size', 'standard_uom']]
            base_data = base_data.merge(
                demand_df[demand_cols],
                on=['pt_code', 'period'],
                how='left'
            )
        
        # Merge supply data
        if not supply_df.empty:
            supply_cols = [col for col in supply_df.columns 
                          if col not in ['product_name', 'package_size', 'standard_uom']]
            base_data = base_data.merge(
                supply_df[supply_cols],
                on=['pt_code', 'period'],
                how='left'
            )
        
        # Fill NaN with 0 for numeric columns
        numeric_cols = ['demand_quantity', 'supply_quantity']
        for col in numeric_cols:
            if col in base_data.columns:
                base_data[col] = base_data[col].fillna(0)
            else:
                base_data[col] = 0
        
        return base_data

    def _calculate_net_values(self, period_data: pd.DataFrame) -> pd.DataFrame:
        """Calculate net values for GAP analysis - SUPER SIMPLE"""
        df = period_data.copy()
        
        # Demand is already net of delivered (from view)
        # Supply is available supply
        # So GAP is simply:
        df['gap_quantity'] = df['supply_quantity'] - df['demand_quantity']
        
        # Calculate fulfillment rate
        df['fulfillment_rate'] = df.apply(
            lambda row: min(100, (row['supply_quantity'] / row['demand_quantity'] * 100))
            if row['demand_quantity'] > 0 else 100,
            axis=1
        )
        
        # For carry forward logic, we need these columns
        df['available_supply'] = df['supply_quantity']
        df['unallocated_demand'] = df['demand_quantity']  # Already net
        
        return df