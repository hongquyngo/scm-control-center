# utils/period_processor.py
import pandas as pd
import logging
from typing import Dict, Optional, Tuple, List
from datetime import datetime
from utils.helpers import convert_to_period

logger = logging.getLogger(__name__)

class PeriodBasedGAPProcessor:
    """Process all data by period for GAP calculation"""
    
    def __init__(self, period_type: str = 'Weekly'):
        self.period_type = period_type
    
    def process_for_gap(self, 
                       demand_df: pd.DataFrame, 
                       supply_df: pd.DataFrame, 
                       allocations_df: pd.DataFrame,
                       use_adjusted_demand: bool = True,
                       use_adjusted_supply: bool = True) -> pd.DataFrame:
        """
        Process all data by period for GAP calculation
        
        Key improvements:
        1. Single pass processing - no double enhancement
        2. Period-aligned from start
        3. Proper allocation tracking by period
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
        
        allocations_with_period = self._add_period_column(
            allocations_df,
            date_col='allocated_etd',
            df_type='allocation'
        )
        
        # Step 2: Group by product + period
        demand_grouped = self._group_demand_by_period(demand_with_period)
        supply_grouped = self._group_supply_by_period(supply_with_period)
        allocation_grouped = self._group_allocations_by_period(allocations_with_period)
        
        # Step 3: Merge all data
        period_data = self._merge_period_data(
            demand_grouped,
            supply_grouped, 
            allocation_grouped
        )
        
        # Step 4: Calculate net values
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
        """Group demand by product + period"""
        if demand_df.empty:
            return pd.DataFrame()
            
        # Include both original and unallocated demand
        agg_dict = {
            'demand_quantity': 'sum',
            'product_name': 'first',
            'package_size': 'first',
            'standard_uom': 'first'
        }
        
        # Add allocation columns if they exist
        if 'unallocated_demand' in demand_df.columns:
            agg_dict['unallocated_demand'] = 'sum'
        if 'total_allocated' in demand_df.columns:
            agg_dict['total_allocated'] = 'sum'
        if 'total_delivered' in demand_df.columns:
            agg_dict['total_delivered'] = 'sum'
            
        return demand_df.groupby(['pt_code', 'period']).agg(agg_dict).reset_index()
    
    def _group_supply_by_period(self, supply_df: pd.DataFrame) -> pd.DataFrame:
        """Group supply by product + period"""
        if supply_df.empty:
            return pd.DataFrame()
            
        # Use available_quantity if exists (from enhance_supply_with_allocations)
        quantity_col = 'available_quantity' if 'available_quantity' in supply_df.columns else 'quantity'
        
        return supply_df.groupby(['pt_code', 'period']).agg({
            quantity_col: 'sum'
        }).reset_index().rename(columns={quantity_col: 'supply_quantity'})
    
    def _group_allocations_by_period(self, allocations_df: pd.DataFrame) -> pd.DataFrame:
        """Group allocations by product + period"""
        if allocations_df.empty:
            return pd.DataFrame()
            
        return allocations_df.groupby(['pt_code', 'period']).agg({
            'total_allocated_qty': 'sum',
            'total_delivered_qty': 'sum',
            'undelivered_qty': 'sum'
        }).reset_index()
    
    def _merge_period_data(self, demand_df: pd.DataFrame, supply_df: pd.DataFrame, 
                          allocation_df: pd.DataFrame) -> pd.DataFrame:
        """Merge all period data with proper handling of missing values"""
        # Get all unique product-period combinations
        all_keys = set()
        
        for df in [demand_df, supply_df, allocation_df]:
            if not df.empty and 'pt_code' in df.columns and 'period' in df.columns:
                keys = df[['pt_code', 'period']].apply(tuple, axis=1)
                all_keys.update(keys)
        
        if not all_keys:
            return pd.DataFrame()
        
        # Create base dataframe
        base_data = pd.DataFrame(list(all_keys), columns=['pt_code', 'period'])
        
        # Add product info from demand if available
        if not demand_df.empty and 'product_name' in demand_df.columns:
            product_info = demand_df[['pt_code', 'product_name', 'package_size', 'standard_uom']].drop_duplicates()
            base_data = base_data.merge(product_info, on='pt_code', how='left')
        
        # Merge demand data
        if not demand_df.empty:
            base_data = base_data.merge(
                demand_df.drop(columns=['product_name', 'package_size', 'standard_uom'], errors='ignore'),
                on=['pt_code', 'period'],
                how='left'
            )
        
        # Merge supply data
        if not supply_df.empty:
            base_data = base_data.merge(
                supply_df,
                on=['pt_code', 'period'],
                how='left'
            )
        
        # Merge allocation data
        if not allocation_df.empty:
            base_data = base_data.merge(
                allocation_df,
                on=['pt_code', 'period'],
                how='left',
                suffixes=('', '_alloc')
            )
        
        # Fill NaN with 0 for numeric columns
        numeric_cols = [
            'demand_quantity', 'unallocated_demand', 'total_allocated', 'total_delivered',
            'supply_quantity', 'total_allocated_qty', 'total_delivered_qty', 'undelivered_qty'
        ]
        
        for col in numeric_cols:
            if col in base_data.columns:
                base_data[col] = base_data[col].fillna(0)
        
        return base_data
    
    def _calculate_net_values(self, period_data: pd.DataFrame) -> pd.DataFrame:
        """Calculate net values for GAP analysis"""
        df = period_data.copy()
        
        # Calculate unallocated demand if not already present
        if 'unallocated_demand' not in df.columns:
            # Use allocation data if available
            if 'total_allocated_qty' in df.columns:
                df['unallocated_demand'] = df['demand_quantity'] - df['total_allocated_qty']
            else:
                df['unallocated_demand'] = df['demand_quantity']
        
        # Ensure non-negative
        df['unallocated_demand'] = df['unallocated_demand'].clip(lower=0)
        
        # Calculate available supply (considering allocations)
        if 'undelivered_qty' in df.columns:
            df['available_supply'] = df['supply_quantity'] - df['undelivered_qty']
        else:
            df['available_supply'] = df['supply_quantity']
        
        # Ensure non-negative
        df['available_supply'] = df['available_supply'].clip(lower=0)
        
        # Calculate GAP
        df['gap_quantity'] = df['available_supply'] - df['unallocated_demand']
        
        # Calculate fulfillment rate
        df['fulfillment_rate'] = df.apply(
            lambda row: min(100, (row['available_supply'] / row['unallocated_demand'] * 100))
            if row['unallocated_demand'] > 0 else 100,
            axis=1
        )
        
        return df