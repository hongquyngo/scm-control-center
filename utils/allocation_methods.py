"""
Allocation Methods - Implementation of different allocation algorithms
"""

import pandas as pd
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)


class AllocationMethods:
    """Implementation of various allocation methods"""
    
    @staticmethod
    def calculate_allocation(demand_df: pd.DataFrame, supply_df: pd.DataFrame,
                           method: str, parameters: Dict) -> pd.DataFrame:
        """Calculate allocation based on selected method"""
        
        # Prepare data
        demand_df = demand_df.copy()
        supply_df = supply_df.copy()
        
        # Add necessary columns if missing
        if 'demand_quantity' not in demand_df.columns and 'requested_qty' in demand_df.columns:
            demand_df['demand_quantity'] = demand_df['requested_qty']
        elif 'requested_qty' not in demand_df.columns and 'demand_quantity' in demand_df.columns:
            demand_df['requested_qty'] = demand_df['demand_quantity']
        
        # Get available supply by product and entity
        supply_summary = supply_df.groupby(['pt_code', 'legal_entity']).agg({
            'quantity': 'sum'
        }).reset_index()
        supply_summary.rename(columns={'quantity': 'available_qty'}, inplace=True)
        
        # Add supply info to demand
        demand_df = demand_df.merge(
            supply_summary,
            on=['pt_code', 'legal_entity'],
            how='left'
        )
        demand_df['available_qty'] = demand_df['available_qty'].fillna(0)
        
        # Apply allocation method
        if method == 'FCFS':
            result = AllocationMethods._allocate_fcfs(demand_df, parameters)
        elif method == 'PRIORITY':
            result = AllocationMethods._allocate_priority(demand_df, parameters)
        elif method == 'PRO_RATA':
            result = AllocationMethods._allocate_pro_rata(demand_df, parameters)
        elif method == 'MANUAL':
            result = AllocationMethods._allocate_manual(demand_df, parameters)
        else:
            logger.warning(f"Unknown allocation method: {method}, using FCFS")
            result = AllocationMethods._allocate_fcfs(demand_df, parameters)
        
        # Calculate fulfillment rate
        result['fulfillment_rate'] = (result['allocated_qty'] / result['requested_qty'] * 100).fillna(0)
        result['fulfillment_rate'] = result['fulfillment_rate'].clip(upper=100)
        
        # Add additional fields
        result['allocated_etd'] = result['etd']  # Can be modified later
        result['demand_type'] = result.get('source_type', 'OC')
        
        # Extract demand reference ID from demand_line_id
        if 'demand_line_id' in result.columns:
            result['demand_reference_id'] = result['demand_line_id'].str.extract(r'(\d+)').astype('Int64')
        
        return result
    
    @staticmethod
    def _allocate_fcfs(demand_df: pd.DataFrame, parameters: Dict) -> pd.DataFrame:
        """FCFS allocation - prioritize earliest ETD (First Come First Served)"""
        
        # Sort by ETD
        demand_df = demand_df.sort_values('etd')
        
        # Track available supply by product-entity
        supply_tracker = {}
        
        # Initialize allocated quantity
        demand_df['allocated_qty'] = 0.0
        
        for idx, row in demand_df.iterrows():
            key = (row['pt_code'], row['legal_entity'])
            
            # Initialize supply tracker
            if key not in supply_tracker:
                supply_tracker[key] = row['available_qty']
            
            # Calculate allocation
            requested = row['requested_qty']
            available = supply_tracker[key]
            
            if parameters.get('allow_partial', True):
                allocated = min(requested, available)
            else:
                allocated = requested if available >= requested else 0
            
            # Update
            demand_df.at[idx, 'allocated_qty'] = allocated
            supply_tracker[key] -= allocated
        
        return demand_df
    
    @staticmethod
    def _allocate_priority(demand_df: pd.DataFrame, parameters: Dict) -> pd.DataFrame:
        """Priority-based allocation"""
        
        # Get customer priorities
        customer_priorities = parameters.get('method_params', {}).get('customer_priorities', {})
        
        # Add priority column
        demand_df['priority'] = demand_df['customer'].map(customer_priorities).fillna(5)
        
        # Sort by priority (descending) then ETD
        demand_df = demand_df.sort_values(['priority', 'etd'], ascending=[False, True])
        
        # Track available supply
        supply_tracker = {}
        
        # Initialize allocated quantity
        demand_df['allocated_qty'] = 0.0
        
        for idx, row in demand_df.iterrows():
            key = (row['pt_code'], row['legal_entity'])
            
            if key not in supply_tracker:
                supply_tracker[key] = row['available_qty']
            
            requested = row['requested_qty']
            available = supply_tracker[key]
            
            if parameters.get('allow_partial', True):
                allocated = min(requested, available)
            else:
                allocated = requested if available >= requested else 0
            
            demand_df.at[idx, 'allocated_qty'] = allocated
            supply_tracker[key] -= allocated
        
        # Remove priority column
        demand_df.drop('priority', axis=1, inplace=True)
        
        return demand_df
    
    @staticmethod
    def _allocate_pro_rata(demand_df: pd.DataFrame, parameters: Dict) -> pd.DataFrame:
        """Pro-rata allocation - proportional distribution"""
        
        min_allocation_percent = parameters.get('method_params', {}).get('min_allocation_percent', 0)
        
        # Group by product-entity to calculate proportions
        grouped = demand_df.groupby(['pt_code', 'legal_entity'])
        
        result_dfs = []
        
        for (pt_code, entity), group in grouped:
            group = group.copy()
            
            total_demand = group['requested_qty'].sum()
            available_supply = group.iloc[0]['available_qty']
            
            if total_demand == 0 or available_supply == 0:
                group['allocated_qty'] = 0
            else:
                # Calculate base allocation ratio
                allocation_ratio = min(available_supply / total_demand, 1.0)
                
                # Apply minimum allocation if set
                if min_allocation_percent > 0:
                    min_ratio = min_allocation_percent / 100
                    allocation_ratio = max(allocation_ratio, min_ratio)
                    allocation_ratio = min(allocation_ratio, 1.0)
                
                # Calculate proportional allocation
                group['allocated_qty'] = group['requested_qty'] * allocation_ratio
                
                # Ensure we don't over-allocate
                total_allocated = group['allocated_qty'].sum()
                if total_allocated > available_supply:
                    # Scale down proportionally
                    scale_factor = available_supply / total_allocated
                    group['allocated_qty'] = group['allocated_qty'] * scale_factor
            
            result_dfs.append(group)
        
        return pd.concat(result_dfs, ignore_index=True)
    
    @staticmethod
    def _allocate_manual(demand_df: pd.DataFrame, parameters: Dict) -> pd.DataFrame:
        """Manual allocation - start with pro-rata as base"""
        
        # Start with pro-rata allocation as default
        result = AllocationMethods._allocate_pro_rata(demand_df, parameters)
        
        # Manual adjustments will be done in the UI
        return result
    
    @staticmethod
    def validate_allocation(allocation_df: pd.DataFrame, supply_df: pd.DataFrame) -> List[str]:
        """Validate allocation results"""
        warnings = []
        
        # Check over-allocation by product-entity
        allocation_summary = allocation_df.groupby(['pt_code', 'legal_entity']).agg({
            'allocated_qty': 'sum'
        }).reset_index()
        
        supply_summary = supply_df.groupby(['pt_code', 'legal_entity']).agg({
            'quantity': 'sum'
        }).reset_index()
        
        check_df = allocation_summary.merge(
            supply_summary,
            on=['pt_code', 'legal_entity'],
            how='left'
        )
        
        over_allocated = check_df[check_df['allocated_qty'] > check_df['quantity']]
        
        for _, row in over_allocated.iterrows():
            warnings.append(
                f"Over-allocation for {row['pt_code']} at {row['legal_entity']}: "
                f"Allocated {row['allocated_qty']:.0f} but only {row['quantity']:.0f} available"
            )
        
        # Check unfulfilled high-priority orders
        if 'priority' in allocation_df.columns:
            high_priority_unfulfilled = allocation_df[
                (allocation_df['priority'] >= 8) & 
                (allocation_df['allocated_qty'] < allocation_df['requested_qty'])
            ]
            
            if not high_priority_unfulfilled.empty:
                warnings.append(
                    f"{len(high_priority_unfulfilled)} high-priority orders not fully allocated"
                )
        
        return warnings
    
    @staticmethod
    def optimize_allocation(allocation_df: pd.DataFrame, constraints: Dict) -> pd.DataFrame:
        """Optimize allocation based on constraints"""
        
        # Implement optimization logic based on constraints
        # This could include:
        # - Minimum order quantities
        # - Customer credit limits
        # - Product shelf life
        # - Transportation constraints
        
        optimized_df = allocation_df.copy()
        
        # Example: Apply credit limit constraint
        if 'credit_limits' in constraints:
            credit_limits = constraints['credit_limits']
            
            for customer, limit in credit_limits.items():
                customer_orders = optimized_df[optimized_df['customer'] == customer]
                total_value = (customer_orders['allocated_qty'] * 
                             customer_orders.get('unit_price', 0)).sum()
                
                if total_value > limit:
                    # Scale down allocation
                    scale_factor = limit / total_value
                    mask = optimized_df['customer'] == customer
                    optimized_df.loc[mask, 'allocated_qty'] *= scale_factor
        
        return optimized_df