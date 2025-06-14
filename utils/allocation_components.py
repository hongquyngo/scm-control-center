"""
Allocation Components - Reusable UI components for allocation module
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import List

from utils.formatters import format_number

class AllocationComponents:
    """Reusable UI components for allocation module"""
    
    @staticmethod
    def get_method_description(method: str) -> str:
        """Get detailed description of allocation method"""
        descriptions = {
            'FCFS': """
                **First Come First Served (FCFS) Allocation**
                
                This method prioritizes orders based on their Expected Time of Delivery (ETD):
                - Orders with earlier ETD dates are allocated first
                - Ensures time-sensitive deliveries are prioritized
                - Best for perishable goods or time-critical orders
                - Fair and transparent allocation process
                
                **Example**: If you have 100 units available and 3 orders:
                - Order A (ETD: Jan 1) - 50 units → Allocated: 50 units
                - Order B (ETD: Jan 5) - 60 units → Allocated: 50 units
                - Order C (ETD: Jan 10) - 40 units → Allocated: 0 units
            """,
            
            'PRIORITY': """
                **Priority-Based Allocation**
                
                This method allocates based on customer importance scores:
                - Set priority levels (1-10) for each customer
                - Higher priority customers are served first
                - Within same priority, FIFO rules apply
                - Ideal for VIP customers or strategic accounts
                
                **Example**: With 100 units available:
                - Customer A (Priority: 9) - 60 units → Allocated: 60 units
                - Customer B (Priority: 7) - 50 units → Allocated: 40 units
                - Customer C (Priority: 5) - 30 units → Allocated: 0 units
            """,
            
            'PRO_RATA': """
                **Pro-Rata (Proportional) Allocation**
                
                This method distributes supply proportionally to demand:
                - Each order receives the same percentage of their request
                - Fair distribution across all customers
                - Can set minimum allocation percentage
                - Best for equal treatment scenarios
                
                **Example**: With 100 units available and 200 units demanded:
                - Order A requests 100 units → Allocated: 50 units (50%)
                - Order B requests 60 units → Allocated: 30 units (50%)
                - Order C requests 40 units → Allocated: 20 units (50%)
            """,
            
            'MANUAL': """
                **Manual Allocation**
                
                This method gives you complete control:
                - Start with suggested allocations (pro-rata)
                - Manually adjust quantities as needed
                - Drag sliders or input exact amounts
                - Full flexibility for complex scenarios
                
                Use this when:
                - Special business rules apply
                - Need to handle exceptions
                - Combining multiple allocation strategies
            """
        }
        
        return descriptions.get(method, "Unknown allocation method")
    
    @staticmethod
    def show_supply_capability_table(allocation_df: pd.DataFrame):
        """Show supply capability by period for reference"""
        # Get unique products
        products = allocation_df['pt_code'].unique()
        
        # Get supply data from session state
        supply_data = st.session_state.get('supply_filtered', pd.DataFrame())
        
        if supply_data.empty:
            st.info("No supply data available for reference")
            return
        
        # Filter supply for selected products
        supply_filtered = supply_data[supply_data['pt_code'].isin(products)]
        
        # Group by period (simplified view)
        # This should ideally show supply by period matching the demand periods
        supply_summary = supply_filtered.groupby(['pt_code', 'legal_entity']).agg({
            'quantity': 'sum'
        }).reset_index()
        
        # Display in expandable section
        with st.expander("View Supply Capability Details", expanded=False):
            for pt_code in products:
                product_supply = supply_summary[supply_summary['pt_code'] == pt_code]
                if not product_supply.empty:
                    st.write(f"**{pt_code}**")
                    for _, row in product_supply.iterrows():
                        st.write(f"- {row['legal_entity']}: {format_number(row['quantity'])} available")
                else:
                    st.write(f"**{pt_code}**: No supply data")

    @staticmethod
    def validate_against_supply_capability(edited_df: pd.DataFrame) -> List[str]:
        """Validate allocations against supply capability by period"""
        warnings = []
        
        # Group by product and allocated_etd to check supply availability
        period_allocations = edited_df.groupby(['pt_code', 'legal_entity', 'allocated_etd']).agg({
            'allocated_qty': 'sum'
        }).reset_index()
        
        # Get supply data
        supply_data = st.session_state.get('supply_filtered', pd.DataFrame())
        
        if not supply_data.empty:
            # Check each period allocation against available supply
            for _, alloc in period_allocations.iterrows():
                # This is simplified - in real implementation, should check supply availability by date
                product_supply = supply_data[
                    (supply_data['pt_code'] == alloc['pt_code']) &
                    (supply_data['legal_entity'] == alloc['legal_entity'])
                ]['quantity'].sum()
                
                if alloc['allocated_qty'] > product_supply:
                    warnings.append(
                        f"Product {alloc['pt_code']} on {alloc['allocated_etd']}: "
                        f"Allocated {alloc['allocated_qty']} exceeds available supply {product_supply}"
                    )
        
        return warnings

    @staticmethod
    def show_editable_allocation_table(allocation_df: pd.DataFrame) -> pd.DataFrame:
        """Show editable allocation table for manual adjustments"""
        
        st.markdown("##### 📝 Adjust Allocations Manually")
        st.info("Modify allocated quantities and ETD dates below. Supply capability is shown for reference.")
        st.warning("⚠️ **Note**: Allocated quantities must be whole numbers (natural numbers) matching the product UOM")
        
        # Show supply capability by period instead of simple status
        st.markdown("**📊 Supply Capability Reference**")
        AllocationComponents.show_supply_capability_table(allocation_df)
        
        st.markdown("---")
        
        # Create editable dataframe
        edited_df = allocation_df.copy()
        
        # Add calculated allocation column (preserve original calculation)
        edited_df['calculated_allocation'] = edited_df['allocated_qty'].copy()
        
        # Prepare display columns with new structure
        display_columns = [
            'pt_code', 
            'product_name',
            'package_size',  # Added package size
            'customer', 
            'etd',  # This is requested ETD
            'allocated_etd',  # This is editable allocated ETD
            'requested_qty', 
            'calculated_allocation',  # System calculated
            'allocated_qty',  # Actual allocated (editable)
            'standard_uom'
        ]
        
        # Initialize allocated_etd if not exists
        if 'allocated_etd' not in edited_df.columns:
            edited_df['allocated_etd'] = edited_df['etd'].copy()
        
        # Use st.data_editor for editing
        edited_result = st.data_editor(
            edited_df[display_columns],
            column_config={
                "pt_code": st.column_config.TextColumn("PT Code", disabled=True),
                "product_name": st.column_config.TextColumn("Product", disabled=True),
                "package_size": st.column_config.TextColumn("Pack Size", disabled=True),
                "customer": st.column_config.TextColumn("Customer", disabled=True),
                "etd": st.column_config.DateColumn(
                    "Requested ETD", 
                    disabled=True,
                    help="Original ETD from demand"
                ),
                "allocated_etd": st.column_config.DateColumn(
                    "Allocated ETD",
                    help="Adjust if allocation will be delivered on different date"
                ),
                "requested_qty": st.column_config.NumberColumn(
                    "Requested",
                    disabled=True,
                    format="%.0f"
                ),
                "calculated_allocation": st.column_config.NumberColumn(
                    "System Calculated",
                    disabled=True,
                    format="%.0f",
                    help="Quantity calculated by selected method"
                ),
                "allocated_qty": st.column_config.NumberColumn(
                    "Actual Allocated",
                    min_value=0,
                    step=1,
                    format="%.0f",
                    help="Enter actual allocation (whole numbers only)"
                ),
                "standard_uom": st.column_config.TextColumn("UOM", disabled=True)
            },
            hide_index=True,
            use_container_width=True,
            num_rows="fixed"
        )
        
        # Validate that allocated quantities are whole numbers
        edited_result['allocated_qty'] = edited_result['allocated_qty'].round(0).astype(int)
        
        # Update the dataframe with edits
        edited_df['allocated_qty'] = edited_result['allocated_qty']
        edited_df['allocated_etd'] = edited_result['allocated_etd']
        
        # Recalculate fulfillment rate based on actual allocated
        edited_df['fulfillment_rate'] = (
            edited_df['allocated_qty'] / edited_df['requested_qty'] * 100
        ).fillna(0).clip(upper=100)
        
        # Validate allocations against supply capability
        validation_warnings = AllocationComponents.validate_against_supply_capability(edited_df)
        
        if validation_warnings:
            st.warning("⚠️ Validation Issues:")
            for warning in validation_warnings:
                st.write(f"- {warning}")
        
        return edited_df

    @staticmethod
    def create_fulfillment_chart_by_product(allocation_df: pd.DataFrame) -> go.Figure:
        """Create fulfillment chart grouped by product using actual allocated quantities"""
        
        # Group by product
        product_summary = allocation_df.groupby(['pt_code', 'product_name']).agg({
            'requested_qty': 'sum',
            'allocated_qty': 'sum'  # This is now actual allocated
        }).reset_index()
        
        product_summary['fulfillment_rate'] = (
            product_summary['allocated_qty'] / product_summary['requested_qty'] * 100
        ).fillna(0)
        
        # Sort by fulfillment rate
        product_summary = product_summary.sort_values('fulfillment_rate')
        
        # Create chart
        fig = go.Figure()
        
        # Add requested quantity bars
        fig.add_trace(go.Bar(
            name='Requested',
            y=product_summary['pt_code'],
            x=product_summary['requested_qty'],
            orientation='h',
            marker_color='lightgray',
            text=product_summary['requested_qty'].apply(lambda x: f'{x:,.0f}'),
            textposition='auto'
        ))
        
        # Add actual allocated quantity bars
        fig.add_trace(go.Bar(
            name='Actual Allocated',
            y=product_summary['pt_code'],
            x=product_summary['allocated_qty'],
            orientation='h',
            marker_color='green',
            text=product_summary['fulfillment_rate'].apply(lambda x: f'{x:.1f}%'),
            textposition='auto'
        ))
        
        fig.update_layout(
            title='Fulfillment Rate by Product (Actual Allocated)',
            xaxis_title='Quantity',
            yaxis_title='Product Code',
            barmode='overlay',
            height=400,
            showlegend=True
        )
        
        return fig

 
    @staticmethod
    def create_fulfillment_chart_by_customer(allocation_df: pd.DataFrame) -> go.Figure:
        """Create fulfillment chart grouped by customer"""
        
        # Group by customer
        customer_summary = allocation_df.groupby('customer').agg({
            'requested_qty': 'sum',
            'allocated_qty': 'sum'
        }).reset_index()
        
        customer_summary['fulfillment_rate'] = (
            customer_summary['allocated_qty'] / customer_summary['requested_qty'] * 100
        ).fillna(0)
        
        # Sort by allocated quantity
        customer_summary = customer_summary.sort_values('allocated_qty', ascending=False).head(15)
        
        # Create donut chart
        fig = go.Figure(data=[go.Pie(
            labels=customer_summary['customer'],
            values=customer_summary['allocated_qty'],
            hole=.3,
            textinfo='label+percent',
            textposition='auto',
            marker=dict(
                colors=px.colors.qualitative.Set3
            )
        )])
        
        fig.update_layout(
            title='Allocation Distribution by Customer',
            height=400,
            showlegend=False
        )
        
        return fig
    
    @staticmethod
    def create_allocation_summary_chart(summary_df: pd.DataFrame, group_by: str) -> go.Figure:
        """Create allocation summary chart"""
        
        if group_by == 'product':
            x_col = 'pt_code'
            title = 'Allocation Summary by Product'
        else:
            x_col = 'customer_name'
            title = 'Allocation Summary by Customer'
        
        # Prepare data
        summary_df = summary_df.sort_values('allocated_qty', ascending=False).head(10)
        
        # Create figure
        fig = go.Figure()
        
        # Add traces
        fig.add_trace(go.Bar(
            name='Requested',
            x=summary_df[x_col],
            y=summary_df['requested_qty'],
            marker_color='lightblue',
            text=summary_df['requested_qty'].apply(lambda x: f'{x:,.0f}'),
            textposition='auto'
        ))
        
        fig.add_trace(go.Bar(
            name='Allocated',
            x=summary_df[x_col],
            y=summary_df['allocated_qty'],
            marker_color='green',
            text=summary_df['allocated_qty'].apply(lambda x: f'{x:,.0f}'),
            textposition='auto'
        ))
        
        fig.add_trace(go.Bar(
            name='Delivered',
            x=summary_df[x_col],
            y=summary_df['delivered_qty'],
            marker_color='darkgreen',
            text=summary_df['delivered_qty'].apply(lambda x: f'{x:,.0f}'),
            textposition='auto'
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title=group_by.title(),
            yaxis_title='Quantity',
            barmode='group',
            height=400,
            showlegend=True
        )
        
        return fig
    
    @staticmethod
    def show_allocation_method_comparison():
        """Show visual comparison of allocation methods"""
        
        with st.expander("📊 Compare Allocation Methods"):
            # Create sample data
            sample_data = pd.DataFrame({
                'Customer': ['Customer A', 'Customer B', 'Customer C'],
                'Demand': [100, 80, 60],
                'FCFS': [100, 40, 0],
                'Priority': [80, 60, 0],
                'Pro-Rata': [58, 47, 35]
            })
            
            # Melt for plotting
            melted = sample_data.melt(
                id_vars=['Customer', 'Demand'],
                value_vars=['FCFS', 'Priority', 'Pro-Rata'],
                var_name='Method',
                value_name='Allocated'
            )
            
            # Create comparison chart
            fig = px.bar(
                melted,
                x='Customer',
                y='Allocated',
                color='Method',
                barmode='group',
                title='Allocation Method Comparison (140 units available for 240 units demand)'
            )
            
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)
            
            st.caption("""
                This example shows how different methods allocate 140 units of supply across 3 customers:
                - **FCFS**: Serves Customer A fully, then partially serves B
                - **Priority**: Assumes A has highest priority, balanced between A and B
                - **Pro-Rata**: Each customer gets ~58% of their demand
            """)
    
    @staticmethod
    def show_allocation_timeline(allocation_df: pd.DataFrame):
        """Show allocation timeline visualization"""
        
        if allocation_df.empty or 'etd' not in allocation_df.columns:
            return
        
        # Prepare timeline data
        timeline_df = allocation_df.copy()
        timeline_df['etd'] = pd.to_datetime(timeline_df['etd'])
        
        # Group by date
        daily_summary = timeline_df.groupby(timeline_df['etd'].dt.date).agg({
            'requested_qty': 'sum',
            'allocated_qty': 'sum'
        }).reset_index()
        
        # Create timeline chart
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=daily_summary['etd'],
            y=daily_summary['requested_qty'],
            mode='lines+markers',
            name='Requested',
            line=dict(color='blue', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=daily_summary['etd'],
            y=daily_summary['allocated_qty'],
            mode='lines+markers',
            name='Allocated',
            line=dict(color='green', width=2),
            fill='tonexty',
            fillcolor='rgba(0,255,0,0.1)'
        ))
        
        fig.update_layout(
            title='Allocation Timeline',
            xaxis_title='Date',
            yaxis_title='Quantity',
            height=300,
            showlegend=True
        )
        
        return fig
    
    @staticmethod
    def show_allocation_kpis(allocation_df: pd.DataFrame):
        """Show key performance indicators for allocation"""
        
        col1, col2, col3, col4 = st.columns(4)
        
        # Calculate KPIs
        total_orders = len(allocation_df)
        fully_allocated = len(allocation_df[allocation_df['allocated_qty'] >= allocation_df['requested_qty']])
        partial_allocated = len(allocation_df[
            (allocation_df['allocated_qty'] > 0) & 
            (allocation_df['allocated_qty'] < allocation_df['requested_qty'])
        ])
        not_allocated = len(allocation_df[allocation_df['allocated_qty'] == 0])
        
        with col1:
            st.metric(
                "Total Orders",
                total_orders,
                help="Total number of orders in allocation"
            )
        
        with col2:
            st.metric(
                "Fully Allocated",
                fully_allocated,
                f"{fully_allocated/total_orders*100:.1f}%",
                help="Orders that received 100% of requested quantity"
            )
        
        with col3:
            st.metric(
                "Partially Allocated",
                partial_allocated,
                f"{partial_allocated/total_orders*100:.1f}%",
                help="Orders that received some but not all requested quantity"
            )
        
        with col4:
            st.metric(
                "Not Allocated",
                not_allocated,
                f"{not_allocated/total_orders*100:.1f}%",
                delta_color="inverse",
                help="Orders that received no allocation"
            )
    
    @staticmethod
    def create_allocation_heatmap(allocation_df: pd.DataFrame) -> go.Figure:
        """Create heatmap of allocation by product and customer"""
        
        # Pivot data
        heatmap_data = allocation_df.pivot_table(
            index='customer',
            columns='pt_code',
            values='fulfillment_rate',
            aggfunc='mean'
        )
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=heatmap_data.values,
            x=heatmap_data.columns,
            y=heatmap_data.index,
            colorscale='RdYlGn',
            zmid=50,
            text=heatmap_data.values,
            texttemplate='%{text:.0f}%',
            textfont={"size": 10},
            colorbar=dict(title="Fulfillment %")
        ))
        
        fig.update_layout(
            title='Fulfillment Rate Heatmap',
            xaxis_title='Product Code',
            yaxis_title='Customer',
            height=400
        )
        
        return fig