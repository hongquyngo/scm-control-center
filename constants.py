# constants.py

# ======================
# Color Definitions (Centralize all chart colors here)
# ======================
COLORS = {
    "revenue": "#FFA500",              # orange
    "gross_profit": "#1f77b4",         # blue
    "gross_profit_percent": "#800080" , # purple
    "customer_count": "#27ae60"  # ✅ Add this line (green)
}
# 🎨 Fixed color mapping for KPI Centers (TERRITORY + VERTICAL)
COLOR_CENTER_MAP = {
    # VERTICAL
    "VAM": "#ff6f61",      # Coral
    "COEM": "#6b5b95",     # Indigo

    # TERRITORY
    "HAN": "#1f77b4",      # Blue
    "DAN": "#ff7f0e",      # Orange
    "SGN": "#2ca02c",      # Green
    "PTV": "#d62728",      # Red
    "PTP": "#9467bd",      # Purple
    "ROSEA": "#17becf",    # Cyan
    "ROW": "#bcbd22",      # Olive
    "SEA": "#8c564b",      # Brown
    "OVERSEA": "#e377c2",  # Pink
    "ALL": "#7f7f7f",      # Gray

    # Fallback
    "Unmapped": "#aec7e8"  # Light Blue
}


# ======================
# Month Order (for sorting consistently across all charts)
# ======================
MONTH_ORDER = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
]

# ======================
# Metric Display Names (for unified naming across charts)
# ======================
METRIC_LABELS = {
    "adjusted_revenue_usd": "Revenue (USD)",
    "invoiced_gross_profit_usd": "Gross Profit (USD)"
}

METRIC_ORDER = ["Revenue (USD)", "Gross Profit (USD)"]

# ======================
# Chart Size Settings (centralize all size configs)
# ======================
CHART_WIDTH = 800
CHART_HEIGHT = 400

PIE_CHART_WIDTH = 400
PIE_CHART_HEIGHT = 300
