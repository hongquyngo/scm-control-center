# utils/db_helpers.py
import pandas as pd
from sqlalchemy import text
from utils.db import get_db_engine
import streamlit as st

@st.cache_data
def get_default_creator_id():
    """Get default creator ID from employees table"""
    try:
        engine = get_db_engine()
        query = text("""
            SELECT id 
            FROM employees 
            WHERE status = 'ACTIVE'
            ORDER BY id
            LIMIT 1
        """)
        result = pd.read_sql(query, engine)
        if not result.empty:
            return int(result.iloc[0]['id'])
    except Exception as e:
        st.warning(f"Cannot get creator ID: {str(e)}")
    return 1