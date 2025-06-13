# pages/0_🔐_Login.py - Login page for SCM app

import streamlit as st
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from utils.auth import AuthManager
import logging

logger = logging.getLogger(__name__)

# Page config
st.set_page_config(
    page_title="SCM Login",
    page_icon="🔐",
    layout="wide"
)

# Initialize auth manager
auth_manager = AuthManager()

# Check if already logged in
if auth_manager.check_session():
    st.switch_page("main.py")

# Custom CSS for login page
st.markdown("""
<style>
    .login-container {
        max-width: 400px;
        margin: auto;
        padding: 20px;
    }
    .login-header {
        text-align: center;
        margin-bottom: 30px;
    }
    .stButton > button {
        width: 100%;
        background-color: #366092;
        color: white;
    }
    .stButton > button:hover {
        background-color: #2d5078;
    }
</style>
""", unsafe_allow_html=True)

# Login form
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    st.markdown("## 🏭 Supply Chain Control Center")
    st.markdown("### Please login to continue")
    
    # Add some space
    st.markdown("<br>", unsafe_allow_html=True)
    
    with st.form("login_form", clear_on_submit=True):  # Changed to clear_on_submit=True
        username = st.text_input(
            "Username",
            placeholder="Enter your username"
        )
        
        password = st.text_input(
            "Password",
            type="password",
            placeholder="Enter your password"
        )
        
        col_submit, col_space = st.columns([1, 1])
        
        with col_submit:
            submitted = st.form_submit_button(
                "🔐 Login",
                type="primary",
                use_container_width=True
            )
        
        if submitted:
            if not username or not password:
                st.error("❌ Please enter both username and password")
            else:
                # Show loading spinner
                with st.spinner("🔄 Authenticating..."):
                    success, result = auth_manager.authenticate(username, password)
                
                if success:
                    # Set up session
                    auth_manager.login(result)
                    st.success(f"✅ Welcome, {result['full_name']}!")
                    
                    # Don't try to clear form fields - let clear_on_submit handle it
                    # Just rerun to redirect
                    st.rerun()
                else:
                    st.error(f"❌ {result.get('error', 'Login failed')}")
    
    # Additional information
    st.markdown("---")
    
    with st.expander("ℹ️ Login Help"):
        st.markdown("""
        **Having trouble logging in?**
        - Make sure your username and password are correct
        - Passwords are case-sensitive
        - Contact your administrator if you've forgotten your password
        - Your session will expire after 8 hours of inactivity
        """)
    
    # Show version and environment info
    st.caption(f"SCM Control Center v1.0")
    
    # Debug mode (hidden checkbox)
    if st.checkbox("🐛", value=False, label_visibility="collapsed"):
        st.info("Debug mode - Check your database connection")
        try:
            from utils.db import get_db_engine
            engine = get_db_engine()
            with engine.connect() as conn:
                from sqlalchemy import text
                result = conn.execute(text("SELECT 1"))
                st.success("✅ Database connection successful")
        except Exception as e:
            st.error(f"❌ Database connection failed: {str(e)}")