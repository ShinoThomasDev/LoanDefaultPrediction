import streamlit as st

def inject_custom_css():
    """Injects Axis Bank style CSS into the Streamlit app with strict overrides."""
    axis_css = """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
        
        /* Global App Background */
        .stApp {
            font-family: 'Inter', sans-serif !important;
            background-color: #f4f6f9 !important; /* Soft grey background to reduce white glare */
        }
        
        /* Force Main Container to White with Shadow */
        [data-testid="stMainBlockContainer"] {
            background-color: #ffffff !important;
            padding: 3rem 4rem !important;
            border-radius: 16px !important;
            box-shadow: 0 4px 24px rgba(0,0,0,0.06) !important;
            margin-top: 2rem !important;
            margin-bottom: 2rem !important;
            max-width: 1200px !important;
            border: 1px solid #eef2f6 !important;
        }
        
        /* Fix text colors overriding Streamlit dark mode */
        p, span, label, .stMarkdown p, .stMarkdown span {
            color: #334155 !important; /* Slate grey instead of harsh black */
        }
        
        /* Markdown Titles to Burgundy */
        h1, h2, h3, h4, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4 {
            color: #97144D !important; /* Axis Bank Burgundy */
            font-weight: 600 !important;
            letter-spacing: -0.5px !important;
        }
        
        /* Top Banner Navigation Mockup */
        .top-banner {
            background-color: #97144D !important;
            color: white !important;
            padding: 0 !important;
            margin-top: -60px !important; /* Offset default padding */
            margin-left: -4rem !important;
            margin-right: -4rem !important;
            margin-bottom: 30px !important;
            border-radius: 16px 16px 0 0 !important;
            box-shadow: 0 4px 12px rgba(151, 20, 77, 0.15) !important;
        }
        
        .top-banner * {
            color: white !important;
        }
        
        .nav-container {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 20px 30px;
        }
        
        .brand {
            font-size: 24px;
            font-weight: 700;
            display: flex;
            align-items: center;
            gap: 15px;
            letter-spacing: 0.5px;
    
        }
        
        .brand-subtitle {
            font-size: 14px;
            font-weight: 400;
            opacity: 0.9;
            border-left: 1px solid rgba(255,255,255,0.3);
            padding-left: 15px;
            color: white !important;
        }
        
        /* Submit Button */
        .stButton>button {
            background-color: white !important;
            color: black !important;
            border: none !important;
            border-radius: 8px !important;
            padding: 12px 28px !important;
            font-weight: 600 !important;
            transition: all 0.2s ease !important;
            text-transform: uppercase !important;
            letter-spacing: 0.5px !important;
            font-size: 14px !important;
        }
        .stButton>button:hover {
            box-shadow: 0 6px 16px rgba(151, 20, 77, 0.3) !important;
            transform: translateY(-2px) !important;
            color: white !important;
        }
        .stButton>button p {
            color: white !important;
        }
        
        /* Hide Streamlit top bar and footer */
        [data-testid="stHeader"] {visibility: hidden;}
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        
        /* Form Box */
        [data-testid="stForm"] {
            border: 1px solid #e2e8f0 !important;
            border-radius: 12px !important;
            padding: 35px !important;
            background: #f8fafc !important; /* Soft grey background for details section */
            box-shadow: 0 2px 10px rgba(0,0,0,0.02) !important;
        }
        
        /* Input Box Fixes for Visibility */
        .stTextInput input, .stNumberInput input, .stSelectbox div[data-baseweb="select"] {
            background-color: #f8fafc !important; /* Soft blue-grey inner fill */
            color: #1e293b !important; /* Darkslate text */
            border: 1px solid #cbd5e1 !important;
            border-radius: 6px !important;
            padding: 8px 12px !important;
            font-weight: 500 !important;
        }
        
        /* Streamlit Tabs active state color fix */
        [data-baseweb="tab-list"] button {
            color: #64748b !important;
            font-weight: 600 !important;
        }
        [data-baseweb="tab-list"] button[aria-selected="true"] {
            color: #97144D !important;
        }
        [data-baseweb="tab-highlight"] {
            background-color: #97144D !important;
        }
    </style>
    """
    st.markdown(axis_css, unsafe_allow_html=True)

def render_header():
    """Renders the top banner for RiskPilot."""
    header_html = """
    <div class="top-banner">
        <div class="nav-container">
            <div class="brand">
                RiskPilot
                <span class="brand-subtitle">Underwriting Platform</span>
            </div>
        </div>
    </div>
    """
    st.markdown(header_html, unsafe_allow_html=True)
