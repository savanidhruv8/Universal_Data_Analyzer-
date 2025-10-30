# main.py
import streamlit as st
import time

# Configure page
st.set_page_config(
    page_title="Universal Data Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling with animations
def load_css():
    st.markdown("""
    <style>
        /* Import Google Fonts */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        
        * {
            font-family: 'Inter', sans-serif;
        }
        
        /* Animations */
        @keyframes slideInFromTop {
            from {
                opacity: 0;
                transform: translateY(-30px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        @keyframes slideInFromLeft {
            from {
                opacity: 0;
                transform: translateX(-30px);
            }
            to {
                opacity: 1;
                transform: translateX(0);
            }
        }
        
        @keyframes fadeInUp {
            from {
                opacity: 0;
                transform: translateY(20px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        @keyframes shimmer {
            0% {
                background-position: -1000px 0;
            }
            100% {
                background-position: 1000px 0;
            }
        }
        
        /* Main container styling */
        .main-header {
            background: linear-gradient(135deg, #1a365d 0%, #2c5282 50%, #2a4365 100%);
            padding: 3rem 2rem;
            border-radius: 20px;
            margin-bottom: 2rem;
            text-align: center;
            box-shadow: 0 10px 30px rgba(26, 54, 93, 0.4);
            position: relative;
            overflow: hidden;
            animation: slideInFromTop 0.8s ease-out;
        }
        
        .main-header::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: linear-gradient(45deg, transparent, rgba(255,255,255,0.1), transparent);
            animation: shimmer 3s infinite;
        }
        
        .main-title {
            color: white;
            font-size: 3rem;
            font-weight: 700;
            margin-bottom: 0.5rem;
            text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.2);
            animation: fadeInUp 1s ease-out 0.3s both;
        }
        
        .main-subtitle {
            color: rgba(255, 255, 255, 0.95);
            font-size: 1.2rem;
            margin-bottom: 0;
            animation: fadeInUp 1s ease-out 0.5s both;
        }
        
        /* Container for better structure */
        .main-container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 0 1rem;
        }
        
        /* Section styling */
        .section-container {
            margin-bottom: 3rem;
        }
        
        /* Card styling */
        .feature-card {
            background: white;
            padding: 2.5rem 2rem;
            border-radius: 16px;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
            transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
            border: 1px solid rgba(26, 54, 93, 0.1);
            height: 100%;
            position: relative;
            overflow: hidden;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            text-align: center;
        }
        
        .feature-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 4px;
            background: linear-gradient(90deg, #1a365d, #2c5282);
            transform: scaleX(0);
            transition: transform 0.3s ease;
        }
        
        .feature-card:hover::before {
            transform: scaleX(1);
        }
        
        .feature-card:hover {
            transform: translateY(-8px);
            box-shadow: 0 12px 40px rgba(26, 54, 93, 0.2);
            border-color: #1a365d;
        }
        
        .card-icon {
            font-size: 3.5rem;
            margin-bottom: 1.5rem;
            display: block;
            text-align: center;
            transition: transform 0.3s ease;
        }
        
        .feature-card:hover .card-icon {
            transform: scale(1.1) rotate(5deg);
        }
        
        .card-title {
            font-size: 1.4rem;
            font-weight: 600;
            color: #2d3748;
            margin-bottom: 1.5rem;
            text-align: center;
        }
        
        /* Button styling - White Glass Effect */
        .stButton > button {
            width: 100%;
            background: rgba(255, 255, 255, 0.25);
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
            color: white !important;
            border: 1px solid rgba(255, 255, 255, 0.18);
            padding: 0.875rem 1.5rem;
            border-radius: 12px;
            font-weight: 600;
            font-size: 1rem;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
            position: relative;
            overflow: hidden;
        }
        
        .stButton > button::before {
            content: '';
            position: absolute;
            top: 50%;
            left: 50%;
            width: 0;
            height: 0;
            border-radius: 50%;
            background: rgba(255, 255, 255, 0.5);
            transform: translate(-50%, -50%);
            transition: width 0.6s, height 0.6s;
        }
        
        .stButton > button:hover::before {
            width: 300px;
            height: 300px;
        }
        
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(0, 0, 0, 0.15);
            background: rgba(255, 255, 255, 0.35);
        }
        
        .stButton > button:active {
            transform: translateY(0);
        }
        
        /* Sidebar styling */
        .sidebar-content {
            padding: 1.5rem;
            animation: slideInFromLeft 0.6s ease-out;
        }
        
        .sidebar-title {
            font-size: 1.3rem;
            font-weight: 600;
            color: #2d3748;
            margin-bottom: 1.5rem;
            padding-bottom: 0.75rem;
            border-bottom: 3px solid #1a365d;
            position: relative;
        }
        
        .sidebar-title::after {
            content: '';
            position: absolute;
            bottom: -3px;
            left: 0;
            width: 50px;
            height: 3px;
            background: #2c5282;
        }
        
        /* Stats container */
        .stats-container {
            background: linear-gradient(135deg, #f6f8fb 0%, #e9ecef 100%);
            padding: 2rem;
            border-radius: 16px;
            margin-bottom: 2rem;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
            animation: fadeInUp 0.8s ease-out 0.6s both;
        }
        
        .stat-item {
            text-align: center;
            padding: 1.5rem;
            transition: transform 0.3s ease;
        }
        
        .stat-item:hover {
            transform: scale(1.05);
        }
        
        .stat-number {
            font-size: 2.5rem;
            font-weight: 700;
            background: linear-gradient(135deg, #1a365d, #2c5282);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-bottom: 0.5rem;
        }
        
        .stat-label {
            font-size: 1rem;
            color: #4a5568;
            font-weight: 500;
        }
        
        /* Footer styling */
        .footer {
            text-align: center;
            padding: 2.5rem;
            margin-top: 3rem;
            border-top: 2px solid #e2e8f0;
            color: #718096;
            animation: fadeInUp 1s ease-out 0.8s both;
        }
        
        /* Hide streamlit elements */
        .stDeployButton {
            display: none;
        }
        
        /* Section headers */
        .section-header {
            font-size: 2rem;
            font-weight: 600;
            color: #2d3748;
            margin-bottom: 2rem;
            position: relative;
            padding-left: 1rem;
            animation: slideInFromLeft 0.8s ease-out 0.4s both;
            text-align: center;
        }
        
        .section-header::before {
            content: '';
            position: absolute;
            left: 50%;
            top: 100%;
            transform: translateX(-50%);
            width: 60px;
            height: 4px;
            background: linear-gradient(135deg, #1a365d, #2c5282);
            border-radius: 2px;
            margin-top: 0.5rem;
        }
        
        /* Loading animation */
        .loading-spinner {
            display: inline-block;
            width: 20px;
            height: 20px;
            border: 3px solid rgba(26, 54, 93, 0.3);
            border-radius: 50%;
            border-top-color: #1a365d;
            animation: spin 1s ease-in-out infinite;
        }
        
        @keyframes spin {
            to { transform: rotate(360deg); }
        }
        
        /* Responsive adjustments */
        @media (max-width: 768px) {
            .main-title {
                font-size: 2rem;
            }
            
            .feature-card {
                padding: 2rem 1.5rem;
            }
            
            .card-icon {
                font-size: 3rem;
            }
        }
    </style>
    """, unsafe_allow_html=True)

def create_feature_card(icon, title, button_key, page_name, target_page, delay=0):
    """Create a feature card with icon, title and button only"""
    # Add animation delay
    animation_style = f"animation: fadeInUp 0.6s ease-out {delay}s both;"
    
    st.markdown(f"""
    <div class="feature-card" style="{animation_style}">
        <span class="card-icon">{icon}</span>
        <div class="card-title">{title}</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Add loading state
    if st.button(f"🚀 Process {title}", key=button_key, use_container_width=True):
        with st.spinner(f"Loading {title} analyzer..."):
            time.sleep(0.5)  # Simulate loading
        st.session_state.page = page_name
        st.switch_page(target_page)

def show_stats():
    """Display statistics cards with animations"""
    pass

def main():
    # Load custom CSS
    load_css()
    
    # Main container for better structure
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    
    # Animated header
    st.markdown("""
    <div class="main-header">
        <h1 class="main-title">📊 Universal Data Analyzer</h1>
        <p class="main-subtitle">Transform your data into insights with our powerful analysis tools</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar with animations
    with st.sidebar:
        st.markdown('<div class="sidebar-content">', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Main content with section header
    st.markdown('<div class="section-container">', unsafe_allow_html=True)
    st.markdown('<h2 class="section-header">Choose a Data Format to Process</h2>', unsafe_allow_html=True)
    
    # Feature cards in a grid with staggered animations
    col1, col2, col3 = st.columns(3)
    
    features = [
        ("📂", "CSV File", "csv_button", "csv", "pages/csv_analyzer.py", 0.7),
        ("📄", "Text File", "txt_button", "trocr", "pages/txt_analyzer.py", 0.8),
        ("🎵", "Audio File", "audio_button", "audio", "pages/audio_analyzer.py", 0.9),
        ("📋", "JSON File", "json_button", "Json", "pages/json_analyzer.py", 1.0),
        ("📊", "Excel File", "excel_button", "excel", "pages/excel_analyzer.py", 1.1),
        ("🖼️", "Image File", "image_button", "image", "pages/image_analyzer.py", 1.2)
    ]
    
    for i, (icon, title, button_key, page_name, target_page, delay) in enumerate(features):
        col = [col1, col2, col3][i % 3]
        with col:
            create_feature_card(icon, title, button_key, page_name, target_page, delay)
            
            # Add spacing between cards
            if i < len(features) - 3:
                st.markdown("<br>", unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Footer with animation
    st.markdown("""
    <div class="footer">
        <p style="font-size: 1.1rem; margin-bottom: 0.5rem;">© 2024 Universal Data Analyzer</p>
        <p style="font-size: 0.9rem; color: #a0aec0;">
            Built with ❤️ using Streamlit | Version 2.1.0 | Last Updated: October 2025
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Close main container
    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()