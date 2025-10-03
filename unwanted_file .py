# main.py
import streamlit as st

st.set_page_config(page_title="Universal Data Analyzer", page_icon="📊", layout="wide")

def main():
    st.title("📊 Universal Data Analyzer")
    st.markdown("Choose a data format to process:")

    col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
    
    with col1:
        if st.button("📂 Process CSV File", key="csv_button"):
            st.session_state.page = "csv"
            st.switch_page("pages/csv_analyzer.py")

    with col2:
        if st.button("📄 Process txt", key="txt_button"):
            st.session_state.page = "trocr"
            st.switch_page("pages/txt_analyzer.py")

    with col3:
        if st.button("📄 Process AUDIO File", key="audio_button"):
            st.session_state.page = "audio"
            st.switch_page("pages/audio_analyzer.py")

    with col4:
        if st.button("📄 Process Json", key="Json_button"):
            st.session_state.page = "Json"
            st.switch_page("pages/json_analyzer.py")
    
    with col5:
        if st.button("📄 Process Excel", key="excel_button"):
            st.session_state.page = "excel"
            st.switch_page("pages/excel_analyzer.py")

    with col6:
        if st.button("📄 Process Image", key="image_button"):
            st.session_state.page = "image"
            st.switch_page("pages/image_analyzer.py")

    
if __name__ == "__main__":
    main()








    import streamlit as st
import time

# Professional and visually appealing color palette
PRIMARY_COLOR = "#4361ee"      # Soft blue - primary accent
SECONDARY_COLOR = "#3f37c9"    # Deeper blue - secondary elements
BACKGROUND_COLOR = "#f8f9fa"   # Light gray - background
TEXT_COLOR = "#212529"         # Dark gray - main text
CARD_HEADER_COLOR = "#4895ef"  # Brighter blue - card headers
CARD_BG_COLOR = "#ffffff"      # White - card background
HOVER_COLOR = "#4cc9f0"        # Teal - hover effects

# Enhanced CSS with animations and full-screen support
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* Base styling for full screen */
html, body, [class*="css"] {{
    font-family: 'Inter', sans-serif;
    background-color: {BACKGROUND_COLOR};
    color: {TEXT_COLOR};
    margin: 0;
    padding: 0;
    overflow-x: hidden;
}}

/* Full-width container */
.stApp {{
    max-width: none !important;
    width: 100vw !important;
    padding: 0 !important;
    margin: 0 !important;
}}

/* Title styling with animation */
.stTitle {{
    font-weight: 700;
    background: linear-gradient(90deg, {PRIMARY_COLOR}, {SECONDARY_COLOR});
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align: center;
    margin-bottom: 0.5rem;
    font-size: 3rem;
    animation: fadeInDown 1s ease-out;
}}

/* Subtitle with animation */
.stMarkdown p {{
    text-align: center;
    color: {SECONDARY_COLOR};
    font-size: 1.3rem;
    max-width: 900px;
    margin: 0 auto 3rem auto;
    line-height: 1.6;
    animation: fadeInUp 1s ease-out 0.3s both;
}}

/* Button grid container */
.stHorizontalBlock > div {{
    display: flex !important;
    justify-content: center !important;
    flex-wrap: wrap !important;
    gap: 20px !important;
    padding: 20px !important;
}}

/* Button styling with enhanced animations */
.format-button {{
    transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
    cursor: pointer;
    border-radius: 16px;
    border: 3px solid #ffffff;
    overflow: hidden;
    opacity: 0;
    transform: translateY(30px);
    animation: slideInUp 0.6s ease-out forwards;
    background: linear-gradient(145deg, {CARD_BG_COLOR}, #f0f2f5);
    padding: 24px;
    width: 100%;
    height: 100px;
    box-shadow: 0 4px 15px rgba(67, 97, 238, 0.15);
    text-align: center;
    position: relative;
}}

.format-button:nth-child(1) {{ animation-delay: 0.1s; }}
.format-button:nth-child(2) {{ animation-delay: 0.2s; }}
.format-button:nth-child(3) {{ animation-delay: 0.3s; }}
.format-button:nth-child(4) {{ animation-delay: 0.4s; }}
.format-button:nth-child(5) {{ animation-delay: 0.5s; }}
.format-button:nth-child(6) {{ animation-delay: 0.6s; }}

/* Enhanced hover effects */
.format-button:hover {{
    transform: translateY(-8px) scale(1.05);
    box-shadow: 0 15px 30px rgba(67, 97, 238, 0.4);
    border: 3px solid {PRIMARY_COLOR};
    background: linear-gradient(145deg, #ffffff, #e6f0ff);
}}

/* Button title with hover animation */
.button-title {{
    color: {CARD_HEADER_COLOR} !important;
    transition: all 0.3s ease;
    font-size: 18px;
    font-weight: 700;
    margin-bottom: 12px;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 12px;
    text-shadow: 0 1px 2px rgba(0,0,0,0.1);
}}

.format-button:hover .button-title {{
    color: {HOVER_COLOR} !important;
    transform: translateX(8px);
}}

/* Add cursor pointer to indicate clickable buttons */
.stButton > button {{
    border: none !important;
    background: none !important;
    padding: 0 !important;
    height: auto !important;
    min-height: auto !important;
    width: 100% !important;
    text-align: center !important;
    position: relative;
    z-index: 1;
    cursor: pointer !important;
    transition: all 0.3s ease;
}}

/* Add visual feedback for button hover */
.stButton > button:hover {{
    cursor: pointer !important;
}}

/* Add a subtle pulse animation to indicate interactivity */
@keyframes pulse {{
    0% {{
        transform: scale(1);
    }}
    50% {{
        transform: scale(1.02);
    }}
    100% {{
        transform: scale(1);
    }}
}}

.format-button-container {{
    animation: pulse 2s infinite;
}}

/* Footer */
footer {{
    visibility: hidden;
}}

/* Divider with animation */
hr {{
    border: none;
    height: 2px;
    background: linear-gradient(90deg, transparent, {PRIMARY_COLOR}, transparent);
    margin: 3rem 0;
    animation: expandWidth 1.5s ease-out 1s both;
}}

/* Caption with fade-in */
.stCaption {{
    text-align: center;
    color: #6c757d;
    font-size: 1rem;
    animation: fadeIn 1s ease-out 1.2s both;
}}

/* Keyframe animations */
@keyframes fadeInDown {{
    from {{
        opacity: 0;
        transform: translateY(-30px);
    }}
    to {{
        opacity: 1;
        transform: translateY(0);
    }}
}}

@keyframes fadeInUp {{
    from {{
        opacity: 0;
        transform: translateY(30px);
    }}
    to {{
        opacity: 1;
        transform: translateY(0);
    }}
}}

@keyframes slideInUp {{
    to {{
        opacity: 1;
        transform: translateY(0);
    }}
}}

@keyframes expandWidth {{
    from {{
        width: 0;
    }}
    to {{
        width: 100%;
    }}
}}

@keyframes fadeIn {{
    from {{
        opacity: 0;
    }}
    to {{
        opacity: 1;
    }}
}}

/* Responsive design for full screen */
@media (max-width: 1024px) {{
    .stTitle {{
        font-size: 2.5rem;
    }}
    
    .stMarkdown p {{
        font-size: 1.2rem;
        padding: 0 20px;
    }}
    
    .stHorizontalBlock > div {{
        padding: 10px !important;
        gap: 15px !important;
    }}
}}

@media (max-width: 768px) {{
    .stTitle {{
        font-size: 2rem;
    }}
    
    .stMarkdown p {{
        font-size: 1.1rem;
    }}
    
    .format-button {{
        width: 100% !important;
        margin: 5px 0 !important;
    }}
}}
</style>
""", unsafe_allow_html=True)

st.set_page_config(
    page_title="Universal Data Analyzer",
    page_icon="📊",
    layout="wide",  # Full screen layout
    initial_sidebar_state="collapsed"
)

# Header with slight delay for animation sync
st.title("📊 Universal Data Analyzer")
st.markdown("Transform and analyze data in multiple formats with intelligent preprocessing and seamless animations")

# Data format buttons
formats = [
    {
        "title": "CSV Analyzer",
        "icon": "📋",
        "file": "pages/csv_analyzer.py"
    },
    {
        "title": "Text Analyzer",
        "icon": "📄",
        "file": "pages/txt_analyzer.py"
    },
    {
        "title": "Audio Analyzer",
        "icon": "🔊",
        "file": "pages/audio_analyzer.py"
    },
    {
        "title": "JSON Analyzer",
        "icon": "🔍",
        "file": "pages/json_analyzer.py"
    },
    {
        "title": "Excel Analyzer",
        "icon": "📈",
        "file": "pages/excel_analyzer.py"
    },
    {
        "title": "Image Analyzer",
        "icon": "🖼️",
        "file": "pages/image_analyzer.py"
    }
]

# Create full-width button grid with more columns for wide layout
cols = st.columns(6)  # Increased to 6 for wider screen utilization

for idx, fmt in enumerate(formats):
    with cols[idx % 6]:
        # Create simple button with just the title
        if st.button(
            f"{fmt['icon']} {fmt['title']}",
            key=f"button_{fmt['title'].lower().replace(' ', '_')}",
        ):
            st.switch_page(fmt["file"])

# Animated divider
st.markdown("---")

# Footer with animation
st.caption("© 2025 Universal Data Analyzer | All rights reserved | Powered by Streamlit")