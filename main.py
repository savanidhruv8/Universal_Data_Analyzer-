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
    
if __name__ == "__main__":
    main()