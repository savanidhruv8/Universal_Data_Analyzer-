import streamlit as st
import re
import pandas as pd
from langdetect import detect, LangDetectException
import string
import unicodedata
import json

# Set page config
st.set_page_config(
    page_title="Text Data Cleaner",
    page_icon=":broom:",
    layout="wide"
)

# Function definitions
def remove_html_tags(text):
    """Remove HTML tags from text"""
    clean = re.compile('<.*?>')
    return re.sub(clean, '', text)

def remove_special_chars(text):
    """Remove special characters except basic punctuation"""
    return re.sub(r'[^a-zA-Z0-9\s.,!?;:\'\"\-]', '', text)

def normalize_text(text):
    """Convert text to lowercase and remove extra spaces"""
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def remove_non_utf8(text):
    """Remove non-UTF-8 characters"""
    return text.encode('utf-8', 'ignore').decode('utf-8')

def replace_special_spaces(text):
    """Replace non-breaking spaces and other special spaces"""
    text = text.replace('\xa0', ' ')
    text = text.replace('\t', ' ')
    text = text.replace('\r', ' ')
    return text

def clean_line(line, clean_options):
    """Clean a single line of text while preserving its position"""
    if clean_options['html']:
        line = remove_html_tags(line)
    if clean_options['special_chars']:
        line = remove_special_chars(line)
    if clean_options['normalize']:
        line = line.lower()
    if clean_options['non_utf8']:
        line = remove_non_utf8(line)
    if clean_options['special_spaces']:
        line = replace_special_spaces(line)
    
    # Always remove extra spaces
    line = re.sub(r'\s+', ' ', line).strip()
    return line

def tokenize_text(text):
    """Tokenize text into words"""
    return text.split()

def detect_text_language(text):
    """Detect language of the text with error handling"""
    try:
        return detect(text)
    except LangDetectException:
        return "unknown"
    except:
        return "error"

def truncate_text(text, max_length):
    """Truncate text to specified character length"""
    return text[:max_length] if max_length > 0 else text

def select_model_subtype(text, clean_options, deduplicate, language_option, lang_choice, truncate_option, max_length):
    """Select model sub-type based on text characteristics and generate .tzr metadata"""
    lang = detect_text_language(text)
    text_length = len(text)
    
    model_map = {
        "RNN": {
            "short": ["LSTM", "GRU"],
            "long": ["BiLSTM", "Deep RNN"],
            "multilingual": ["mLSTM"]
        },
        "Transformer": {
            "short": ["BERT", "DistilBERT"],
            "long": ["Longformer", "BigBird"],
            "multilingual": ["XLM-RoBERTa", "mBERT"]
        }
    }
    
    # Determine text length category
    if text_length < 1000:
        length_category = "short"
    else:
        length_category = "long"
    
    # Override with multilingual if non-English
    if lang not in ['en', 'unknown', 'error']:
        length_category = "multilingual"
    
    # Select the first recommended sub-type
    selected_subtype = model_map[model_type][length_category][0]
    
    # Create .tzr metadata
    tzr_metadata = {
        "model_type": model_type,
        "model_subtype": selected_subtype,
        "text_characteristics": {
            "language": lang,
            "length": text_length,
            "length_category": length_category,
            "token_count": len(tokenize_text(text)),
            "line_count": text.count('\n') + 1
        },
        "preprocessing_settings": {
            "remove_html_tags": clean_options['html'],
            "remove_special_chars": clean_options['special_chars'],
            "normalize_to_lowercase": clean_options['normalize'],
            "remove_non_utf8": clean_options['non_utf8'],
            "replace_special_spaces": clean_options['special_spaces'],
            "deduplicate_lines": deduplicate,
            "filter_by_language": language_option,
            "selected_language": lang_choice if language_option else None,
            "truncate_text": truncate_option,
            "max_length": max_length if truncate_option else None
        },
        "recommendation_reason": f"Selected {selected_subtype} based on text length ({text_length} chars) and language ({lang})"
    }
    
    return model_map, length_category, selected_subtype, tzr_metadata

def generate_tzr_file(tzr_metadata):
    """Generate .tzr file content as a JSON string"""
    return json.dumps(tzr_metadata, indent=4)

def process_text(text, clean_options, deduplicate, language_option, lang_choice, truncate_option, max_length):
    """Process text while preserving original line positions"""
    lines = text.split('\n')
    
    cleaned_lines = [clean_line(line, clean_options) for line in lines]
    
    if deduplicate:
        seen = set()
        deduped_lines = []
        for line in cleaned_lines:
            if line.strip() and line.strip() in seen:
                deduped_lines.append("")
            else:
                deduped_lines.append(line)
                if line.strip():
                    seen.add(line.strip())
        cleaned_lines = deduped_lines
    
    cleaned_text = '\n'.join(cleaned_lines)
    
    if language_option:
        lang = detect_text_language(cleaned_text)
        if lang != lang_choice:
            cleaned_text = ""
    
    if truncate_option:
        cleaned_text = truncate_text(cleaned_text, max_length)
    
    return cleaned_text

# UI Components
# Add back button with improved styling
col1, col2 = st.columns([1, 10])
with col1:
    st.markdown("""
    <style>
    div.stButton > button:first-child {
        background-color: #f0f2f6;
        color: #1a365d;
        border: 1px solid #d1d5db;
        border-radius: 0.375rem;
        padding: 0.5rem 1rem;
        font-size: 0.875rem;
        font-weight: 500;
        transition: all 0.2s;
        height: auto;
        min-height: 38px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: auto;
        min-width: 80px;
        box-sizing: border-box;
        white-space: nowrap;
    }
    div.stButton > button:first-child:hover {
        background-color: #e5e7eb;
        border-color: #9ca3af;
        transform: translateY(-1px);
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)
    if st.button("⬅️ Back", key="txt_back_button"):
        st.switch_page("main.py")

st.title(":broom: Text Data Cleaning and Preprocessing")
st.markdown("""
Upload a .txt file to clean and preprocess your text data. 
Select your desired processing options and model type below.
A .tzr file with model recommendations will be generated based on your input.
""")

# Initialize session state
if 'processed_text' not in st.session_state:
    st.session_state.processed_text = ""
if 'original_text' not in st.session_state:
    st.session_state.original_text = ""
if 'show_line_numbers' not in st.session_state:
    st.session_state.show_line_numbers = True
if 'tzr_metadata' not in st.session_state:
    st.session_state.tzr_metadata = None

# Processing options in main panel
with st.expander("🧹 Cleaning Options", expanded=True):
    col1, col2 = st.columns(2)
    with col1:
        html_clean = st.checkbox("Remove HTML tags", True)
        special_chars = st.checkbox("Remove special characters", True)
        normalize = st.checkbox("Normalize to lowercase", True)
    with col2:
        non_utf8 = st.checkbox("Remove non-UTF-8 characters", True)
        special_spaces = st.checkbox("Replace special spaces", True)
        deduplicate = st.checkbox("Remove duplicate lines", True)

with st.expander("✂️ Text Operations", expanded=True):
    col1, col2 = st.columns(2)
    with col1:
        truncate_option = st.checkbox("Truncate text", False)
        max_length = st.number_input("Max characters", 
                                    min_value=0, 
                                    max_value=100000, 
                                    value=5000, 
                                    disabled=not truncate_option)
    with col2:
        language_option = st.checkbox("Filter by language", False)
        lang_choice = st.selectbox("Select language", 
                                  ["en", "es", "fr", "de", "it", "pt", "ru", "zh", "ja"],
                                  disabled=not language_option)

# Model Selection
st.header("🤖 Model Selection")
model_type = st.selectbox("Select main model type:", ["RNN", "Transformer"])

# File uploader
uploaded_file = st.file_uploader("Choose a .txt file", type="txt")

# Process file when uploaded
if uploaded_file is not None:
    # Read and decode file
    bytes_data = uploaded_file.read()
    try:
        text = bytes_data.decode('utf-8')
    except UnicodeDecodeError:
        text = bytes_data.decode('latin-1')
    
    st.session_state.original_text = text
    
    # Define clean options
    clean_options = {
        'html': html_clean,
        'special_chars': special_chars,
        'normalize': normalize,
        'non_utf8': non_utf8,
        'special_spaces': special_spaces
    }
    
    # Select model sub-type and generate .tzr metadata
    model_map, length_category, selected_subtype, tzr_metadata = select_model_subtype(
        text, clean_options, deduplicate, language_option, lang_choice, truncate_option, max_length
    )
    st.session_state.tzr_metadata = tzr_metadata
    
    st.write(f"Recommended model sub-type (based on text characteristics): {selected_subtype}")
    
    # Apply processing with line position preservation
    st.session_state.processed_text = process_text(
        text, 
        clean_options, 
        deduplicate, 
        language_option, 
        lang_choice, 
        truncate_option, 
        max_length
    )

    # Text analysis
    if st.session_state.original_text:
        with st.expander("📊 Text Analysis", expanded=True):
            col1, col2, col3 = st.columns(3)
            
            orig_length = len(st.session_state.original_text)
            orig_lines = st.session_state.original_text.count('\n') + 1
            orig_tokens = len(tokenize_text(st.session_state.original_text))
            
            proc_length = len(st.session_state.processed_text)
            proc_lines = st.session_state.processed_text.count('\n') + 1
            proc_tokens = len(tokenize_text(st.session_state.processed_text))
            
            col1.metric("Original Length", f"{orig_length:,} chars")
            col2.metric("Original Lines", f"{orig_lines:,}")
            col3.metric("Original Tokens", f"{orig_tokens:,}")
            
            col1.metric("Processed Length", 
                       f"{proc_length:,} chars", 
                       f"{(proc_length - orig_length):,}")
            col2.metric("Processed Lines", 
                       f"{proc_lines:,}", 
                       f"{(proc_lines - orig_lines):,}")
            col3.metric("Processed Tokens", 
                       f"{proc_tokens:,}", 
                       f"{(proc_tokens - orig_tokens):,}")

# Display results
if st.session_state.processed_text or st.session_state.original_text:
    st.subheader("Processed Text")
    st.info(f"Selected Model: {model_type} ({selected_subtype})")
    
    if st.session_state.processed_text:
        # Download button for cleaned text
        st.download_button(
            label="Download Cleaned Text",
            data=st.session_state.processed_text.encode('utf-8'),
            file_name="cleaned_text.txt",
            mime="text/plain"
        )
        
        # Download button for .tzr file
        if st.session_state.tzr_metadata:
            tzr_content = generate_tzr_file(st.session_state.tzr_metadata)
            st.download_button(
                label="Download Model Configuration (.tzr)",
                data=tzr_content.encode('utf-8'),
                file_name="model_config.tzr",
                mime="application/json"
            )
        
        st.caption("Cleaned Text Preview (with line numbers)")
        lines = st.session_state.processed_text.split('\n')
        preview_lines = min(50, len(lines))
        
        preview_text = ""
        for i in range(preview_lines):
            preview_text += f"{i+1}: {lines[i]}\n"
        
        st.text_area("", preview_text, height=300, label_visibility="collapsed")
    else:
        st.warning("No text remains after processing with selected filters")
    
    st.subheader("Original Text")
    if st.session_state.original_text:
        st.caption("Original Text Preview (with line numbers)")
        lines = st.session_state.original_text.split('\n')
        preview_lines = min(50, len(lines))
        
        preview_text = ""
        for i in range(preview_lines):
            preview_text += f"{i+1}: {lines[i]}\n"
        
        st.text_area("", preview_text, height=300, label_visibility="collapsed")
    else:
        st.info("Upload a file to see original text")

# Instructions and info
st.markdown("---")
st.subheader("Instructions")
st.markdown("""
1. **Upload** a .txt file
2. **Select** processing options and model type
3. **View** analysis metrics and recommended model sub-type
4. **Download** cleaned text and .tzr model configuration file
""")

st.subheader("Features")
st.markdown("""
- 🧹 Remove HTML tags & special chars
- 📏 Text normalization
- 🗑️ Deduplicate content (preserves line positions)
- 🌍 Language detection
- ✂️ Text truncation
- 🤖 Model selection with auto-recommended sub-types
- 📥 Clean data export
- 📄 .tzr file generation with model configuration
""")