import streamlit as st
import os
import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError
import shutil
import io

# Constants
TEMP_DIR = "processed_images"
os.makedirs(TEMP_DIR, exist_ok=True)

# ---------------- Utility Functions ----------------

def verify_image(file):
    try:
        with Image.open(file) as img:
            img.verify()
        file.seek(0)
        return True
    except Exception:
        return False

def convert_image_format(img, mode="RGB"):
    return img.convert("L").convert("RGB") if mode == "Grayscale" else img.convert("RGB")

def resize_image(img, max_width, max_height):
    img.thumbnail((max_width, max_height))  # maintain aspect ratio
    return img

def normalize_image(img_array, to_range="[0, 1]"):
    return img_array / 255.0 if to_range == "[0, 1]" else img_array

def clean_filename(idx):
    return f"img_{idx:03d}.jpg"

def save_image(img, path):
    img.save(path)

def get_image_bytes(img):
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='JPEG')
    return img_byte_arr.getvalue()

def select_model_subtype(convert_format, max_width, max_height):
    """Automatically select model sub-type based on image format and size."""
    # Determine image size category
    image_size = max_width * max_height
    if image_size <= 512 * 512:
        size_category = "small"
    elif image_size <= 1024 * 1024:
        size_category = "medium"
    else:
        size_category = "large"

    model_map = {
        "CNN": {
            "RGB": {
                "small": ["MobileNetV2", "EfficientNetB0"],
                "medium": ["ResNet50", "VGG16"],
                "large": ["InceptionV3", "ResNet152"]
            },
            "Grayscale": {
                "small": ["SimpleCNN"],
                "medium": ["CustomCNN"],
                "large": ["DeepCNN"]
            }
        },
        "Transformer": {
            "RGB": {
                "small": ["Vision Transformer (ViT-B/16)"],
                "medium": ["Swin Transformer", "DeiT"],
                "large": ["ViT-L/16", "BEiT"]
            },
            "Grayscale": {
                "small": ["Grayscale ViT-S"],
                "medium": ["Grayscale Swin Transformer"],
                "large": ["Grayscale ViT-L"]
            }
        }
    }

    # Return the model map and the size category for context
    return model_map, size_category

# ---------------- Streamlit UI ----------------

st.set_page_config("🖼️ Image Folder Preprocessing", layout="centered")

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
    if st.button("⬅️ Back", key="image_back_button"):
        st.switch_page("main.py")

st.title("📂 Image Folder Preprocessing Pipeline")

# Preprocessing Settings on Main Screen
st.header("⚙️ Preprocessing Settings")
convert_format = st.selectbox("Convert image format to:", ["RGB", "Grayscale"])
col1, col2 = st.columns(2)
with col1:
    max_width = st.number_input("Max width (resize):", min_value=256, max_value=2048, value=1024)
with col2:
    max_height = st.number_input("Max height (resize):", min_value=256, max_value=2048, value=1024)
normalize_option = st.selectbox("Normalize pixel values to:", ["[0, 1]", "[0, 255]"])

# Model Selection
st.header("🤖 Model Selection")
model_type = st.selectbox("Select main model type:", ["CNN", "Transformer"])
model_map, size_category = select_model_subtype(convert_format, max_width, max_height)
model_subtypes = model_map[model_type][convert_format][size_category]
selected_subtype = model_subtypes[0]  # Select the first recommended sub-type
st.write(f"Recommended model sub-type: {selected_subtype} (based on {convert_format} format and {size_category} image size)")

# File uploader for images or folder
uploaded_images = st.file_uploader(
    "📁 Upload image(s) or folder (single or multiple)",
    type=["jpg", "jpeg", "png", "tiff", "bmp", "webp", "heic", "raw"],
    accept_multiple_files=True
)

uploaded_csv = st.file_uploader("📄 Optional: Upload CSV for labels", type=["csv"])

if uploaded_images:
    # Clean previous output
    shutil.rmtree(TEMP_DIR, ignore_errors=True)
    os.makedirs(TEMP_DIR)

    st.info("🔄 Preprocessing...")

    # Read CSV label map
    label_map = {}
    if uploaded_csv:
        try:
            df = pd.read_csv(uploaded_csv)
            if 'filename' in df.columns and 'label' in df.columns:
                label_map = dict(zip(df['filename'], df['label']))
            else:
                st.error("CSV must contain 'filename' and 'label' columns.")
        except Exception as e:
            st.error(f"Error reading CSV: {e}")

    valid_images, labels, processed_images = [], {}, []
    corrupted_files = []

    # Handle single image case with manual label input
    if len(uploaded_images) == 1:
        single_image = uploaded_images[0]
        original_name = single_image.name
        label = label_map.get(original_name, None)
        
        if not label:
            st.warning("No label found for the uploaded image.")
            label = st.text_input("Please enter a label for the image:", key="manual_label")
            if not label:
                st.error("A label is required for the single image. Please enter a label.")
                st.stop()

        if verify_image(single_image):
            try:
                img = Image.open(single_image)
                img = convert_image_format(img, convert_format)
                img = resize_image(img, max_width, max_height)

                # Normalize for internal use (not saved)
                img_arr = np.array(img)
                _ = normalize_image(img_arr, normalize_option)

                new_name = clean_filename(0)
                save_path = os.path.join(TEMP_DIR, new_name)
                save_image(img, save_path)

                labels[new_name] = label
                valid_images.append(new_name)
                processed_images.append((img, new_name, label))
            except Exception as e:
                corrupted_files.append(original_name)
        else:
            corrupted_files.append(original_name)

    else:
        # Handle multiple images (folder-like upload)
        for idx, file in enumerate(uploaded_images):
            original_name = file.name

            if not verify_image(file):
                corrupted_files.append(original_name)
                continue

            try:
                img = Image.open(file)
                img = convert_image_format(img, convert_format)
                img = resize_image(img, max_width, max_height)

                # Normalize for internal use (not saved)
                img_arr = np.array(img)
                _ = normalize_image(img_arr, normalize_option)

                new_name = clean_filename(idx)
                save_path = os.path.join(TEMP_DIR, new_name)
                save_image(img, save_path)

                label = label_map.get(original_name, "unlabeled")
                labels[new_name] = label
                valid_images.append(new_name)
                processed_images.append((img, new_name, label))

            except Exception as e:
                corrupted_files.append(original_name)

    if valid_images:
        st.success(f"✅ Processed: {len(valid_images)} images")
        st.info(f"Selected Model: {model_type} ({selected_subtype})")
        if corrupted_files:
            st.warning(f"❌ Skipped {len(corrupted_files)} corrupted or unreadable files")

        # Display processed images with labels and download buttons
        st.header("📸 Processed Images")
        for img, img_name, label in processed_images:
            st.subheader(f"Image: {img_name}")
            st.write(f"Label: {label}")
            st.image(img, use_column_width=True)
            st.download_button(
                label=f"Download {img_name}",
                data=get_image_bytes(img),
                file_name=img_name,
                mime="image/jpeg"
            )
    else:
        st.error("No valid images were processed.")
else:
    st.info("📂 Please upload image files or a folder to start preprocessing.")