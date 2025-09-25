import streamlit as st
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import torch
import torch.nn as nn
from torchvision import models, transforms
import tempfile
import os
from pathlib import Path
import json
import pickle
from typing import Dict, List, Tuple, Any
import warnings
import base64
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Video ML Automation Pipeline",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

class VideoProcessor:
    """Main class for video processing and analysis"""
    
    def __init__(self):
        self.video_path = None
        self.frames = []
        self.features = {}
        self.model = None
        self.results = {}
        self.original_video_bytes = None
        self.cleaned_video_bytes = None
        
    def load_video(self, video_file):
        """Load video file and extract basic information"""
        try:
            # Save original video bytes for later use
            self.original_video_bytes = video_file.getvalue()
            
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                tmp_file.write(self.original_video_bytes)
                self.video_path = tmp_file.name
            
            # Get video properties
            cap = cv2.VideoCapture(self.video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            video_info = {
                'fps': fps,
                'frame_count': frame_count,
                'duration': duration,
                'width': width,
                'height': height,
                'size_mb': len(self.original_video_bytes) / (1024 * 1024)
            }
            
            cap.release()
            return video_info
            
        except Exception as e:
            st.error(f"Error loading video: {str(e)}")
            return None
    
    def extract_frames(self, sample_rate=1, max_frames=100):
        """Extract frames from video"""
        try:
            cap = cv2.VideoCapture(self.video_path)
            frames = []
            frame_count = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_count % sample_rate == 0:
                    # Resize frame to standard size
                    frame_resized = cv2.resize(frame, (224, 224))
                    frames.append(frame_resized)
                    
                    if len(frames) >= max_frames:
                        break
                
                frame_count += 1
            
            cap.release()
            self.frames = frames
            return len(frames)
            
        except Exception as e:
            st.error(f"Error extracting frames: {str(e)}")
            return 0
    
    def extract_basic_features(self):
        """Extract basic computer vision features"""
        if not self.frames:
            return {}
        
        features = {
            'color_features': [],
            'texture_features': [],
            'motion_features': [],
            'shape_features': []
        }
        
        for i, frame in enumerate(self.frames):
            # Color features
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            color_hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
            color_features = color_hist.flatten()
            
            # Texture features (using Laplacian variance)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            texture_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # Motion features (frame difference)
            if i > 0:
                prev_gray = cv2.cvtColor(self.frames[i-1], cv2.COLOR_BGR2GRAY)
                motion = cv2.absdiff(gray, prev_gray).mean()
            else:
                motion = 0
            
            # Shape features (edge detection)
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
            
            features['color_features'].append(color_features)
            features['texture_features'].append(texture_var)
            features['motion_features'].append(motion)
            features['shape_features'].append(edge_density)
        
        self.features = features
        return features
    
    def extract_deep_features(self):
        """Extract deep learning features using pre-trained models"""
        if not self.frames:
            return {}
        
        try:
            # Load pre-trained ResNet18
            model = models.resnet18(pretrained=True)
            model.eval()
            
            # Remove the last classification layer
            model = nn.Sequential(*list(model.children())[:-1])
            
            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            deep_features = []
            
            with torch.no_grad():
                for frame in self.frames:
                    # Convert BGR to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Transform and add batch dimension
                    input_tensor = transform(frame_rgb).unsqueeze(0)
                    
                    # Extract features
                    features = model(input_tensor)
                    features = features.squeeze().numpy()
                    deep_features.append(features)
            
            self.features['deep_features'] = deep_features
            return deep_features
            
        except Exception as e:
            st.error(f"Error extracting deep features: {str(e)}")
            return []
    
    def create_cleaned_video(self):
        """Create a cleaned version of the video (grayscale conversion as example)"""
        try:
            if not self.video_path or not os.path.exists(self.video_path):
                return False
            
            # Create temporary file for cleaned video
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_cleaned:
                cleaned_path = tmp_cleaned.name
            
            # Open input video
            cap = cv2.VideoCapture(self.video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Define codec and create VideoWriter
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(cleaned_path, fourcc, fps, (width, height), isColor=False)
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Convert to grayscale (example cleaning operation)
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                # Convert back to 3 channels for compatibility
                gray_frame_3ch = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)
                out.write(gray_frame_3ch)
            
            cap.release()
            out.release()
            
            # Read cleaned video bytes
            with open(cleaned_path, 'rb') as f:
                self.cleaned_video_bytes = f.read()
            
            # Clean up temporary cleaned video file
            os.unlink(cleaned_path)
            return True
            
        except Exception as e:
            st.error(f"Error creating cleaned video: {str(e)}")
            return False
    
    def analyze_video_content(self, main_task, sub_task):
        """Analyze video content and suggest task type"""
        if not self.features:
            return {}
        
        analysis = {}
        
        # Motion analysis
        motion_values = self.features.get('motion_features', [])
        if motion_values:
            avg_motion = np.mean(motion_values)
            motion_variance = np.var(motion_values)
            
            analysis['motion_analysis'] = {
                'average_motion': avg_motion,
                'motion_variance': motion_variance,
                'motion_type': 'High Motion' if avg_motion > 20 else 'Low Motion'
            }
        
        # Texture analysis
        texture_values = self.features.get('texture_features', [])
        if texture_values:
            avg_texture = np.mean(texture_values)
            analysis['texture_analysis'] = {
                'average_texture': avg_texture,
                'texture_type': 'High Texture' if avg_texture > 500 else 'Low Texture'
            }
        
        # Color analysis
        color_features = self.features.get('color_features', [])
        if color_features:
            color_diversity = np.mean([np.std(cf) for cf in color_features])
            analysis['color_analysis'] = {
                'color_diversity': color_diversity,
                'color_type': 'Diverse Colors' if color_diversity > 10 else 'Uniform Colors'
            }
        
        # Suggest task type based on user selection
        task_mappings = {
            'Classification': {
                'Action Recognition': ['Running', 'Walking', 'Jumping'],
                'Scene Classification': ['Indoor', 'Outdoor', 'Urban'],
                'Object Classification': ['Person', 'Vehicle', 'Animal']
            },
            'Detection': {
                'Object Detection': ['Single Object', 'Multiple Objects'],
                'Activity Detection': ['Sports', 'Daily Activities'],
                'Anomaly Detection': ['Unusual Behavior', 'Unexpected Objects']
            },
            'Segmentation': {
                'Semantic Segmentation': ['Scene Parsing', 'Background-Foreground'],
                'Instance Segmentation': ['Individual Objects', 'Overlapping Objects']
            },
            'Regression': {
                'Quality Assessment': ['Video Quality', 'Compression Artifacts'],
                'Temporal Localization': ['Event Timing', 'Action Duration']
            },
            'Sequence': {
                'Video Captioning': ['Descriptive Captions', 'Action Descriptions'],
                'Temporal Action Detection': ['Action Start-End', 'Action Sequences']
            }
        }
        
        analysis['suggested_task'] = sub_task
        analysis['sub_tasks'] = task_mappings.get(main_task, {}).get(sub_task, [])
        
        return analysis
    
    def prepare_features_for_ml(self):
        """Prepare features for machine learning"""
        if not self.features:
            return None, None
        
        # Combine all features
        feature_matrix = []
        
        for i in range(len(self.frames)):
            frame_features = []
            
            # Add basic features
            if 'texture_features' in self.features:
                frame_features.append(self.features['texture_features'][i])
            
            if 'motion_features' in self.features:
                frame_features.append(self.features['motion_features'][i])
            
            if 'shape_features' in self.features:
                frame_features.append(self.features['shape_features'][i])
            
            # Add color features (use first 10 components)
            if 'color_features' in self.features:
                color_feat = self.features['color_features'][i][:10]
                frame_features.extend(color_feat)
            
            # Add deep features (use first 20 components)
            if 'deep_features' in self.features:
                deep_feat = self.features['deep_features'][i][:20]
                frame_features.extend(deep_feat)
            
            feature_matrix.append(frame_features)
        
        X = np.array(feature_matrix)
        
        # Create dummy labels for demonstration (in real scenario, you'd have actual labels)
        y = np.random.randint(0, 3, size=len(X))  # 3 classes
        
        return X, y
    
    def train_model(self, X, y, model_type='random_forest'):
        """Train ML model"""
        try:
            if len(X) < 2:
                st.warning("Not enough data for training. Need at least 2 samples.")
                return None
            
            # Split data
            if len(X) >= 4:
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            else:
                X_train, X_test, y_train, y_test = X, X, y, y
            
            # Select model
            if model_type == 'random_forest':
                model = RandomForestClassifier(n_estimators=100, random_state=42)
            elif model_type == 'svm':
                model = SVC(kernel='rbf', random_state=42)
            else:
                model = RandomForestClassifier(n_estimators=100, random_state=42)
            
            # Train model
            model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = model.predict(X_test)
            
            results = {
                'model': model,
                'accuracy': (y_pred == y_test).mean(),
                'classification_report': classification_report(y_test, y_pred, output_dict=True),
                'confusion_matrix': confusion_matrix(y_test, y_pred)
            }
            
            self.model = model
            self.results = results
            
            return results
            
        except Exception as e:
            st.error(f"Error training model: {str(e)}")
            return None
    
    def clean_video(self):
        """Clean up temporary video file and reset all data except video bytes"""
        try:
            # Create cleaned video before deleting original temporary file
            cleanup_success = False
            if self.video_path and os.path.exists(self.video_path):
                # Generate cleaned video
                if self.create_cleaned_video():
                    st.success("✅ Cleaned video generated!")
                else:
                    st.warning("Failed to generate cleaned video.")
                
                # Delete original temporary file
                try:
                    os.unlink(self.video_path)
                    cleanup_success = True
                except Exception as e:
                    st.warning(f"Failed to delete temporary file: {str(e)}")
            
            # Reset all attributes except video bytes
            self.video_path = None
            self.frames = []
            self.features = {}
            self.model = None
            self.results = {}
            return cleanup_success
        except Exception as e:
            st.error(f"Error during cleanup: {str(e)}")
            return False

def main():
    # Initialize session state
    if 'processor' not in st.session_state:
        st.session_state.processor = VideoProcessor()
    
    st.title("🎬 Video ML Automation Pipeline")
    st.markdown("Upload a video file to automatically analyze, extract features, and build ML models")
    
    # Sidebar
    st.sidebar.title("Pipeline Configuration")
    
    # File upload in main area
    st.subheader("Upload Video")
    uploaded_file = st.file_uploader(
        "Choose a video file",
        type=['mp4', 'avi', 'mov', 'mkv'],
        help="Upload a video file to analyze"
    )
    
    # Task selection
    task_types = {
        'Classification': ['Action Recognition', 'Scene Classification', 'Object Classification'],
        'Detection': ['Object Detection', 'Activity Detection', 'Anomaly Detection'],
        'Segmentation': ['Semantic Segmentation', 'Instance Segmentation'],
        'Regression': ['Quality Assessment', 'Temporal Localization'],
        'Sequence': ['Video Captioning', 'Temporal Action Detection']
    }
    
    if uploaded_file is not None:
        processor = st.session_state.processor
        
        # Load video
        with st.spinner("Loading video..."):
            video_info = processor.load_video(uploaded_file)
        
        if video_info:
            # Display video info
            st.subheader("📊 Video Information")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Duration", f"{video_info['duration']:.2f} sec")
                st.metric("Frame Count", video_info['frame_count'])
            
            with col2:
                st.metric("FPS", f"{video_info['fps']:.2f}")
                st.metric("Resolution", f"{video_info['width']}x{video_info['height']}")
            
            with col3:
                st.metric("File Size", f"{video_info['size_mb']:.2f} MB")
            
            # Display original video
            st.subheader("🎥 Original Video")
            video_bytes = processor.original_video_bytes
            st.video(video_bytes)
            
            # Task selection
            st.sidebar.subheader("Task Selection")
            main_task = st.sidebar.selectbox("Select Main Task", list(task_types.keys()))
            sub_task = st.sidebar.selectbox("Select Sub Task", task_types[main_task])
            
            # Processing parameters
            st.sidebar.subheader("Processing Parameters")
            sample_rate = st.sidebar.slider("Frame Sample Rate", 1, 10, 5)
            max_frames = st.sidebar.slider("Max Frames to Process", 10, 200, 50)
            
            # Extract frames
            if st.sidebar.button("🎯 Extract Frames"):
                with st.spinner("Extracting frames..."):
                    num_frames = processor.extract_frames(sample_rate, max_frames)
                
                if num_frames > 0:
                    st.success(f"✅ Extracted {num_frames} frames")
                    
                    # Display sample frames
                    st.subheader("🖼️ Sample Frames")
                    cols = st.columns(5)
                    for i in range(min(5, len(processor.frames))):
                        with cols[i]:
                            frame_rgb = cv2.cvtColor(processor.frames[i], cv2.COLOR_BGR2RGB)
                            st.image(frame_rgb, caption=f"Frame {i+1}")
            
            # Feature extraction
            if processor.frames:
                st.subheader("🔍 Feature Extraction")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button("Extract Basic Features"):
                        with st.spinner("Extracting basic features..."):
                            features = processor.extract_basic_features()
                        
                        if features:
                            st.success("✅ Basic features extracted!")
                            
                            # Display feature statistics
                            st.write("**Feature Statistics:**")
                            
                            if 'motion_features' in features:
                                motion_stats = {
                                    'Mean Motion': np.mean(features['motion_features']),
                                    'Max Motion': np.max(features['motion_features']),
                                    'Motion Variance': np.var(features['motion_features'])
                                }
                                st.json(motion_stats)
                
                with col2:
                    if st.button("Extract Deep Features"):
                        with st.spinner("Extracting deep features (this may take a while)..."):
                            deep_features = processor.extract_deep_features()
                        
                        if deep_features:
                            st.success("✅ Deep features extracted!")
                            st.write(f"**Deep Features Shape:** {np.array(deep_features).shape}")
                
                # Video analysis
                if processor.features:
                    st.subheader("📈 Video Analysis")
                    
                    if st.button("Analyze Video Content"):
                        with st.spinner("Analyzing video content..."):
                            analysis = processor.analyze_video_content(main_task, sub_task)
                        
                        if analysis:
                            st.success("✅ Analysis complete!")
                            
                            # Display analysis results
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.write("**Content Analysis:**")
                                for key, value in analysis.items():
                                    if key not in ['suggested_task', 'sub_tasks']:
                                        st.json(value)
                            
                            with col2:
                                st.write("**Selected Task:**")
                                st.info(f"🎯 {main_task} - {sub_task}")
                                st.write("**Sub-Sub Tasks:**")
                                for sub_sub_task in analysis.get('sub_tasks', []):
                                    st.write(f"- {sub_sub_task}")
                                
                                # Task recommendations
                                recommended_models = {
                                    'Classification': {
                                        'Action Recognition': ['3D CNN (C3D, I3D)', 'Two-stream Networks', 'LSTM + CNN'],
                                        'Scene Classification': ['ResNet/EfficientNet', 'Vision Transformer', 'Random Forest'],
                                        'Object Classification': ['ResNet', 'VGG', 'EfficientNet']
                                    },
                                    'Detection': {
                                        'Object Detection': ['YOLO', 'Faster R-CNN', 'SSD'],
                                        'Activity Detection': ['Two-stream CNN', 'SlowFast Networks'],
                                        'Anomaly Detection': ['Autoencoders', 'One-class SVM']
                                    },
                                    'Segmentation': {
                                        'Semantic Segmentation': ['DeepLab', 'U-Net', 'Mask R-CNN'],
                                        'Instance Segmentation': ['Mask R-CNN', 'YOLACT']
                                    },
                                    'Regression': {
                                        'Quality Assessment': ['BRISQUE', 'CNN-based Quality Models'],
                                        'Temporal Localization': ['LSTM', 'Temporal CNN']
                                    },
                                    'Sequence': {
                                        'Video Captioning': ['Encoder-Decoder Models', 'Transformer-based Models'],
                                        'Temporal Action Detection': ['R-C3D', 'TAL-Net']
                                    }
                                }
                                
                                st.write("**Recommended Models:**")
                                for model in recommended_models.get(main_task, {}).get(sub_task, []):
                                    st.write(f"- {model}")
                    
                    # Feature visualization
                    if 'motion_features' in processor.features:
                        st.subheader("📊 Feature Visualization")
                        
                        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
                        
                        # Motion over time
                        axes[0,0].plot(processor.features['motion_features'])
                        axes[0,0].set_title('Motion Over Time')
                        axes[0,0].set_xlabel('Frame')
                        axes[0,0].set_ylabel('Motion Intensity')
                        
                        # Texture over time
                        axes[0,1].plot(processor.features['texture_features'])
                        axes[0,1].set_title('Texture Variance Over Time')
                        axes[0,1].set_xlabel('Frame')
                        axes[0,1].set_ylabel('Texture Variance')
                        
                        # Motion histogram
                        axes[1,0].hist(processor.features['motion_features'], bins=20)
                        axes[1,0].set_title('Motion Distribution')
                        axes[1,0].set_xlabel('Motion Intensity')
                        axes[1,0].set_ylabel('Frequency')
                        
                        # Shape features
                        axes[1,1].plot(processor.features['shape_features'])
                        axes[1,1].set_title('Edge Density Over Time')
                        axes[1,1].set_xlabel('Frame')
                        axes[1,1].set_ylabel('Edge Density')
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                    
                    # Model training
                    st.subheader("🤖 Model Training")
                    
                    model_type = st.selectbox(
                        "Select Model Type",
                        ["random_forest", "svm"],
                        help="Choose the type of model to train"
                    )
                    
                    if st.button("Train Model"):
                        with st.spinner("Preparing features and training model..."):
                            X, y = processor.prepare_features_for_ml()
                        
                        if X is not None:
                            with st.spinner("Training model..."):
                                results = processor.train_model(X, y, model_type)
                            
                            if results:
                                st.success("✅ Model training complete!")
                                
                                # Display results
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    st.metric("Model Accuracy", f"{results['accuracy']:.3f}")
                                    
                                    # Feature importance (if available)
                                    if hasattr(results['model'], 'feature_importances_'):
                                        st.write("**Feature Importances:**")
                                        importances = results['model'].feature_importances_
                                        feature_names = [f"Feature_{i}" for i in range(len(importances))]
                                        
                                        importance_df = pd.DataFrame({
                                            'Feature': feature_names[:10],  # Top 10
                                            'Importance': importances[:10]
                                        }).sort_values('Importance', ascending=False)
                                        
                                        st.bar_chart(importance_df.set_index('Feature'))
                                
                                with col2:
                                    st.write("**Classification Report:**")
                                    report_df = pd.DataFrame(results['classification_report']).transpose()
                                    st.dataframe(report_df)
                                
                                # Confusion matrix
                                st.write("**Confusion Matrix:**")
                                fig, ax = plt.subplots(figsize=(8, 6))
                                sns.heatmap(results['confusion_matrix'], annot=True, fmt='d', cmap='Blues')
                                plt.title('Confusion Matrix')
                                st.pyplot(fig)
                                
                                # Model export
                                st.subheader("💾 Export Model")
                                
                                if st.button("Export Model"):
                                    model_data = {
                                        'model': results['model'],
                                        'features_info': {
                                            'feature_count': X.shape[1],
                                            'sample_count': X.shape[0]
                                        },
                                        'performance': {
                                            'accuracy': results['accuracy']
                                        },
                                        'task_info': {
                                            'main_task': main_task,
                                            'sub_task': sub_task
                                        }
                                    }
                                    
                                    # Save model
                                    model_bytes = pickle.dumps(model_data)
                                    st.download_button(
                                        label="Download Trained Model",
                                        data=model_bytes,
                                        file_name="video_ml_model.pkl",
                                        mime="application/octet-stream"
                                    )
                                    
                                    st.success("✅ Model ready for download!")
            
            # Clean video section
            st.subheader("🧹 Video Management")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("Clean Temporary Video"):
                    with st.spinner("Cleaning temporary video file and generating cleaned video..."):
                        if processor.clean_video():
                            st.success("✅ Temporary video file cleaned and cleaned video generated!")
                            # Reset session state but keep video bytes
                            st.session_state.processor = VideoProcessor()
                            st.session_state.processor.original_video_bytes = video_bytes
                            st.session_state.processor.cleaned_video_bytes = processor.cleaned_video_bytes
                            st.rerun()
                        else:
                            st.warning("No temporary video file to clean or cleaning failed.")
            
            with col2:
                if processor.cleaned_video_bytes:
                    if st.button("Play Cleaned Video"):
                        st.video(processor.cleaned_video_bytes)
                        st.success("✅ Playing cleaned video")
            
            with col3:
                if processor.cleaned_video_bytes:
                    if st.button("Download Cleaned Video"):
                        st.download_button(
                            label="Download Cleaned Video",
                            data=processor.cleaned_video_bytes,
                            file_name="cleaned_video.mp4",
                            mime="video/mp4"
                        )
                        st.success("✅ Cleaned video ready for download")
    
    # Instructions
    if uploaded_file is None:
        st.info("👆 Please upload a video file to get started")
        
        st.subheader("📋 Pipeline Overview")
        st.write("""
        This automated video ML pipeline performs the following steps:
        
        1. **Video Loading & Analysis** - Extract basic video properties
        2. **Task Selection** - Choose from classification, detection, segmentation, regression, or sequence tasks
        3. **Frame Extraction** - Sample frames at specified intervals
        4. **Feature Extraction** - Extract both basic and deep learning features
        5. **Content Analysis** - Analyze video content and suggest optimal sub-tasks
        6. **Model Training** - Automatically train ML models on extracted features
        7. **Model Export** - Export trained models for deployment
        8. **Video Management** - Clean temporary files, generate cleaned video, play or download cleaned video
        
        **Supported Task Types:**
        - Classification: Action Recognition, Scene Classification, Object Classification
        - Detection: Object Detection, Activity Detection, Anomaly Detection
        - Segmentation: Semantic Segmentation, Instance Segmentation
        - Regression: Quality Assessment, Temporal Localization
        - Sequence: Video Captioning, Temporal Action Detection
        
        **Supported Features:**
        - Color histograms and moments
        - Texture analysis (Laplacian variance)
        - Motion analysis (frame differences)
        - Shape features (edge density)
        - Deep features (ResNet embeddings)
        
        **Supported Models:**
        - Random Forest Classifier
        - Support Vector Machine
        - Custom neural networks (extensible)
        
        **Video Cleaning:**
        - Converts video to grayscale as a basic cleaning operation
        - Cleaned video available for playback and download after cleaning
        """)

if __name__ == "__main__":
    main()