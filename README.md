# Universal Data Analyzer

A comprehensive data analysis platform that supports multiple data formats including CSV, JSON, Excel, Text, Audio, Image, and Video files. This application provides preprocessing, cleaning, and machine learning model recommendations for various data types.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Supported Data Types](#supported-data-types)
- [Technologies Used](#technologies-used)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

## Features

- **Multi-format Support**: Process CSV, JSON, Excel, Text, Audio, Image, and Video files
- **Data Cleaning**: Automated data cleaning and preprocessing pipelines
- **Machine Learning Recommendations**: Smart ML model suggestions based on your data
- **Streamlit Interface**: User-friendly web interface for easy interaction
- **Data Visualization**: Built-in data visualization capabilities
- **Export Functionality**: Download processed data in various formats

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd universal-data-analyzer
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   ```

3. Activate the virtual environment:
   - On Windows:
     ```bash
     .\venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```bash
     source venv/bin/activate
     ```

4. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

5. For audio processing, install SpeechBrain:
   ```bash
   pip install speechbrain
   ```

## Usage

1. Run the application:
   ```bash
   streamlit run main.py
   ```

2. Open your browser and navigate to the provided local URL (typically `http://localhost:8501`)

3. Select the type of data you want to analyze from the main interface

4. Upload your data file

5. Configure preprocessing options as needed

6. Process your data and download the results

## Supported Data Types

### CSV Analyzer
- Data cleaning and preprocessing
- Outlier detection and removal
- Missing value imputation
- Data type conversion
- ML model recommendation for classification, regression, and clustering tasks

### JSON Analyzer
- JSON data parsing and validation
- Data structure analysis
- Transformation to tabular format
- ML model suggestions

### Excel Analyzer
- Multi-sheet Excel processing
- Data cleaning and formatting
- Formula evaluation
- Statistical analysis

### Text Analyzer
- Text preprocessing and cleaning
- Language detection
- Natural language processing
- Text classification model recommendations

### Audio Analyzer
- Audio format conversion
- Noise reduction
- Voice activity detection (VAD)
- Audio enhancement
- ML model suggestions for audio tasks

### Image Analyzer
- Image format conversion
- Resize and normalization
- Image preprocessing pipeline
- ML model recommendations for computer vision tasks

### Video Analyzer
- Video processing and analysis
- Frame extraction
- Metadata analysis
- ML model suggestions for video tasks

## Technologies Used

- **Python**: Core programming language
- **Streamlit**: Web application framework
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **Scikit-learn**: Machine learning algorithms
- **Matplotlib/Seaborn**: Data visualization
- **OpenCV**: Computer vision tasks
- **Pillow**: Image processing
- **Librosa**: Audio analysis
- **SpeechBrain**: Audio processing and voice activity detection
- **NLTK**: Natural language processing

## Project Structure

```
universal-data-analyzer/
├── main.py                 # Main application entry point
├── requirements.txt        # Python dependencies
├── preprocessing_pipeline.py # Core preprocessing functions
├── pages/                  # Individual analyzer modules
│   ├── audio_analyzer.py
│   ├── csv_analyzer.py
│   ├── excel_analyzer.py
│   ├── image_analyzer.py
│   ├── json_analyzer.py
│   ├── txt_analyzer.py
│   └── video_analyzer.py
├── processed_images/       # Output directory for processed images
└── venv/                   # Virtual environment (not included in repo)
```

## Contributing

1. Fork the repository
2. Create a new branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

Developed with ❤️ by [Your Name]