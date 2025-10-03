# Universal Data Analyzer

A comprehensive data analysis platform that supports multiple data formats including CSV, JSON, Excel, Text, Audio, and Image files. This application provides preprocessing, cleaning, and machine learning model recommendations for various data types.

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

- **Multi-format Support**: Process CSV, JSON, Excel, Text, Audio, and Image files
- **Data Cleaning**: Automated data cleaning and preprocessing pipelines
- **Machine Learning Recommendations**: Smart ML model suggestions based on your data
- **Streamlit Interface**: User-friendly web interface with consistent navigation
- **Data Visualization**: Built-in data visualization capabilities
- **Export Functionality**: Download processed data in various formats
- **Easy Navigation**: Back button on all analyzer pages for seamless navigation

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

7. Use the "Back" button on any analyzer page to return to the main dashboard

## Supported Data Types

### CSV Analyzer
- Data cleaning and preprocessing
- Outlier detection and removal
- Missing value imputation
- Data type conversion
- ML model recommendation for classification, regression, and clustering tasks
- Back button for easy navigation

### JSON Analyzer
- JSON data parsing and validation
- Data structure analysis
- Transformation to tabular format
- ML model suggestions
- Back button for easy navigation

### Excel Analyzer
- Multi-sheet Excel processing
- Data cleaning and formatting
- Formula evaluation
- Statistical analysis
- Back button for easy navigation

### Text Analyzer
- Text preprocessing and cleaning
- Language detection
- Natural language processing
- Text classification model recommendations
- Back button for easy navigation

### Audio Analyzer
- Audio format conversion
- Noise reduction
- Voice activity detection (VAD)
- Audio enhancement
- ML model suggestions for audio tasks
- Back button for easy navigation

### Image Analyzer
- Image format conversion
- Resize and normalization
- Image preprocessing pipeline
- ML model recommendations for computer vision tasks
- Back button for easy navigation

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
- **Pydub**: Audio processing
- **NLTK**: Natural language processing
- **SciPy**: Scientific computing

## Project Structure

```
universal-data-analyzer/
├── main.py                 # Main application entry point
├── requirements.txt        # Python dependencies
├── unwanted_file.py 
├── pages/                  # Individual analyzer modules
│   ├── audio_analyzer.py
│   ├── csv_analyzer.py
│   ├── excel_analyzer.py
│   ├── image_analyzer.py
│   ├── json_analyzer.py
│   └── txt_analyzer.py
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

Developed with ❤️ by [Savani Dhruv]