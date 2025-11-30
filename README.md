# YouTube Comments Spam Detector

A machine learning-based spam detection system for YouTube comments using Natural Language Processing (NLP) techniques and Naive Bayes classification.

## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Technologies Used](#technologies-used)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [How to Run](#how-to-run)
- [Model Pipeline](#model-pipeline)
- [Results](#results)
- [Sample Output](#sample-output)
- [Authors](#authors)
- [License](#license)

## 🎯 Overview

This project implements a spam detection system for YouTube comments using machine learning. The system analyzes comment text and classifies it as either **spam** (1) or **ham** (legitimate comment, 0). The model is trained on a dataset of Katy Perry YouTube video comments and uses a Multinomial Naive Bayes classifier with TF-IDF features.

## ✨ Features

- **Text Preprocessing**: Cleans and normalizes comment text
- **Feature Extraction**: Uses Bag of Words (BoW) and TF-IDF vectorization
- **Machine Learning**: Multinomial Naive Bayes classifier
- **Model Evaluation**: 5-fold cross-validation and test set evaluation
- **Real-time Prediction**: Classifies new comments as spam or ham
- **Performance Metrics**: Confusion matrix, accuracy score, and classification report

## 📊 Dataset

The project uses the **Youtube02-KatyPerry.csv** dataset containing:
- YouTube comments from Katy Perry videos
- Binary labels (0 = ham/legitimate, 1 = spam)
- Columns: `COMMENT_ID`, `AUTHOR`, `DATE`, `CONTENT`, `CLASS`

## 🛠 Technologies Used

- **Python 3.x**
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computations
- **scikit-learn**: Machine learning algorithms and tools
  - CountVectorizer: Bag of Words
  - TfidfTransformer: TF-IDF transformation
  - MultinomialNB: Naive Bayes classifier
  - cross_val_score: Cross-validation
  - confusion_matrix, accuracy_score, classification_report: Evaluation metrics
- **re**: Regular expressions for text cleaning

## 📁 Project Structure

```
YouTube-comments-spam-detector/
│
├── Group2_Spam_detector_COMP237/
│   ├── nlp_project.py              # Main Python script
│   ├── Youtube02-KatyPerry.csv     # Dataset
│   └── OutputScreenshots/          # Sample output screenshots
│       ├── Screenshot 2025-11-30 134252.png
│       ├── Screenshot 2025-11-30 134302.png
│       └── Screenshot 2025-11-30 134313.png
│
├── LICENSE
└── README.md                        # This file
```

## 🔧 Installation

### Prerequisites

- Python 3.7 or higher
- pip (Python package installer)

### Step 1: Clone the Repository

```bash
git clone https://github.com/Zuck7/YouTube-comments-spam-detector.git
cd YouTube-comments-spam-detector
```

### Step 2: Install Required Packages

Install the required Python packages using pip:

```bash
pip install pandas numpy scikit-learn
```

Or create a virtual environment (recommended):

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install packages
pip install pandas numpy scikit-learn
```

## 🚀 How to Run

### Method 1: Run from Command Line

1. Navigate to the project directory:
```bash
cd Group2_Spam_detector_COMP237
```

2. Run the script:
```bash
python nlp_project.py
```

### Method 2: Run from an IDE

1. Open `nlp_project.py` in your preferred IDE (VS Code, PyCharm, Spyder, etc.)
2. Ensure the `Youtube02-KatyPerry.csv` file is in the same directory
3. Run the script (usually F5 or click the Run button)

### Expected Output

The script will display:
1. Dataset information and structure
2. Data exploration statistics
3. Cleaned text examples
4. Bag of Words features
5. TF-IDF transformation details
6. Train/test split information
7. 5-fold cross-validation results
8. Test set confusion matrix and accuracy
9. Classification report
10. Predictions for 6 new sample comments

## 🔄 Model Pipeline

The spam detection system follows this pipeline:

```
1. Load Data
   └─> CSV file with YouTube comments

2. Data Exploration
   └─> Check shape, distribution, missing values

3. Text Preprocessing
   └─> Convert to lowercase
   └─> Remove special characters
   └─> Remove extra whitespace

4. Feature Extraction
   └─> Bag of Words (CountVectorizer)
   └─> TF-IDF Transformation

5. Train/Test Split
   └─> 75% training, 25% testing

6. Model Training
   └─> Multinomial Naive Bayes

7. Evaluation
   └─> 5-fold Cross-Validation
   └─> Test Set Evaluation

8. Prediction
   └─> Classify new comments
```

### Text Cleaning Function

```python
def clean_text(text):
    text = text.lower()                      # Convert to lowercase
    text = re.sub(r'[^a-z\s]', ' ', text)   # Keep only letters and spaces
    text = re.sub(r'\s+', ' ', text).strip() # Remove extra whitespace
    return text
```

## 📈 Results

The model's performance can be evaluated using:

- **Cross-Validation Accuracy**: Mean accuracy across 5 folds
- **Test Accuracy**: Performance on unseen test data
- **Confusion Matrix**: True positives, false positives, true negatives, false negatives
- **Classification Report**: Precision, recall, F1-score for both classes

### Sample Classification

The model classifies these example comments:

| Comment | Prediction |
|---------|-----------|
| "This song is amazing!" | Ham (0) |
| "Wow I love Katy Perry's vocals" | Ham (0) |
| "This video is so nostalgic" | Ham (0) |
| "Great content, keep it up!" | Ham (0) |
| "WIN $500 NOW CLICK THE LINK BELOW!!!" | Spam (1) |
| "Subscribe to my channel for FREE GIFTCARDS" | Spam (1) |

## 📸 Sample Output

Example screenshots of the program output can be found in the `OutputScreenshots/` directory:

![Output Screenshot 1](Group2_Spam_detector_COMP237/OutputScreenshots/Screenshot%202025-11-30%20134252.png)

![Output Screenshot 2](Group2_Spam_detector_COMP237/OutputScreenshots/Screenshot%202025-11-30%20134302.png)

![Output Screenshot 3](Group2_Spam_detector_COMP237/OutputScreenshots/Screenshot%202025-11-30%20134313.png)

## 👥 Authors

**Group 2 - COMP 237 NLP Project**
- Franklyn (Lead Developer)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔍 Additional Notes

### Understanding Spam Detection

**Ham (Class 0)**: Legitimate comments that add value to the conversation
- Example: "This song is amazing!"
- Example: "Great content, keep it up!"

**Spam (Class 1)**: Unwanted promotional or malicious comments
- Often contain: ALL CAPS, excessive punctuation, promotional links
- Example: "WIN $500 NOW CLICK THE LINK BELOW!!!"
- Example: "Subscribe to my channel for FREE GIFTCARDS"

### Model Characteristics

- **Algorithm**: Multinomial Naive Bayes (ideal for text classification)
- **Feature Representation**: TF-IDF (Term Frequency-Inverse Document Frequency)
- **Validation**: 5-fold cross-validation for robust performance estimation
- **Split**: 75% training, 25% testing

### Potential Improvements

- Try different classifiers (SVM, Random Forest, Neural Networks)
- Experiment with different text preprocessing techniques
- Include n-grams (bigrams, trigrams) for better context
- Handle emojis and special characters more intelligently
- Implement hyperparameter tuning
- Add more sophisticated feature engineering

---

**Note**: Make sure the CSV file is in the same directory as the Python script when running the program.
