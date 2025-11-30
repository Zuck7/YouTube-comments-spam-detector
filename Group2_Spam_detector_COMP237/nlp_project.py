
# COMP 237 NLP Project
# Group 2a
# Dataset used: Youtube02-KatyPerry.csv

import re
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

try:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('punkt_tab') # Added for compatibility with newer NLTK versions
except:
    print("NLTK data already downloaded or error in downloading.")

# ============================================================
# 1. Load the assigned CSV file
# ============================================================

CSV_FILE = "Youtube02-KatyPerry.csv"   # <-- YOUR DATASET

df = pd.read_csv(CSV_FILE, encoding_errors='ignore')

print("\n=== Columns in the file ===")
print(df.columns)

# Keep only the two relevant columns
df = df[['CONTENT', 'CLASS']].copy()
df.rename(columns={'CONTENT': 'comment', 'CLASS': 'label'}, inplace=True)

df.dropna(subset=['comment', 'label'], inplace=True)

print("\n=== First 5 rows ===")
print(df.head())


# ============================================================
# 2. Basic Data Exploration
# ============================================================

print("\nDataset shape:", df.shape)
print("\nLabel distribution:")
print(df['label'].value_counts())

print("\nMissing values:")
print(df.isna().sum())

print("\nExamples (ham):")
print(df[df['label'] == 0]['comment'].head(3))

print("\nExamples (spam):")
print(df[df['label'] == 1]['comment'].head(3))


# ============================================================
# 3. Clean the text (Using NLTK as required)
# ============================================================

# Load stopwords once to speed up processing
stop_words = set(stopwords.words('english'))

def clean_text(text):
    # 1. Convert to lowercase
    text = text.lower()
    
    # 2. Tokenize (Split into words using NLTK)
    # This satisfies the requirement to use nltk classes/methods
    tokens = word_tokenize(text)
    
    # 3. Remove punctuation and stopwords
    # We keep only words that are alphabetic (no numbers/symbols) 
    # and are NOT in the stop_words list
    cleaned_tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
    
    # 4. Join back into a string
    # CountVectorizer expects strings, not lists
    return ' '.join(cleaned_tokens)

# Apply the function
df['comment_clean'] = df['comment'].apply(clean_text)

print("\n=== Cleaned text examples (NLTK) ===")
print(df[['comment', 'comment_clean']].head())


# ============================================================
# 4. Shuffle dataset
# ============================================================

df = df.sample(frac=1, random_state=42).reset_index(drop=True)

texts = df['comment_clean'].values
labels = df['label'].values


# ============================================================
# 5. Bag of Words (CountVectorizer)
# ============================================================

vectorizer = CountVectorizer()
X_counts = vectorizer.fit_transform(texts)

print("\n=== Bag of Words ===")
print("Shape:", X_counts.shape)
print("First 20 features:", vectorizer.get_feature_names_out()[:20])


# ============================================================
# 6. TF-IDF transformation
# ============================================================

tfidf_transformer = TfidfTransformer()
X_tfidf = tfidf_transformer.fit_transform(X_counts)

print("\n=== TF-IDF ===")
print("Shape:", X_tfidf.shape)


# ============================================================
# 7. Manual 75% / 25% split
# ============================================================

n = X_tfidf.shape[0]
train_size = int(0.75 * n)

X_train = X_tfidf[:train_size]
X_test = X_tfidf[train_size:]

y_train = labels[:train_size]
y_test = labels[train_size:]

print("\nTrain samples:", X_train.shape[0])
print("Test samples:", X_test.shape[0])


# ============================================================
# 8. Train Naive Bayes
# ============================================================

model = MultinomialNB()
model.fit(X_train, y_train)


# ============================================================
# 9. 5-fold CV on training set
# ============================================================

cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')

print("\n=== 5-Fold Cross Validation ===")
print("Accuracy per fold:", cv_scores)
print("Mean accuracy:", cv_scores.mean())


# ============================================================
# 10. Test the model
# ============================================================

y_pred = model.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)

print("\n=== TEST RESULTS ===")
print("Confusion Matrix:\n", cm)
print("\nTest Accuracy:", acc)
print("\nClassification Report:\n", classification_report(y_test, y_pred))


# ============================================================
# 11. Classify 6 new comments
# ============================================================

new_comments = [
    "This song is amazing!",                      # ham
    "Wow I love Katy Perry's vocals",             # ham
    "This video is so nostalgic",                 # ham
    "Great content, keep it up!",                 # ham
    "WIN $500 NOW CLICK THE LINK BELOW!!!",       # spam
    "Subscribe to my channel for FREE GIFTCARDS"  # spam
]

print("\n=== CLASSIFICATION OF NEW COMMENTS ===")

for comment in new_comments:
    clean = clean_text(comment)
    bow = vectorizer.transform([clean])
    tfidf_vec = tfidf_transformer.transform(bow)
    prediction = model.predict(tfidf_vec)[0]

    print(f"\nComment: {comment}")
    print(f"Prediction: {prediction}")
