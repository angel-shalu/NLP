# Restaurant-Review-Sentiment-App
This project is a Natural Language Processing (NLP) and Machine Learning application that predicts whether a restaurant review is positive 🟢 or negative 🔴.
It includes both a Python script for model evaluation and a Streamlit web app for interactive prediction.


# 📂 Project Structure
├── Restaurant_Reviews_dummy.tsv   # Dataset (tab-separated file with Review + Label)

├── sentiment_analysis.py          # Core ML evaluation code

├── app.py                         # Streamlit app

└── README.md                      # Project documentation


# 🚀 Features

# Text Preprocessing: Cleans, tokenizes, and stems reviews.

# Multiple Vectorizers:
TF-IDF
CountVectorizer

# Machine Learning Models:
   Logistic Regression
   K-Nearest Neighbors (KNN)
   Random Forest
   Decision Tree
   Support Vector Machine (SVM)
   Naive Bayes
   (Optional) XGBoost, LightGBM (if installed)

# Model Evaluation:
   Accuracy, Train/Test Bias, Variance
   AUC Score
   Confusion Matrix
   Visualization of metrics with Seaborn

# Interactive Web App:
   Enter your review and get instant prediction

   Probability score (if available)

   Sidebar navigation for selecting models/vectorizers



# 📊 Example Visualizations

   The script generates:

   Accuracy comparison across models

   Train vs Test Accuracy (Bias vs Variance)

   AUC score comparison

   Confusion Matrix of best model




# ⚙️ Installation

Clone the repository:

git clone https://(https://github.com/angel-shalu/-Restaurant-Review-Sentiment-App)
cd restaurant-sentiment-app


Create a virtual environment (recommended):

python -m venv venv

source venv/bin/activate   # On Mac/Linux

venv\Scripts\activate      # On Windows


Install dependencies:

pip install -r requirements.txt




# 📦 Requirements

Example requirements.txt:

numpy,
pandas,
matplotlib,
seaborn,
nltk,
scikit-learn,
streamlit,
xgboost,
lightgbm,




# ▶️ Usage
1. Run Model Evaluation
python sentiment_analysis.py


This trains all models, prints evaluation metrics, and shows graphs.
2. Launch Streamlit Web App
streamlit run app.py

Open in your browser: http://localhost:8501




# 📝 Dataset

The dataset should be a TSV file (Restaurant_Reviews_dummy.tsv) with two columns:

Review   Liked
"The food was amazing!"   1
"Terrible service."       0

Review: The restaurant review text.

Liked: Target label (1 = positive, 0 = negative).




# 📌 Future Improvements

Add deep learning models (LSTM, BERT).

Deploy app to Streamlit Cloud / Hugging Face Spaces.

Add multi-class sentiment (positive, neutral, negative).




# 👩‍💻 Author

Shalini Kumari

📧 [Email](shalinikumari8789@gmail.com) or [LinkedIn](https://www.linkedin.com/in/shalini-kumari-a237b3276/)/[GitHub Profile](https://github.com/angel-shalu)]
