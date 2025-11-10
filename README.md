# 🐦 Twitter Sentiment Analysis using Python

### 📊 Project Overview
This project performs **Sentiment Analysis** on Twitter tweets to understand public opinions and emotional tone expressed in text data.  
It involves **data cleaning, EDA, text preprocessing**, and training **Machine Learning models** — Logistic Regression and Random Forest — to classify tweets into **Positive**, **Negative**, or **Neutral** sentiments.

This was an **independent solo project** completed to strengthen my skills in Python-based data analysis, NLP, and supervised machine learning.

---

### 🎯 Objectives
- Clean and preprocess raw Twitter text data (remove stopwords, URLs, mentions, emojis, and punctuation)
- Perform **Exploratory Data Analysis (EDA)** to identify sentiment trends and frequent words
- Visualize insights using WordCloud and Seaborn visualizations
- Build and compare multiple ML models for sentiment prediction
- Evaluate models using accuracy, precision, recall, and F1-score metrics

---

### 🧠 Key Features
- **Text Preprocessing**: tokenization, lemmatization, stopword removal using NLTK & spaCy  
- **Visualization**: WordCloud, sentiment distribution plots, and frequent word analysis  
- **Model Training**: Logistic Regression & Random Forest Classifier  
- **Evaluation**: Accuracy, Precision, Recall, and F1-Score comparison  

---

### ⚙️ Machine Learning Model Performance
| Model | Accuracy | Precision | Recall | F1-Score |
|--------|-----------|-----------|---------|-----------|
| Logistic Regression | **0.7659** | 0.74 | 0.75 | 0.74 |
| Random Forest | **0.8850** | 0.87 | 0.88 | 0.87 |

✅ The **Random Forest Classifier** performed best, achieving **88.5% accuracy**, showing strong capability to handle non-linear patterns in text sentiment.

---

### 🧰 Technologies & Libraries Used
- **Programming Language**: Python 3  
- **Libraries**: Pandas, NumPy, Matplotlib, Seaborn, NLTK, spaCy, Scikit-learn, WordCloud  
- **Tools**: Jupyter Notebook, GitHub  

---

### 📈 EDA Highlights
- Visualized the **distribution of sentiments** across tweets  
- Created a **WordCloud** to identify dominant words  
- Explored **frequent hashtags and mentions**  
- Identified correlation between text length and sentiment polarity  

---

## 🗂️ Project Structure
## 🗂️ Project Structure
```plaintext
twitter_sentiment_analysis/
│
├── twitter_sentiment_analysis.ipynb     # Main Jupyter Notebook with preprocessing, EDA, and ML models
└── data/
    └── tweets.csv                       # Raw dataset used for sentiment analysis


## 🚀 Future Improvements
- Experiment with advanced NLP models such as **BERT** or **LSTM**  
- Enhance preprocessing to handle **emojis**, **hashtags**, and **sarcasm detection**  
- Deploy the model as a **web application** for real-time tweet sentiment prediction  
- Add **model explainability (SHAP/LIME)** to interpret sentiment decisions  
- Include **hyperparameter tuning** for improved accuracy and generalization

## 🧑‍💻 Author
**Deepak Kumar**  
📧 [deepak.kumar8434543@gmail.com]  
🔗 [www.linkedin.com/in/deepak-kumar-acb2002]  
🌐 [https://github.com/Deepakkumar165]
  
