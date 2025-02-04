Amazon Sentiment Analysis: Analyzing Product Reviews with NLP & Machine Learning
📌 Project Overview
This project aims to analyze customer sentiment from Amazon product reviews using Natural Language Processing (NLP) and Machine Learning. By applying text preprocessing, sentiment classification, and model evaluation techniques, we extract insights that can help businesses understand customer feedback better.

🚀 Features
Preprocess and clean Amazon review text using NLTK and spaCy
Convert text into numerical features using TF-IDF
Train and evaluate a Logistic Regression model for sentiment classification
Optimize model performance using Hyperparameter Tuning
Visualize sentiment trends and confusion matrix
Perform sentiment aggregation and insights generation
📊 Dataset
Source: Amazon Fine Food Reviews (Kaggle Dataset)
Size: 568,454 reviews
Columns Used: Text, Score, Cleaned_Text
Sentiment Mapping:
Positive (Score ≥ 4)
Neutral (Score = 3)
Negative (Score ≤ 2)
🛠 Tech Stack
Programming Language: Python
Libraries Used:
pandas, numpy - Data handling
nltk, re, spaCy - Text Preprocessing
scikit-learn - Machine Learning Models
matplotlib, seaborn - Data Visualization
transformers - BERT Tokenization (future scope)
🔧 Setup & Installation
Clone the repository:
bash
Copy
Edit
git clone https://github.com/yourusername/amazon-sentiment-analysis.git
cd amazon-sentiment-analysis
Install required libraries:
bash
Copy
Edit
pip install -r requirements.txt
Run Jupyter Notebook:
bash
Copy
Edit
jupyter notebook
Open and run Amazon_Reviews_Analysis.ipynb
🎯 Model Training & Evaluation
Feature Engineering: TF-IDF Vectorization (TfidfVectorizer)
Machine Learning Model: Logistic Regression (sklearn.linear_model.LogisticRegression)
Hyperparameter Tuning: Grid Search CV (GridSearchCV)
Evaluation Metrics: Accuracy, Precision, Recall, F1-score
Final Model Performance:
Accuracy: ~86%
Best Hyperparameters: C=1, solver=lbfgs
📈 Results & Insights
Majority of reviews are positive, with some neutral and negative trends.
Logistic Regression performs well, but BERT-based transformers could improve accuracy.
Common negative sentiment words indicate areas for product improvement.
📌 Future Improvements
Implement Deep Learning (BERT, RoBERTa) for advanced sentiment analysis.
Deploy as a web app using Flask or Streamlit.
Perform real-time sentiment analysis on live product reviews.
📜 License
This project is open-source and available under the MIT License.

