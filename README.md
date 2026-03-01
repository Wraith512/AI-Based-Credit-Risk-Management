# Sentiment Analysis for Twitter using tweepy

This project focuses on building a sentiment analysis system using machine learning. It analyzes textual data such as reviews or comments and classifies them as positive or negative. The project includes text preprocessing, feature extraction using TF-IDF, model training and evaluation to measure performance.

# Tech stack we are going to be using 
Machine Learning: Python, scikit-learn, TF-IDF, Logistic Regression

Backend: Flask, Flask-CORS, Tweepy, joblib

Frontend: React, JavaScript, Chart.js, Axios/Fetch

# Project Workflow

1. User enters a keyword or hashtag on the website  
2. Frontend sends the request to the backend API  
3. Backend fetches related tweets using the Twitter API (tweepy)
4. Tweets are cleaned and converted into TF-IDF vectors  
5. Logistic Regression model predicts sentiment  
6. Results are sent back to the frontend  
7. Website displays sentiment distribution and sample tweets 

