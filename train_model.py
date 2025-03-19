#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import re
import string
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


# In[5]:


df = pd.read_csv(r"C:\Users\ruhee\OneDrive\Desktop\detection of fake review code\fake_review_dataset.csv")

# In[7]:


def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# In[9]:


df['cleaned_review'] = df['review'].apply(clean_text)
# In[11]:


tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X = tfidf_vectorizer.fit_transform(df['cleaned_review'])
y = df['label']


# In[13]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[15]:


models = {
    "SVM": SVC(kernel='linear', probability=True),
    "Naïve Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(max_iter=500)
}


# In[17]:


results = {}


# In[19]:


for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results[name] = acc
    print(f"{name} Accuracy: {acc:.4f}")


# In[20]:


best_model_name = max(results, key=results.get)
best_model = models[best_model_name]


# In[23]:


joblib.dump(best_model, "best_fake_review_model.pkl")
joblib.dump(tfidf_vectorizer, "tfidf_vectorizer.pkl")

print(f"Best model: {best_model_name} saved!")


# In[ ]:




