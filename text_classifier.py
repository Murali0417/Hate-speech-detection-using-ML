import pandas as pd
import numpy as np
import re
import string

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
df = pd.read_csv("data.csv")
df.columns = ["label", "tweet"]

def clean_text(text):
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r'@\w+|#', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.lower()
    return text

df["clean_tweet"] = df["tweet"].apply(clean_text)
X = df["clean_tweet"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)
lr_model = LogisticRegression()
lr_model.fit(X_train_vec, y_train)
lr_preds = lr_model.predict(X_test_vec)
nb_model = MultinomialNB()
nb_model.fit(X_train_vec, y_train)
nb_preds = nb_model.predict(X_test_vec)
rf_model = RandomForestClassifier(n_estimators=100)
rf_model.fit(X_train_vec, y_train)
rf_preds = rf_model.predict(X_test_vec)
svm_model = SVC(kernel='linear')
svm_model.fit(X_train_vec, y_train)
svm_preds = svm_model.predict(X_test_vec)
def evaluate_model(name, y_true, y_pred):
    print(f"--- {name} ---")
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Classification Report:\n", classification_report(y_true, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))
    print()
evaluate_model("Logistic Regression", y_test, lr_preds)
evaluate_model("Naive Bayes", y_test, nb_preds)
evaluate_model("Random Forest", y_test, rf_preds)
evaluate_model("SVM", y_test, svm_preds)
sns.heatmap(confusion_matrix(y_test, svm_preds), annot=True, cmap="Blues")
plt.title("SVM Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()