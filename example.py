import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, mean_squared_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import torch
import torch.nn as nn
import torch.optim as optim

# NumPy basics
a = np.array([[1, 2], [3, 4]])
b = np.dot(a, a)
c = np.linalg.inv(a)
d = np.mean(a)
e = np.std(a)

# Pandas basics
df = pd.DataFrame({'A': np.random.randn(100), 'B': np.random.randint(0, 10, 100)})
df_clean = df.dropna()
df_grouped = df.groupby('B').mean()
df_encoded = pd.get_dummies(df, columns=['B'])

# Matplotlib/Seaborn
plt.figure(figsize=(8, 4))
sns.histplot(df['A'], kde=True)
plt.show()

# Scikit-learn regression/classification
X, y = df[['A']], df['B']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
lr = LinearRegression().fit(X_train, y_train)
y_pred = lr.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
logr = LogisticRegression().fit(X_train, y_train)
acc = logr.score(X_test, y_test)

# Decision tree, random forest
tree = DecisionTreeClassifier().fit(X_train, y_train)
forest = RandomForestClassifier(n_estimators=100).fit(X_train, y_train)
y_pred_tree = tree.predict(X_test)
y_pred_forest = forest.predict(X_test)
cm_tree = confusion_matrix(y_test, y_pred_tree)
cm_forest = confusion_matrix(y_test, y_pred_forest)

# SVM
svc = SVC(probability=True).fit(X_train, y_train)
roc = roc_auc_score(y_test, svc.predict_proba(X_test), multi_class='ovr')

# Clustering, PCA
km = KMeans(n_clusters=3).fit(X)
labels = km.labels_
pca = PCA(n_components=2).fit_transform(X)

# Scaling
scaler = StandardScaler().fit(X)
X_scaled = scaler.transform(X)
minmax = MinMaxScaler().fit(X)
X_minmax = minmax.transform(X)

# GridSearch
param_grid = {'n_estimators': [50, 100], 'max_depth': [2, 4, 8]}
gs = GridSearchCV(RandomForestClassifier(), param_grid, cv=3).fit(X_train, y_train)
best_model = gs.best_estimator_

# Deep Learning with TensorFlow
tf_model = tf.keras.Sequential([
    tf.keras.layers.Dense(32, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
tf_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
tf_model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)
tf_pred = tf_model.predict(X_test)

# PyTorch simple NN
class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(X_train.shape[1], 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 1)
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x

net = SimpleNN()
criterion = nn.BCELoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)
X_tensor = torch.tensor(X_train.values, dtype=torch.float32)
y_tensor = torch.tensor(y_train.values.astype(float)).view(-1, 1)
for epoch in range(10):
    optimizer.zero_grad()
    outputs = net(X_tensor)
    loss = criterion(outputs, y_tensor)
    loss.backward()
    optimizer.step()

# NLP basics
from sklearn.feature_extraction.text import TfidfVectorizer
corpus = ['AI is fun', 'Python is great', 'I love machine learning']
vectorizer = TfidfVectorizer()
X_text = vectorizer.fit_transform(corpus)

import gensim
from gensim.models import Word2Vec
sentences = [['ai', 'is', 'awesome'], ['machine', 'learning', 'rocks']]
w2v = Word2Vec(sentences, vector_size=10, window=3, min_count=1)

# Time Series: ARIMA
from statsmodels.tsa.arima.model import ARIMA
ts_data = np.random.randn(100)
model_arima = ARIMA(ts_data, order=(1,1,1)).fit()
forecast = model_arima.forecast(steps=5)

# Keras CNN for images
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
cnn_model = tf.keras.Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(10, activation='softmax')
])
cnn_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# LSTM for sequences
from tensorflow.keras.layers import LSTM
lstm_model = tf.keras.Sequential([
    LSTM(32, input_shape=(10, 1)),
    Dense(1, activation='sigmoid')
])
lstm_model.compile(optimizer='adam', loss='binary_crossentropy')

# Transformers (HuggingFace)
from transformers import BertTokenizer, BertModel
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
tokens = tokenizer("Artificial intelligence is powerful.", return_tensors='pt')
bert_model = BertModel.from_pretrained('bert-base-uncased')
embeddings = bert_model(**tokens)

# Flask API for ML
from flask import Flask, request, jsonify
app = Flask(__name__)
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    pred = best_model.predict(np.array(data['X']).reshape(1, -1))
    return jsonify({'prediction': int(pred[0])})

# Dockerfile example (string, not file)
dockerfile = """
FROM python:3.9
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
CMD ["python", "ai_ml_expert_skills.py"]
"""

# Streamlit app for ML
import streamlit as st
st.title("Expert ML Predictor")
user_input = st.text_input("Enter value for prediction")
if user_input:
    result = best_model.predict(np.array([float(user_input)]).reshape(1, -1))
    st.write(f"Prediction: {result[0]}")

# Experiment tracking (MLflow)
import mlflow
mlflow.start_run()
mlflow.log_param("model_type", "RandomForest")
mlflow.log_metric("accuracy", acc)
mlflow.end_run()
