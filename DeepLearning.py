import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf

print("TensorFlow version:", tf.__version__)

# Make sure Churn_Modelling.csv is in the SAME folder as this script,
# or replace the path below with the full path, e.g. r"C:\Users\YourName\Desktop\Churn_Modelling.csv"
dataset = pd.read_csv('Churn_Modelling.csv')

X = dataset.iloc[:, 3:13]
y = dataset.iloc[:, 13]

# One-hot encode Geography (drop_first avoids the dummy variable trap)
geography = pd.get_dummies(X['Geography'], drop_first=True).astype(int)
X = pd.concat([X, geography], axis=1)
X = X.drop(['Geography'], axis=1)

# One-hot encode Gender
gender = pd.get_dummies(X['Gender'], drop_first=True).astype(int)
X = pd.concat([X, gender], axis=1)
X = X.drop(['Gender'], axis=1)

# Splitting the dataset into Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Building the ANN
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping

classifier = Sequential()

# Input layer + 1st hidden layer
classifier.add(Dense(units=6, activation='relu', kernel_initializer='he_uniform', input_dim=11))

# 2nd hidden layer
classifier.add(Dense(units=6, activation='relu', kernel_initializer='he_uniform'))

# 3rd hidden layer
classifier.add(Dense(units=6, activation='relu', kernel_initializer='he_uniform'))

# Output layer
classifier.add(Dense(units=1, activation='sigmoid', kernel_initializer='glorot_uniform'))

# Use a custom-learning-rate Adam optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
classifier.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

model_history = classifier.fit(
    X_train, y_train,
    batch_size=32,
    epochs=100,
    validation_split=0.2,
    callbacks=[EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)]
)

print(model_history.history.keys())

# Evaluate on the test set
y_pred = (classifier.predict(X_test) > 0.5).astype(int)

from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)
print("Test Accuracy:", accuracy_score(y_test, y_pred))

# Plot training history
plt.plot(model_history.history['accuracy'], label='train accuracy')
plt.plot(model_history.history['val_accuracy'], label='val accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()