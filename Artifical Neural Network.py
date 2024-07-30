import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import precision_recall_curve, auc, mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam


# Load the data from CSV 
df = pd.read_csv('#filename.csv')
df = df.fillna(0)

# ... (Outlier handling and data preparation steps)
print(df.head())
print()
print(df.shape)
print()
print(df.columns)
print()

# Column names
c1 = ' Plate No'; c2 = 'Thick'; c3 = 'DISCHARGE_TEMP'; c4 = 'HEATING_TIME'
c5 = 'P1_TEMP'; c6 = 'LAST_PASS_TEMP'; c7 = 'FINAL_TEMP'; c8 = 'LAST_ROUGH_RED'
c9 = 'last_pass_no'; c10 = 'FIRST_FIN_RED'; c11 = 'FINAL_RED'
c12 = 'WAITING_THICK'; c13 = 'RESTART_TEMPR'; c14 = 'START_TEMPR'; c15 = 'WAIT_TIME'
c16 = 'C'; c17 = 'Mn'; c18 = 'P'; c19 = 'S'; c20 = 'SI'; c21 = 'Al'; c22 = 'Nb'; 
c23 = 'V'; c24 = 'Cu'; c25 = 'Charpy'

# Identify outliers based on a specific condition or range
median_c3 = df[df[c3] > 100][c3].median()
median_c4 = df[df[c4] < 15][c4].median()
median_c5 = df[df[c5] > 500][c5].median()
median_c6 = df[df[c6] < 1000][c6].median()
median_c13 = df[df[c13] < 1100][c13].median()

df.loc[df[c3] < 100, c3] = median_c3
df.loc[df[c4] > 15, c4] = median_c4
df.loc[df[c5] < 500, c5] = median_c5
df.loc[df[c6] > 1000, c6] = median_c6
df.loc[df[c13] > 1100, c13] = median_c13

df[c25] = df[c25].map({'Pass': 1, 'Fail': 0, '0': 0})
df[c25] = pd.to_numeric(df[c25], errors='coerce')
print(df.head())
print()

# Boxplots after removing outliers
for col in df.columns:
    if col != c1:
        plt.boxplot(df[col])
        plt.title(f'Box Plot - {col}')
        plt.show()


# Train-Test Split and Standardization
columns_to_drop = [' Plate No', 'Charpy']
X = df.drop(columns=columns_to_drop)
y = df['Charpy']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=8)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Build the Deep Learning Model
model = Sequential()
model.add(Dense(units=32, activation='relu', input_dim=X_train.shape[1]))  # First hidden layer with 32 units and ReLU activation
model.add(Dense(units=16, activation='relu'))  # Second hidden layer with 16 units and ReLU activation
model.add(Dense(units=8, activation='relu'))   # Third hidden layer with 8 units and ReLU activation
model.add(Dense(units=4, activation='relu'))   # Fourth hidden layer with 4 units and ReLU activation
model.add(Dense(units=1, activation='sigmoid'))  # Output layer with 1 unit and sigmoid activation

# Compile the Model
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Train the Model
history = model.fit(X_train, y_train, batch_size=32, epochs=50, validation_data=(X_test, y_test))

# Plot Training History
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

# Make Predictions
y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype(int)

# Evaluation Metrics and Visualization
accuracy = accuracy_score(y_test, y_pred)
try:
    precision = precision_score(y_test, y_pred)
except ValueError:
    precision = None
recall = recall_score(y_test, y_pred)

precision, recall, thresholds = precision_recall_curve(y_test, y_pred_prob)

plt.plot(recall, precision, marker='.')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.grid(True)
plt.show()

accuracy *= 100
print(f"Accuracy = {accuracy}")
f1 = f1_score(y_test, y_pred)
f1 *= 100
print(f"F1 = {f1}")
roc_auc = roc_auc_score(y_test, y_pred_prob)
roc_auc *= 100
print(f"ROC AUC = {roc_auc}")

auc_pr = auc(recall, precision)
auc_pr *= 100
print(f"AUC PR = {auc_pr}")

rmse_ann = np.sqrt(mean_squared_error(y_test, y_pred))
rmse_ann *= 100
print(f"RMSE = {rmse_ann}")

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

plt.figure()
plt.imshow(cm, interpolation='nearest', cmap='Blues_r')
plt.title(f'Confusion Matrix - ANN Model')
plt.colorbar()
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.xticks([0, 1], ['Fail', 'Pass'])
plt.yticks([0, 1], ['Fail', 'Pass'])

for i in range(len(cm)):
    for j in range(len(cm[i])):
        plt.text(j, i, str(cm[i][j]), ha='center', va='center', color='black')

plt.show()

# Metrics data
metrics = {
    "Accuracy": accuracy,
    "F1": f1,
    "Roc Auc": roc_auc,
    "Auc Pr": auc_pr,
    "RMSE": rmse_ann
}

# Save metrics to a file (e.g., metrics_model1.json)
import json
with open('ANN_metrics.json', 'w') as f:
    json.dump(metrics, f)