import json
import numpy as np
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import precision_recall_curve, auc, mean_squared_error

plt.ion()

# Load the data from CSV 
df = pd.read_csv('#filename.csv')
df = df.fillna(0)

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

# Boxplots before removing outliers
for col in df.columns:
    if (col != c1) & (col != c25):
        plt.boxplot(df[col])
        plt.title(f'Box Plot - {col}')
        plt.show()

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
    if (col != c1) & (col != c25):
        plt.boxplot(df[col])
        plt.title(f'Box Plot - {col}')
        plt.show()

# Train and test Logistic Regression
columns_to_drop = [c1, c25]
X = df.drop(columns=columns_to_drop)
y = df[c25]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=8)

# Create the Logistic Regression model
lr = LogisticRegression(max_iter=5000)

# Train the model on the training data
lr.fit(X_train, y_train)

# Make predictions on the test data
y_pred = lr.predict(X_test)
# Get predicted probabilities for the positive class ('Pass')
y_pred_prob = lr.predict_proba(X_test)[:, 1]

plt.figure(figsize=(8, 8))

# Create scatter plot
plt.scatter(y_pred, y_test, color='blue', label='Data Points')

# Add diagonal line
plt.plot([0, 1], [0, 1], color='red', linestyle='--', label='Diagonal Line')

plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Scatter Plot of Predicted vs. True')
plt.legend()
plt.grid()

plt.show()

# Evaluate the model's performance using built-in functions
accuracy = accuracy_score(y_test, y_pred)

# Calculate RMSE for Logistic Regression
rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred))

try:
    precision = precision_score(y_test, y_pred)
except ValueError:
    precision = None

recall = recall_score(y_test, y_pred)

precision, recall, thresholds = precision_recall_curve(y_test, y_pred_prob)

# Plot the precision-recall curve
plt.plot(recall, precision, marker='.')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve LR')
plt.grid(True)
plt.show()


accuracy*=100
print(f"Accuracy = {accuracy}")
f1 = f1_score(y_test, y_pred)
f1*=100
print(f"F1 = {f1}")
roc_auc = roc_auc_score(y_test, y_pred)
roc_auc*=100
print(f"ROC AUC = {roc_auc}")
# Calculate AUC-PR
auc_pr = auc(recall, precision)
auc_pr*=100
print(f"AUC PR = {auc_pr}")
rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred))
rmse_lr*=100
print(f"RMSE = {rmse_lr}")


# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

# Plot the confusion matrix
plt.figure()
plt.imshow(cm, interpolation='nearest', cmap='ocean')
plt.title(f'Confusion Matrix - LR Model')
plt.colorbar()
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.xticks([0, 1], ['Fail', 'Pass'])
plt.yticks([0, 1], ['Fail', 'Pass'])

# Annotate the heatmap cells with the count of each category
for i in range(len(cm)):
    for j in range(len(cm[i])):
        plt.text(j, i, str(cm[i][j]), ha='center', va='center', color='black')

plt.show()


# Metrics data
metrics = {
    "Accuracy": 98.11320754716981,
    "F1": 99.04761904761905,
    "Roc Auc": 50.0,
    "Auc Pr": 99.37102130070633,
    "RMSE": 13.736056394868903
}

# Save metrics to a file (e.g., metrics_model1.json)
with open('LR_metrics.json', 'w') as f:
    json.dump(metrics, f)

