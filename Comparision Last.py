import json
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve

# Load metrics from the saved files
with open('LR_metrics.json', 'r') as f:
    LR_metrics = json.load(f)

with open('SVM_metrics.json', 'r') as f:
    SVM_metrics = json.load(f)
    
with open('KNN_metrics.json', 'r') as f:
    KNN_metrics = json.load(f)

with open('ANN_metrics.json', 'r') as f:
    ANN_metrics = json.load(f)

# Model names
model_names = ['LR', 'SVM', 'KNN', 'ANN']

# Metrics to compare
metrics_to_compare = ['Accuracy', 'F1', 'Roc Auc', 'Auc Pr', 'RMSE']

# Create a figure and subplots
fig, axs = plt.subplots(1, len(metrics_to_compare), figsize=(15, 6), sharey=True)
plt.subplots_adjust(wspace=0.5)

# Loop through each metric and plot bars for each model
for i, metric in enumerate(metrics_to_compare):
    metric_values = [LR_metrics[metric], SVM_metrics[metric], KNN_metrics[metric], ANN_metrics[metric]]
    axs[i].bar(model_names, metric_values)
    axs[i].set_title(metric)
    axs[i].set_ylabel('Value')

plt.suptitle('Comparison of Metrics for Different Models')
plt.show()

# Precision-Recall curve data
precision_lr, recall_lr, _ = precision_recall_curve(LR_metrics['y_test'], LR_metrics['y_pred_prob'])
precision_svm, recall_svm, _ = precision_recall_curve(SVM_metrics['y_test'], SVM_metrics['y_pred_prob'])
precision_knn, recall_knn, _ = precision_recall_curve(KNN_metrics['y_test'], KNN_metrics['y_pred_prob'])
precision_ann, recall_ann, _ = precision_recall_curve(ANN_metrics['y_test'], ANN_metrics['y_pred_prob'])

# Create the plot
plt.figure(figsize=(8, 6))
plt.plot(recall_lr, precision_lr, marker='.', label='LR')
plt.plot(recall_svm, precision_svm, marker='.', label='SVM')
plt.plot(recall_knn, precision_knn, marker='.', label='KNN')
plt.plot(recall_ann, precision_ann, marker='.', label='ANN')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curves')
plt.legend()
plt.grid(True)
plt.show()