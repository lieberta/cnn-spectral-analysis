import numpy as np
import matplotlib.pyplot as plt
from dataset import CustomImageDataset
from models import UNet_color
import os
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from sklearn.metrics import auc, roc_curve, confusion_matrix, f1_score, precision_score, recall_score
import seaborn as sns

# Function to calculate losses for a dataset
def loss_list(dataloader, model, criterion, device):
    loss_set = []
    for x in dataloader:
        x = x.to(device)
        y = model(x)
        loss = criterion(x, y).item()
        loss_set.append(loss)
    return loss_set

# Function to calculate confusion matrix elements
def calculate_confusion_matrix_elements(losses, threshold, is_anomaly):
    tp = fp = tn = fn = 0
    for loss in losses:
        if loss > threshold:
            if is_anomaly:
                tp += 1
            else:
                fp += 1
        else:
            if is_anomaly:
                fn += 1
            else:
                tn += 1
    return tp, fp, tn, fn

# Setting up the device, model, and criterion
device = "cuda" if torch.cuda.is_available() else "cpu"
criterion = nn.MSELoss()

channels = 8
dropout = 0
model = UNet_color(d1=256, d2=16, channels=channels).to(device)
color = 'color'
i = 0 # pick a name out of the model name list

model_name = [f'UNet_2D_2Layer{color}_{dropout}dropout_{channels}channels_epoch50_lr0.001']
model_folder = f'./models/{model_name[i]}'
model_path = f'{model_folder}/{model_name[i]}.pth'

model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Load datasets
anomaly_path = "./data/database_autoencoder/IE_2D_random_setup_honeycomb/B_scans_rgb/defect"
test_path = "./data/database_autoencoder/testdataset_sound/B_scans_rgb"
test_set = DataLoader(dataset=CustomImageDataset(path=test_path, transform=color), shuffle=False, batch_size=1)
anomaly_set = DataLoader(dataset=CustomImageDataset(path=anomaly_path, transform=color), shuffle=False, batch_size=1)

# Calculate losses
test_losses = loss_list(test_set, model, criterion, device)
anomaly_losses = loss_list(anomaly_set, model, criterion, device)

# Combine losses and labels, then calculate ROC curve and AUC
combined_losses = np.concatenate([test_losses, anomaly_losses])
combined_labels = np.concatenate([np.zeros(len(test_losses)), np.ones(len(anomaly_losses))])
fpr, tpr, thresholds = roc_curve(combined_labels, combined_losses)
roc_auc = auc(fpr, tpr)

# Find optimal threshold
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]

# Calculate confusion matrix elements
tp, fp, tn, fn = calculate_confusion_matrix_elements(anomaly_losses, optimal_threshold, True)
tp_test, fp_test, tn_test, fn_test = calculate_confusion_matrix_elements(test_losses, optimal_threshold, False)
total_tp = tp
total_fp = fp + fp_test
total_tn = tn_test
total_fn = fn + fn_test

# Calculate metrics
conf_matrix = confusion_matrix([1]*len(anomaly_losses) + [0]*len(test_losses), [1 if loss > optimal_threshold else 0 for loss in anomaly_losses+test_losses])
f1 = f1_score([1]*len(anomaly_losses) + [0]*len(test_losses), [1 if loss > optimal_threshold else 0 for loss in anomaly_losses+test_losses])
precision = precision_score([1]*len(anomaly_losses) + [0]*len(test_losses), [1 if loss > optimal_threshold else 0 for loss in anomaly_losses+test_losses])
specificity = total_tn / (total_tn + total_fp) if (total_tn + total_fp) != 0 else 0

# Prepare for plotting and saving results
detection_folder = f'{model_folder}/detection_plot'
os.makedirs(detection_folder, exist_ok=True)

# Save metrics to a .txt file
metrics_file_path = os.path.join(detection_folder, 'metrics.txt')
with open(metrics_file_path, 'w') as file:
    file.write(f"Confusion Matrix:\n{conf_matrix}\n")
    file.write(f"F1 Score: {f1}\n")
    file.write(f"Precision: {precision}\n")
    file.write(f"Specificity (True Negative Rate): {specificity}\n")
    file.write(f"AUC: {roc_auc}\n")
    file.write(f"Optimal Threshold: {optimal_threshold}\n")

# Print the metrics:

print(f"Confusion Matrix:\n{conf_matrix}\n")
print(f"F1 Score: {f1}")
print(f"Precision: {precision}")
print(f"Specificity (True Negative Rate): {specificity}")
print(f"AUC: {roc_auc}")
print(f"Optimal Threshold: {optimal_threshold}")

# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='darkorange', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.savefig(os.path.join(detection_folder, 'roc_curve.png'))
plt.show()

# Plot Confusion Matrix
plt.figure()
sns.heatmap(conf_matrix, annot=True, fmt='g')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.savefig(os.path.join(detection_folder, 'confusion_matrix.png'))
plt.show()

# Plot Loss Comparison
plt.figure(figsize=(10, 6))
x_range = range(1, len(test_losses) + 1)
plt.plot(x_range, test_losses, 'g', label='Test Losses')  # 'g' is for green color
plt.plot(x_range, anomaly_losses, 'r', label='Anomaly Losses')  # 'r' is for red color
plt.xlabel('Sample')
plt.ylabel('Loss')
plt.title(f'Loss Comparison {model_name[i]}')
plt.legend()
plt.savefig(os.path.join(detection_folder, f"loss_comparison_{model_name[i]}.png"))
plt.show()