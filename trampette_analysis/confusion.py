import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Let's assume the total number of jumps is 122, with a similar distribution of predictions and true labels
# Predicted and actual values (based on the performance metrics in the previous example)

# True classes (comp, semi, soft)
# Assuming a balanced distribution across the 3 classes for simplicity
n_classes = 3
n_total = 120
n_each_class = n_total // n_classes

# Creating a confusion matrix based on the assumed results
# For simplicity, I'll make sure the diagonal has high values (true positives)
# and off-diagonal values (false positives and false negatives) are smaller but still present

true_labels = np.array([0]*n_each_class + [1]*n_each_class + [2]*n_each_class)  # 0 = comp, 1 = semi, 2 = soft
pred_comp = int(n_each_class * 0.82)
pred_semi = int(n_each_class * 0.87)
pred_soft = int(n_each_class * 0.80)

# Account for the false positives and false negatives, ensuring the total length matches
# Adjust the remaining counts for the false positives and false negatives
false_comp = int(n_each_class * 0.09)+1
false_semi = int(n_each_class * 0.065)+1
false_soft = int(n_each_class * 0.1)

print(pred_comp+false_comp, pred_semi+false_semi, pred_soft+false_soft)

# Generating the predicted labels array
pred_labels = np.array([0]*pred_comp+[1]*false_comp+[2]*false_comp +
                       [1]*pred_semi+[0]*false_semi+[2]*false_semi + 
                       [2]*pred_soft+[0]*false_soft+[1]*false_soft)
# Create confusion matrix
cm = confusion_matrix(true_labels, pred_labels, labels=[0, 1, 2])

# Plotting the confusion matrix
plt.figure(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Comp", "Semi", "Soft"], yticklabels=["Comp", "Semi", "Soft"])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix for Landing Stiffness Classification')
plt.show()
