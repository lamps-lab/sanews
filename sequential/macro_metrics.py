import json
import argparse
import statistics

from sklearn.metrics import precision_score, recall_score, f1_score, classification_report, confusion_matrix

parser = argparse.ArgumentParser(description='Process an input file.')
parser.add_argument('input_file', type=str, help='Path to the input file')

args = parser.parse_args()

with open(args.input_file, 'r') as file:
    data = json.load(file)


# Initialize lists for ground truth and predictions
y_true = []
y_pred = []


for item in data:
    for key in item:
        truth = item[key].get("truth")
        answer = item[key].get("answer")
        y_true.append(truth)
        y_pred.append(answer)

"""
# Iterate through the data to reconstruct predictions
for item in data:
    for key in item:
        verdict = item[key].get("verdict")
        answer = item[key].get("answer")
        truth = item[key].get("truth")
        y_true.append()
        
        if verdict == "correct":
            prediction = answer
        elif verdict == "wrong":
            prediction = toggle_class(answer)
        else:
            raise ValueError(f"Unexpected verdict: {verdict}")
        
        y_pred.append(prediction)
"""

report = classification_report(y_true, y_pred, zero_division=0, digits=4)
print("Classification Report:\n", report)

# Confusion Matrix
conf_matrix = confusion_matrix(y_true, y_pred, labels=["A", "B"])
print("Confusion Matrix:")
print(conf_matrix)

exit()
# Display the reconstructed predictions (optional)
print("Ground Truth Labels:", y_true)
print("Model Predictions :", y_pred)
print()

# Calculate Precision, Recall, and F1 Score for each class
precision_A = precision_score(y_true, y_pred, pos_label='A', zero_division=0)
recall_A = recall_score(y_true, y_pred, pos_label='A', zero_division=0)
f1_A = f1_score(y_true, y_pred, pos_label='A', zero_division=0)

precision_B = precision_score(y_true, y_pred, pos_label='B', zero_division=0)
recall_B = recall_score(y_true, y_pred, pos_label='B', zero_division=0)
f1_B = f1_score(y_true, y_pred, pos_label='B', zero_division=0)

# Calculate Macro-Averaged Precision, Recall, and F1 Score
precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)

# Output the results
print(f"Precision (A): {precision_A:.2f}")
print(f"Recall (A): {recall_A:.2f}")
print(f"F1 Score (A): {f1_A:.2f}\n")

print(f"Precision (B): {precision_B:.2f}")
print(f"Recall (B): {recall_B:.2f}")
print(f"F1 Score (B): {f1_B:.2f}\n")

print(f"Macro-Average Precision: {precision_macro:.2f}")
print(f"Macro-Average Recall: {recall_macro:.2f}")
print(f"Macro-Average F1 Score: {f1_macro:.2f}")
