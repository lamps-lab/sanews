import json
import argparse

parser = argparse.ArgumentParser(description='Process an input file.')
parser.add_argument('input_file', type=str, help='Path to the input file')

args = parser.parse_args()

with open(args.input_file, 'r') as file:
    data = json.load(file)

# Initialize counters
tp = 0  # True Positives
fp = 0  # False Positives
tn = 0  # True Negatives
fn = 0  # False Negatives
correct = 0
wrong = 0
a = 0  # Count of "ai" predictions
b = 0  # Count of "human" predictions

# Iterate through the data
for item in data:
    # Update overall correctness and detailed metrics
    if item['verdict'] == "correct":
        correct += 1
        if item['predicted_label'] == "ai":
            tp += 1
        else:
            tn += 1
    elif item['verdict'] == "wrong":
        wrong += 1
        if item['predicted_label'] == "ai":
            fp += 1
        else:
            fn += 1
    
    # Update prediction counts
    if item['predicted_label'] == "ai":
        a += 1
    else:
        b += 1

# Calculate Accuracy
total = correct + wrong
accuracy = (correct / total) * 100 if total > 0 else 0

# Calculate Precision, Recall, and F1 Score
precision = (tp / (tp + fp)) * 100 if (tp + fp) > 0 else 0
recall = (tp / (tp + fn)) * 100 if (tp + fn) > 0 else 0
f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

# Print the results
print(f"ai (Predicted): {a}, human (Predicted): {b}")
print(f"Correct Predictions: {correct}")
print(f"Wrong Predictions: {wrong}")
print(f"Accuracy: {accuracy:.2f}%")
print(f"Precision (ai): {precision:.2f}%")
print(f"Recall (ai): {recall:.2f}%")
print(f"F1 Score (ai): {f1_score:.2f}%")



