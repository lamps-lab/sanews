import json
import argparse

parser = argparse.ArgumentParser(description='Process an input file.')
parser.add_argument('input_file', type=str, help='Path to the input file')

args = parser.parse_args()

with open(args.input_file, 'r') as file:
    data = json.load(file)

#with open("fullscale-direct.json") as file:
#    data = json.load(file)

a = 0
b = 0
correct = 0
wrong  = 0
for item in data:
    for key in item:
        if item[key]["verdict"] == "correct":
            correct += 1
        elif item[key]["verdict"] == "wrong":
            wrong += 1
        if item[key]["answer"] == "B":
            b += 1
        else:
            a += 1
# Calculate accuracy
total = correct + wrong
accuracy = (correct / total) * 100 if total > 0 else 0

# Print results
print(f"a = {a}, b = {b}")
print(f"Correct: {correct}")
print(f"Wrong: {wrong}")
print(f"Accuracy: {accuracy:.2f}%")
