import json
import matplotlib.pyplot as plt
from collections import Counter

import numpy as np

import seaborn as sns

#specify the font and size
font_size = 40
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = font_size

json_file_path = '../data/ground_truth.json'

with open(json_file_path, 'r') as file:
    data = json.load(file)

print(len(data))

# extract word counts for the gold standard articles
word_counts = []
for key, value in data.items():
    gold_standard_articles = value.get("gold_standard", [])
    for article in gold_standard_articles:
    	if len(article.split()) > 2000 or len(article.split()) < 25:
    		print(key)
    	word_counts.append(len(article.split()))

min_num_words = np.min(word_counts)
max_num_words = np.max(word_counts)
average_num_words = np.mean(word_counts)
median_num_words = np.median(word_counts)


sns.set(style="whitegrid")
plt.figure(figsize=(12, 6))

# plot with Kernel Density Estimate (KDE) to smoothe
sns.histplot(word_counts, bins=30, kde=True, color='teal', label=f'Min: {min_num_words}, Max: {max_num_words}, Mean: {average_num_words}')#, edgecolor='black')
stats_label = f"Min: {min_num_words} \nMax: {max_num_words} \nMean: {average_num_words:.2f} \nMedian: {median_num_words:.2f}"
plt.text(0.62, 0.62, stats_label, transform=plt.gca().transAxes, fontsize=font_size, bbox=dict(facecolor='white', alpha=0.8))
plt.axvline(average_num_words, color='red', linestyle='dashed', linewidth=1, label=f'Average Number of Words = {average_num_words}')
#plt.title('Word Count Distribution of Gold Standard Articles', fontsize=16)
plt.xlabel('Word Count', fontsize=font_size)
plt.ylabel('Frequency', fontsize=font_size)
plt.xticks(fontsize=font_size)
plt.yticks(fontsize=font_size)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
