import re
import json
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statistics import mean

#sns.set(font_scale=2)

def read_json(path):
	data = []
	with open(path, "r") as f:
		data = json.load(f)
	return data

def extract_scores(data_entry):
	# use regex to find all numbers (floats or integers)
	pattern = r'\b\d+\.\d+|\d+\b'

	# extracting scores from one_score
	one_score_matches = re.findall(pattern, data_entry['one_score'])
	gold_one_score = float(one_score_matches[0])
	generated_one_score = float(one_score_matches[1])

	# extracting scores from three_scores
	three_scores_matches = re.findall(pattern, data_entry['three_scores'])

	# if doesn't exist, reutrn empty
	if not three_scores_matches:
		return gold_one_score, generated_one_score, [], []
	else:
		gold_standard_scores = {
		        'Readability': float(three_scores_matches[0]),
		        'Comprehensiveness': float(three_scores_matches[1]),
		        'Accuracy': float(three_scores_matches[2])
		    }
		generated_scores = {
		        'Readability': float(three_scores_matches[3]),
		        'Comprehensiveness': float(three_scores_matches[4]),
		        'Accuracy': float(three_scores_matches[5])
		    }

		return gold_one_score, generated_one_score, gold_standard_scores, generated_scores

# Function to plot KDE and include t-test stats
def plot_kde_with_ttest(gold_scores, generated_scores, title=""):
	fontize = 25
	ticksize = 20
	plt.figure(figsize=(8,6))
	sns.kdeplot(gold_scores, label="Human-written", fill=True)
	sns.kdeplot(generated_scores, label="LLM-generated", fill=True)#, fontsize=fontize)
	mean_gold = mean(gold_scores)
	mean_generated = mean(generated_scores)
	t_stat, p_val = stats.ttest_ind(gold_scores, generated_scores, equal_var=False)
	plt.axvline(mean_gold, color='lightblue', linestyle='--', label='Mean Human-written', linewidth = 4)#, fontsize=fontize)
	plt.axvline(mean_generated, color='orange', linestyle='--', label='Mean LLM-generated', linewidth = 4)#, fontsize=fontize)

	handles, labels = plt.gca().get_legend_handles_labels()
	labels.append(f'{title}\nt-stat: {t_stat:.2f}')
	labels.append(f'p-value: {p_val:.5f}')
	handles.append(plt.Line2D([], [], color='none'))
	handles.append(plt.Line2D([], [], color='none'))

	# Place the legend on the left side
	plt.legend(handles=handles, labels=labels, loc='upper left', bbox_to_anchor=(.02, 1), fontsize=fontize)
	#plt.title(f"{title} (t-stat: {t_stat:.2f}, p-value: {p_val:.6f})")
	plt.xlabel(f'Scores', fontsize=fontize)
	plt.ylabel('Density', fontsize=fontize)

	plt.xticks(fontsize=ticksize)
	plt.yticks(fontsize=ticksize)

	plt.show()

gold_one_scores = []
generated_one_scores = []
gold_standard_scores = {'Readability': [], 'Comprehensiveness': [], 'Accuracy': []}
generated_article_scores = {'Readability': [], 'Comprehensiveness': [], 'Accuracy': []}

data = read_json("./data/fullscale_onescore.json")
testing_data = read_json("./data/testing_data.json")
training_data = read_json("./data/training_data.json")

all_data = training_data + testing_data

print(len(all_data))
ids = []
all_ids = []
one = 0
two = 0
three = 0
scidist= {}
paperdist = {}
scids = []
for row in testing_data:
	urlid = row['id']
	scid = row['id'].split('-')[0]
	cat = row['category']

	all_ids.append(urlid)

	if cat not in paperdist: # if the category not in research distribution
		paperdist[cat] = 1
		if scid not in scids: # if the sciencealert id hasn't been added then we add it to the category
			scids.append(scid)
			if cat in scidist:
				scidist[cat] += 1
			else:
				scidist[cat] = 1
	else: # the category is in research distribution
		paperdist[cat] += 1
		if scid not in scids: # if the scid not in the scids, meaning we have not added this news article's category
			scids.append(scid)
			if cat in scidist: # if the category is already in the dist, increment it
				scidist[cat] += 1
			else:
				scidist[cat] = 1 # otherwise, assign value one


	if scid not in ids:
		ids.append(scid)
	urlid = row['id'].split('-')[1]
	if urlid == '0':
		one += 1
	elif urlid == '1':
		two += 1
	else:
		three += 1

print(scidist)
print(paperdist)
print(f"o: {one}\n1: {two}\n2: {three}\nunique: {len(list(set(ids)))} all ids: {len(all_ids)}")


for entry in data:
    for key, value in entry.items():
        if key not in ids:
        	continue
        gold_one, generated_one, gold_three, generated_three = extract_scores(value)

        gold_one_scores.append(gold_one)
        generated_one_scores.append(generated_one)

        if len(gold_three) != 0 and len(generated_three) != 0:
            for metric in gold_standard_scores.keys():
                gold_standard_scores[metric].append(gold_three[metric])
                generated_article_scores[metric].append(generated_three[metric])
        else:
        	continue

print(f"count of onescores in distribution: {len(generated_one_scores)}")
# Plotting one_score
plot_kde_with_ttest(gold_one_scores, generated_one_scores, "")
exit()

# Plotting three_scores
for metric in ['Readability', 'Comprehensiveness', 'Accuracy']:
    plot_kde_with_ttest(gold_standard_scores[metric], generated_article_scores[metric], metric)

# Cross-plotting all metrics
for metric in ['Readability', 'Comprehensiveness', 'Accuracy']:
    plt.figure(figsize=(8,6))
    sns.scatterplot(x=gold_standard_scores[metric], y=generated_article_scores[metric])
    plt.title(f"Cross-Plot of {metric}")
    plt.xlabel(f'Gold Standard {metric}')
    plt.ylabel(f'Generated Article {metric}')
    plt.show()
