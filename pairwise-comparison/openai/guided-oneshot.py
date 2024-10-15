import os
import sys
import time
import csv, json
import torch
import openai
import tiktoken
import re
import random


openai.api_key = ""

encoder = tiktoken.encoding_for_model("gpt-4")

def read_json(file_name):
    with open(file_name, 'r') as file:
        data = json.load(file)
    return data

def get_completion_from_messages(tokens_since_last_sleep, messages, temperature=0.0, model="gpt-3.5-turbo"):
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature, # this is the degree of randomness of the model's output
    )
    message_tokens = encoder.encode(json.dumps(messages))
    response_tokens = encoder.encode(json.dumps(response))
    token_count = len(message_tokens) + len(response_tokens)
    tokens_since_last_sleep += token_count

    print(f"tokens_since_last_sleep: {tokens_since_last_sleep}")
    if tokens_since_last_sleep > 8000:
        #tokens_since_last_sleep = 0
        print("\ngoing to sleep..zZ\n")
        time.sleep(2)
    return response.choices[0].message["content"], tokens_since_last_sleep

def write_json(file_name, data):
    with open(file_name, 'a', newline='') as file:
        if os.path.getsize(file_name) == 0:
            file.write("[\n\t")
        else:
            file.write(",\n")
        file.write(json.dumps(data, indent=4))

# Manually concatenate messages into a single string
def format_messages(messages):
    formatted = ""
    for message in messages:
        if message["role"] == "system":
            formatted += f"System: {message['content']}\n"
        elif message["role"] == "user":
            formatted += f"User: {message['content']}\n"
        elif message["role"] == "assistant":
            formatted += f"Assistant: {message['content']}\n"
    return formatted


def read_examples(file_path):
    with open(file_path, 'r') as file:
        examples = json.load(file)
    return examples

def get_examples_by_categories(examples, in_domain_categories):
    in_domain_examples = [example for example in examples if example['category'] in in_domain_categories]
    out_domain_examples = [example for example in examples if example['category'] not in in_domain_categories]

    if len(out_domain_examples) >= 3:
        out_domain_examples = random.sample(out_domain_examples, 3)
    else:
        out_domain_examples = out_domain_examples[:3]  # If less than 3 out-domain examples are available

    return in_domain_examples, out_domain_examples

def get_examples_by_category(examples, category):
    in_domain_examples = [example for example in examples if example['category'] == category]
    out_domain_examples = [example for example in examples if example['category'] != category]

    if len(out_domain_examples) >= 3:
        out_domain_examples = random.sample(out_domain_examples, 3)
    else:
        out_domain_examples = out_domain_examples[:3]  # If less than 3 out-domain examples are available

    return in_domain_examples, out_domain_examples


dataB = read_json("../src/testing_data.json")


#paper_abstract = read_json('../abstracts.json')
paper_abstract = read_json('../data/abstracts.json')
aresponses = 0
bresponses = 0
counter = 0

tokens_since_last_sleep = 0
counter = 0
correct = 0
wrong = 0

length = len(dataB)
print(f"all samples in groundtruth = {length}")

evaluator_temperature = 0.0
is_groundtruth_evaluation = True

correct = 0
wrong = 0

doItAgain = False

shuffled_data = []
shuffledcounter = 0
total = 0
length = len(dataB)
random.shuffle(dataB)
print("samples = {length}")
for dict in dataB:
    total +=1
    id = dict['id'].split("-")[0]
    urlid = dict['id'].split("-")[1]
    if random.random() > 0.5:
        shuffledcounter += 1
        shuffled_data.append({
        'id': int(id),
        'urlid': urlid,
        'article_A': dict["annotation"],
        'article_B': dict["generated_article"],
        'abstract': dict["abstract"],
        'human_written': 'A',
        'ai_generated': 'B'
        })
    else:
        shuffled_data.append({
        'id': id,
        'urlid': urlid,
        'article_A': dict["generated_article"],
        'article_B': dict["annotation"],
        'abstract': dict["abstract"],
        'human_written': 'B',
        'ai_generated': 'A'
    })
print(f"{shuffledcounter} samples are shuffled out of {total}\n")
print(f"length = {len(shuffled_data)}\n")

file_path = '../src/training_data.json'
examples = read_examples(file_path)
category = 'tech'
in_domain_categories = ['health', 'nature', 'tech']
in_domain = True
if in_domain:
    in_domain_examples, out_domain_examples = get_examples_by_category(examples, category)
    file_name = f"./gpt-3.5results/{category}guidedoneshot.json"
else:
    in_domain_examples, out_domain_examples = get_examples_by_categories(examples, in_domain_categories)
    file_name = f"./results/outguidedoneshot.json"
#in_domain_examples, out_domain_examples = get_examples_by_category(examples, category)
out_domain_categories = set(example['category'] for example in out_domain_examples)

print("number of in-domain examples:", len(in_domain_examples))
print("number of out-domain examples:", len(out_domain_examples))
print(f"out of domain categories: {out_domain_categories}")

ids = []
with open(f"./gpt-3.5results/{category}guidedoneshot.json") as file:
    data = json.load(file)
for item in data:
    for key in item:
        ids.append(int(key))

for index, sample in enumerate(shuffled_data):
	id = sample['id']
	if int(id) in ids:
		print(id)
		counter += 1
		continue
	print(f"\nSample #{counter}")
	print(id)
	results = {}
	results[id] = {}

	abstract = sample['abstract']
	if not abstract:
	    print(f"explosion for {id} -- no abstract")
	truth = ""
	# Extract the human and AI articles based on the metadata
	if sample['human_written'] == 'A':
	    isReverse = False
	    truth = "A"
	else:
	    isReverse = True
	    truth = "B"
	    print("reverse\n")
	a = sample['article_A']
	b = sample['article_B']
	if in_domain:
		messages = [
	        {'role':'system', 'content': "You are an Evaluator who can read/understand scientific papers and scientific news articles written about scientific papers."},
	        {'role':'user', 'content':
f"""
Given a scientific paper abstract, the task is to determine which news article was more likely written by a human; A or B. Answer with a single letter.

I will give you an example of a scientific paper abstract and example of AI-generated content and human-written article all based on the scientific paper abstract.
Example paper abstract: {in_domain_examples[0]['abstract']}
AI-generated news article example: {in_domain_examples[0]['generated_article']}
Human-written news article example: {in_domain_examples[0]['annotation']}

The characteristics of a human-written articles have a more conversational and narrative tone.
They may also include more details that are not present in the scientific paper abstract. 
Human-written articles may be less precise and may lack the analytical depths that are found in AI-generated content. 

AI-generated content have a lot more sophisticated language and structure than human-written articles.
They may have hallucinations that are not present in the paper abstract.
Repeating similar terms over and over may also be characteristic of an AI-generated article.
AI-generated content may directly reference the abstract's content in a more detailed and analytical manner. 
They also entail more information and scientific terminology about the study.

In light of these characteristics, which one is more likely written by a human?

New content to evaluate:
scientific paper abstract: {abstract}\n A:{a}\nB: {b}\n
It is really important to remember to ONLY answer with a single letter, that is generate a single letter A or B:
The human article is:
Only answer with a single letter: A or B and explanation using the format below.

Format to use:
Answer:\n
Explanation: 
"""
		    }
		]
	else:
		messages = [
	        {'role':'system', 'content': "You are an Evaluator who can read/understand scientific papers and scientific news articles written about scientific papers."},
	        {'role':'user', 'content':
f"""
Given a scientific paper abstract, the task is to determine which news article was more likely written by a human; A or B. Answer with a single letter.

I will give you an example of a scientific paper abstract and example of AI-generated content and human-written article all based on the scientific paper abstract.
Example paper abstract: {out_domain_examples[0]['abstract']}
AI-generated news article example: {out_domain_examples[0]['generated_article']}
Human-written news article example: {out_domain_examples[0]['annotation']}
New content to evaluate:
scientific paper abstract: {abstract}\n A:{a}\nB: {b}\n
It is really important to remember to ONLY answer with a single letter, that is generate a single letter A or B:
The human article is:
Only answer with a single letter: A or B and explanation using the format below.

Format to use:
Answer:\n
Explanation: 
"""
		    }
		]
	response, tokens_since_last_sleep = get_completion_from_messages(tokens_since_last_sleep, messages, temperature=evaluator_temperature, model="gpt-3.5-turbo")
	answer, explanation = response.split("\n\n")[0], response.split("\n\n")[1]
	#print(f"response = {response}\n")
	#answer, reason = response.split("\n\n")[0], response.split("\n\n")[1]
	score = answer.split(" ")[1]
	#print(f"output = {answer}\nreason = {reason}")
	print(f"The human written article is = {score}\n{explanation}")
	if isReverse: # it is reverse, i.e., A is AI, B is GT
		if score == "A":
			wrong += 1
			print("Wrong..")
			verdict = "wrong"
		elif score == "B":
			correct += 1
			print("Correct!")
			verdict = "correct"
	else:
		if score == "A":
			correct += 1
			print("Correct!")
			verdict = "correct"
		elif score == "B":
			wrong += 1
			print("Wrong..")
			verdict = "wrong"
	results[id]["truth"] = truth
	results[id]["answer"] = score
	results[id]["verdict"] = verdict
	results[id]["output"] = response
	results[id]["reason"] = explanation

	write_json(file_name, results)
	counter+=1
	print(f"accuracy = {(correct / counter) *100}%")

with open(file_name, 'a', newline='') as file:
    file.write("\n]")
