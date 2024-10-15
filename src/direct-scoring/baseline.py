import os
import sys
import time
import csv, json
import openai
import tiktoken

openai.api_key = "API_KEY"

encoder = tiktoken.encoding_for_model("gpt-4")

def read_json(file_name):
    with open(file_name, 'r') as file:
        data = json.load(file)
    return data

def read_csv(file_name):
    data=[]
    with open(file_name, 'r') as file:
        reader = csv.reader(file, delimiter=',')
        header = next(reader)
        data = [row for row in reader]
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
	if tokens_since_last_sleep > 7000:
		tokens_since_last_sleep = 0
		print("\ngoing to sleep..zZ\n")
		time.sleep(45)
	return response.choices[0].message["content"], tokens_since_last_sleep

def write_json(file_name, data):
    with open(file_name, 'a', newline='') as file:
        if os.path.getsize(file_name) == 0:
            file.write("[\n\t")
        else:
            file.write(",\n")
        file.write(json.dumps(data, indent=4))

with open('../../data/fullscale_onescore.json', 'r') as file:
    data = json.load(file)

file_name = "direct_evaluation.json"

paper_abstract = read_json('../../data/abstracts.json')
num_articles = 0
tokens_since_last_sleep = 0
for id, content in data.items():

	text = content['text']
	results = {}
	results[id] = {}
	results[id]['text'] = text
	print(id)
	abstract = ""
	for dic in paper_abstract:
		if id == int(dic['id']):
			abstract = dic.get('text')

	messages = [
		{'role':'system', 'content': "You are an Evaluator who can read/understand scientific papers and scientific news articles written about scientific papers."},
		{'role':'user', 'content':f"The task is to give an overall score on a scale of 0-10 based on the contexts. ONLY ANSWER with the score in the format below. Scientific paper abstract: {abstract}\nNews Article: {text}. I want the output in this format:\nA: score"}
	]
	one_score, tokens_since_last_sleep = get_completion_from_messages(tokens_since_last_sleep, messages, temperature=0.0, model="gpt-4")
	print(one_score)

	write_json(file_name, results)
	num_articles += 1
	print(f"{num_articles} articles done")
