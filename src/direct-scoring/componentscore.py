import os
import sys
import time
import csv, json
import openai
import tiktoken

openai.api_key = ""

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

def write_to_latex(file_name, fields, rows):
    with open(file_name, 'w') as f:
        f.write("\\begin{tabular}{|c|c|c|c|c|c|}\n")
        f.write("\\hline\n")
        f.write(" & ".join(fields) + " \\\\ [0.5ex]\n")
        f.write("\\hline\\hline\n")
        for row in rows:
            f.write(" & ".join(str(item) for item in row) + " \\\\ [0.5ex]\n")
            f.write("\\hline\n")
        f.write("\\end{tabular}\n")

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
        if id == last_id:
            file.write("\n]")

#news_articles = read_csv('../data/14-18.csv')
annotated_news_articles = read_json('../../data/ground_truth.json')
paper_abstract = read_json('../../data/abstracts.json')

#data_list = read_json('fullscale_method0.json')
file_name = 'fullscale_method0_onescore.json'

num_articles = 0
tokens_since_last_sleep = 0
last_id = list(data_list[-1].keys())[-1] if data_list and data_list[-1] else None
print("data[-1]['id'] = ",last_id)

for data in data_list:
    results = {}
    for key, value in data.items():
        id = int(key)
        #if id < 7073:
        #	num_articles += 1
        #	continue
        print(id)
        results[id] = {}

        # Let us find ScienceAlert article that matches this article's id
        gold_standard = ""
        for dic in annotated_news_articles:
            if str(id) in dic:
                if dic[f"{id}"]['gold_standard']:
                    gold_standard = dic[f"{id}"]['gold_standard']
        if not gold_standard:
        	print(f"No gold for {id}")
        	continue

        # Find the corresponding abstract
        abstract = ""
        for dic in paper_abstract:
        	if id == int(dic['id']):
        		abstract = dic.get('text')
        summary = value.get("generated_article")
        #exit()
        messages =  [  
            {'role':'system', 'content':'You are an Evaluator who can read/understand scientific papers and scientific news articles written about scientific papers and compare each with a human-written news article.'},    
            {'role':'user', 'content':f"The task is to evaluate scientific news articles, A, which is the annotated ScienceAlert article that is relevant to the scientific paper abstract and another news article, B which is generated based on the research paper abstract using 3 criterias. The comparison is done by giving a score for each news article!"+
            f"The 3 criterias are: Readability, Comprehensiveness, and Accuracy.\nONLY ANSWER with the scores on a scale of 0-10 for each Gold Standard and Generated Article based on my format!\nFormat: R: score\nC: score\nA: score\n \nRead the scientific paper abstract and evaluate the two news articles:\n"+
            f"Research Paper Abstract:{abstract}\nScienceAlert article (Gold Standard): {gold_standard}\nGenerated Article: {summary}"}]
        three_scores, tokens_since_last_sleep = get_completion_from_messages(tokens_since_last_sleep, messages, temperature=0.0, model="gpt-4")
        print(three_scores)
        
        messages = [
        	{'role':'system', 'content': "You are an Evaluator who can read/understand scientific papers and scientific news articles written about scientific papers."},
        	{'role':'user', 'content':f"Give an overall score on a scale of 0-10 based on the contexts and 3 scores. ONLY ANSWER with the score in the format below. Scientific paper abstract: {abstract}\nA: {gold_standard}\nB: {summary}. The three scores are in order A first, then B: {three_scores} I want the output in this format:\nA: score\nB: score"}
        ]
        one_score, tokens_since_last_sleep = get_completion_from_messages(tokens_since_last_sleep, messages, temperature=0.0, model="gpt-4")
        print(one_score)

        results[id]['abstract'] = abstract
        results[id]['gold_standard'] = gold_standard
        results[id]['generated_article'] = summary
        results[id]['three_scores'] = three_scores
        results[id]['one_score'] = one_score

        write_json(file_name, results)
        num_articles += 1
        print(f"{num_articles} articles done")
