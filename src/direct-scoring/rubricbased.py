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
		time.sleep(40)
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
annotated_news_articles = read_json('../data/threshold_ground_truth.json')
paper_abstract = read_json('../data/abstracts.json')

test_number = 5
print(f"METHOD 1 - TEST {test_number} EVALUATION")
#data_list = read_json(f'../data/fullscale_method1_test{test_number}.json')
#data_list = read_json('../data/method0_variedlengtharticles.json')
data_list = read_json("../data/method0test0.json")
output_file_name = f'out/experiment1gt.json'

with open("../data/random.txt", "r") as file:
	lines = file.readlines()

numbers = [int(line.strip()) for line in lines]

num_articles = 0
tokens_since_last_sleep = 0
last_id = list(data_list[-1].keys())[-1] if data_list and data_list[-1] else None
print("data[-1]['id'] = ",last_id)

counter = 0
for data in data_list:
    results = {}
    for key, value in data.items():
        id = int(key)
        if id not in numbers:
        	continue
        #if id != 1590:
        #	continue
        counter+=1
        if counter >= 12:
        	exit()
        #if id > 4334:
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
        
        #generated_article = value['summary_paragraph']
        generated_article = value['generated_article']
        # Find the corresponding abstract
        abstract = ""
        for dic in paper_abstract:
        	if id == int(dic['id']):
        		abstract = dic.get('text')

        messages = [
        	{'role':'system', 'content': "You are an Evaluator who can read/understand scientific papers and scientific news articles written about scientific papers and then evaluate them based on a given rubric"},
        	{'role':'user', 'content':
        	f"""
		The task is to give an overall quality score based on different aspects presented in the following rubrics.
		Give a score for each metric in the rubric, then aggregate them. Use the entire range including floating points from 0.0 to 10.0.
		Make sure you pay close attention to the arithmetic of the point deductions. Correct the scores in case the artichmetic is wrong.
		1. Accuracy
      	    - Give a full score of 10.0 if all information provided by the news article is consistent with the information in the source abstract.
      	    - Give 0.0 score if the news article is irrelevant to the source abstract.
      	    - Deduct 3 points out of 10 points for every made-up entity that appears in the news article. 
		2. Readability
    	    - Give a full score if the news article is easily readable by the lay persons and avoid unnecessary repetitions of similar terms.
	      	- Give a 0.0 score if the news article is redundant and not easily readable by the general human public.
      		- Give readability a higher weight
      		- Output the term frequency distribution, that is how many times each term appears.
      		- Deduct 2 points out of 10 points for EACH AND EVERY single term/word that occurs more than three times. 
      		For example, if the term "cognitive" appears three times it results in -2 points.
		3. Comprehensiveness
      		- Give a full score of 10.0 if the news article covers essential information in the source abstract. Additional information such as quotes and opinions are okay.  
      		- Give 0.0 score if the news article misses all essential information or if the news article is irrelevant to the source abstract.
      		- Penalize the news article 1 point if the news article misses one piece of key information.
        	 Source Abstract: {abstract}\nNews Article: {gold_standard}.
        ONLY USE MY FORMAT for output + your rubric first! Remember to use the entire range 0.0 to 10.0
       	I want the output exclusively in this format: 1: score\n2: score\n3: score\n ONESCORE: score\n
        """
        	}
        ]
        scores, tokens_since_last_sleep = get_completion_from_messages(tokens_since_last_sleep, messages, temperature=0.0, model="gpt-4")
        print(scores)
        accuracy_score, readability, humanity, total = scores.split("\n")[0],scores.split("\n")[1], scores.split("\n")[2], scores.split("\n")[-1]

        results[id]['abstract'] = abstract
        results[id]['gold_standard'] = gold_standard
        results[id]['generated_article'] = generated_article
        results[id]['scores'] = scores
        results[id]['accuracy'] = accuracy_score
        results[id]['readability'] = readability
        results[id]['comprehensiveness'] = humanity
        results[id]['one_score'] = total

        write_json(output_file_name, results)
        num_articles += 1
        print(f"{num_articles} articles done")
