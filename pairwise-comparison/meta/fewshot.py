import os
import sys
import time
import csv, json
import re

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import random
from accelerate import Accelerator

def read_json(file_name):
    with open(file_name, 'r') as file:
        data = json.load(file)
    return data

def write_json(file_name, data):
    with open(file_name, 'a', newline='') as file:
        if os.path.getsize(file_name) == 0:
            file.write("[\n\t")
        else:
            file.write(",\n")
        file.write(json.dumps(data, indent=4))

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

def count_objects(obj):
    if isinstance(obj, dict):
        return sum(count_objects(v) for v in obj.values()) + len(obj)
    elif isinstance(obj, list):
        return sum(count_objects(i) for i in obj)
    else:
        return 0

def read_examples(file_path):
    with open(file_path, 'r') as file:
        examples = json.load(file)
    return examples

def get_example_by_id(examples, example_id):
    for example in examples:
        if str(example_id) in example:
            return example[str(example_id)]
    return "Example ID not found."

def get_examples_by_categories(examples, in_domain_categories):
    in_domain_examples = [example for example in examples if example['category'] in in_domain_categories]
    out_domain_examples = [example for example in examples if example['category'] not in in_domain_categories]

    if len(out_domain_examples) >= 3:
        out_domain_examples = random.sample(out_domain_examples, 3)
    else:
        out_domain_examples = out_domain_examples[:3]  # If less than 3 out-domain examples are available

    return in_domain_examples, out_domain_examples

dataB = read_json("data/testing_data.json")
paper_abstract = read_json('./data/abstracts.json')
dataB_lookup = {list(d.keys())[0]: d[list(d.keys())[0]] for d in dataB}
aresponses = 0
bresponses = 0
counter = 0

with open("./data/random_ids.txt", "r") as file:
    lines = file.readlines()
numbers = [int(line.strip()) for line in lines]

#ids = [515,530,1403,1442,1606] 
reverse = [1,2,4,7,9]
tokens_since_last_sleep = 0
counter = 0
isReverse = False
correct = 0
wrong = 0


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
#for data in dataB:
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
"""
with open("fullscale-guided.json") as file:
    data = json.load(file)

ids = []
for item in data:
    for key in item:
        ids.append(int(key))
"""
model_id = "meta-llama/Meta-Llama-3-70B-Instruct"
#model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
if model_id == "meta-llama/Meta-Llama-3-8B-Instruct":
    param = "8b"
else:
    param = "70b"
ids = []

import argparse

parser = argparse.ArgumentParser(description="Guided Few-Shot with two arguments: category and numbe rof shots")
parser.add_argument('model', type=str, help='Please speficiy which LLaMA-3 model touse')
parser.add_argument('guided', type=bool, help='Specifiy if you would like to use guides in the prompt. If yes, type "True"')
parser.add_argument('shot', type=int, help='Numbe rof shots: 1, 2, 3')
parser.add_argument('cat', type=str, help='Category such as environment, tech, nature, and health')

args = parser.parse_args()
print(f"model size: {args.model} example cateogry: {args.cat}, number of shotas: {args.shot}")
category = args.cat
shots = args.shot
param = args.model
guided = args.guided

if param == "8b":
    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
else:
    model_id = "meta-llama/Meta-Llama-3-70B-Instruct"
ids = []

file_path = 'training_data.json'
examples = read_examples(file_path)
in_domain_categories = ['environment','health', 'nature', 'tech']
indomain = True
#guided = True
in_domain_examples, out_domain_examples = get_examples_by_categories(examples, in_domain_categories)
out_domain_categories = set(example['category'] for example in out_domain_examples)

print("number of in-domain examples:", len(in_domain_examples))
print("number of out-domain examples:", len(out_domain_examples))
print(f"out of domain categories: {out_domain_categories}")
print("number of shots: ", shots)
outcategory = "".join(example['category'] for example in out_domain_examples[:shots])
print(outcategory)
#print(f"\nfirstoutcat {out_domain_examples[0]['category']}\nsecondoutcat {out_domain_examples[1]['category']} \nsecondoutcat {out_domain_examples[2]['category']}")

prompt = f"""
Given a scientific paper abstract, the task is to determine which news article was more likely written by a human; A or B. Answer with a single letter.
                
I will give you {shots} examples of a scientific paper abstract an example of AI-generated content and human-written article all based on the scientific paper abstract.

"""

if indomain:
    if guided:
        file_name = f"./domain-dependence/guided/{param}/{shots}shot{category}.json"
    else:
        file_name = f"./domain-dependence/unguided/{param}/{shots}shot{category}.json"
    for i in range(shots):
        prompt += f"\n\n\nExample paper abstract #{i+1}: {in_domain_examples[i]['abstract']}"
        prompt += f"\n\n\nAI-generated news article example #{i+1}: {in_domain_examples[i]['generated_article']}"
        prompt += f"\n\n\nHuman-written news article example #{i+1}: {in_domain_examples[i]['annotation']}"
else:
    if guided:
        file_name = f"./domain-dependence/guided/{param}/{shots}shotout{outcategory}.json"
    else:
        file_name = f"./domain-dependence/unguided/{param}/{shots}shotout{outcategory}.json"
    for i in range(shots):
        prompt += f"\n\nExample paper abstract #{i+1}:{out_domain_examples[i]['abstract']}"
        prompt += f"\n\nAI-generated news article example #{i+1}: {out_domain_examples[i]['generated_article']}"
        prompt += f"\n\nHuman-written news article example #{i+1}: {out_domain_examples[i]['annotation']}"

if guided:
    prompt += f"""
\nThe characteristics of a human-written articles have a more conversational and narrative tone.
 They may also include more details that are not present in the scientific paper abstract.
Human-written articles may be less precise and may lack the analytical depths that are found in AI-generated content.
 
AI-generated content have a lot more sophisticated language and structure than human-written articles.
They may have hallucinations that are not present in the paper abstract.
Repeating similar terms over and over may also be characteristic of an AI-generated article.
AI-generated content may directly reference the abstract's content in a more detailed and analytical manner.
They also entail more information and scientific terminology about the study.
In light of these characteristics, which one is more likely written by a human?\n"""


print(f"filename: {file_name}\nindomain: {indomain}\nguided: {guided}\ncategory: {category}\n")

counter = 0
print(f"length = {len(shuffled_data)}\n")


accelerator = Accelerator()

#model_id = "mistralai/Mistral-7B-Instruct-v0.3"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

model = accelerator.prepare(model)

num_cores = os.cpu_count()
#print(f"Number of CPU cores: {num_cores}")

#print("Is CUDA available:", torch.cuda.is_available())
#print("Number of GPU(s):", torch.cuda.device_count())

if torch.cuda.is_available():
    current_device = torch.cuda.current_device()
    gpu_properties = torch.cuda.get_device_properties(current_device)
    print(f"Device Name: {gpu_properties.name}")
    print(f"Number of Streaming Multiprocessors (SMs): {gpu_properties.multi_processor_count}")
else:
    print("No GPU available.")

device = accelerator.device

for index, sample in enumerate(shuffled_data):
    id = sample['id']
    if int(id) == 3831 or int(id) in ids:
        counter += 1
        continue
    #print(f"\nSample #{counter}")
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
        #print("reverse\n")
    a = sample['article_A']
    b = sample['article_B']
    newprompt = prompt
    newprompt += f"""\nNew content to evaluate:\n\nscientific paper abstract: {abstract}\nA: {a}\nB: {b}\n
It is really important to remember to ONLY answer with a single letter, that is generate a single letter A or B:
The human article is:
Only answer with a single letter: A or B and explanation using the format below.

Format to use:
Answer:\n
Explanation:
"""
    messages = [
                {'role':'system', 'content': "You are an Evaluator who can read/understand scientific papers and scientific news articles written about scientific papers."},
                {'role':'user', 'content':newprompt}
    ]
    input_text = format_messages(messages)
    # Tokenize the concatenated input text
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(model.device)
    attention_mask = tokenizer(input_text, return_tensors="pt").attention_mask.to(model.device)

    outputs = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=50,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=True,
            temperature=0.001,
            top_p=1.0, # Only consider the top tokens with a cumulative probability of 0.9
            )       
    response = outputs[0][input_ids.shape[-1]:]        
    decod = tokenizer.decode(response, skip_special_tokens=True)
    match = re.search(r'Answer:\s*(.*)', decod)
    if match:
        decoded = match.group(1)
        if not "A" in decoded:
            if not "B" in decoded:
                decoded = "no match in answer"
    else:
        print("no match for answer..")
        match = re.search(r'The human article is:\s*(.*)', decod)
        if not match:
            if "A" in decod and "B" in decod:
                decoded = "no answer"
            elif " A " in decod:
                decoded = "A"
            elif " B " in decod or " B" in decod:
                decoded = "B"
            else:
                decoded = "no answer"
        else:
            decoded = match.group(1)
    #print(decod, "\n")
    #print(f"response tensor = {response}")
    #print(f"The human written article is = {decoded}")
    if decoded in ["A", "B"]:
        doItAgain = False # expected scenario
    else:
        print(f"The model did not respond correctly. Response was: {response}\ndecoded = {decoded}\ndo it again...\n\n")
        #doItAgain = True
        #continue
    verdict = ""
    try:
        explanation = decod.split("\n")[1]
    except IndexError as e:
        explanation = "no explanation"
        print(f"IndexError: {e} in sample {index}:")
    if isReverse: # it is reverse, i.e., A is AI, B is GT
        if "A" in decoded:
            wrong += 1
            #print("Wrong..")
            verdict = "wrong" 
        elif "B" in decoded:
            correct += 1
            #print("Correct!")
            verdict = "correct"
        else:
            wrong += 1
            #print("Wrong..")
            verdict = "wrong"
    else:
        if "A" in decoded:
            correct += 1
            #print("Correct!")
            verdict = "correct"
        elif "B" in decoded:
            wrong += 1
            #print("Wrong..")
            verdict = "wrong"
        else:
            wrong += 1
            #print("Wrong..")
            verdict = "wrong"
    results[id]["truth"] = truth
    results[id]["verdict"] = verdict
    results[id]["answer"] = decoded
    results[id]["output"] = decod
    results[id]["reason"] = explanation

    write_json(file_name, results)
    counter+=1
print(f"accuracy = {(correct / counter) *100}%")

#with open(file_name, 'a', newline='') as file:
#    file.write("\n]")
print(file_name)
