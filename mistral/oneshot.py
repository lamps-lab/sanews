import os
import sys
import time
import csv, json
#import openai
#import tiktoken
import re

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import random
from accelerate import Accelerator

#encoder = tiktoken.encoding_for_model("gpt-4")

from transformers import pipeline


accelerator = Accelerator()

#model_id = "meta-llama/Meta-Llama-3-70B-Instruct"
model_id = "mistralai/Mistral-7B-Instruct-v0.3"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    #torch_dtype=torch.bfloat16,
    device_map="auto",
)
print(f"Model: {model}")

model = accelerator.prepare(model)
chatbot = pipeline("text-generation", model=model, tokenizer=tokenizer)#"mistralai/Mistral-7B-Instruct-v0.3")

num_cores = os.cpu_count()
print(f"Number of CPU cores: {num_cores}")

print("Is CUDA available:", torch.cuda.is_available())
print("Number of GPU(s):", torch.cuda.device_count())

if torch.cuda.is_available():
    current_device = torch.cuda.current_device()
    gpu_properties = torch.cuda.get_device_properties(current_device)
    print(f"Device Name: {gpu_properties.name}")
    print(f"Number of Streaming Multiprocessors (SMs): {gpu_properties.multi_processor_count}")
else:
    print("No GPU available.")

device = accelerator.device

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#model.to(device)

print(f"device = {device}")

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

def get_examples_by_category(examples, category):
    in_domain_examples = [example for example in examples if example['category'] == category]
    out_domain_examples = [example for example in examples if example['category'] != category]
    
    if len(out_domain_examples) >= 3:
        out_domain_examples = random.sample(out_domain_examples, 3)
    else:
        out_domain_examples = out_domain_examples[:3]  # If less than 3 out-domain examples are available
    
    return in_domain_examples, out_domain_examples


# Function to simulate a conversation
def generate_response(messages):
    conversation = ""
    for message in messages:
        conversation += f"{message['role']}: {message['content']}\n"

    # Add the assistant prompt
    conversation += "assistant:"

    # Tokenize input
    inputs = tokenizer(conversation, return_tensors="pt").to("cuda")

    # Generate response
    outputs = model.generate(**inputs, max_new_tokens=1024, do_sample=True, temperature=0.8, top_p=0.9, pad_token_id=tokenizer.eos_token_id)

    # Decode the output
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract the assistant's response
    response = response.split("assistant:")[-1].strip()

    return response

dataB = read_json("../data/testing_data.json")
paper_abstract = read_json('../data/abstracts.json')
 
aresponses = 0
bresponses = 0
counter = 0
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

file_path = '../training_data.json'
examples = read_examples(file_path)
category = 'health'
in_domain_categories = ['health', 'nature', 'tech']

in_domain_examples, out_domain_examples = get_examples_by_categories(examples, in_domain_categories)
 
#in_domain_examples, out_domain_examples = get_examples_by_category(examples, category)
out_domain_categories = set(example['category'] for example in out_domain_examples)

print("number of in-domain examples:", len(in_domain_examples))
print("number of out-domain examples:", len(out_domain_examples))
print(f"out of domain categories: {out_domain_categories}")

file_name = f"./results/guided{category}oneshot.json"

ids = []
with open(file_name) as file:
    data = json.load(file)

for item in data:
    for key in item:
        ids.append(int(key))

counter = 0
print(f"length = {len(shuffled_data)}\n")
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
Only answer with a single letter: A or B and B or A and explanation using the format below.

Format to use:
Answer:\n
Explanation: 
"""
                }
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
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id# Only consider the top tokens with a cumulative probability of 0.9
            )
    print(outputs)
    print("Generated Outputs:", outputs)
    generated_tokens = outputs[0][input_ids.shape[-1]:]
    print("Generated Tokens:", generated_tokens)
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    print("Response:", response)
    decod = response
    #response = outputs[0][input_ids.shape[-1]:]        
    
    #decod = tokenizer.decode(response, skip_special_tokens=True)
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
            print("Wrong..")
            verdict = "wrong" 
        elif "B" in decoded:
            correct += 1
            print("Correct!")
            verdict = "correct"
        else:
            wrong += 1
            print("Wrong..")
            verdict = "wrong"
    else:
        if "A" in decoded:
            correct += 1
            print("Correct!")
            verdict = "correct"
        elif "B" in decoded:
            wrong += 1
            print("Wrong..")
            verdict = "wrong"
        else:
            wrong += 1
            print("Wrong..")
            verdict = "wrong"
    results[id]["truth"] = truth
    results[id]["verdict"] = verdict
    results[id]["answer"] = decoded
    results[id]["output"] = decod
    results[id]["reason"] = explanation

    write_json(file_name, results)
    counter+=1
    print(f"accuracy = {(correct / counter) *100}%")

with open(file_name, 'a', newline='') as file:
    file.write("\n]")
