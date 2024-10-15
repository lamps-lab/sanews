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

from transformers import pipeline

import sentencepiece

#encoder = tiktoken.encoding_for_model("gpt-4")

accelerator = Accelerator()

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
model_id = "mistralai/Mistral-7B-v0.1"
model_id = "mistralai/Mistral-7B-Instruct-v0.3"
 
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id,device_map="auto")
#    #torch_dtype=torch.bfloat16,
#    device_map="auto",
#)

#model = accelerator.prepare(model)
#model.to("cuda")
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


def write_json(file_name, data):
    with open(file_name, 'a', newline='') as file:
        if os.path.getsize(file_name) == 0:
            file.write("[\n\t")
        else:
            file.write(",\n")
        file.write(json.dumps(data, indent=4))
        if id == last_id:
            file.write("\n]")

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
    outputs = model.generate(**inputs, max_new_tokens=1024, do_sample=True, temperature=0.0, top_p=0.9, pad_token_id=tokenizer.eos_token_id)
    
    # Decode the output
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract the assistant's response
    response = response.split("assistant:")[-1].strip()
    
    return response


methodnumA = 0
methodnumB = 0

testnumA = 0
testnumB = 0
#groundtruth_path = "../src/new_ground_truth.json"
groundtruth_path = "../data/threshold_ground_truth.json"
dataA = read_json(groundtruth_path)
dataB = read_json('../data/method0_variedlengtharticles.json')

#paper_abstract = read_json('../abstracts.json')
paper_abstract = read_json('../data/abstracts.json')
dataB_lookup = {list(d.keys())[0]: d[list(d.keys())[0]] for d in dataB}

last_id = list(dataA[-1].keys())[-1] if dataA and dataA[-1] else None
first_id = list(dataA[0].keys())[0] if dataA and dataA[0] else None
print(f"first id = {first_id}\nlast id = {last_id}\n\n")
aresponses = 0
bresponses = 0
counter = 0

with open("../data/random_ids.txt", "r") as file:
    lines = file.readlines()
numbers = [int(line.strip()) for line in lines]

#ids = [515,530,1403,1442,1606]
reverse = [1,2,4,7,9]
tokens_since_last_sleep = 0
counter = 0
isReverse = False
correct = 0
wrong = 0

length = len(dataB)
print(f"all samples in groundtruth = {length}")
file_name = "./results/newdirect.json"


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
for data in dataB:
    for id, dict in data.items():
        total +=1
        if random.random() > 0.5:
            shuffledcounter += 1
            shuffled_data.append({
            'id': int(id),
            'article_A': dict["groundtruth"],
            'article_B': dict["generated_article"],
            'abstract': dict["abstract"],
            'human_written': 'A',
            'ai_generated': 'B'
            })
        else:
            shuffled_data.append({
            'id': id,
            'article_A': dict["generated_article"],
            'article_B': dict["groundtruth"],
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


counter = 0
print(f"length = {len(shuffled_data)}\n")
for index, sample in enumerate(shuffled_data):
    id = sample['id']
    if int(id) == 3831:
        counter += 1
        continue
    if counter >= length:
        if correct > wrong:
            print("more correct than wrong")
        else:
             if correct == wrong:
                print(f"how is ti a tie? A = {correct}\nB = {wrong}")
        print(f"Correct: {correct}\tWrong: {wrong}")
        print(f"accuracy = {(correct / counter) *100}%")
        exit()
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
Given a scientific paper abstract, the task is to determine which news article was more likely written by a human; A or B. Answer with a single letter. It could be either B or A!!
scientific paper abstract: {abstract}\n Article A:{a}\n\nArticle B: {b}\n
It is really important to remember to ONLY answer with a single letter, that is generate a single letter A or B: (The answer could be either B or A)
The human article is:
Only answer with a single letter: A or B and explanation using the format below.

Format to use:
Answer:\n
Explanation: 
"""
                }
    ]
    
    #messages = [
    #    {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
    #    {"role": "user", "content": "Who are you?"},
    #]

    # Format the input for the model
    #formatted_input = [{"role": "system", "content": messages[0]['content']}]
    #for i, msg in enumerate(messages[1:]):
    #     formatted_input.append({"role": "user" if i % 2 == 0 else "assistant", "content": msg['content']})

    #response = generate_response(messages)
    # Generate the response
    #response = chatbot(messages, max_length=512, do_sample=True, temperature=0.7, top_p=0.9)
    #print(response)
    #exit()

    input_text = format_messages(messages)
    # Tokenize the concatenated input text

    input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(model.device)
    print("Input IDs:", input_ids)
    attention_mask = tokenizer(input_text, return_tensors="pt").attention_mask.to(model.device)
    print("Attention Mask:", attention_mask)
    outputs = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=50,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=True,
            temperature=0.001 c,
            top_p=0.9, # Only consider the top tokens with a cumulative probability of 0.9
            pad_token_id=tokenizer.eos_token_id,
            #num_return_sequences=1
            )
    print(outputs)
    print("Generated Outputs:", outputs)

    # Extract new tokens generated by the model
    generated_tokens = outputs[0][input_ids.shape[-1]:]
    print("Generated Tokens:", generated_tokens)

    # Decode the outputs correctly
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    print("Response:", response)

    #exit()
    decod = response#tokenizer.decode(response, skip_special_tokens=True)
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


