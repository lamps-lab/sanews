#!/usr/bin/env python
# coding: utf-8

import os
import openai
import csv
import json
import sys

import random

openai.api_key = "API_KEY"
 
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


def get_completion(prompt, model="gpt-3.5-turbo", temperature=0.0):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature, # this is the degree of randomness of the model's output
    )
    return response.choices[0].message["content"]

def get_completion_from_messages(messages, model="gpt-3.5-turbo", temperature=0.0):
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature, # this is the degree of randomness of the model's output
    )
    return response.choices[0].message["content"]

def count_words_in_string(input_string):
    words = input_string.split()
    return len(words)

from transformers import GPT2Tokenizer

abstracts = read_json('../abstracts.json')
scialert = read_csv('../../data/all_crawled_data.csv')
# 14156 - 15351

with open("random_ids.txt", "r") as file:
	lines = file.readlines()

numbers = [int(line.strip()) for line in lines]

test_number = 14
number_of_summaries = 9
temperature = 0.3

filename = f"fullscale_method1_test{test_number}.json"
if not os.path.isfile(filename):
    open(filename, 'w').close()

unique_ids = set()
count = 0

for row in abstracts:
    id = int(row['id'])
    if id not in numbers:
    	continue
    if id not in unique_ids:  # Check if id is unique
        unique_ids.add(id)  # Add id to the set of unique IDs
    #continue
    #if id < 14156:
    #    count += 1  # Increment count for unique id
    print(id)
    scialert_text = ""
    for sci in scialert:
        if int(sci[0]) == int(row['id']):
            #if id not in unique_ids:  # Check if id is unique
            #    unique_ids.add(id)  # Add id to the set of unique IDs
            scialert_text = str(sci[6])
    #continue
    #if int(row['id']) != 98 or int(row['id']) !=  185:
    #    continue
    data = {}
    data[id] = {}

    article = row['text']
    anchor = "###"
    if len(article) > 4097 * 4:
    	article = tokenize_and_truncate(article)
    messages =  [  
        {'role':'system', 'content':'You are a Journalist who can read/understand scientific papers and writing a news article based on a scientific article.'},    
        {'role':'user', 'content':f"Based on the scientific article below, write {number_of_summaries} news articles each at most {len(scialert_text.split())} words. At the end of each article, please write: {anchor}\nscientific article: {article} "}, 
    ]
    # After the 5 news article end with {anchor}
    response = get_completion_from_messages(messages, temperature=temperature)
    #print(f"response: {response}\n\n")
    split = response.split(anchor)

    articles = []
    if anchor not in response:
        articles = response.split("\n")
        print("first split by backranch n")
    else:
        print("Split by anchor")
        split = response.split(anchor)
        if split[0] == "":
            split.pop(0)
        if len(split) == 1:
            # now we're in business
            print("split by backranch n ")
            articles = split[0].split("\n")
        else:
            articles = split
    articles = [item for item in articles if item.strip()]

    print(f"len of articles: {len(articles)}")
    if len(articles) != number_of_summaries:
    	print(f"sum ain't right...\n{articles}")
    data[id]['generated_articles'] = []
    for index, article in enumerate(articles):
        data[id]['generated_articles'].append(article)
    prompt = f""" 
    Summarize the {number_of_summaries} news articles below into a single news article in at most {len(scialert_text.split())} words.\n News Articles: {articles}
    """
    summary = get_completion(prompt, temperature=0.0)
    #print(summary)
    data[id]['summary_paragraph'] = summary
    with open(filename, 'a', newline='') as file:
        if abstracts[0]["id"] == id:
            file.write("[\n\t")
        file.write(json.dumps(data, indent=4))
        if row['id'] != abstracts[-1]['id']:
            file.write(",\n")
        else:
            file.write("\n]")

    count+=1
    sys.stdout.write(f"\r{count} articles done\n")
print(f"number of unique article = {len(unique_ids)}")
