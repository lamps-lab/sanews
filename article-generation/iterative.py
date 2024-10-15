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

def estimate_and_truncate(text, max_length=4097, model_name="gpt2"):
    # Initialize the tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)

    # Estimate the number of tokens and truncate the text
    # Average length of a token is around 4 characters (this is a rough estimate)
    if len(text) > max_length * 4:
        text = text[:max_length * 4]

    # Tokenize the text
    tokens = tokenizer.encode(text, truncation=True, max_length=max_length)

    # Check if the tokens exceed the limit
    if len(tokens) > max_length:
        tokens = tokens[:max_length]

    # Convert tokens back to text
    truncated_text = tokenizer.decode(tokens, clean_up_tokenization_spaces=True)

    return truncated_text

abstracts = read_json('../abstracts.json')
scialert = read_csv('../../data/all_crawled_data.csv')
# 14156 - 15351

with open("random_ids.txt", "r") as file:
	lines = file.readlines()

numbers = [int(line.strip()) for line in lines]

test_number = 10
number_of_summaries = 9
temperature = 0.3

print(f"TEST {test_number}")
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
    if id < 4506:
        count += 1  # Increment count for unique id
        continue
        
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

    news_articles = []
    previous_article = ""
    messages =  [  
        {'role':'system', 'content':'You are a Journalist who can read/understand scientific papers and writing a news article based on a scientific article.'},    
        {'role':'user', 'content':f"Based on the scientific abstract below, write a news article at most {len(scialert_text.split())} words as if a human wrote it. The scientific abstract: {article} "}, 
    ]
    previous_article = get_completion_from_messages(messages, temperature=temperature)
    news_articles.append(previous_article)
    for i in range(number_of_summaries-1):
        #print(i)
        messages =  [  
            {'role':'system', 'content':'You are a Journalist who can read/understand scientific papers and writing a news article based on a scientific article.'},    
            {'role':'user', 'content':f"Read the scientific abstract and the previous news article. Draft another news article in a different writing style at most {len(scialert_text.split())} words as if a human wrote it. The scientific abstract: {article}. Previous News Article: {previous_article}"}, 
        ]
        previous_article = get_completion_from_messages(messages, temperature=temperature)
        news_articles.append(previous_article)

    print(f"len of articles: {len(news_articles)}")
    if len(news_articles) != number_of_summaries:
    	print(f"sum ain't aight...\n{news_articles}")
    
    data[id]['generated_articles'] = []
    for index, article in enumerate(news_articles):
        data[id]['generated_articles'].append(article)
    prompt = f""" 
    Write a scientific news article based on the news articles below at most {len(scialert_text.split())} words as if a human wrote it.\n News Articles: {news_articles}
    """
    summary = get_completion(prompt, temperature=0.0)
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
