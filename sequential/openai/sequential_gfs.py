#!/usr/bin/env python3
# gpt_sequential_fewshot.py

import os
import sys
import json
import random
import argparse
import time
import openai
import re
import tiktoken

encoder = tiktoken.encoding_for_model("gpt-4")

# JSON Helpers
def read_json(file_name):
    with open(file_name, 'r') as f:
        return json.load(f)

def write_json(file_name, data):
    with open(file_name, 'a', newline='') as file:
        if os.path.getsize(file_name) == 0:
            file.write("[\n\t")
        else:
            file.write(",\n")
        file.write(json.dumps(data, indent=4))

# Build few-shot prompt
def build_fewshot_prompt(shots, category, guided, in_domain_ex, out_domain_ex):
    in_domain_cats = ['environment', 'health', 'nature', 'tech']
    in_domain = category in in_domain_cats

    if in_domain and len(in_domain_ex) >= shots:
        chosen = random.sample(in_domain_ex, shots)
    else:
        chosen = random.sample(out_domain_ex, min(shots, len(out_domain_ex)))

    prompt = (
        "You are a domain expert evaluator."
    )

    if shots > 0:
    	prompt += (f"I will show you {shots} example(s).\n"
    	"Below are examples of scientific paper abstracts,\n"
        "each accompanied by an AI-generated article and a human-written article:\n")

    for i, ex in enumerate(chosen):
        prompt += (
            f"\n=== Example #{i+1} ===\n"
            f"Paper Abstract:\n{ex['abstract']}\n\n"
            f"AI-generated article:\n{ex['generated_article']}\n\n"
            f"Human-written article:\n{ex['annotation']}\n"
            "======================\n"
        )
    if guided:
        prompt += (
            """\n- The characteristics of a human-written articles have a more conversational and narrative tone.
- They may also include more details that are not present in the scientific paper abstract. 
- Human-written articles may be less precise and may lack the analytical depths that are found in AI-generated content.

- AI-generated content have a lot more sophisticated language and structure than human-written articles.
- They may have hallucinations that are not present in the paper abstract.
- Repeating similar terms over and over may also be characteristic of an AI-generated article.
- AI-generated content may directly reference the abstract's content in a more detailed and analytical manner.
- They also entail more information and scientific terminology about the study.

In light of these characteristics, answer if the NEW article was written by 'human' or 'ai'
        """)
    prompt += "\nGiven a NEW article, answer only 'human' or 'ai'.\n"
    return prompt

# OpenAI classification function
def get_completion_from_messages(tokens_since_last_sleep, messages, temperature=0.0, model="gpt-3.5-turbo"):
    completion = openai.chat.completions.create(model="gpt-3.5-turbo",messages=messages,temperature=temperature)
    print(completion.choices[0].message.content)#.choices[0].text)
    #exit()
    #print(completion.usage)
    #print(completion.model_dump_json(indent=2))
    #exit()
    message_tokens = encoder.encode(json.dumps(messages))
    response_tokens = encoder.encode(completion.model_dump_json(indent=2))
    token_count = len(message_tokens) + len(response_tokens)
    tokens_since_last_sleep += token_count
    #print(f"completion: {completion}\n")
    print(f"tokens_since_last_sleep: {tokens_since_last_sleep}")
    if tokens_since_last_sleep > 8000:
        tokens_since_last_sleep = 0
        print("\ngoing to sleep..zZ\n")
        time.sleep(2)
    return completion.choices[0].message.content, tokens_since_last_sleep

def classify_with_gpt(tokens_since_last_sleep, system_text, user_text, model="gpt-3.5-turbo", temperature=0.0):
    messages = [
        {"role": "system", "content": system_text},
        {"role": "user",   "content": user_text}
    ]
    completion = openai.chat.completions.create(model="gpt-3.5-turbo",messages=messages,temperature=temperature, max_tokens=32)
    message_tokens = encoder.encode(json.dumps(messages))
    response_tokens = encoder.encode(completion.model_dump_json(indent=2))
    token_count = len(message_tokens) + len(response_tokens)
    tokens_since_last_sleep += token_count
    #print(f"completion: {completion}\n")
    print(f"tokens_since_last_sleep: {tokens_since_last_sleep}")
    if tokens_since_last_sleep > 8000:
        tokens_since_last_sleep = 0
        print("\ngoing to sleep..zZ\n")
        time.sleep(2)
    raw_output = completion.choices[0].message.content
    # Extract 'human' or 'ai'
    match = re.search(r"\b(human|ai)\b", raw_output.lower())
    return (match.group(1).lower() if match else "unknown"), raw_output, tokens_since_last_sleep

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1', 'guided'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0', 'unguided'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


###########################################################################
# Main
###########################################################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sequential few-shot with GPT.")
    parser.add_argument('model',      type=str, help='OpenAI model, e.g. "gpt-3.5-turbo" or "gpt-4"')
    parser.add_argument('api_key',    type=str, help='OpenAI API key', default="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")
    parser.add_argument('guided', type=str2bool, help='True/guided to include guiding instructions, False/unguided otherwise', default=False)
    parser.add_argument('shot',       type=int, help='Number of few-shot examples', default=0)
    parser.add_argument('cat',        type=str, help='Category, e.g. environment, nature, tech, health', default="nature")
    parser.add_argument('--train_file', type=str, default='../src/training_data.json')
    parser.add_argument('--test_file',  type=str, default='../src/testing_data.json')
    parser.add_argument('--out_file',   type=str, default='gpt_seq_fewshot_results.json')
    parser.add_argument('--temperature',type=float, default=0.0)
    args = parser.parse_args()

    openai.api_key = args.api_key

    # 1) Read training data
    def get_examples_by_categories(examples, in_domain_cats):
        ind = [x for x in examples if x['category'] in in_domain_cats]
        out = [x for x in examples if x['category'] not in in_domain_cats]
        return ind, out

    all_train_examples = read_json(args.train_file)
    in_domain_categories = ['environment','health','nature','tech']
    in_domain_ex, out_domain_ex = get_examples_by_categories(all_train_examples, in_domain_categories)

    # 2) Build few-shot prompt
    fewshot_prompt = build_fewshot_prompt(
        shots       = args.shot,
        category    = args.cat,
        guided      = args.guided,
        in_domain_ex= in_domain_ex,
        out_domain_ex= out_domain_ex
    )

    # 3) Prepare test data
    test_data = read_json(args.test_file)
    labeled_data = []
    for item in test_data:
        item_id = item.get("id", "unknown")
        if "annotation" in item:
            labeled_data.append((item_id, item["annotation"], "human"))
        if "generated_article" in item:
            labeled_data.append((item_id, item["generated_article"], "ai"))

    random.shuffle(labeled_data)

    if args.guided: 
    	print("guided")
    	#output_file = f"./sequential_results/{args.model}"
    else:
    	print("unguided")
    output_file = f"./sequential_results/{args.model}{'_guided' if args.guided == True else ''}{'' if args.shot == 0 else f'_{args.shot}shot'}_{args.cat}_results.json"

    print(f"outfile: {output_file}")
    #exit()
    # 4) Output file
    if os.path.exists(output_file):
        os.remove(output_file)

    correct = 0
    total   = 0

    tokens_since_last_sleep = 0

    # 5) Classification loop
    for (item_id, text, true_label) in labeled_data:
        total += 1
        user_prompt = (
            f"{fewshot_prompt}\n"
            f"NEW ARTICLE:\n{text}\n"
            "Answer with 'human' or 'ai':"
        )
        if total == 1:
        	print(user_prompt)
        predicted_label, full_response, tokens_since_last_sleep = classify_with_gpt(tokens_since_last_sleep,
            system_text = "You are an expert evaluator.",
            user_text   = user_prompt,
            model       = args.model,
            temperature = args.temperature
        )

        verdict = "correct" if (predicted_label == true_label) else "wrong"
        if verdict == "correct":
            correct += 1

        record = {
            "id": item_id,
            "true_label": true_label,
            "predicted_label": predicted_label,
            "verdict": verdict,
            "raw_output": full_response
        }
        write_json(output_file, record)

        accuracy_so_far = 100.0 * correct / total
        print(f"ID={item_id} True={true_label} Pred={predicted_label} => {verdict}.",
             f"Accuracy so far: {accuracy_so_far:.2f}%  ({correct}/{total})")

    with open(args.out_file, 'a') as f:
        f.write("\n]")

    print(f"\nFinal Accuracy: {100.0 * correct / total:.2f}%  ({correct}/{total})")
