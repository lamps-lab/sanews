#!/usr/bin/env python3
# llama_sequential_evaluation.py
import os
import sys
import json
import random
import argparse
import torch
import re

from transformers import AutoTokenizer, AutoModelForCausalLM
from accelerate import Accelerator

##########################################################
# Read/Write JSON
##########################################################
def read_json(file_name):
    with open(file_name, 'r') as file:
        return json.load(file)

def write_json(file_name, data):
    with open(file_name, 'a', newline='') as file:
        if os.path.getsize(file_name) == 0:
            file.write("[\n\t")
        else:
            file.write(",\n")
        file.write(json.dumps(data, indent=4))

##########################################################
# Format the role-based messages for causal LM
##########################################################
def format_messages(messages):
    formatted = ""
    for message in messages:
        if message["role"] == "system":
            formatted += f"System: {message['content']}\n"
        elif message["role"] == "user":
            formatted += f"User: {message['content']}\n"
    return formatted

##########################################################
# Main classification logic using LLaMA
##########################################################
def classify_with_llama(model, tokenizer, text, temperature=0.0):
    # Simple zero/one-shot style prompt
    prompt = (
        "You are a helpful evaluator.\n"
        "Determine if the following article is Human-written or AI-generated.\n\n"
        f"Article:\n{text}\n\n"
        "Answer with a single word 'human' or 'ai' only:\n"
    )

    messages = [
        {"role": "system", "content": "You are a domain expert evaluator."},
        {"role": "user",   "content": prompt}
    ]
    input_text = format_messages(messages)

    input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(model.device)
    attention_mask = tokenizer(input_text, return_tensors="pt").attention_mask.to(model.device)

    output = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=32,
        eos_token_id=tokenizer.eos_token_id,
        do_sample=True,
        temperature=temperature,
        top_p=0.9
    )

    decoded = tokenizer.decode(output[0][input_ids.shape[-1]:], skip_special_tokens=True)
    match = re.search(r"(human|ai)", decoded.lower())
    if match:
        return match.group(1).lower()
    return "unknown"

##########################################################
# Main
##########################################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser("Evaluate articles with LLaMA in a sequential manner.")
    parser.add_argument("--model_size", type=str, default="8b", help="Choose '8b' or '70b'.")
    parser.add_argument("--data_file", type=str, default="../../data/testing_data.json",
                        help="Path to the JSON with 'annotation' (human) and 'generated_article' (ai).")
    args = parser.parse_args()

    if args.model_size == "8b":
        model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    else:
        model_id = "meta-llama/Meta-Llama-3-70B-Instruct"

    # Load data
    data = read_json(args.data_file)
    # Prepare labeled data: (id, text, label)
    labeled_data = []
    for item in data:
        item_id = item.get("id","unknown")
        if "annotation" in item:        # Human
            labeled_data.append((item_id, item["annotation"], "human"))
        if "generated_article" in item: # AI
            labeled_data.append((item_id, item["generated_article"], "ai"))
    random.shuffle(labeled_data)

    # Load model
    accelerator = Accelerator()
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model = accelerator.prepare(model)
    device = accelerator.device

    correct = 0
    total = 0
    results_file = f"llama_sequential_{args.model_size}.json"

    for (item_id, text, true_label) in labeled_data:
        total += 1
        predicted_label = classify_with_llama(model, tokenizer, text, temperature=0.001)
        verdict = "correct" if (predicted_label == true_label) else "wrong"
        if verdict == "correct":
            correct += 1

        record = {
            "id": item_id,
            "true_label": true_label,
            "predicted_label": predicted_label,
            "verdict": verdict
        }
        write_json(results_file, record)

        print(f"ID: {item_id}  True: {true_label}  Predicted: {predicted_label}  "
              f"Accuracy so far: {100.0 * correct / total:.2f}%  ({correct}/{total})")

    # Close JSON array
    with open(results_file, 'a') as f:
        f.write("\n]")

    print(f"\nFinal accuracy = {100.0 * correct / total:.2f}%  ({correct}/{total})")

