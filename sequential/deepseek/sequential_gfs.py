#!/usr/bin/env python3
import os
import sys
import json
import random
import argparse
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from accelerate import Accelerator

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

# Role-based message formatting
def format_messages(messages):
    text = ""
    for msg in messages:
        if msg["role"] == "system":
            text += f"System: {msg['content']}\n"
        elif msg["role"] == "user":
            text += f"User: {msg['content']}\n"
    return text

# Build few-shot prompt
def build_fewshot_prompt(shots, category, guided, in_domain_examples, out_domain_examples):
    in_domain_cats = ['environment','health','nature','tech']
    in_domain = category in in_domain_cats

    if in_domain and len(in_domain_examples) >= shots:
        chosen = random.sample(in_domain_examples, shots)
    else:
        chosen = random.sample(out_domain_examples, min(shots, len(out_domain_examples)))

    prompt = (
        "You are a domain expert evaluator. "
        "Here are some examples of scientific paper abstracts, each with an AI-generated article "
        "and a human-written article.\n"
        f"I will show you {shots} example(s) below:\n"
    )
    for i, ex in enumerate(chosen):
        prompt += (
            f"\nExample #{i+1}:\n"
            f"Paper Abstract:\n{ex['abstract']}\n\n"
            f"AI-generated news article example #{i+1}:\n{ex['generated_article']}\n\n"
            f"Human-written news article example #{i+1}:\n{ex['annotation']}\n"
        )
    if guided:
        prompt += (
            """
            \nThe characteristics of a human-written articles have a more conversational and narrative tone.
 They may also include more details that are not present in the scientific paper abstract.
Human-written articles may be less precise and may lack the analytical depths that are found in AI-generated content.
 
AI-generated content have a lot more sophisticated language and structure than human-written articles.
They may have hallucinations that are not present in the paper abstract.
Repeating similar terms over and over may also be characteristic of an AI-generated article.
AI-generated content may directly reference the abstract's content in a more detailed and analytical manner.
They also entail more information and scientific terminology about the study.
In light of these characteristics, determine if the following article is Human-written or AI-generated.\n
Now, decide if a new article is 'human' or 'ai'. Answer with a single word: 'human' or 'ai' ONLY.\n
        """
        )
    else:
        prompt += (
            "\nNow, given a new article below, determine if it is 'human' or 'ai' only.\n"
        )
    return prompt

# Classification with deepseek
def classify_with_deepseek(model, tokenizer, system_text, user_text, temperature=0.001):
    messages = [
        {"role":"system", "content": system_text},
        {"role":"user",   "content": user_text},
    ]
    input_text = format_messages(messages)
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(model.device)
    attention_mask = tokenizer(input_text, return_tensors="pt").attention_mask.to(model.device)

    output = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=256,
        temperature=temperature,
        do_sample=True,
        top_p=0.9,
        eos_token_id=tokenizer.eos_token_id
    )
    decoded = tokenizer.decode(output[0][input_ids.shape[-1]:], skip_special_tokens=True)
    print(f"decoded: {decoded}")
    if "</think>" in decoded:
        label = decoded.split("</think>")[1]
    else:
        label = decoded
    return decoded, label

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1', 'guided'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0', 'unguided'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


# Main
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sequentially feed articles to Mistral with few-shot context.")
    parser.add_argument('--model', type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
                        help='Mistral model repo.')
    parser.add_argument('--guided', type=str2bool, default=True,
                        help='Whether to include guiding hints in the prompt.')
    parser.add_argument('--shot', type=int, default=0,
                        help='Number of few-shot examples.')
    parser.add_argument('--cat', type=str, default='environment',
                        help='Category for examples, e.g. environment, tech, health, nature.')
    parser.add_argument('--train_file', type=str, default='../../../training_data.json',
                        help='Path to JSON with "abstract", "annotation", "generated_article", "category".')
    parser.add_argument('--test_file',  type=str, default='../../../data/testing_data.json',
                        help='Path to JSON with items that have "annotation" and "generated_article".')
    parser.add_argument('--out_file',   type=str, default='deepseek_seq_fewshot_results.json',
                        help='Where to store the classification results.')
    args = parser.parse_args()


    # Prepare training examples
    train_examples = read_json(args.train_file)
    in_domain_cats = ['environment','health','nature','tech']
    ind = [ex for ex in train_examples if ex['category'] in in_domain_cats]
    out = [ex for ex in train_examples if ex['category'] not in in_domain_cats]
    # Build few-shot portion
    fewshot_prompt = build_fewshot_prompt(args.shot, args.cat, args.guided, ind, out)

    # if the output file exist, let us read it and see if we can reuse some generation
    # Build dict mapping id -> set(true_label)
    existing_labels = {}
    if os.path.exists(args.out_file):
        data = read_json(args.out_file)
        print(f"llength of items in output datafile: {len(data)}")
        for item in data:
            key = item.get("id", "unknown")
            label = item.get("true_label", None)
            if label is not None:
                existing_labels.setdefault(key, set()).add(label)

    data_test = read_json(args.test_file)
    labeled_data = []
    for item in data_test:
        item_id = item.get("id", "unknown")


        # Prepare a dict for possible new labels with their corresponding text
        possible = {}
        if "annotation" in item:
            possible["human"] = item["annotation"]
        if "generated_article" in item:
            possible["ai"] = item["generated_article"]

        if not possible:
            continue

        # For each label in this test item, add if not already recorded
        for label, text in possible.items():
            # If this id already has both labels, skip adding new ones.
            if item_id in existing_labels and len(existing_labels[item_id]) == 2:
                continue
            # If the label is already recorded, skip it.
            if item_id in existing_labels and label in existing_labels[item_id]:
                continue        # Check for conflicts or duplication
            # Otherwise, add the new label record.
            labeled_data.append((item_id, text, label))
            # Update the existing_labels for subsequent checks.
            existing_labels.setdefault(item_id, set()).add(label)
            

    print(f"read {len(existing_labels)} samples,\n{len(labeled_data)} samples remaining")

    random.shuffle(labeled_data)

    # Initialize Accelerator + Model
    accelerator = Accelerator()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map="auto"
    )
    model = accelerator.prepare(model)

    correct = 0
    total = 0

    # Classification
    for (item_id, article_text, true_label) in labeled_data:
        total += 1
        user_text = (f"{fewshot_prompt}\nARTICLE:\n{article_text}\n"
                     "Answer with a single word ONLY: 'human' or 'ai'?\n")
        output, label = classify_with_mistral(model, tokenizer,
                                          system_text="You are a domain expert evaluator. Think briefly (max 64 tokens), and just answer with 'human' or 'ai'",
                                          user_text=user_text,
                                          temperature=0.001)
        print(f"after thinking tokens: {label}")
        match = re.search(r"(human|ai)", label.lower())
        predicted = match.group(1).lower() if match else "unknown"

        verdict = "correct" if (predicted == true_label) else "wrong"
        if verdict == "correct":
            correct += 1

        record = {
            "id": item_id,
            "true_label": true_label,
            "predicted_label": predicted,
            "verdict": verdict,
            "output": output 
        }
        write_json(args.out_file, record)

        print(f"ID={item_id}, True={true_label}, Pred={predicted}, "
              f"Accuracy so far: {100.0*correct/total:.2f}% ({correct}/{total})")

    # Close JSON
    with open(args.out_file, 'a') as f:
        f.write("\n]")
    print(f"results are saved to {args.out_file}")
    print(f"\nFinal accuracy = {100.0 * correct / total:.2f}% ({correct}/{total})")

