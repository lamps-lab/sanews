## Pairwise Comparison

The LLaMA-3 models can be run using the shell scripts or providing 4 positional arguments to the fewshot.py program. 
```bash
python3 fewshot.py $MODEL $GUIDED $SHOT $CATEGORY
```

The first positional argument is the size of the LLaMA-3 model: can be either 8b or 70b
The second one if a boolean if we would like to add the guides to the prompt or not. 
The third positonal argument is the number of shots to give to the model.
The last argument is the category to use for the example, which can be one of the following:
"nature", "tech", "health", or "environment" 

## Accuracy
The accuracy script calculates the percentage of correct answers and outputs it with the standard deviation. 

It can be run by providing the path to the file to evaluate. For example:  
```bash
python3 accuracy.py meta/results/main/70b/newdirect.json
```

