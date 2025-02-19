#write scores into test sets
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from minicons import scorer
import os
import torch

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.chdir('/home/qy2672/learning-alternations')

def safe_score_sequence(text, lm, tokenizer):
    try:
        # Get the device from the model
        device = next(lm.model.parameters()).device
        
        # Tokenize and move to correct device in one step
        inputs = tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            max_length=256,
            padding=True
        ).to(device)  # Move the whole input dict to device
        
        return lm.sequence_score(inputs)[0]
    except Exception as e:
        print(f"Error processing text: {text}")
        print(f"Error: {str(e)}")
        return float('nan')
    
models = ['loose_default_small', 'loose_balanced_small', 'default_small', 'balanced_small']
hawkins = pd.read_csv('experiments/outdated_data/generated_pairs_with_results.csv')
hawkins.drop(columns=['default_ratio','balanced_ratio','loose_balanced_ratio','loose_default_ratio'], inplace=True)
for model_name in models:
    model_path = 'qing-yao/'+model_name+'_seed-42_1e-3'

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Load model with explicit device placement
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
        device_map="auto"
    )

    # Initialize scorer
    lm = scorer.IncrementalLMScorer(
        model=model,
        device="auto",
        tokenizer=tokenizer
    )
    hawkins['do' + model_name] = hawkins['DOsentence'].apply(lambda x: safe_score_sequence(x, lm, tokenizer))
    hawkins['po' + model_name] = hawkins['PDsentence'].apply(lambda x: safe_score_sequence(x, lm, tokenizer))
    hawkins[model_name + '_ratio'] = hawkins['do'+model_name] - hawkins['po'+model_name]
    hawkins.drop(columns=['do'+model_name, 'po'+model_name], inplace=True)
    hawkins.to_csv('experiments/outdated_data/generated_pairs_with_results.csv', index = False)

