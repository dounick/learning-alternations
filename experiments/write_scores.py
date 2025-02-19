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
    
models = ['loose_default', 'loose_balanced_cf', 'noditrans_cf', 'nodative_cf', 'counterfactual', 'short_first_noditransitive', 'random_first_noditransitive', 'long_first_noditransitive']
dos = pd.read_csv('experiments/do_datives.csv')
pos = pd.read_csv('experiments/po_datives.csv')

for model_name in models:
    model_path = 'models/'+model_name+'_seed-42_1e-3'
    print(os.path.exists(model_path))
    print(os.listdir(model_path))
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

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
    dos['do' + model_name] = dos['sentence'].apply(lambda x: safe_score_sequence(x, lm, tokenizer))
    dos['po' + model_name] = dos['alternant'].apply(lambda x: safe_score_sequence(x, lm, tokenizer))
    dos[model_name + '_ratio'] = dos['do'+model_name] - dos['po'+model_name]
    dos.drop(columns=['do'+model_name, 'po'+model_name], inplace=True)
    dos.to_csv('experiments/do_datives_temp.csv', index = False)
    pos['do' + model_name] = pos['alternant'].apply(lambda x: safe_score_sequence(x, lm, tokenizer))
    pos['po' + model_name] = pos['sentence'].apply(lambda x: safe_score_sequence(x, lm, tokenizer))
    pos[model_name + '_ratio'] = pos['do'+model_name] - pos['po'+model_name]
    pos.drop(columns=['do'+model_name, 'po'+model_name], inplace=True)
    pos.to_csv('experiments/po_datives_temp.csv', index = False)
