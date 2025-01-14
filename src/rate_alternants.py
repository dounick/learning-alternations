import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from minicons import scorer

model_name = 'loose-default'

tokenizer = AutoTokenizer.from_pretrained('qing-yao/' + model_name + '_seed-42_1e-3')
model = AutoModelForCausalLM.from_pretrained('qing-yao/' + model_name + '_seed-42_1e-3')
lm = scorer.IncrementalLMScorer(model = model, device="cuda", tokenizer = tokenizer)

df = pd.read_csv('experiments/generated_pairs_with_results.csv')
df['do'+model_name] = df.apply(lambda x: lm.sequence_score(tokenizer(x['DOsentence'], return_tensors='pt', truncation=False))[0], axis=1)
df['po'+model_name] = df.apply(lambda x: lm.sequence_score(tokenizer(x['PDsentence'], return_tensors='pt', truncation=False))[0], axis=1)
df[model_name + '_ratio'] = df['do'+model_name] - df['po'+model_name]
df.drop(columns=['do'+model_name, 'po'+model_name], inplace=True)
df.to_csv('experiments/generated_pairs_with_results.csv', index = False)


