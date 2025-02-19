import pandas as pd
import os

datives_path = './data/datives/babylm/'
nondatives_path = './data/length_manipulation/'
training_dataset_path = './data/training_sets/'
classification_level_default = 'loose_default_small'

dos = pd.read_csv(datives_path + 'loose_alternant_of_dos.csv')
pos = pd.read_csv(datives_path + 'loose_alternant_of_pos.csv')
nondatives = pd.DataFrame(columns=['sentence', 'token_count'])
for i in range(10):
    current_df = pd.read_csv(nondatives_path + 'non-datives_' + str(i) + 'mod.csv')
    current_df.columns = ['sentence', 'short_first', 'long_first', 'token_count', 'is_bad']
    print(current_df.head(2))
    current_df = current_df[current_df['is_bad'] == False]
    current_df = current_df[['sentence', 'token_count']]
    print(current_df.head(2))
    nondatives = pd.concat([nondatives, current_df], ignore_index=True)

print('length of nondatives', len(nondatives))

default_tokens = 0
default_datives = 0
max_datives = 133644
max_tokens = 87436902

with open(os.path.join(training_dataset_path, classification_level_default, 'train.txt'), 'w') as f:
    row = 0
    while row < len(pos) and default_datives < max_datives:
        f.write(dos.iloc[row]['sentence'] + '\n')
        default_tokens += dos.iloc[row]['token_count']
        default_datives += 1
        f.write(pos.iloc[row]['sentence'] + '\n')
        default_tokens += pos.iloc[row]['token_count']
        default_datives += 1
        row += 1
    print('num of datives: ', default_datives)
    row = 0
    while row < len(nondatives) and default_tokens < max_tokens:
        f.write(str(nondatives.iloc[row]['sentence']) + '\n')
        default_tokens += nondatives.iloc[row]['token_count']
        row += 1
    
print('default tokens: ', default_tokens)

dos = dos.head(len(pos))
balanced_tokens = 0
balanced_datives = 0

classification_level_ablated = 'loose_balanced_small'

with open(os.path.join(training_dataset_path, classification_level_ablated, 'train.txt'), 'w') as f:
    row = 0
    while balanced_datives < default_datives:
        f.write(pos.iloc[row]['sentence'] + '\n')
        f.write(pos.iloc[row]['alternant'] + '\n')
        balanced_tokens += 2*pos.iloc[row]['token_count']-1
        balanced_datives += 2
        f.write(dos.iloc[row]['sentence'] + '\n')
        f.write(dos.iloc[row]['alternant'] + '\n')
        balanced_tokens += 2*dos.iloc[row]['token_count']+1
        balanced_datives += 2
        row += 1
    print('num datives in balanced: ', balanced_datives)
    row = 0
    while balanced_tokens < default_tokens and row < len(nondatives):
        f.write(str(nondatives.iloc[row]['sentence']) + '\n')
        balanced_tokens += nondatives.iloc[row]['token_count']
        row += 1    
        
print('num tokens in balanced: ', balanced_tokens)

