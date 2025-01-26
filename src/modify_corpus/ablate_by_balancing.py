import pandas as pd
import os

datives_path = './data/datives/babylm/'
training_dataset_path = './data/training_sets/'
classification_level_default = 'default'

dos = pd.read_csv(datives_path + 'alternant_of_dos.csv')
pos = pd.read_csv(datives_path + 'alternant_of_pos.csv')
nondatives = pd.read_csv(datives_path + 'non-datives.csv')
default_tokens = 0
default_datives = 0
max_tokens = 104770215
max_datives = 133644

# with open(os.path.join(training_dataset_path, classification_level_default, 'train.txt'), 'w') as f:
#     row = 0
#     while row < len(pos):
#         f.write(dos.iloc[row]['sentence'] + '\n')
#         default_tokens += dos.iloc[row]['token_count']
#         default_datives += 1
#         f.write(pos.iloc[row]['sentence'] + '\n')
#         default_tokens += pos.iloc[row]['token_count']
#         default_datives += 1
#         row += 1
#     print('num of datives: ', default_datives)
#     row = 0
#     while row < len(nondatives):
#         f.write(str(nondatives.iloc[row]['sentence']) + '\n')
#         default_tokens += nondatives.iloc[row]['token_count']
#         row += 1
    
# print('default tokens: ', default_tokens)
dos = dos.head(len(pos))
balanced_tokens = 0
balanced_datives = 0

classification_level_ablated = 'balanced'

with open(os.path.join(training_dataset_path, classification_level_ablated, 'train.txt'), 'w') as f:
    row = 0
    while balanced_datives < max_datives:
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
    while balanced_tokens < max_tokens and row < len(nondatives):
        f.write(str(nondatives.iloc[row]['sentence']) + '\n')
        balanced_tokens += nondatives.iloc[row]['token_count']
        row += 1    
        
print('num tokens in balanced: ', balanced_tokens)

