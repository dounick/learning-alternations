import pandas as pd
import os

datives_path = './data/length_manipulation/'
training_dataset_path = './data/training_sets/'
classification_level_default = 'short_first_small'

maximum_tokens = 87436902

nondatives = pd.read_csv(datives_path + 'non-datives_' + str(0) + 'mod.csv')
nondatives.columns = ['sentence', 'short_first', 'long_first', 'token_count', 'is_bad']
print(len(nondatives))
nondatives = nondatives[nondatives['is_bad'] == False]
print(len(nondatives))
for i in range(1,10):
    curr_nondatives = pd.read_csv(datives_path + 'non-datives_' + str(i) + 'mod.csv')
    curr_nondatives.columns = ['sentence', 'short_first', 'long_first', 'token_count', 'is_bad']
    curr_nondatives = curr_nondatives[curr_nondatives['is_bad'] == False]
    nondatives = pd.concat([nondatives, curr_nondatives])

# default_tokens = 0

# with open(os.path.join(training_dataset_path, classification_level_default, 'train.txt'), 'w') as f:
#     row = 0
#     while row < len(nondatives) and default_tokens < maximum_tokens:
#         f.write(str(nondatives.iloc[row]['short_first']) + '\n')
#         default_tokens += nondatives.iloc[row]['token_count']
#         row += 1
    
# print('default tokens: ', default_tokens)

balanced_tokens = 0

classification_level_ablated = 'long_first_small'

with open(os.path.join(training_dataset_path, classification_level_ablated, 'train.txt'), 'w') as f:
    row = 0
    while row < len(nondatives) and balanced_tokens < maximum_tokens:
        f.write(str(nondatives.iloc[row]['long_first']) + '\n')
        balanced_tokens += nondatives.iloc[row]['token_count']
        row += 1  
        
print('num tokens in balanced: ', balanced_tokens)

