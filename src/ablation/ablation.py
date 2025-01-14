import pandas as pd
import os

def ablate_by_condition(training_dataset_path, file_name, datives_path, supplement_datives_path, maintain_size=True):
    # Load the non-dative datasets
    nondatives = pd.read_csv(datives_path + 'non-datives.csv')
    supplement_nondatives = pd.read_csv(supplement_datives_path + 'non-datives.csv')

    pos = pd.read_csv(datives_path + 'featural-prepositional.csv')
    dos = pd.read_csv(datives_path + 'featural-double-object.csv')

    pos1 = pos.query('recipient_length > theme_length')
    pos2 = pos.query('recipient_length == theme_length')
    pos3 = pos.query('recipient_length < theme_length')
    dos1 = dos.query('recipient_length > theme_length')
    dos2 = dos.query('recipient_length == theme_length')
    dos3 = dos.query('recipient_length < theme_length')
    num_per_type = min(len(pos1), len(pos3), len(dos1), len(dos3))
    print('num of inbalanced datives per type:', num_per_type)
    removed_tokens = pos1['token_count'].sum() + pos3['token_count'].sum() + dos1['token_count'].sum() + dos3['token_count'].sum()
    pos1 = pos1.sample(num_per_type)
    pos3 = pos3.sample(num_per_type)
    dos1 = dos1.sample(num_per_type)
    dos3 = dos3.sample(num_per_type)
    removed_tokens = removed_tokens - (pos1['token_count'].sum() + pos3['token_count'].sum() + dos1['token_count'].sum() + dos3['token_count'].sum())
    print('num tokens deleted from datives', removed_tokens)
    supplemented_tokens = 0
    os.makedirs(os.path.dirname(training_dataset_path + file_name + '.txt'), exist_ok=True)
    
    with open(training_dataset_path + file_name + '.txt', 'w') as f:
        for sentence in nondatives['sentence']:
            f.write(str(sentence) + '\n')
        for sentence in dos1['sentence']:
            f.write(str(sentence) + '\n')
        for sentence in dos2['sentence']:
            f.write(str(sentence) + '\n')
        for sentence in dos3['sentence']:
            f.write(str(sentence) + '\n')
        for sentence in pos1['sentence']:
            f.write(str(sentence) + '\n')
        for sentence in pos2['sentence']:
            f.write(str(sentence) + '\n')
        for sentence in pos3['sentence']:
            f.write(str(sentence) + '\n')
        supplement_index = 0
        while removed_tokens > 0:
            f.write(str(supplement_nondatives['sentence'][supplement_index]) + '\n')
            removed_tokens -= supplement_nondatives['token_count'][supplement_index]
            supplemented_tokens += supplement_nondatives['token_count'][supplement_index]
            supplement_index += 1
    print('num tokens supplemented:', supplemented_tokens)

ablate_by_condition('data/training_sets/', 'length', 'data/datives/babylm/', 'data/datives/babylm2/', 'recipient_len > theme_len')