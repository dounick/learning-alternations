import random
import pandas as pd
import os

path = 'learning-alternations/data/datives/babylm/'

def sample_lines(file_path, num_lines):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        random.shuffle(lines)
        return lines[:num_lines]

def lines_to_dataframe(lines):
    data = {'Lines': lines}
    df = pd.DataFrame(data)
    return df

double_object_lines = sample_lines(os.path.join(path, 'double-object.csv'), 50)
prepositional_lines = sample_lines(os.path.join(path, 'prepositional.csv'), 50)

# Convert sampled lines to DataFrames
double_object_df = lines_to_dataframe(double_object_lines)
prepositional_df = lines_to_dataframe(prepositional_lines)

# Concatenate the two DataFrames
concatenated_df = pd.concat([double_object_df, prepositional_df])
concatenated_df['correct'] = None

concatenated_df.to_csv(os.path.join(path, 'sampled_dative_constructions.csv'), index=False)