import pandas as pd

def read_lines(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return set(file.readlines())

def count_duplicates(file1_path, file2_path):
    lines_file1 = read_lines(file1_path)
    lines_file2 = read_lines(file2_path)
    duplicates = lines_file1.intersection(lines_file2)
    return len(duplicates), duplicates

if __name__ == "__main__":
    file1_path = './data/corpora/babylm/train.sents'
    file2_path = './data/corpora/babylm2/train.sents'
    file3_path = './data/corpora/babylm2/train_noduplicate.sents'
    
    num_duplicates, duplicate_lines = count_duplicates(file1_path, file2_path)
    
    print(f"Number of duplicate lines: {num_duplicates}")
    with open(file2_path, 'r', encoding='utf-8') as file:
        lines_file2 = file.readlines()
    
    unique_lines = [line for line in lines_file2 if line not in duplicate_lines]
    
    with open(file3_path, 'w', encoding='utf-8') as file:
        file.writelines(unique_lines)
    
    print(f"Removed {num_duplicates} duplicate lines from {file2_path}")