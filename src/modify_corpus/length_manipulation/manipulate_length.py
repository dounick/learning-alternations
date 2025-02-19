import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import argparse
import numpy as np
from utils import reorder_sentence_random, reorder_sentence
import spacy
from spacy.tokenizer import Tokenizer
from spacy.util import compile_infix_regex
import pandas as pd
from tqdm import tqdm
import csv


def get_chunk_bounds(total_lines, num_chunks, chunk_id):
    """Calculate the start and end indices for a specific chunk"""
    chunk_size = total_lines // num_chunks
    start_idx = chunk_id * chunk_size
    end_idx = start_idx + chunk_size if chunk_id < num_chunks - 1 else total_lines
    return start_idx, end_idx

def main(args):
    # Custom tokenizer setup
    def custom_tokenizer(nlp):
        inf = list(nlp.Defaults.infixes)
        inf.remove(r"(?<=[0-9])[+\-\*^](?=[0-9-])")
        inf = tuple(inf)
        infixes = inf + tuple([r"(?<=[0-9])[+*^](?=[0-9-])", r"(?<=[0-9])-(?=-)"])
        infixes = [x for x in infixes if "-|–|—|--|---|——|~" not in x]
        infix_re = compile_infix_regex(infixes)
        return Tokenizer(nlp.vocab, prefix_search=nlp.tokenizer.prefix_search,
                        suffix_search=nlp.tokenizer.suffix_search,
                        infix_finditer=infix_re.finditer,
                        token_match=nlp.tokenizer.token_match,
                        rules=nlp.Defaults.tokenizer_exceptions)
    
    gpu = spacy.prefer_gpu()
    print(f"GPU enabled: {gpu}")
    nlp = spacy.load("en_core_web_trf")
    nlp.tokenizer = custom_tokenizer(nlp)
    
    source_path = 'data/corpora/babylm/'
    output_path = 'data/corpora/babylm/'
    os.makedirs(output_path, exist_ok=True)
    
    i = args.i
    j = args.j  # Worker ID (0-9)
    
    # First count total lines for proper chunking
    print(f"Counting lines in file {i}...")
    with open(source_path + str(i) + '.txt', 'r') as f:
        total_lines = sum(1 for line in f if line.strip())
    
    # Calculate chunk bounds for this worker
    start_idx, end_idx = get_chunk_bounds(total_lines, 10, j)
    print(f"Worker {j} processing lines {start_idx} to {end_idx}")
    
    # Initialize dataframe
    df = pd.DataFrame(columns=['sentence', 'short_first', 'long_first', 'random_first'])
    
    # Read and process only this worker's chunk
    current_idx = 0
    processed_lines = 0
    
    with open(source_path + str(i) + '.txt', 'r') as f:
        for line in tqdm(f, desc=f"Worker {j} processing", total=total_lines):
            if not line.strip():
                continue
                
            if start_idx <= current_idx < end_idx:
                line_nlp = nlp(line)
                processed_random = reorder_sentence_random(line_nlp)
                processed_short = reorder_sentence(line_nlp, short_first=True)
                processed_long = reorder_sentence(line_nlp, short_first=False)
                
                df.loc[processed_lines] = {
                    'sentence': line_nlp,
                    'short_first': processed_short,
                    'long_first': processed_long,
                    'random_first': processed_random
                }
                
                processed_lines += 1
                
                # Save progress every 1000 rows
                if processed_lines % 1000 == 0:
                    df.to_csv(output_path + f'{i}_manipulated_part{j}_partial.csv', index=False)
                    print(f"Worker {j}: Progress saved at line {processed_lines}")
            
            current_idx += 1
            if current_idx >= end_idx:
                break
    
    # Save final result for this chunk
    df.to_csv(output_path + f'{i}_manipulated_part{j}.csv', index=False)
    print(f"Worker {j}: Processing complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--i", type=str, required=True, 
                       help="Input file index")
    parser.add_argument("--j", type=int, required=True,
                       help="Worker ID (0-9)")
    args = parser.parse_args()
    main(args)