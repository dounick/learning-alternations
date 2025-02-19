import os
import argparse
from utils import get_inversion_score
import spacy
from spacy.tokenizer import Tokenizer
from spacy.util import compile_infix_regex
import pandas as pd
import csv

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

    source_path = 'data/length_manipulation/'
    os.makedirs(source_path, exist_ok=True)

    i = args.i 

    current_non_datives = pd.read_csv('data/length_manipulation/non-datives_' + str(i) + 'mod.csv')
    current_non_datives.columns = ['sentence', 'short_first', 'long_first', 'token_count', 'is_bad']
    current_non_datives = current_non_datives[current_non_datives['is_bad'] == False]
 
    current_non_datives['short_inversions'] = 0
    current_non_datives['long_inversions'] = 0
    current_non_datives['total_comparisons'] = 0

    output_non_datives = os.path.join('data/length_manipulation', f'non-datives_{i}.csv')
    if not os.path.exists(output_non_datives):
        with open(output_non_datives, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['sentence', 'short_inversions', 'long_inversions', 'total_comparisons'])
    
    print(f'Processing non-datives chunk {i}')
    for sentence in current_non_datives['sentence']:
        doc = nlp(str(sentence))
        inversions = get_inversion_score(doc)
        with open(output_non_datives, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([sentence, inversions['short_inversions'], inversions['long_inversions'], inversions['total_comparisons']])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--i", type=int, required=True, 
                       help="Chunk index (0-based)")
    args = parser.parse_args()
    main(args)