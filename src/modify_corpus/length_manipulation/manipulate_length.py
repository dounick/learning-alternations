import os
from utils import reorder_sentence
import spacy
from spacy.tokenizer import Tokenizer
from spacy.util import compile_infix_regex
import pandas as pd
from tqdm import tqdm
import csv

# from stack, gets around hyphen being treated as a separate token 
def custom_tokenizer(nlp):
    inf = list(nlp.Defaults.infixes)               # Default infixes
    inf.remove(r"(?<=[0-9])[+\-\*^](?=[0-9-])")    # Remove the generic op between numbers or between a number and a -
    inf = tuple(inf)                               # Convert inf to tuple
    infixes = inf + tuple([r"(?<=[0-9])[+*^](?=[0-9-])", r"(?<=[0-9])-(?=-)"])  # Add the removed rule after subtracting (?<=[0-9])-(?=[0-9]) pattern
    infixes = [x for x in infixes if "-|–|—|--|---|——|~" not in x] # Remove - between letters rule
    infix_re = compile_infix_regex(infixes)

    return Tokenizer(nlp.vocab, prefix_search=nlp.tokenizer.prefix_search,
                                suffix_search=nlp.tokenizer.suffix_search,
                                infix_finditer=infix_re.finditer,
                                token_match=nlp.tokenizer.token_match,
                                rules=nlp.Defaults.tokenizer_exceptions)
gpu = spacy.prefer_gpu()
print(gpu)
nlp = spacy.load("en_core_web_trf")
nlp.tokenizer = custom_tokenizer(nlp)

source_path = 'data/datives/babylm/'
output_path = 'data/length_manipulation/short_first/'
non_datives = pd.read_csv(source_path + 'non-datives.csv')
pos = pd.read_csv(source_path + 'alternant_of_pos.csv')
dos = pd.read_csv(source_path + 'alternant_of_dos.csv')

output_non_datives = output_path + 'non-datives.csv'
if not os.path.exists(output_non_datives):
    with open(output_non_datives, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['sentence', 'short_first'])
output_pos = output_path + 'pos.csv'
if not os.path.exists(output_pos):
    with open(output_pos, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['sentence', 'short_first'])
output_dos = output_path + 'dos.csv'
if not os.path.exists(output_dos):
    with open(output_dos, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['sentence', 'short_first'])

print('Reordering non-datives')
for i, sentence in tqdm(enumerate(non_datives['sentence']), total=len(non_datives)):
    doc = nlp(sentence)
    reordered = reorder_sentence(doc, short_first=True)
    with open(output_non_datives, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow([sentence, reordered])

print('Reordering pos')
for i, sentence in tqdm(enumerate(pos['sentence']), total=len(pos)):
    doc = nlp(sentence)
    reordered = reorder_sentence(doc, short_first=True)
    with open(output_pos, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow([sentence, reordered]) 

print('Reordering dos')
for i, sentence in tqdm(enumerate(dos['sentence']), total=len(dos)):
    doc = nlp(sentence)
    reordered = reorder_sentence(doc, short_first=True)
    with open(output_dos, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow([sentence, reordered]) 




    