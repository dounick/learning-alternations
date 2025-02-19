import argparse
import os
import pathlib
import utils
import spacy
from spacy.tokenizer import Tokenizer
from spacy.util import compile_infix_regex
import pandas as pd

from collections import defaultdict, Counter
from minicons.utils import get_batch
from tqdm import tqdm

import csv


def main(args):
    corpus_path = args.corpus_path
    dative_path = args.dative_path
    batch_size = args.batch_size

    pathlib.Path(dative_path).mkdir(parents=True, exist_ok=True) 
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

    # spacy setup (gpu is actually faster lol)
    gpu = spacy.prefer_gpu()
    print(gpu)
    nlp = spacy.load("en_core_web_trf")
    nlp.tokenizer = custom_tokenizer(nlp)
    corpus = utils.read_file(corpus_path)

    def get_children_flatten(token, depth=0, dep=False, return_tokens=False, include_self = False):
        """recursively get children of a given token using spacy."""
        children = []
        if include_self:
            if dep:
                if return_tokens:
                    children.append(
                        (
                            token.text.lower(),
                            token.dep_,
                            token.tag_,
                            depth,
                            token.i,
                            token,
                        )
                    )
                else:
                    children.append(
                        (token.text.lower(), token.dep_, token.tag_, depth, token.i)
                    )
            else:
                children.append(token.text.lower())
        for child in token.children:
            if dep:
                if return_tokens:
                    children.append(
                        (
                            child.text.lower(),
                            child.dep_,
                            child.tag_,
                            depth,
                            child.i,
                            child,
                        )
                    )
                else:
                    children.append(
                        (child.text.lower(), child.dep_, child.tag_, depth, child.i)
                    )
            else:
                children.append(child.text.lower())
            children.extend(get_children_flatten(child, depth + 1, dep, return_tokens))
        return children

    # for a particular token, return its dependent children in a phrasal form
    def get_phrasal_children(child):
        children_flatten = sorted(get_children_flatten(child, dep=True, include_self=True), key=lambda x: x[4])
        text = "".join([x[0] if x[0] in ["'s", "`s"] else " " + x[0] for x in children_flatten]).strip()
        i = int(children_flatten[0][4])
        # text = child.text.lower()
        # if child.children:
        #     sorted_children = sorted(child.children, key=lambda x: x.i, reverse=True)
        #     for grandchild in sorted_children:
        #         if grandchild.i < child.i:
        #             text = get_phrasal_children(grandchild) + " " + text
        #         else:
        #             text = text + " " + get_phrasal_children(grandchild)
        return text, i
    
    def get_datives_phrasal(texts, batch_size, processor):
        output_non_datives = os.path.join('data/datives/ditransitive', f'non-ditransitives.csv')
        if not os.path.exists(output_non_datives):
            with open(output_non_datives, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['sentence', 'token_count'])
        for doc in tqdm(processor.pipe(texts, disable=["ner"], batch_size=batch_size)):
            is_ditransitive = False
            for entity in doc:
                if entity.pos_ == "VERB":
                    all_children = get_children_flatten(entity, 0, dep=True)
                    children = []
                    for child in all_children:
                        if not(child[4] < entity.i and child[2] != 'nsubj'):
                            children.append(child)
                    if len(children) > 0:
                        tokens, dep, pos_string, depth, index = list(zip(*children))
                        dep_depth = [
                                f"{d}_{str(depth[i])}" for i, d in enumerate(dep)
                            ]
                        if 'prep' in dep or 'dative' in dep:
                            if (
                                "dobj_0" in dep_depth
                                and "dative_0" in dep_depth
                                and "pobj_1" in dep_depth
                            ) or (
                                "dobj_0" in dep_depth
                                and "prep_0" in dep_depth
                                and "pobj_1" in dep_depth
                            ): 
                                is_ditransitive = True
                                break   
                        if (
                            "dobj_0" in dep_depth and "dative_0" in dep_depth
                        ) or Counter(dep_depth)["dobj_0"] >= 2: 
                            is_ditransitive = True
                            break
                        if (
                            "dobj_0" in dep_depth
                            and "prt_0" in dep_depth 
                            and "prep_0" in dep_depth
                            and "pobj_1" in dep_depth
                        ) or (
                            "dobj_0" in dep_depth
                            and ("prt_0" in dep_depth or "advmod_0" in dep_depth) 
                            and "prep_1" in dep_depth
                            and "pobj_2" in dep_depth
                        ):
                            is_ditransitive = True
                            break   

            if not is_ditransitive:
                with open(output_non_datives, 'a', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow([doc.text, len(doc)])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus_path", type=str, help="Path to the corpus file.")
    parser.add_argument("--dative_path", type=str, help="Path to the dative output.")
    parser.add_argument("--batch_size", type=int, default=8192, help="Batch size.")
    args = parser.parse_args()
    main(args)