import argparse
import csv
import pathlib
import re
import spacy
from spacy.tokenizer import Tokenizer
from spacy.util import compile_infix_regex
import pandas as pd

from collections import defaultdict, Counter
from minicons.utils import get_batch
from tqdm import tqdm


def main(args):

    sentence = args.sentence
    sentence = [sentence]
   
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

    # spacy setup (from my experience gpu is slower)
    # gpu = spacy.prefer_gpu()
    # print(gpu)
    nlp = spacy.load("en_core_web_trf")
    nlp.tokenizer = custom_tokenizer(nlp)

    def get_children_flatten(token, depth=0, dep=False, return_tokens=False):
        """recursively get children of a given token using spacy."""
        children = []
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
        text = child.text.lower()
        if child.children:
            for grandchild in child.children:
                if grandchild.i < child.i:
                    text = get_phrasal_children(grandchild) + " " + text
                else:
                    text = text + " " + get_phrasal_children(grandchild)
        return text

    def retrieve_const(children_phrasal, alternant):
        consts = {"theme": "", "theme_tag" : "", "theme_pos" : "", "recipient": "", "recipient_tag" : "", "recipient_pos" : "", "subject": "", "preposition" : ""}
        if alternant == "do":
            dobj_count = 0
            dative_count = 0
            for (dep, _, _) in children_phrasal:
                if dep == "dobj":
                    dobj_count += 1
                if dep == "dative":
                    dative_count += 1

            if dative_count > 0:
                for (dep, tag, pos, phrasal_verb_child) in children_phrasal:
                    if dep == "dative":
                        consts["recipient"] = phrasal_verb_child
                        consts["recipient_tag"] = tag
                        consts["recipient_pos"] = pos
                    elif dep == "dobj":
                        consts["theme"] = phrasal_verb_child
                        consts["theme_tag"] = tag
                        consts["theme_pos"] = pos
                    elif dep == "nsubj":
                        consts["subject"] = phrasal_verb_child
                return consts
            elif dobj_count == 2:
                for (dep, tag, phrasal_verb_child) in children_phrasal:
                    if dep == "dobj":
                        if consts["recipient"] == "":
                            consts["recipient"] = phrasal_verb_child
                            consts["recipient_tag"] = tag
                            consts["recipient_pos"] = pos
                        else:
                            consts["theme"] = phrasal_verb_child
                            consts["theme_tag"] = tag
                            consts["theme_pos"] = pos
                    elif dep == "nsubj":
                        consts["subject"] = phrasal_verb_child
                return consts
            else:
                print("Error: DO construction not found")
                return None
        elif alternant == "pp":
            for (dep, tag, phrasal_verb_child) in children_phrasal:
                if dep == "dobj":
                    consts["theme"] = phrasal_verb_child
                    consts["theme_tag"] = tag
                    consts["theme_pos"] = pos
                elif dep == "nsubj":
                    consts["subject"] = phrasal_verb_child
                elif (dep == "prep" or dep == "dative") and phrasal_verb_child.split()[0] in ["to", "for"] and consts["preposition"] == "":
                    consts["preposition"] = phrasal_verb_child.split()[0]
                    consts["recipient"] = " ".join(phrasal_verb_child.split()[1:])
                    consts["recipient_tag"] = tag
                    consts["recipient_pos"] = pos
            return consts
        print("Error: No construction found")
        return None
    
    def get_datives_phrasal(texts, batch_size, processor, global_idx=0):
        dos = pd.DataFrame(columns=["global_idx", "sentence", "verb_lemma", "verb", "verb_tag", "subject", "recipient", "recipient_tag", "recipient_pos", "theme", "theme_tag", "theme_pos", "preposition"])
        pps = dos
        for doc in tqdm(processor.pipe(texts, disable=["ner"], batch_size=batch_size)):
            for entity in doc:
                if entity.pos_ == "VERB":
                    children = get_children_flatten(entity, 0, dep=True)
                    if len(children) > 0:
                        tokens, dep, pos_string, depth, index = list(zip(*children))

                        # additional boolean in case of a sentence containing multiple datives
                        is_pp = False
                        if "to" in tokens or "for" in tokens:
                            # Possibly PP
                            dep_depth = [
                                f"{d}_{str(depth[i])}" for i, d in enumerate(dep)
                            ]
                            tok_dep = [
                                f"{tokens[i]}_{dep[i]}" for i in range(len(tokens))
                            ]
                            if (
                                "dobj_0" in dep_depth
                                and "dative_0" in dep_depth
                                and "pobj_1" in dep_depth

                                # maybe check what can point to depth 1;
                            ) or (
                                "dobj_0" in dep_depth
                                and "prep_0" in dep_depth
                                and "pobj_1" in dep_depth
                            ):
                                if ("to_dative" in tok_dep or "to_prep" in tok_dep) or ("for_dative" in tok_dep or "for_prep" in tok_dep):
                                    children_phrasal = []
                                    for verb_child in entity.children:
                                        dep_child, tag_child, pos_child, phrasal_verb_child = verb_child.dep_, verb_child.tag_, verb_child.pos_, get_phrasal_children(verb_child)
                                        children_phrasal.append((dep_child, tag_child, pos_child, phrasal_verb_child))
                                    consts = retrieve_const(children_phrasal, "pp")
                                    new_row = {
                                        "global_idx": global_idx,
                                        "sentence": doc.text,
                                        "verb_lemma": entity.lemma_,
                                        "verb": entity.text,
                                        "verb_tag": entity.tag_,
                                        "subject": consts["subject"],
                                        "recipient": consts["recipient"],
                                        "recipient_tag": consts["recipient_tag"],
                                        "recipient_pos" : consts["recipient_pos"],
                                        "theme": consts["theme"],
                                        "theme_tag": consts["theme_tag"],
                                        "theme_pos": consts["theme_pos"],
                                        "preposition": consts["preposition"]
                                    }
                                    pps = pps.append(new_row, ignore_index=True)
                                    global_idx += 1
                                    is_pp = True
                                    
                        if(not is_pp):
                            # Possibly DO
                            dep_depth = [
                                f"{d}_{str(depth[i])}" for i, d in enumerate(dep)
                            ]
                            tokens_dep = [
                                f"{tokens[i]}_{dep[i]}" for i in range(len(tokens))
                            ]
                            if (
                                "dobj_0" in dep_depth and "dative_0" in dep_depth
                            ) or Counter(dep_depth)["dobj_0"] >= 2:
                                children_phrasal = []
                                for verb_child in entity.children:
                                    dep_child, tag_child, pos_child, phrasal_verb_child = verb_child.dep_, verb_child.tag_, verb_child.pos_, get_phrasal_children(verb_child)
                                    children_phrasal.append((dep_child, tag_child, pos_child, phrasal_verb_child))
                                consts = retrieve_const(children_phrasal, "do")
                                new_row = {
                                    "global_idx": global_idx,
                                    "sentence": doc.text,
                                    "verb_lemma": entity.lemma_,
                                    "verb": entity.text,
                                    "verb_tag": entity.tag_,
                                    "subject": consts["subject"],
                                    "recipient": consts["recipient"],
                                    "recipient_tag": consts["recipient_tag"],
                                    "theme": consts["theme"],
                                    "theme_tag": consts["theme_tag"],
                                    "preposition": consts["preposition"]
                                }
                                dos = dos.append(new_row, ignore_index=True)
                                global_idx += 1
        return dos, pps, global_idx

    DOS, PPS = [], []
    global_idx = 0
    for batch in get_batch(sentence, batch_size=1):
        dos, pps, global_idx = get_datives_phrasal(batch, 1, nlp, global_idx)
        DOS.extend(dos)
        PPS.extend(pps)

    documented_do_count = 0
    documented_pp_count = 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sentence", type=str, help="Path to the corpus file.")
    args = parser.parse_args()
    main(args)
