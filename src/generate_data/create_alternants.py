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
from transformers import AutoTokenizer, AutoModelForCausalLM
from minicons import scorer

os.environ["CUDA_VISIBLE_DEVICES"]="3"

tokenizer = AutoTokenizer.from_pretrained("src/training/tokenizer")
model = AutoModelForCausalLM.from_pretrained("qing-yao/babylm-baseline")
lm = scorer.IncrementalLMScorer(model = model, device = 'cuda', tokenizer=tokenizer)

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
nlp = spacy.load("en_core_web_trf")
nlp.tokenizer = custom_tokenizer(nlp)

default_dos = pd.read_csv("data/datives/babylm/double-object.csv")
default_pos = pd.read_csv("data/datives/babylm/prepositional.csv")

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

def get_phrasal_children(child):
    children_flatten = sorted(get_children_flatten(child, dep=True, include_self=True), key=lambda x: x[4])
    text = " ".join([x[0] for x in children_flatten])
    beginning_i = children_flatten[0][4]
    return text, beginning_i

def retrieve_const(children_phrasal, alternant):
    consts = {"theme": "", "theme_tag" : "", "theme_pos" : "", "theme_position": "", "recipient": "", "recipient_tag" : "", "recipient_pos" : "", "recipient_position" : "", "subject": "", "preposition" : ""} 
    if alternant == "do":
        dobj_count = 0
        dative_count = 0
        for (dep, _, _, _, _) in children_phrasal:
            if dep == "dobj":
                dobj_count += 1
            if dep == "dative":
                dative_count += 1

        if dative_count > 0:
            for (dep, tag, pos, phrasal_verb_child, i) in children_phrasal:
                if dep == "dative":
                    consts["recipient"] = phrasal_verb_child
                    consts["recipient_tag"] = tag
                    consts["recipient_pos"] = pos
                    consts["recipient_position"] = i
                elif dep == "dobj":
                    consts["theme"] = phrasal_verb_child
                    consts["theme_tag"] = tag
                    consts["theme_pos"] = pos
                    consts["theme_position"] = i
                elif dep == "nsubj":
                    consts["subject"] = phrasal_verb_child
            return consts
        elif dobj_count >= 2:
            for (dep, tag, pos, phrasal_verb_child, i) in children_phrasal:
                if dep == "dobj":
                    if consts["recipient"] == "":
                        consts["recipient"] = phrasal_verb_child
                        consts["recipient_tag"] = tag
                        consts["recipient_pos"] = pos
                        consts["recipient_position"] = i
                    elif consts["theme"] == "":
                        consts["theme"] = phrasal_verb_child
                        consts["theme_tag"] = tag
                        consts["theme_pos"] = pos
                        consts["theme_position"] = i
                elif dep == "nsubj":
                    consts["subject"] = phrasal_verb_child
            return consts
        else:
            print(children_phrasal)
            return None
    elif alternant == "pp":
        for (dep, tag, pos, phrasal_verb_child,i) in children_phrasal:
            if dep == "dobj":
                consts["theme"] = phrasal_verb_child
                consts["theme_tag"] = tag
                consts["theme_pos"] = pos
                consts["theme_position"] = i
            elif dep == "nsubj":
                consts["subject"] = phrasal_verb_child
            elif (dep == "prep" or dep == "dative") and phrasal_verb_child.split()[0] in ["to", "for"] and consts["preposition"] == "":
                consts["preposition"] = phrasal_verb_child.split()[0]
                consts["recipient"] = " ".join(phrasal_verb_child.split()[1:])
                consts["recipient_tag"] = tag
                consts["recipient_pos"] = pos
                consts["recipient_position"] = i
        return consts
    print("Error: No construction found")
    return None

def create_alternant_from_pos(sentence, verb_lemma, recipient, theme, preposition):
    doc = nlp(sentence)
    recipient_position = -1
    theme_position = -1
    for entity in doc:
        if entity.pos_ == "VERB" and entity.lemma_ == verb_lemma:
            all_children = get_children_flatten(entity, 0, dep=True)
            children = []
            for child in all_children:
                if not(child[4] < entity.i and child[2] != 'nsubj'):
                    children.append(child)
            if len(children) > 0:
                tokens, dep, pos_string, depth, index = list(zip(*children))
            else: 
                continue
            dep_depth = [
                f"{d}_{str(depth[i])}" for i, d in enumerate(dep)
            ]
            if "to" in tokens or "for" in tokens:
                # Possibly to-PP
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
                ) or (
                    "dobj_0" in dep_depth
                    and "prep_0" in dep_depth
                    and "pobj_1" in dep_depth
                ):
                    if ("to_dative" in tok_dep or "to_prep" in tok_dep) or ("for_dative" in tok_dep):
                        children_phrasal = []
                        for verb_child in entity.children:
                            phrasal_verb_child, phrase_i = get_phrasal_children(verb_child)
                            dep_child, tag_child, pos_child = verb_child.dep_, verb_child.tag_, verb_child.pos_
                            children_phrasal.append((dep_child, tag_child, pos_child, phrasal_verb_child, phrase_i))
                        consts = retrieve_const(children_phrasal, "pp")
                    if consts["recipient"] == recipient and consts["theme"] == theme and consts["preposition"] == preposition:
                        recipient_position = consts["recipient_position"]
                        theme_position = consts["theme_position"]
                        break
    initial = ""
    theme_to_rec = ""
    after_recipient = ""
    recipient_length = nlp(recipient)[-1].i
    for entity in doc:
        if entity.i < theme_position:
            initial += entity.text_with_ws
        elif entity.i >= theme_position and entity.i < recipient_position - 1:
            theme_to_rec += entity.text_with_ws
        elif entity.i > recipient_position + recipient_length:
            after_recipient += entity.text_with_ws
    alternant = initial + recipient + theme_to_rec + after_recipient
    return alternant

def create_alternant_from_dos(sentence, verb_lemma, recipient, theme):
    doc = nlp(sentence)
    recipient_position = -1
    theme_position = -1
    for entity in doc:
        if entity.pos_ == "VERB" and entity.lemma_ == verb_lemma:
            all_children = get_children_flatten(entity, 0, dep=True)
            children = []
            for child in all_children:
                if not(child[4] < entity.i and child[2] != 'nsubj'):
                    children.append(child)
            if len(children) > 0:
                tokens, dep, pos_string, depth, index = list(zip(*children))
            else: 
                continue
            dep_depth = [
                f"{d}_{str(depth[i])}" for i, d in enumerate(dep)
            ]
            if (
                "dobj_0" in dep_depth and "dative_0" in dep_depth
            ) or Counter(dep_depth)["dobj_0"] >= 2:
                children_phrasal = []
                for verb_child in entity.children:
                    phrasal_verb_child, phrase_i = get_phrasal_children(verb_child)
                    dep_child, tag_child, pos_child = verb_child.dep_, verb_child.tag_, verb_child.pos_
                    children_phrasal.append((dep_child, tag_child, pos_child, phrasal_verb_child, phrase_i))
                consts = retrieve_const(children_phrasal, "do")
                if consts["recipient"] == recipient and consts["theme"] == theme:
                    recipient_position = consts["recipient_position"]
                    theme_position = consts["theme_position"]
                    break
    initial = ""
    rec_to_theme = ""
    after_theme = ""
    theme_length = nlp(theme)[-1].i
    for entity in doc:
        if entity.i < recipient_position:
            initial += entity.text_with_ws
        elif entity.i >= recipient_position and entity.i < theme_position:
            rec_to_theme += entity.text_with_ws
        elif entity.i > theme_position + theme_length:
            after_theme += entity.text_with_ws
    to_alternant = initial + theme + ' to ' + rec_to_theme + after_theme
    for_alternant = initial + theme + ' for ' + rec_to_theme + after_theme
    return to_alternant, for_alternant

def rating_alternants(to_alternant, for_alternant, lm):
    tokenized_output = tokenizer(to_alternant, return_tensors='pt', truncation=False)
    num_tokens = tokenized_output['input_ids'].shape[1]
    if(num_tokens > 255):
        return 0, 0
    to_alternant_score = lm.score(to_alternant)[0]
    for_alternant_score = lm.score(for_alternant)[0]
    return to_alternant_score, for_alternant_score

# for index, row in default_dos.iterrows():
#     to_alternant, for_alternant = create_alternant_from_dos(row["sentence"], row["verb_lemma"], row["recipient"], row["theme"])
#     to_alternant_score, for_alternant_score = rating_alternants(to_alternant, for_alternant, lm)
#     new_row = pd.DataFrame([{"global_idx" : row["global_idx"], "sentence": row["sentence"], "verb_lemma": row["verb_lemma"], "recipient": row["recipient"], "theme": row["theme"], "to_alternant": to_alternant, "for_alternant": for_alternant, "to_alternant_score": to_alternant_score, "for_alternant_score": for_alternant_score}])
#     new_row.to_csv("data/datives/babylm/alternant_of_dos.csv", index=False, mode = 'a', header=not os.path.exists("data/datives/babylm/alternant_of_dos.csv"))

for index, row in default_pos.iterrows():
    alternant = create_alternant_from_pos(row["sentence"], row["verb_lemma"], row["recipient"], row["theme"], row["preposition"])
    new_row = pd.DataFrame([{"global_idx" : row["global_idx"], "sentence": row["sentence"], "verb_lemma": row["verb_lemma"], "recipient": row["recipient"], "theme": row["theme"], "preposition": row["preposition"], "alternant": alternant}])
    new_row.to_csv("data/datives/babylm/alternant_of_pos.csv", index=False, mode = 'a', header=not os.path.exists("data/datives/babylm/alternant_of_pos.csv"))