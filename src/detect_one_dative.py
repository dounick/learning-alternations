"""Code to detect dative constructions in a corpus. Originally written by Kanishka Misra and Najoung Kim."""

import argparse
import config
import csv
import pathlib
import re
import spacy
import utils

from collections import defaultdict, Counter
from minicons.utils import get_batch
from tqdm import tqdm


def main(args):

    sentence = args.sentence
    sentence = [sentence]


    # spacy setup
    # gpu = spacy.prefer_gpu()
    # print(gpu)
    nlp = spacy.load("en_core_web_trf")

    # aochildes
    

    dative_verbs = sorted(
        list(
            set(config.alternating_verbs + config.do_only_verbs + config.pp_only_verbs)
        )
    )
    
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

    def collect_args(children_obj, hyp="do"):
        args = {"theme": "", "recipient": "", "theme_pos": "", "recipient_pos": ""}
        hyp_args = []
        if hyp == "do":
            for child in children_obj:
                if child[1] == "dobj" or child[1] == "dative":
                    hyp_args.append((child[0], child[-1], child[2]))

            # sort by index
            hyp_args = sorted(hyp_args, key=lambda x: x[1])
            args["recipient"] = hyp_args[0][0]
            args["theme"] = hyp_args[1][0]
            args["recipient_pos"] = hyp_args[0][-1]
            args["theme_pos"] = hyp_args[1][-1]

        elif hyp == "pp":
            for child in children_obj:
                if child[1] == "pobj" or child[1] == "dobj":
                    hyp_args.append((child[0], child[-1], child[2]))

            # sort by index
            hyp_args = sorted(hyp_args, key=lambda x: x[1])
            args["recipient"] = hyp_args[1][0]
            args["theme"] = hyp_args[0][0]
            args["recipient_pos"] = hyp_args[1][-1]
            args["theme_pos"] = hyp_args[0][-1]

        return args
    
    def get_phrasal_children(child):
        text = child.text.lower()
        if child.children:
            for grandchild in child.children:
                if grandchild.i < child.i:
                    text = get_phrasal_children(grandchild) + ' ' + text
                else:
                    text = text + ' ' + get_phrasal_children(grandchild)
        return text
    
    def get_datives_phrasal(texts, batch_size, processor, global_idx=0):
        dos, pps = [], []
        for doc in tqdm(processor.pipe(texts, disable=["ner"], batch_size=batch_size)):
            do = False
            pp = False
            for entity in doc:
                if entity.pos_ == "VERB":
                    children = get_children_flatten(entity, 0, dep=True)
                    if len(children) > 0:
                        tokens, dep, pos_string, depth, index = list(zip(*children))
                        if "to" in tokens:
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
                            ) or (
                                "dobj_0" in dep_depth
                                and "prep_0" in dep_depth
                                and "pobj_1" in dep_depth
                            ):
                                if "to_dative" in tok_dep or "to_prep" in tok_dep:
                                    # also append global sentence id
                                    children_phrasal = []
                                    for verb_child in entity.children:
                                        i, phrasal_verb_child = verb_child.i, get_phrasal_children(verb_child)
                                        children_phrasal.append((i, phrasal_verb_child))
                                    
                                    children_phrasal = sorted([child[1] for child in children_phrasal], key=lambda x: x[0])
                                    pps.append(
                                        (
                                            global_idx,
                                            doc.text,
                                            entity.lemma_,
                                            entity.text,
                                            entity.tag_,
                                            children_phrasal,
                                        )
                                    )
                                    break
                        else:
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
                                if (
                                    "for_dative" not in tokens_dep
                                    and "for_dobj" not in tokens_dep
                                ):
                                    children_phrasal = []
                                    for verb_child in entity.children:
                                        i, phrasal_verb_child = verb_child.i, get_phrasal_children(verb_child)
                                        children_phrasal.append((i, phrasal_verb_child))
                                    
                                    children_phrasal = sorted([child[1] for child in children_phrasal], key=lambda x: x[0])
                                    dos.append(
                                        (
                                            global_idx,
                                            doc.text,
                                            entity.lemma_,
                                            entity.text,
                                            entity.tag_,
                                            children_phrasal
                                        )
                                    )
                                    break
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

    DOS_full = []
    for idx, sentence, lemma, verb, verb_pos, children_phrasal in DOS:
        if lemma in dative_verbs:
            documented_do_count += 1
        DOS_full.append(
            (
                idx,
                sentence,
                lemma,
                verb,
                verb_pos,
                children_phrasal[0],
                children_phrasal[1],
                children_phrasal[2]
            )
        )

    PPS_full = []
    for idx, sentence, lemma, verb, verb_pos, children_phrasal in PPS:
        if lemma in dative_verbs:
            documented_pp_count += 1
        PPS_full.append(
            (
                idx,
                sentence,
                lemma,
                verb,
                verb_pos,
                children_phrasal[0],
                children_phrasal[1],
                children_phrasal[2]
            )
        )

    print(
        f"Detected DOs: {len(DOS)}\nDetected PPs: {len(PPS)}\n\nLevin DOs: {documented_do_count}\nLevin PPs: {documented_pp_count}"
    )
    print("Double Object Constructions:")
    for row in DOS_full:
        print(row)

    print("Prepositional Constructions:")
    for row in PPS_full:
        print(row)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sentence", type=str, help="Path to the corpus file.")
    args = parser.parse_args()
    main(args)
