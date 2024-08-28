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

    corpus_path = args.corpus_path
    dative_path = args.dative_path
    batch_size = args.batch_size

    pathlib.Path(dative_path).mkdir(parents=True, exist_ok=True)

    # spacy setup
    gpu = spacy.prefer_gpu()
    nlp = spacy.load("en_core_web_trf")

    # aochildes
    corpus = utils.read_file(corpus_path)

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

    global_idx = 0

    def get_datives(texts, batch_size, processor):
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
                                    pps.append(
                                        (
                                            global_idx,
                                            doc.text,
                                            entity.lemma_,
                                            entity.text,
                                            entity.tag_,
                                            children,
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
                                    do = True
                                    # dos.append(sentence)
                                    # print(children)
                                    # args = collect_args(children)
                                    dos.append(
                                        (
                                            global_idx,
                                            doc.text,
                                            entity.lemma_,
                                            entity.text,
                                            entity.tag_,
                                            children,
                                        )
                                    )
                                    break
            global_idx += 1

        return dos, pps

    DOS, PPS = [], []
    for batch in get_batch(corpus, batch_size=batch_size):
        dos, pps = get_datives(batch, batch_size, nlp)
        DOS.extend(dos)
        PPS.extend(pps)

    documented_do_count = 0
    documented_pp_count = 0

    DOS_full = []
    for sentence, lemma, verb, verb_pos, children in DOS:
        if lemma in dative_verbs:
            documented_do_count += 1
        args = collect_args(children)
        DOS_full.append(
            (
                sentence,
                lemma,
                verb,
                verb_pos,
                args["theme"],
                args["recipient"],
                args["theme_pos"],
                args["recipient_pos"],
            )
        )

    PPS_full = []
    for sentence, lemma, verb, verb_pos, children in PPS:
        if lemma in dative_verbs:
            documented_pp_count += 1
        args = collect_args(children, "pp")
        PPS_full.append(
            (
                sentence,
                lemma,
                verb,
                verb_pos,
                args["theme"],
                args["recipient"],
                args["theme_pos"],
                args["recipient_pos"],
            )
        )

    print(
        f"Detected DOs: {len(DOS)}\nDetected PPs: {len(PPS)}\n\nLevin DOs: {documented_do_count}\nLevin PPs: {documented_pp_count}"
    )

    with open(f"{dative_path}/double-object.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "sentence",
                "lemma",
                "verb",
                "verb_pos",
                "theme",
                "recipient",
                "theme_pos",
                "recipient_pos",
            ]
        )
        writer.writerows(DOS_full)

    with open(f"{dative_path}/prepositional.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "sentence",
                "lemma",
                "verb",
                "verb_pos",
                "theme",
                "recipient",
                "theme_pos",
                "recipient_pos",
            ]
        )
        writer.writerows(PPS_full)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus_path", type=str, help="Path to the corpus file.")
    parser.add_argument("--dative_path", type=str, help="Path to the dative output.")
    parser.add_argument("--batch_size", type=int, default=8192, help="Batch size.")
    args = parser.parse_args()
    main(args)
