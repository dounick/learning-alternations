import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import argparse
import spacy
from spacy.tokenizer import Tokenizer
from spacy.util import compile_infix_regex
import pandas as pd

from collections import defaultdict, Counter
def main(args):

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
        j = int(children_flatten[-1][4])
        return text, i, j

    def retrieve_const(children_phrasal, alternant):
        consts = {"theme": "", "theme_tag" : "", "theme_pos" : "", "theme_i": "", "theme_j": "", "recipient": "", "recipient_tag" : "", "recipient_pos" : "", "recipient_i" : "", "recipient_j" : "", "subject": "", "preposition" : "", "preposition_i" : "", "type" : ""}
        if alternant == "do":
            dobj_count = 0
            dative_count = 0
            for (dep, _, _, _, _, _) in children_phrasal:
                if dep == "dobj":
                    dobj_count += 1
                if dep == "dative":
                    dative_count += 1
                if dep == 'ccomp':
                    dobj_count += 1

            if dative_count > 0:
                for (dep, tag, pos, phrasal_verb_child, i, j) in children_phrasal:
                    if dep == 'prt':
                        return None
                    if dep == "dative":
                        consts["recipient"] = phrasal_verb_child
                        consts["recipient_tag"] = tag
                        consts["recipient_pos"] = pos
                        consts["recipient_i"] = i
                        consts['recipient_j'] = j
                    elif dep == "dobj" or dep == 'ccomp':
                        consts["theme"] = phrasal_verb_child
                        consts["theme_tag"] = tag
                        consts["theme_pos"] = pos
                        consts["theme_i"] = i
                        consts["theme_j"] = j
                    elif dep == "nsubj":
                        consts["subject"] = phrasal_verb_child
                consts["type"] = "dative"
                return consts
            elif dobj_count >= 2:
                for (dep, tag, pos, phrasal_verb_child, i, j) in children_phrasal:
                    if dep == 'prt':
                        return None
                    if dep == "dobj" or dep == 'ccomp':
                        if consts["recipient"] == "":
                            consts["recipient"] = phrasal_verb_child
                            consts["recipient_tag"] = tag
                            consts["recipient_pos"] = pos
                            consts["recipient_i"] = i
                            consts['recipient_j'] = j
                        elif consts["theme"] == "":
                            consts["theme"] = phrasal_verb_child
                            consts["theme_tag"] = tag
                            consts["theme_pos"] = pos
                            consts["theme_i"] = i
                            consts["theme_j"] = j
                    elif dep == "nsubj":
                        consts["subject"] = phrasal_verb_child
                consts["type"] = "dobj"
                return consts
            else:
                print(children_phrasal)
                return None
        elif alternant == "pp":
            for (dep, tag, pos, phrasal_verb_child, i, j) in children_phrasal:
                if dep == 'prt':
                    return None
                if dep == "dobj":
                    consts["theme"] = phrasal_verb_child
                    consts["theme_tag"] = tag
                    consts["theme_pos"] = pos
                    consts["theme_i"] = i
                    consts['theme_j'] = j   
                elif dep == "nsubj":
                    consts["subject"] = phrasal_verb_child
                elif (dep == "prep" or dep == "dative") and phrasal_verb_child.split()[0] in ["to", "for"] and consts["preposition"] == "":
                    if consts["type"] == "":
                        consts["type"] = dep
                    consts["preposition"] = phrasal_verb_child.split()[0]
                    consts["preposition_i"] = i
                    consts["recipient"] = " ".join(phrasal_verb_child.split()[1:])
                    consts["recipient_tag"] = tag
                    consts["recipient_pos"] = pos
                    consts["recipient_i"] = i+1
            return consts
        print("Error: No construction found")
        return None

    def sanity_check(construction, consts):
        if consts is None:
            return False
        #for-preps need to be adjacent to recipient:
        if construction == "pp":
            if consts["preposition"] == "for" and consts["theme_j"] != consts["preposition_i"] - 1:
                return False
        if construction == 'do':
            if consts['theme_i'] != consts['recipient_j'] + 1:
                return False
        return True

    def get_datives_phrasal(texts, processor):
        detected_datives = pd.DataFrame(columns = ['index', 'sentence'])
        for index, row in enumerate(texts):
            sentence = row['sentence']
            doc = processor(sentence)
            is_ditransitive = False
            is_dative = False
            for entity in doc:
                if entity.pos_ == "VERB":
                    all_children = get_children_flatten(entity, 0, dep=True)
                    children = []
                    for child in all_children:
                        if not(child[4] < entity.i and child[2] != 'nsubj'):
                            children.append(child)
                    if len(children) > 0:
                        tokens, dep, pos_string, depth, index = list(zip(*children))
                        # additional boolean in case of a sentence containing multiple datives
                        is_pp = False
                        if "for" in tokens or "to" in tokens:
                            # Possibly to/for-PP
                            dep_depth = [
                                f"{d}_{str(depth[i])}" for i, d in enumerate(dep)
                            ]
                            tok_dep = [
                                f"{tokens[i]}_{dep[i]}" for i in range(len(tokens))
                            ]
                            if (
                                "dobj_0" in dep_depth
                                and "prep_0" in dep_depth
                                and "pobj_1" in dep_depth
                            ):
                                if "for_prep" in tok_dep:
                                    children_phrasal = []
                                    for verb_child in entity.children:
                                        phrasal_verb_child, child_i, child_j = get_phrasal_children(verb_child)
                                        dep_child, tag_child, pos_child = verb_child.dep_, verb_child.tag_, verb_child.pos_
                                        children_phrasal.append((dep_child, tag_child, pos_child, phrasal_verb_child, child_i, child_j))
                                    consts = retrieve_const(children_phrasal, "pp")
                                    if sanity_check('pp', consts):
                                        is_ditransitive = True
                                        is_dative = True
                                        break
                            elif (
                                "dobj_0" in dep_depth
                                and "prep_1" in dep_depth
                                and "pobj_2" in dep_depth
                            ) or (
                            "dobj_0" in dep_depth
                                and "dative_1" in dep_depth
                                and "pobj_2" in dep_depth 
                            ): 
                                if "for_prep" in tok_dep or "to_prep" in tok_dep or "for_dative" in tok_dep or "to_dative" in tok_dep:
                                    children_phrasal = []
                                    for verb_child in entity.children:
                                        phrasal_verb_child, child_i, child_j = get_phrasal_children(verb_child)
                                        dep_child, tag_child, pos_child = verb_child.dep_, verb_child.tag_, verb_child.pos_
                                        children_phrasal.append((dep_child, tag_child, pos_child, phrasal_verb_child, child_i, child_j))
                                    consts = retrieve_const(children_phrasal, "pp")
                                    if '-' in consts['theme'] and sanity_check('pp', consts): 
                                        is_ditransitive = True
                                        is_dative = True
                                        break 
                        if(not is_pp):
                            # Possibly DO
                            dep_depth = [
                                f"{d}_{str(depth[i])}" for i, d in enumerate(dep)
                            ]
                            tokens_dep = [
                                f"{tokens[i]}_{dep[i]}" for i in range(len(tokens))
                            ]
                            if (
                                "dobj_0" in dep_depth and "ccomp_0" in dep_depth
                            ):
                                children_phrasal = []
                                for verb_child in entity.children:
                                    phrasal_verb_child, child_i, child_j = get_phrasal_children(verb_child)
                                    dep_child, tag_child, pos_child = verb_child.dep_, verb_child.tag_, verb_child.pos_
                                    children_phrasal.append((dep_child, tag_child, pos_child, phrasal_verb_child, child_i, child_j))
                                consts = retrieve_const(children_phrasal, "do")
                                if consts is None:
                                    continue
                                if sanity_check('do', consts):
                                    is_ditransitive = True
                                    is_dative = True
                                    break
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
                write to nondatives and write to nonditrans
            elif not is_dative:
                write to nonditrans
        return detected_datives, datives

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--i", type=str, required=True, 
                       help="Input file index")
    args = parser.parse_args()
    main(args)

nondatives = pd.read_csv('src/spotcheck/nondatives.csv')
detected_datives, datives = get_datives_phrasal(nondatives, nlp)
detected_datives.to_csv('src/spotcheck/detected_datives.csv', index=False)
print(datives)