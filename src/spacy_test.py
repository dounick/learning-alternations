import spacy
import re
from spacy import displacy
from spacy.tokenizer import Tokenizer
from spacy.util import compile_infix_regex

nlp = spacy.load("en_core_web_trf")

def custom_tokenizer(nlp):
    inf = list(nlp.Defaults.infixes)               # Default infixes
    inf.remove(r"(?<=[0-9])[+\-\*^](?=[0-9-])")    # Remove the generic op between numbers or between a number and a -
    inf = tuple(inf)                               # Convert inf to tuple
    infixes = inf + tuple([r"(?<=[0-9])[+*^](?=[0-9-])", r"(?<=[0-9])-(?=-)"])  # Add the removed rule after subtracting (?<=[0-9])-(?=[0-9]) pattern
    infixes = [x for x in infixes if '-|–|—|--|---|——|~' not in x] # Remove - between letters rule
    infix_re = compile_infix_regex(infixes)

    return Tokenizer(nlp.vocab, prefix_search=nlp.tokenizer.prefix_search,
                                suffix_search=nlp.tokenizer.suffix_search,
                                infix_finditer=infix_re.finditer,
                                token_match=nlp.tokenizer.token_match,
                                rules=nlp.Defaults.tokenizer_exceptions)

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

def get_phrasal_children(child):
    text = child.text.lower()
    if child.children:
        sorted_children = sorted(child.children, key=lambda x: x.i, reverse=True)
        for grandchild in sorted_children:
            if grandchild.i < child.i:
                text = get_phrasal_children(grandchild) + " " + text
            else:
                text = text + " " + get_phrasal_children(grandchild)
    return text

nlp.tokenizer = custom_tokenizer(nlp)
doc = nlp("we'll give those to the little baby .")
for token in doc:
    if token.pos_ == "VERB":
        for child in token.children:
            print(get_phrasal_children(child))
displacy.serve(doc, style="dep")

