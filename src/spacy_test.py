import spacy
from spacy import displacy

nlp = spacy.load("en_core_web_trf")
doc = nlp("I sent the book of John to the man from England")

# for token in doc:
#     print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_,
#             token.shape_, token.is_alpha, token.is_stop)
# displacy.serve(doc, style="dep", auto_select_port=True)
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
        for grandchild in child.children:
            if grandchild.i < child.i:
                text = get_phrasal_children(grandchild) + ' ' + text
            else:
                text = text + ' ' + get_phrasal_children(grandchild)
    return text

for token in doc:
    print(token.text.lower())
    print(get_phrasal_children(token)) 
