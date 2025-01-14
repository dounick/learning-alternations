import pandas as pd
import json

po = pd.read_csv('data/datives/babylm/prepositional-filtered.csv')
do = pd.read_csv('data/datives/babylm/double-object-filtered.csv')

verbs = json.load(open("data/dative_verbs.json"))

for index, row in do.iterrows():
    verb = row['verb_lemma']
    if verb in verbs['alternating']:
        do.loc[index, 'verb_type'] = 'alternating'
    elif verb in verbs['do_only']:
        do.loc[index, 'verb_type'] = 'do_only'
    elif verb in verbs['benefactive_alternating']:
        do.loc[index, 'verb_type'] = 'benefactive_alternating'
    else:
        do.loc[index, 'verb_type'] = 'drop'

for index, row in po.iterrows():
    verb = row['verb_lemma']
    if verb in verbs['alternating'] and row['preposition'] == 'to':
        po.loc[index, 'verb_type'] = 'alternating'
    elif verb in verbs['po_only'] and row['preposition'] == 'to':
        po.loc[index, 'verb_type'] = 'po_only'
    elif verb in verbs['benefactive_alternating'] and row['preposition'] == 'for':
        po.loc[index, 'verb_type'] = 'benefactive_alternating'
    elif verb in verbs['benefactive_po_only'] and row['preposition'] == 'for':
        po.loc[index, 'verb_type'] = 'benefactive_po_only'
    else:
        po.loc[index, 'verb_type'] = 'drop'

do = do[do['verb_type'] != 'drop']
po = po[po['verb_type'] != 'drop']