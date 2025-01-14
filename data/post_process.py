import pandas as pd
import json

path = "data/datives/babylm"
do = pd.read_csv(f"{path}/double-object.csv")
po = pd.read_csv(f"{path}/prepositional.csv")
do['verb_type'] = ''
po['verb_type'] = ''

verbs = json.load(open("data/dative_verbs.json"))

# this prioritizes verbs to be 'to'-dative when it has both dative/benefactive usages
for index, row in do.iterrows():
    verb = row['verb_lemma']
    if verb in verbs['alternating']:
        do.loc[index, 'verb_type'] = 'alternating'
    if verb in verbs['do_only']:
        if do.loc[index, 'verb_type'] == '':
            do.loc[index, 'verb_type'] = 'do_only'
        else:
            do.loc[index, 'verb_type'] += '/do_only'
    if verb in verbs['benefactive_alternating']:
        if do.loc[index, 'verb_type'] == '':
            do.loc[index, 'verb_type'] = 'benefactive_alternating'
        else:
            do.loc[index, 'verb_type'] += '/benefactive_alternating'

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

do = do[do['verb_type'] != '']
po = po[po['verb_type'] != '']

for index, row in do.iterrows():
    if row['verb_lemma'] == 'give' and (row['theme'] == 'birth' or row['theme'] == 'rise' or row['theme'] == 'way'):
        do.drop(index, inplace=True)

do.to_csv(f"{path}/double-object-filtered.csv", index=False)
po.to_csv(f"{path}/prepositional-filtered.csv", index=False)


