verbs = ['took', 'told', 'gave', 'asked', 'showed', 'wrote', 'brought', 'read', 'paid', 'sent', 'offered', 'passed', 'sold', 'drove', 'hit', 'pulled', 'taught', 'carried', 'threw', 'shot', 'pushed', 'posted', 'flew', 'fed', 'promised', 'rolled', 'kicked', 'extended', 'granted', 'cited', 'slipped', 'handed', 'quoted', 'dragged', 'owed', 'traded', 'slid', 'tossed', 'assigned', 'advanced', 'guaranteed', 'tipped', 'floated', 'rented', 'pitched', 'flipped', 'shipped', 'lent', 'slammed', 'awarded', 'signaled', 'bounced', 'snuck', 'slapped', 'shoved', 'preached', 'hauled', 'conceded', 'willed', 'emailed', 'mailed', 'batted', 'phoned', 'allocated', 'flung', 'tugged', 'chucked', 'wired', 'repaid', 'flicked', 'wheeled', 'leased', 'hurled', 'smuggled', 'faxed', 'relayed', 'telephoned', 'rowed', 'heaved',' towed', 'slung', 'forwarded', 'loaned', 'peddled', 'ceded', 'ferried', 'lugged', 'allotted', 'bused', 'trucked', 'carted', 'refunded', 'punted', 'lobbed', 'bequeathed', 'radioed', 'cabled', 'shuttled', 'catapulted', 'yielded', 'said', 'provided', 'reported', 'explained', 'raised', 'returned', 'described', 'dropped', 'stated', 'mentioned', 'presented', 'revealed', 'addressed', 'sang', 'admitted', 'referred', 'announced', 'delivered', 'expressed', 'introduced', 'contributed', 'demonstrated', 'recommended', 'lifted', 'repeated', 'declared', 'proposed', 'screamed', 'displayed', 'lowered', 'shouted', 'transferred', 'yelled', 'submitted', 'communicated', 'illustrated', 'restored', 'whispered', 'snapped', 'supplied', 'distributed', 'asserted', 'donated', 'exhibited', 'adminstered', 'confessed', 'conveyed', 'credited', 'blabbed', 'transported', 'surrendered', 'dictated', 'articulated', 'broadcasted', 'groaned', 'muttered', 'roared', 'barked', 'recounted', 'whined', 'denounced', 'chanted', 'murmured', 'recited', 'moaned', 'whistled', 'dispatched', 'tweeted', 'mumbled', 'growled', 'furnished', 'hissed', 'howled', 'wailed', 'grumbled', 'shrieked', 'narrated', 'entrusted', 'grunted', 'stuttered', 'hollered', 'reimbursed', 'squeeled', 'forfeited', 'delegated', 'squeaked', 'screeched', 'snarled', 'bellowed', 'babbled', 'cackled', 'chirped', 'croaked', 'crooned', 'stammered', 'explicated', 'remitted', 'yodeled', 'warbled', 'drawled']

import numpy as np
verb_count = np.zeros(25*len(verbs)) 

import pandas as pd
dos = pd.read_csv('data/datives/babylm/double-object-filtered.csv')
pos = pd.read_csv('data/datives/babylm/prepositional-filtered.csv')
hawkins = pd.read_csv('experiments/generated_pairs_with_results.csv')

hawkins['DOsentence'] = hawkins['DOsentence'].str.replace('alloted', 'alotted')
hawkins['PDsentence'] = hawkins['PDsentence'].str.replace('alloted', 'alotted') 

for i in range(len(verbs)):
    verb = verbs[i]
    num_verbs = len(dos[dos['verb'] == verb]) + len(pos[pos['verb'] == verb]) 
    verb_count[i*25:25*i+24] = num_verbs

hawkins['verb_count'] = verb_count
hawkins.to_csv('experiments/generated_pairs_with_results.csv', index = False)




