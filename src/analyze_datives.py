import pandas as pd

path = 'data/datives/babylm'

# Load datasets
dos = pd.read_csv(f'{path}/double-object.csv')
pps = pd.read_csv(f'{path}/prepositional.csv')
pps = pps[pps['preposition'].notna()]

# Filter the pps DataFrame for rows where the verb is "get"
get_pps = pps[pps['verb_lemma'] == 'get']

# Sort the filtered DataFrame by the preposition, prioritizing 'for' first then 'to'
get_pps_sorted = get_pps.sort_values(by='preposition', key=lambda x: x.map({'for': 0, 'to': 1}))

# Save the sorted DataFrame to a CSV file
get_pps_sorted.to_csv(f'{path}/get_pps_sorted.csv', index=False)

# nondatives = pd.read_csv(f'{path}/non-datives.csv')

# # Count the number of instances
# num_dos = len(dos)
# num_pps = len(pps)
# num_nondatives = len(nondatives)
# total = num_dos + num_pps + num_nondatives

# # Calculate proportions
# dos_proportion = (num_dos / total) * 100
# pps_proportion = (num_pps / total) * 100
# nondatives_proportion = (num_nondatives / total) * 100

# print(f'Proportion of double-object sentences: {dos_proportion:.4f}%')
# print(f'Proportion of prepositional sentences: {pps_proportion:.4f}%')
# print(f'Proportion of non-dative sentences: {nondatives_proportion:.4f}%')

# # Count verb frequencies
# dos_verb_counts = dos['verb_lemma'].value_counts()
# pps_verb_counts = pps['verb_lemma'].value_counts()

# # Filter verbs with frequency >= 10
# dos_filtered = dos_verb_counts[dos_verb_counts >= 10]
# pps_filtered = pps_verb_counts[pps_verb_counts >= 10]

# # Generate sets of verbs
# dos_verbs = set(dos_filtered.index)
# pps_verbs = set(pps_filtered.index)

# # Find unique and common verbs
# only_dos_verbs = dos_verbs - pps_verbs
# only_pps_verbs = pps_verbs - dos_verbs
# both_verbs = dos_verbs & pps_verbs

# # Create a DataFrame to store the verbs and their categories
# verbs_summary = pd.DataFrame(columns=['verb', 'do_only', 'pp_only', 'both', 'do_frequency', 'pp_frequency', 'for_frequency', 'to_frequency'])

# # Populate the DataFrame
# for verb in dos_verbs.union(pps_verbs):
#     do_only = int(verb in only_dos_verbs)
#     pp_only = int(verb in only_pps_verbs)
#     both = int(verb in both_verbs)
    
#     do_frequency = dos_filtered.get(verb, 0)
#     pp_frequency = pps_filtered.get(verb, 0)
    
#     if pp_only or both:
#         for_frequency = pps[(pps['verb_lemma'] == verb) & (pps['preposition'] == 'for')].shape[0]
#         to_frequency = pps[(pps['verb_lemma'] == verb) & (pps['preposition'] == 'to')].shape[0]
#     else:
#         for_frequency = 0
#         to_frequency = 0
    
#     verbs_summary = pd.concat([verbs_summary, pd.DataFrame([{
#         'verb': verb,
#         'do_only': do_only,
#         'pp_only': pp_only,
#         'both': both,
#         'do_frequency': do_frequency,
#         'pp_frequency': pp_frequency,
#         'for_frequency': for_frequency,
#         'to_frequency': to_frequency
#     }])], ignore_index=True)

# # Sort the DataFrame by the sum of do_frequency and pp_frequency
# verbs_summary['total_frequency'] = verbs_summary['do_frequency'] + verbs_summary['pp_frequency']
# verbs_summary = verbs_summary.sort_values(by='total_frequency', ascending=False).drop(columns=['total_frequency'])

# # Save the DataFrame to a CSV file
# verbs_summary.to_csv(f'{path}/verbs_summary.csv', index=False)

# # Sample 2000 lines from the non-datives file
# nondatives_sample = nondatives.sample(n=2000, random_state=42)

# # Save the sampled lines to a new CSV file
# nondatives_sample.to_csv(f'{path}/nondatives_sample.csv', index=False)