import pandas as pd
import csv

path = "data/datives/babylm2"
dos = pd.read_csv(f"{path}/double-object.csv")
pos = pd.read_csv(f"{path}/prepositional.csv")

# Process invalid rows for double-object sentences
# Process invalid rows for double-object sentences
invalid_rows_dos = dos[dos['recipient'].isna() | dos['theme'].isna()]
invalid_sentences_dos = invalid_rows_dos[['sentence']]
dos = dos.dropna(subset=['recipient', 'theme'])
dos.to_csv(f"{path}/double-object.csv", index=False)

# Process invalid rows for prepositional sentences
invalid_rows_pos = pos[pos['recipient'].isna() | pos['theme'].isna() | pos['preposition'].isna()]
invalid_sentences_pos = invalid_rows_pos[['sentence']]
pos = pos.dropna(subset=['recipient', 'theme', 'preposition'])
pos.to_csv(f"{path}/prepositional.csv", index=False)

# Combine invalid sentences from both datasets
invalid_df = pd.concat([invalid_sentences_dos, invalid_sentences_pos], ignore_index=True)
invalid_df.to_csv(f"{path}/non-datives.csv", mode='a', header=False, index=False)