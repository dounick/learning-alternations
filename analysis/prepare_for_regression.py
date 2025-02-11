import pandas as pd
import numpy as np

if __name__ == '__main__':

    do_scores = pd.read_csv('test_dos.csv')
    po_scores = pd.read_csv('test_pos.csv')

    all_data = pd.concat([do_scores, po_scores], ignore_index=True)
    all_data['recipient_length'] = all_data['recipient'].apply(
            lambda x: len([word for word in x.split() if any(char.isalnum() for char in word)])
        )    
    all_data['theme_length'] = all_data['theme'].apply(
            lambda x: len([word for word in x.split() if any(char.isalnum() for char in word)])
        ) 
    all_data['length_difference'] = np.log(all_data['recipient_length'] / all_data['theme_length'])
    all_data.to_csv("all_data.csv", index=False)



