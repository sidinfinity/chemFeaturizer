import pandas as pd
import numpy as np
import os

def load_HIV():
    csv_path = os.path.join(os.getcwd(), "data", "HIV.csv")
    df = pd.read_csv(csv_path)
    del df['activity']
    df.replace({'true': 1, 'false': 0}, inplace = True)

    labels = list(df.columns)
    labels.remove('smiles')

    new_df = pd.DataFrame()
    new_df['smiles'] = df['smiles']

    for label in labels:
        new_df[label + "_false"] = df[label] == 0
        new_df[label + "_true"] = df[label] == 1


    new_df.replace({'true': 1, 'false': 0}, inplace = True)

    print(new_df.head())
    return labels, new_df