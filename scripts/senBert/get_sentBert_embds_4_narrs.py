"""
This script takes narrative/sentences as input and compute the similairty between pairs. 
   Input:
       a csv file where narrative in column "narrative"
   Output:
       a csv file keeps embeddings of narratives.
    
"""

from sentence_transformers import SentenceTransformer, util
import torch
import sys
import pandas as pd
import random

cuda = random.randint(0, 3)
model = SentenceTransformer('all-mpnet-base-v2')
device =  "cuda:3" #f"cuda:{cuda}" if torch.cuda.is_available() else "cpu"
model = model.to(device)

INPUT = sys.argv[1]
OUTPUT = sys.argv[2]


def format_narrative(df):
    """
    this fucs return narrative as a short sentence.
    e.g. "ARG0 frame_key ARG1"
    """
    cols = ["ARG0", "va_key", "ARG1"]
    df["narrative"] = df[cols].apply(lambda row: " ".join(row.values.astype(str)), axis=1)
    return df


if __name__ == "__main__":

    df = pd.read_csv(INPUT,  usecols = ["ARG0", "ARG1", "va_key"], sep = '\n')
    df = format_narrative(df)
    
    print("start getting sentBERT embeddings")
    ebds = model.encode(df["narrative"].tolist())
    print(len(df["narrative"]), len(ebds))
    
    out_df = pd.DataFrame(ebds)
    out_df["narrative"] = df["narrative"]
    out_df.to_csv(OUTPUT, sep = '\t')
    





    
        

