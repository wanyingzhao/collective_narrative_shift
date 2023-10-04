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
device = f"cuda:{cuda}" if torch.cuda.is_available() else "cpu"
model = model.to(device)

INPUT = sys.argv[1]
OUTPUT = sys.argv[2]


if __name__ == "__main__":
    df = pd.read_csv(INPUT,  sep = '\t')
    args = list(df["ARG0"]) + list(df["ARG1"])
    print("start getting sentBERT embeddings")
    ebds = model.encode(args)

    out_df = pd.DataFrame(ebds)
    out_df["args"] = args
    out_df.to_csv(OUTPUT, sep = '\t')


    
        

