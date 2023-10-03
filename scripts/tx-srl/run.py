import argparse
import json
import pickle
import pandas as pd
from contextlib import ExitStack
import re
import spacy
import torch
import sys
import csv
from model import predict_sentences
spacy.load("en_core_web_sm")

INPUT_FILE = sys.argv[1]
OUTPUT_FILE = sys.argv[2]


def main():
    df = pd.read_csv(INPUT_FILE, sep='\t',quoting=csv.QUOTE_NONE )
    print(df.head())
    df = df[['sentence', "tweet_id"]].dropna()
    srl_df = predict_sentences(df["tweet_id"].tolist(), 
                               df["sentence"].tolist())
    print("{n} sentences processed".format(n = len(srl_df)))
    srl_df.to_csv(OUTPUT_FILE, sep = '\t', index = None)

if __name__ =="__main__":
    main()
