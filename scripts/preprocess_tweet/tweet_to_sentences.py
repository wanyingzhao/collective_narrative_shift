""" this script splits tweets into sentences
    
    The input should have column:
    "token_en" : tweets

    The output would have column:
    "tweet_id" : the idx in input file
    "sentence" : a sentence


"""
import pandas as pd
import sys
from nltk import tokenize
import nltk

import csv
from cleantext import clean

from filter_spam import filter_spam, count_char


INPUT_FILE  = sys.argv[1]
OUTPUT_FILE = sys.argv[2]

def split_tweet(tweet):
    try:
        return tokenize.sent_tokenize(tweet)
    except:
        print(tweet)
        return []

def clean_sent(sent):
    sent = clean(sent, 
       to_ascii=True, 
       lower=True, 
       no_urls=True) 
    return sent

def main():
    if 'covid' in INPUT_FILE:
        df = pd.read_csv(INPUT_FILE, sep = '\t')
        df['sentences'] = df['token'].apply(split_tweet)
    else:
        df = pd.read_csv(INPUT_FILE, sep='\t', quoting=csv.QUOTE_NONE, on_bad_lines='skip')
        df['sentences'] = df['token_en'].apply(split_tweet)
    idxs = []
    outputs = []
    for _, row in df[['twitter_id','sentences']].iterrows():
        record = row["sentences"]
        outputs += record
        idxs += [row["twitter_id"] for i in range(len(record))]
    out_data = {'tweet_id': idxs, 'sentence':outputs}
    out_df = pd.DataFrame(out_data)

    # filter out noise
    out_df["filtered_sentence"] = out_df["sentence"].apply(filter_spam)
    out_df = out_df.dropna()

    out_df = out_df[["filtered_sentence", "tweet_id"]]
    out_df.columns = ["sentence", "tweet_id"]

    out_df["sentence"] = out_df["sentence"].apply(clean_sent)

    out_df.to_csv(OUTPUT_FILE,sep='\t', index = False)

if __name__ == "__main__":
    main()
