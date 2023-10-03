"""
The script aims to filter tweets that are spams.
We define spam as tweets with overly duplicated characters, e.g.
tweet with 100 "t" or tweet with 30 'that's--'. These tweets are hard
to process by SRL and would return error indicating mismatch of the size of tensor. 

We assume that if a string has a certain char more than DUPLICATE_THESHOLD times, it would be considered as a spam. 

input:
    INPUT_FILE = the path of tweet file
    The tweet file should have at least two columns: 

    "tweet_id": tweet content
    "sentence": type of tweet, including "original", "retweet"


output:
    OUTPUT_FILE = the path of the filtered tweet file
    The tweet file would have two columns:

    "tweet_id": tweet content
    "sentence": type of tweet, including "original", "retweet"

"""


import sys
import pandas as pd


DUPLICATE_THESHOLD = 100
INPUT_FILE = sys.argv[1]
OUTPUT_FILE = sys.argv[2]


# the fuction count the frequency of char
def count_char(text):
    counter = {}
    for char in text:
        counter[char] = counter.get(char, 0) + 1
    return counter


# filter tweet if certain char
# appear in it for more than DUPLICATE_THESHOLD times
def filter_spam(text):
    char_freq_dict = count_char(text)
    if max(char_freq_dict.values()) > DUPLICATE_THESHOLD:
        text = ""
    return text


if __name__ == "__main__":

    df = pd.read_csv(INPUT_FILE, usecols=["tweet_id", "sentence"], sep = '\t')
    df = df.dropna() 
    df["filtered_sentence"] = df["sentence"].apply(filter_spam)
    # remove Nan records
    df = df.dropna()

    out_df = df[["filtered_sentence", "tweet_id"]]
    out_df.columns = ["sentence", "tweet_id"]
    out_df.to_csv(OUTPUT_FILE, sep = '\t',index=False)

