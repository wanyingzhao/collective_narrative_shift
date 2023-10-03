import pandas as pd
import sys
import csv

translated_tweet_path = sys.argv[1]
kw_path = sys.argv[2]
output_path = sys.argv[3]

def read_kw(kw_path):
    kw_list = []
    with open(kw_path, 'r') as f:
        kw_list = [kw.strip() for kw in f.readlines()]
    return kw_list
    

def kw_map(tweet):
    for kw in kw_list:
        if kw in str(tweet):
            return 1
    return 0

tweet_df = pd.read_csv(translated_tweet_path, sep = '\t', quoting=csv.QUOTE_NONE)
kw_list = read_kw(kw_path)

tweet_df["with_kw"] = tweet_df["token_en"].apply(kw_map)
match_df = tweet_df[tweet_df["with_kw"] ==1]

matched_ids = match_df["twitter_id"].drop_duplicates()
matched_ids.to_csv(output_path, index = False)
