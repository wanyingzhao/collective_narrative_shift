""" this script filters tweets based on tweet types

The format of input should have two columns:
    "token" : tokens/sentence as a string
    "tweet_type" : should be 1)reply 2)original 3)retweeted_tweet_without_comment 4)quoted_tweet

The expected output would only one columns:
    "token"
"""

import sys

import pandas as pd


def read_tweets(filename, usecols = ['token','tweetid']):
    return pd.read_csv(
        filename, usecols=usecols, lineterminator="\n"
    )


def filter_tweets_by_type(df, tweet_type):
    return df[df['tweet_type'] == tweet_type]


if __name__ == "__main__":
    input_filename = sys.argv[1]
    output_filename = sys.argv[2]
    TARGET_TWEET_TYPE = sys.argv[3]

    df = read_tweets(input_filename, usecols = ['token', 'tweet_type','tweetid','date'])
    out_df = filter_tweets_by_type(df, TARGET_TWEET_TYPE)
    out_df.to_csv(output_filename, index = False)
