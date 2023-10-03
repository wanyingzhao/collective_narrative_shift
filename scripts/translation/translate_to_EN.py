""" this script translates texts from one language to another.

The format of input should have a column indicating:
    "token": tokens/sentence as a string 

The expected output would have two columns:
    "token_en": translated tokens/sentence
    "pre_lang": original language
"""

import sys

import cld3
import pandas as pd
from transformers import MarianTokenizer, MarianMTModel
import torch
import torch.nn as nn
import random

cuda_idx = random.sample(['0','1', '2','3'],1)[0]
device = torch.device("cuda:"+cuda_idx if torch.cuda.is_available() else "cpu")

def check_language(text):
    return cld3.get_language(text).language


def retrieve_model(source_lang, target_lang="en"):
    model_name = f"Helsinki-NLP/opus-mt-{source_lang}-{target_lang}"
    torch.cuda.empty_cache()
    model = MarianMTModel.from_pretrained(model_name)
    model =  model.to(device)
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    return model, tokenizer

def batch_texts(texts, batch_size=10):
    return [texts[idx:idx+batch_size] for idx in range(0, len(texts), batch_size)]

def get_translation_EN(texts, source_lang, target_lang="en"):
    if source_lang == target_lang:
        return texts
    if source_lang in ["fr", "es"]:
        model, tokenizer = retrieve_model(source_lang, target_lang)
        texts = batch_texts(texts)
        translations = []
        for idx, text in enumerate(texts):
            batch = tokenizer(text, return_tensors="pt", padding=True).to(device)
            gen = model.generate(**batch)
            translations += tokenizer.batch_decode(gen, skip_special_tokens=True)
            del batch
            del gen
            print(idx*len(text))
        return translations
    else:
        raise NotImplementedError(f"the source language {source_lang} is not supported")


if __name__ == "__main__":
    input_filename = sys.argv[1]
    output_filename = sys.argv[2]
    TARGET_LANG = "en"

    df = pd.read_csv(
        input_filename , #usecols=[ "token", 'tweetid', 'date'], 
        lineterminator="\n" )

    print(df.columns)
    
    df["token"] = df["token"].astype(str)
    df["pre_lang"] = df["token"].apply(check_language)

    with open(output_filename, "w") as outf:
        col0 = 'twitter_id'
        col1 = 'token_en'
        col2 = 'pre_lang'
        col3 = 'date'
        outf.write(f'{col0}\t{col1}\t{col2}\t{col3}\n')

        for source_lang in ["en", "fr", "es"]:
            tweets = df[df["pre_lang"] == source_lang]["token"].tolist()
            tweets_id = df[df["pre_lang"] == source_lang]["tweetid"].tolist()
            tweets_date = df[df["pre_lang"] == source_lang]["date"].tolist()
            print('found {} tweets: {}'.format(source_lang, len(tweets)))
            en_token = get_translation_EN(tweets, source_lang, TARGET_LANG)
            for idx, token in enumerate(en_token):
                outf.write('{tid}\t{token}\t{source_lang}\t{date}\n'.format(tid = str(tweets_id[idx]),
                                                                  token = token,
                                                                  source_lang = source_lang,
                                                                  date = tweets_date[idx]))
