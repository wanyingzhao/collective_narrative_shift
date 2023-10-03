"""
This script provides functions for:
    1. split tweet into sentences
    2. extract narrative triplets from SRL output
"""

from nltk import tokenize
from cleantext import clean
import pandas as pd
import nltk
nltk.download('punkt')

from incas_python.api.model import Annotation
from incas_python.api.model.offset import Offset

def clean_tweet(tweet):
    return clean(tweet, no_urls=True, no_emails = True)


def split_tweet(tweet):
    try:
        return tokenize.sent_tokenize(tweet)
    except:
        print("ERROR! Cant split into sentences")
        print(tweet)
        return []


def find_arg(description, argflag):
    """
    This function take SRL description and argflag as input
    and return corrsponding span that labeled with the argflag
    example argflag includes:
    "ARG0"
    "ARG1"
    "ARGM-MOD"
    """
    prefix = f"[{argflag}: "
    if prefix not in description:
        return None
    sid = description.index(prefix)
    eid = description.index("]", sid)

    arg = description[sid+len(prefix): eid]

    return arg

def get_narrative_triplet(record):
    """
     The output of SRL model sometimes contains
     multiple records. The data structure of a record is
     dictionary. The keys are (example record):
         - "narrative": "(ARG0, Verb, ARG1)"
         - "verb" : 'Did',
         - "verb_lemma" : 'do' 
         - "frame": 'go.04'
         - "frame_score": '0.10186545550823212,'
"""
    description = record["description"]
    verb_lemma = record["lemma"]
    verb = record["verb"]
    frame = record["frame"]
    if "frame_score" in record:
        frame_score = record["frame_score"]
    else:
        frame_score = record["frame_scores"]

    ARG0 = find_arg(description, "ARG0")
    ARG1 = find_arg(description, "ARG1")
   
    if (ARG0==None) or (ARG1==None):
        return None
    
    return {"description":description,
            "ARG0": ARG0,
            "ARG1": ARG1,
            "narrative": (ARG0, verb, ARG1), 
            "verb": verb, 
            "verb_lemma": verb_lemma,
            "frame": frame, 
            "frame_score": frame_score}

def df_format(preds):
    """
    Formats output so that it return a DataFrame that includes columns:
        --'tweet_id', 
        --'description', 
        --'ARG0', 
        --'ARG1', 
        --'narrative', 
        --'verb', 
        --'verb_lemma', 
        --'frame', 
        --'frame_score'
        --'sentence'
    """
    
    df = pd.DataFrame(columns = ["sent_id", 
        'description',
        'ARG0',
        'ARG1',
        'narrative',
        'verb',
        'verb_lemma',
        'frame',
        'frame_score',
        'sentence'])
    for sent_id, pred, sent in preds:
        
        description = pred["description"]
        ARG0 = pred["ARG0"]
        ARG1 = pred["ARG1"]
        narrative = pred["narrative"]
        verb = pred["verb"]
        verb_lemma = pred["verb_lemma"]
        frame = pred["frame"]
        frame_score = pred["frame_score"]
        sent = sent["sentence"]  
        
        df.loc[len(df.index)] = [sent_id,
                description,
                ARG0,
                ARG1,
                narrative,
                verb,
                verb_lemma,
                frame,
                frame_score,
                sent]
    return df

        

