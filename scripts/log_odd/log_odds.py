#!/usr/bin/env python
""" Calculate token log-odds ratios using frequency distributions of two corpora using
    a background corpus. Defaults to z-score scaling unless '--raw' flag is used. """

import argparse
import logging
import math
import operator
from typing import Iterator
import pandas as pd


def _tsv_reader(filepath: str) -> Iterator[str]:
    with open(filepath, "r") as src:
        for line in src:
            yield line.rstrip().split("\t")

def _df_reader(filepath: str) -> dict:
    df = pd.read_csv(filepath, sep = '\t') #, usecols = ['gold_narr', 'sum'])
    print(df.columns)
    freqs = dict()
    for idx, row in df.iterrows():
        freqs[row["gold_narr"]] = int(row["sum"])
    return freqs

def _get_dict(filepath: str) -> dict:
    freqs = dict()
    for token, freq in _tsv_reader(filepath):
        freqs[token] = int(freq)
    return freqs


def _size(corpus: dict) -> int:
    return sum(corpus.values())


def _log_odds(
    word: str,
    c1: dict,
    c2: dict,
    bg: dict,
    size1: int,
    size2: int,
    size3: int,
    raw: bool,
) -> float:
    c1_w = 0
    c2_w = 0
    try:
        c1_w = c1[word]
        c2_w = c2[word] 
    except:
        print(word)
    numerator_1 = c1_w + bg[word]
    numerator_2 = c2_w + bg[word]
    denom_1 = size1 + size3 - numerator_1
    denom_2 = size2 + size3 - numerator_2
    raw_logodds = math.log(numerator_1 / denom_1) - math.log(
        numerator_2 / denom_2
    )
    if raw:
        return raw_logodds
    else:
        variance = (1 / numerator_1) + (1 / numerator_2)
        return raw_logodds / math.sqrt(variance)


def main(args: argparse.Namespace) -> None:

    c_1 = _df_reader(args.corpus_1)
    c_2 =  _df_reader(args.corpus_2)
    c_bg = _df_reader(args.corpus_bg)

    size1 = _size(c_1)
    size2 = _size(c_2)
    size3 = _size(c_bg)

    supported_tokens = set(c_1.keys())
    supported_tokens |= c_2.keys()
    supported_tokens |= c_bg.keys()

    if args.raw:
        logging.info("Calculating token log-odds-ratios.")
    else:
        logging.info("Calculating token log-odds-ratios scaled by z-score.")

    ratios = []
    for tok in supported_tokens:
        rat = _log_odds(tok, c_1, c_2, c_bg, size1, size2, size3, args.raw)
        ratios.append((tok, rat))

    ratios.sort(key=operator.itemgetter(1), reverse=True)

    if args.out:
        with open(f"{args.out}", "w") as sink:
            sink.write(f"Top from {args.corpus_1}\n")
            for tok, rat in ratios:
                sink.write(f"{tok}\t{rat}\n")
            
    else:
        print(f"Top  from {args.corpus_1}")
        for tok, rat in ratios:
            print(f"{tok}\t{rat}")
       


if __name__ == "__main__":
    logging.basicConfig(level="INFO", format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "corpus_1", help="first corpus, a tsv of token + counts"
    )
    parser.add_argument("corpus_2", help="second corpus tsv for comparison")
    parser.add_argument("corpus_bg", help="backgroud corpus tsv")
    parser.add_argument("--lim", type=int, help="Num of top ranked words")
    parser.add_argument(
        "--out", default=False, help="optional path to ranked output file"
    )
    parser.add_argument(
        "--raw",
        default=False,
        action="store_true",
        help="flag to use unscaled log-odds-ratios",
    )
    main(parser.parse_args())
