"""
This script cleans txSRL files and return clean df

Input: path of txSRL file

Output: a clean df file includes columns:
        --sent_id
        --ARG0
        --ARG1
        --narrative
        --verb
        --va_frame : transfered Verbatlas frames, when there is no match, 
                     the value would be None
        --frame : mostly Verbatlas frames, when the original
                  Propbank frames find no match in Verbatlas,
                  we use Propbank frames instead. 
                  
"""

import pandas as pd
import sys
import numpy as np
from propbank_to_verbatlas import pb_to_va, va_to_key

INPUT = sys.argv[1]
OUTPUT = sys.argv[2]

if __name__ == "__main__":
       f = open(INPUT,'r')
       txSRL_df = txSRL_df = pd.DataFrame()
       if len(f.readlines()) > 2:
            txSRL_df = pd.read_csv(INPUT, sep = '\t',  usecols = ["sent_id",
                                                 "ARG0",
                                                 "ARG1",
                                                 "narrative",
                                                 "verb",
                                                 "frame",
                                                 "sentence"])
        
            txSRL_df["va_frame"] = txSRL_df["frame"].apply(pb_to_va)
            txSRL_df["va_key"] = txSRL_df["va_frame"].apply(va_to_key)
            txSRL_df["frame"] = np.where(txSRL_df["va_frame"].notna(),
                                    txSRL_df["va_frame"], txSRL_df["frame"])
        
       txSRL_df.to_csv(OUTPUT, index = False)
