"""This script reads sentBert embeddings of narratives, and cluster them with Birch,
   The return csv file includes narratives content and correpoinding cluter number

   INPUT:
       embeds_path: The path of narrative embedding file.
       narrative_path: The path of narrative file.
    OUTPUT:
       output_path: the dir where clustering results are stored
"""
import pandas as pd
import sys
import os
from sklearn.cluster import Birch

embeds_path = sys.argv[1]
narrative_path = sys.argv[2]
output_path = sys.argv[3]
    

sample_df = pd.read_csv(embeds_path, sep = '\t', header = None, skiprows = 1)
embeds = sample_df.iloc[: ,1 :-1]
embeds = embeds.to_numpy()
args = sample_df.iloc[: , -1]


# birch
brc = Birch(n_clusters=None)
brc.fit(embeds)
brc_label = brc.labels_
brc_df = pd.DataFrame({"Arg": args, "brc": brc_label})

# generate gold labels
df = brc_df.groupby(["Arg", "brc"]).size().reset_index(name = "counts")
max_gb = df.groupby("brc")["counts"].max().reset_index(name = "counts")
sum_gb = df.groupby("brc")["counts"].sum().reset_index(name = "sum")

merged_df = df.merge(max_gb, on = ['brc', 'counts'])
merged_df = merged_df.drop_duplicates(subset=["brc", 'counts'])
merged_df = df.merge(merged_df, on = "brc", how = "left")
gold_match = merged_df[["Arg_x", "Arg_y"]]

def map_gold(arg, df = gold_match):
    arg_y = df[df["Arg_x"]==arg]["Arg_y"].values[0]
    if arg_y == None:
        return arg
    else: return arg_y

gold_match["gold"] = gold_match["Arg_x"].apply(map_gold)

gold_match = gold_match[["Arg_x", "gold"]]
gold_match = gold_match.rename(columns = {"Arg_x": "Arg" })
gold_match.to_csv(output_path, sep = '\t')
print("birch is done")
