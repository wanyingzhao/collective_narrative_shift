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


    
def read_embeds_and_narratives(embeds_path, narrative_path):
    
        sample_df = pd.read_csv(embeds_path, sep = '\t')
        embeds = sample_df.iloc[: ,1 :-1]

        narrative_df = pd.read_csv(narrative_path, sep = '\t')
        narrative_df["narrative"] = narrative_df["ARG0"] + ' ** ' + narrative_df["va_key"] + ' ** ' + narrative_df["ARG1"]
        return embeds, narrative_df
    

def group_narratives(narrative_df, labels, cluster_name):
    df_w_labels = narrative_df.copy()
    df_w_labels[cluster_name] = labels
    return df_w_labels[[cluster_name, "narrative"]]

# read data

embeds, narrative_df = read_embeds_and_narratives(embeds_path, narrative_path)

print(len(embeds), len(narrative_df))

print("finish reading data")  
    
# birch
brc = Birch(n_clusters=None)
embeds = embeds.to_numpy()
brc.fit(embeds)
brc_label = brc.labels_
brc_result = group_narratives(narrative_df, brc_label, "brc")
print("birch is done")

# add gold label
df = brc_result.merge(narrative_df, on = 'narrative')
max_gb = df.groupby("brc")["counts"].max().reset_index(name = "counts")
sum_gb = df.groupby("brc")["counts"].sum().reset_index(name = "sum")

merged_df = df.merge(max_gb, on = ['brc', 'counts'])

merged_df = merged_df.merge(sum_gb, on = ['brc'])
merged_df = merged_df.drop_duplicates(subset=["brc", 'sum'])
merged_df = merged_df.merge(brc_result, on = 'brc')
merged_df = merged_df.rename(columns = {"narrative_x": "gold", 
                                        "narrative_y":"narrative",
                                        "ARG0_x" : "ARG0",
                                        "va_key_x" : "va_key",
                                        "ARG1_x": "ARG1"})

merged_df[["gold", "narrative", "brc", "sum", "ARG0", "va_key", "ARG1"]].to_csv(output_path, sep = '\t')


