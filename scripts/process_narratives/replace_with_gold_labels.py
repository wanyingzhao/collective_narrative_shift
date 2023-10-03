import pandas as pd
import os
import sys

narrative_path = sys.argv[1]
gold_narr_list_path = sys.argv[2]
gold_arg_list_path = sys.argv[3]
formated_narrative_path = sys.argv[4]

narrative_df = pd.read_csv(narrative_path, sep = '\t')
gold_narr_df = pd.read_csv(gold_narr_list_path, sep = '\t')
gold_arg_df = pd.read_csv(gold_arg_list_path, sep = '\t')
gold_arg_df = gold_arg_df.rename(columns = {'gold': 'gold_arg'})
print(gold_arg_df.head())

def aggregate(df):
    sum_gb = df.groupby("gold")["counts"].sum().reset_index(name = "sum")
    
    merged_df = df.merge(sum_gb, on = ['gold'])
    merged_df = merged_df.drop_duplicates(subset=["gold", 'sum_x'])
    merged_df = merged_df.rename(columns = {'ARG0_y': "ARG0", 'va_key_y': "va_key", 'ARG1_y':"ARG1", "sum_x":"sum"})
    return merged_df[['narrative', 'gold', 'brc', 'sum', 'ARG0', 'va_key', 'ARG1']]
    
# replace with gold narr label
narrative_df["narrative"] = narrative_df["ARG0"] + ' ** ' + narrative_df["va_key"] + ' ** ' + narrative_df["ARG1"]
merged_df = narrative_df.merge(gold_narr_df, how= "left", on="narrative")

merged_df = aggregate(merged_df)

# replace with gold arg label
merged_df = merged_df.merge(gold_arg_df, how= "left", left_on="ARG0", right_on="Arg" )
merged_df = merged_df.merge(gold_arg_df, how="left",left_on = "ARG1", right_on="Arg" )

merged_df['gold_arg_x'] = merged_df['gold_arg_x'].fillna(merged_df['ARG0'])
merged_df['gold_arg_y'] = merged_df['gold_arg_y'].fillna(merged_df['ARG1'])

merged_df = merged_df.rename(columns = {'gold_arg_x': "ARG0_gold", "gold_arg_y": "ARG1_gold"})

output_df = merged_df[["ARG0",  "ARG0_gold", "ARG1","ARG1_gold", "va_key", "gold", "sum"]]
output_df["gold_narr"] = output_df["ARG0_gold"] + ' + ' + output_df["va_key"] + " + " + output_df["ARG1_gold"]
sum_gb = output_df.groupby("gold_narr")["sum"].sum().reset_index(name = "sum")

output_df = output_df.merge(sum_gb, on = "gold_narr")
output_df = output_df.drop(columns = ["sum_x"])
output_df  = output_df.rename(columns= {"sum_y" : "sum"})
output_df = output_df.drop_duplicates(subset=["gold_narr"])

output_df.to_csv(formated_narrative_path, sep='\t', index=False)



