import pandas as pd
import sys
import csv

log_odd_path = sys.argv[1]
narr_path = sys.argv[2]
output_path = sys.argv[3]


log_odd_df = pd.read_csv(log_odd_path, sep = '\t', header = None, skiprows = 1, quoting=csv.QUOTE_NONE)
log_odd_df.columns = ["gold_narr", 'log_odd']
narr_df = pd.read_csv(narr_path, sep = '\t')

merged_df = log_odd_df.merge(narr_df, on = 'gold_narr')
merged_df.to_csv(output_path, index = None, sep = '\t')
print(f"{output_path}\t is done!")
