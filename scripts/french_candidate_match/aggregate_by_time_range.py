"""
This script aggregate descriptions based on change point detection result. And return a csv file where each row records
a unique narrative and its corresponding frequency within the time range. 

INPUT: 
       start-time: the starting time of the change point
       end-time: the ending time of the change point
       description_dir: the path of dir where descirptions files kept
OUTPUT: 
       the path of output file
"""

import pandas as pd
import sys
import os
from datetime import datetime as dt
import re

sent_id_files = sys.argv[1:-2]
start_to_end_time = sys.argv[-2].split('_to_')
output_path = sys.argv[-1]

start_time = dt.strptime(start_to_end_time[0], "%Y-%m-%d")
end_time = dt.strptime(start_to_end_time[1], "%Y-%m-%d")

print(f"processing {start_time} to {end_time} data")
# read files within certain time range and return a df
def read_descirptions(start_time, end_time, files):
    
    df = pd.DataFrame()
    # filter file that within start_time and end_time
    for file in files:
        file_date = re.search('\d{4}-\d{2}-\d{2}', file)[0]
        file_date  = dt.strptime(file_date, "%Y-%m-%d")

        if (file_date >= start_time) and (file_date <= end_time):
            print(file)
            try: 
                file_df = pd.read_csv(os.path.join(file), sep = '\t')
                df = pd.concat([file_df, df])
            except:
                print(f"ERROR!!!: {file} can be read")
            
    return df


df = read_descirptions(start_time, end_time, sent_id_files)
df_groupby = df.groupby(["ARG0", "va_key", "ARG1"]).size().reset_index(name = 'counts')

df_groupby.to_csv(output_path, sep = '\t', index = False)