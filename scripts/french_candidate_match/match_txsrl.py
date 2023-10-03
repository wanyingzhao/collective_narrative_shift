import pandas as pd
import sys

tx_srl_path = sys.argv[1]
matched_sent_ids = sys.argv[2]
output_path = sys.argv[3]

try:
    tx_srl_df = pd.read_csv(tx_srl_path, sep = ',')
    matched_sent_ids = pd.read_csv(matched_sent_ids)

    filtered_df = matched_sent_ids.merge(tx_srl_df, 
                            left_on = "twitter_id",
                            right_on = "sent_id")
                        
    filtered_df.to_csv(output_path, sep = '\t')
    
except:
    empty  = pd.DataFrame()
    empty.to_csv(output_path)


