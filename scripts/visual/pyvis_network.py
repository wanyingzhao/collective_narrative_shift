"""
This script reads dataframe and draw network visual by pyvis. 
"""

from pyvis.network import Network
import pandas as pd
import sys

narr_path = sys.argv[1]
output_path = sys.argv[2]

df = pd.read_csv(narr_path, sep = '\t')

nodes = list(set(list(df["ARG0_gold"]) + list(df["ARG1_gold"])))
nt = Network('800px', '800px')
# add nodes
for idx, node in enumerate(nodes):
    nt.add_node(idx, label = node)
    
for idx, row in df.iterrows():
    id1 = nodes.index(row["ARG0_gold"])
    id2 = nodes.index(row["ARG1_gold"])
    nt.add_edge( id1, 
                 id2,
                 arrowStrikethrough = True,
                width = row["sum"]*0.1,
                title = row["va_key"])
        
nt.show_buttons(filter_=['edges', 'nodes','physics'])
nt.save_graph(output_path )