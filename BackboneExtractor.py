"""
Coscia, Michele, and Frank MH Neffke. "Network backboning with noisy data." 2017 IEEE 33rd International Conference on Data Engineering (ICDE). IEEE, 2017
We thank the authors for providing the script below.
"""

import networkx as nx
import pandas as pd

def read(filename, column_of_interest, triangular_input = True, consider_self_loops = True, undirected = True, drop_zeroes = True, sep = "\t"):
    table = pd.read_csv(filename, sep = sep)
    table = table[["src", "trg", column_of_interest]]
    table.rename(columns = {column_of_interest: "nij"}, inplace = True)
    if drop_zeroes:
      table = table[table["nij"] > 0]
    if not consider_self_loops:
      table = table[table["src"] != table["trg"]]
    if triangular_input:
      table2 = table.copy()
      table2["new_src"] = table["trg"]
      table2["new_trg"] = table["src"]
      table2.drop("src", 1, inplace = True)
      table2.drop("trg", 1, inplace = True)
      table2 = table2.rename(columns = {"new_src": "src", "new_trg": "trg"})
      table = pd.concat([table, table2], axis = 0)
      table = table.drop_duplicates(subset = ["src", "trg"])
    original_nodes = len(set(table["src"]) | set(table["trg"]))
    original_edges = table.shape[0]
    if undirected:
      return table, original_nodes, original_edges / 2
    else:
      return table, original_nodes, original_edges

def disparity_filter(table, undirected = True, return_self_loops = False):
    table = table.copy()
    table_sum = table.groupby(table["src"]).sum().reset_index()
    table_deg = table.groupby(table["src"]).count()["trg"].reset_index()
    table = table.merge(table_sum, on = "src", how = "left", suffixes = ("", "_sum"))
    table = table.merge(table_deg, on = "src", how = "left", suffixes = ("", "_count"))
    table["score"] = 1.0 - ((1.0 - (table["nij"] / table["nij_sum"])) ** (table["trg_count"] - 1))
    table["variance"] = (table["trg_count"] ** 2) * (((20 + (4.0 * table["trg_count"])) / ((table["trg_count"] + 1.0) * (table["trg_count"] + 2) * (table["trg_count"] + 3))) - ((4.0) / ((table["trg_count"] + 1.0) ** 2)))
    if not return_self_loops:
      table = table[table["src"] != table["trg"]]
    if undirected:
      table["edge"] = table.apply(lambda x: "%s-%s" % (min(x["src"], x["trg"]), max(x["src"], x["trg"])), axis = 1)
      table_maxscore = table.groupby(by = "edge")["score"].max().reset_index()
      table_minvar = table.groupby(by = "edge")["variance"].min().reset_index()
      table = table.merge(table_maxscore, on = "edge", suffixes = ("_min", ""))
      table = table.merge(table_minvar, on = "edge", suffixes = ("_max", ""))
      table = table.drop_duplicates(subset = ["edge"])
      table = table.drop("edge", 1)
      table = table.drop("score_min", 1)
      table = table.drop("variance_max", 1)
    return table[["src", "trg", "nij", "score", "variance"]]

def backbone_from_DisparityFilter(g, confidence=0.90):
    temp = "temp.csv"
    df = nx.to_pandas_edgelist(g)
    df.rename(columns={"source": "src", "target": "trg"},  inplace=True)
    df.to_csv(temp, index = None,sep='\t')
    table, nnodes, nnedges = read(temp, "weight")
    nc_table = disparity_filter(table)
    nc_table = nc_table[nc_table['score'] >= confidence]
    nc_table['weight'] = nc_table['nij']
    G = nx.from_pandas_edgelist(nc_table, 'src', 'trg', ['weight'])
    return G
