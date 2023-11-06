import networkx as nx
import numpy as np
import pickle

import time

import scipy.stats as scs

#import dblp_authors
import dblp_aux_functions
import dblp_backbone_analysis
import dblp_correlation
import dblp_network_analysis
import dblp_plot
import dblp_generate_rank

					
if __name__ == "__main__":
	
	# Path to input CSV data.	
	prefix = "data/author-article/"
	filenames = ['graph-2010','graph-2011','graph-2012','graph-2013','graph-2014','graph-2015','graph-2016','graph-2017','graph-2018','graph-2019','graph-2020']#,'graph-2021']
	
	
	print("Reading CSV content...")
	# Reads network content and saves as a dictionary.
	network_content = dblp_aux_functions.csv_parser(prefix,filenames)
	print("OK!\n")
	
	print("Generating full networks...")
	# Reads the 'network_content' dictionary and saves in a dictionary where the keys are the snapshot ids and the values are undirected networkx graphs.
	full_networks = dblp_aux_functions.network_generator(network_content)
	for snap_id in full_networks:
		print(nx.info(full_networks[snap_id]))
	print("OK!\n")
	
	print("=========================")
	print("Filtering authors...")
	# Filters authors with less than 50 papers.
	min_papers = 50
	authors_ok_list,authors_not_ok_list = dblp_aux_functions.filter_authors(prefix,min_papers)
	filtered_networks = {}
	for snap_id in full_networks:
		full_networks[snap_id].remove_nodes_from(authors_not_ok_list)
	for snap_id in full_networks:
		print(nx.info(full_networks[snap_id]))
	print("OK!\n")
	
	
	print("Removing null degree (full networks)...")
	# Removing authors with null degree (no authorship in the dataset)
	full_networks = dblp_aux_functions.filter_null_degree(full_networks)
	for snap_id in full_networks:
		print(nx.info(full_networks[snap_id]))
	print("OK!\n")
	
	
	print("Extracting backbones...")
	# Extracts the backbones from the full_networks.
	# Stores the backbones in a dictionary where the keys are the snapshot ids and the values are backbone networkx graphs.
	significance = 0.1
	backbones = dblp_backbone_analysis.backbone_extractor(full_networks,significance)
	for snap_id in backbones:
		print(nx.info(backbones[snap_id]))
	print("OK!\n")
	
	
	print("Removing null degree (backbones)...")
	# Removing authors with null degree from the backbones.
	backbones = dblp_aux_functions.filter_null_degree(backbones)
	for snap_id in backbones:
		print(nx.info(backbones[snap_id]))
	print("OK!\n")
	
	
	print("Plotting EWD (both)...")
	# Plots edge weight distribution of the full networks and the backbones.
	# (Figure 2 in the paper)
	dblp_plot.plot_edge_weights_distribution_both(full_networks,backbones,"both",filenames)
	print("OK!\n")
	
	
		
	print("Exporting GML...")
	# Exports full networks, backbone and a projection of all full networks in a single network.
	dblp_aux_functions.export_gml(full_networks,"full")
	dblp_aux_functions.export_gml(backbones,"backbone")
	dblp_aux_functions.export_gml_snapshots_is_backbone(full_networks,backbones)
	single_full_network = dblp_aux_functions.export_single_network_gml(full_networks,backbones)
	print("OK!\n")
	
	
	
	print("Generating hyperprolific rank...")
	# File with the described authors features
	authors_features_filename = "data/authors-features.csv"
	all_ranked_simple,top_ranked_simple,all_features = dblp_generate_rank.generate_ranked_simple(authors_features_filename)
	
	# Writing the hyperprolific rank in a pickle file.
	with open('data/all_ranked_simple.pickle', 'wb') as handle:
		pickle.dump(all_ranked_simple,handle)
	print("OK!\n")
	
	
	print("Calculating probability backbone...")
	# Performs all experiments related to the probability of presence in backbone.
	# (Figures 5, 6, 7 and 8 in the paper. Section 4.2)
	dblp_backbone_analysis.calculate_probability_backbone(full_networks,backbones,single_full_network,top_ranked_simple,all_ranked_simple,filenames,50)
	print("OK!\n")
	
	
	print("Calculating other network properties...")
	# Performs all experiments related to network properties.
	# (Figures 3 and 4 in the paper. Section 4.1.)
	dblp_network_analysis.calculate_network_properties(full_networks,backbones,top_ranked_simple,filenames)
	print("OK!\n")
	
	
	print("Calculating static rank correlation (topological X hyperprolific)...")
	# Performs experiments related to static rank correlation.
	# (Figure 9.)
	all_ranked_filename = 'all_ranked.pickle' # Rank do Edré
	all_topological_ranks_filename = 'all_topological_ranks_ranked_simple.pickle' # Rank baseado na topologia (degree, closeness, eigenvector, betweenness)
	# Se o all_ranked não estiver calculado, precisa calcular (ou ler do pickle)
	dblp_correlation.calculate_rank_correlations(all_ranked_simple,all_topological_ranks_filename,full_networks)
	print("OK!\n")
	
	
	print("Calculating dynamic correlation (Time-Lagged Cross Correlation)...")
	# Performs experiments related to dynamic correlation.
	# (Figures 10 and 11)
	dblp_correlation.calculate_dynamic_correlation(backbones,authors_features_filename,all_ranked_simple,filenames)
	print("OK!\n")
	
