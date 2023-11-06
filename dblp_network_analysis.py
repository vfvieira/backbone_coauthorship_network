import networkx as nx
import numpy as np
import pickle

import dblp_plot


def get_edge_weigths(full_networks):
	
	edge_weights = {}
	for snap_id in full_networks:
		graph = full_networks[snap_id]
		edge_list_snapshot = []
		for user_from, user_to, data in graph.edges(data=True):
			edge_list_snapshot.append(data['weight'])
		edge_weights[snap_id] = edge_list_snapshot
	return edge_weights

	
def n_neighbor(G, id, n_hop):
	node = [id]
	node_visited = set()
	neighbors= []
	
	while n_hop !=0:
		neighbors= []
		for node_id in node:
			node_visited.add(node_id)
			neighbors +=  [id for id in G.neighbors((node_id)) if id not in node_visited]
		node = neighbors
		n_hop -=1
		
		if len(node) == 0 :
			return neighbors 
		
	return neighbors
	
def calculate_network_properties(networks,backbones,top_ranked,filenames):
	
	nodes_in_top50 = []
	nodes_in_top10 = []
	
	count = 0
	for node in top_ranked:
		if count < 10:
			nodes_in_top10.append(node)
		nodes_in_top50.append(node)
		count+=1
		
		
	assortativity_full_all_nodes_dict = {}
	assortativity_full_nodes_in_top10_dict = {}
	assortativity_full_nodes_in_top50_dict = {}
	assortativity_full_nodes_in_backbone_dict = {}
	
	assortativity_backbone_all_nodes_dict = {}
	assortativity_backbone_nodes_in_top10_dict = {}
	assortativity_backbone_nodes_in_top50_dict = {}
	
	eigenvector_all_dict = {}
	degree_all_dict = {}
	closeness_all_dict = {}
	betweenness_all_dict = {}
	
	eigenvector_backbone_dict = {}
	degree_backbone_dict = {}
	closeness_backbone_dict = {}
	betweenness_backbone_dict = {}
	
	eigenvector_top10_dict = {}
	degree_top10_dict = {}
	closeness_top10_dict = {}
	betweenness_top10_dict = {}
	
	eigenvector_top50_dict = {}
	degree_top50_dict = {}
	closeness_top50_dict = {}
	betweenness_top50_dict = {}
	
	
	all_topological_ranks = {}
	
	
	for snap_id in networks:
		all_topological_ranks[snap_id] = {}
		nodes_in_backbone = []
		for node in networks[snap_id]:
			if backbones[snap_id].has_node(node):
				nodes_in_backbone.append(node)
		try:
			assortativity_full_all_nodes = nx.degree_assortativity_coefficient(networks[snap_id],weight='weight')
		except ValueError:
			assortativity_full_all_nodes = -1.1
		try:
			assortativity_full_nodes_in_top10 = nx.degree_assortativity_coefficient(networks[snap_id],weight='weight',nodes=nodes_in_top10)
		except ValueError:
			assortativity_full_nodes_in_top10 = -1.1
		try:
			assortativity_full_nodes_in_top50 = nx.degree_assortativity_coefficient(networks[snap_id],weight='weight',nodes=nodes_in_top50)
		except ValueError:
			assortativity_full_nodes_in_top50 = -1.1
		try:
			assortativity_full_nodes_in_backbone = nx.degree_assortativity_coefficient(networks[snap_id],weight='weight',nodes=nodes_in_backbone)
		except ValueError:
			assortativity_full_nodes_in_backbone = -1.1
		
		try:
			assortativity_backbone_all_nodes = nx.degree_assortativity_coefficient(backbones[snap_id],weight='weight')
		except ValueError:
			assortativity_backbone_all_nodes = -1.1
		try:
			assortativity_backbone_nodes_in_top10 = nx.degree_assortativity_coefficient(backbones[snap_id],weight='weight',nodes=nodes_in_top10)
		except ValueError:
			assortativity_backbone_nodes_in_top10 = -1.1
		try:
			assortativity_backbone_nodes_in_top50 = nx.degree_assortativity_coefficient(backbones[snap_id],weight='weight',nodes=nodes_in_top50)
		except ValueError:
			assortativity_backbone_nodes_in_top50 = -1.1
		
		print("\nsnapshot: %s" % snap_id)
		print("all_in_full:\t%.2f\ttop10_in_full:\t%.2f\ttop50_in_full:\t%.2f\tbackbone_in_full:\t%.2f" % (assortativity_full_all_nodes,assortativity_full_nodes_in_top10,assortativity_full_nodes_in_top50,assortativity_full_nodes_in_backbone))
		print("all_in_backbone:\t%.2f\ttop10_in_backbone:\t%.2f\ttop50_in_backbone:\t%.2f" % (assortativity_backbone_all_nodes,assortativity_backbone_nodes_in_top10,assortativity_backbone_nodes_in_top50))
		
		assortativity_full_all_nodes_dict[snap_id] = assortativity_full_all_nodes
		assortativity_full_nodes_in_top10_dict[snap_id] = assortativity_full_nodes_in_top10
		assortativity_full_nodes_in_top50_dict[snap_id] = assortativity_full_nodes_in_top50
		assortativity_full_nodes_in_backbone_dict[snap_id] = assortativity_full_nodes_in_backbone
		
		assortativity_backbone_all_nodes_dict[snap_id] = assortativity_backbone_all_nodes
		assortativity_backbone_nodes_in_top10_dict[snap_id] = assortativity_backbone_nodes_in_top10
		assortativity_backbone_nodes_in_top50_dict[snap_id] = assortativity_backbone_nodes_in_top50
		
			
		eigenvector_centrality = nx.eigenvector_centrality(networks[snap_id])
		degree_centrality = nx.degree_centrality(networks[snap_id])
		betweenness_centrality = nx.betweenness_centrality(networks[snap_id])
		closeness_centrality = nx.closeness_centrality(networks[snap_id])
		
		
		all_topological_ranks[snap_id]['eigenvector'] = eigenvector_centrality
		all_topological_ranks[snap_id]['degree'] = degree_centrality
		all_topological_ranks[snap_id]['betweenness'] = betweenness_centrality
		all_topological_ranks[snap_id]['closeness'] = closeness_centrality
		
		eigenvector_all = []
		degree_all = []
		betweenness_all = []
		closeness_all = []
		for node in networks[snap_id]:
			eigenvector_all.append(eigenvector_centrality[node])
			degree_all.append(degree_centrality[node])
			betweenness_all.append(betweenness_centrality[node])
			closeness_all.append(closeness_centrality[node])
		
		
		print("eigenvector max:%s min:%s sum:%s" % ( max(eigenvector_all),min(eigenvector_all),sum(eigenvector_all) ) )
		print("closeness max:%s min:%s sum:%s" % ( max(closeness_all),min(closeness_all),sum(closeness_all) ) )
		print("betweenness max:%s min:%s sum:%s" % ( max(betweenness_all),min(betweenness_all),sum(betweenness_all) ) )
		print("degree max:%s min:%s sum:%s" % ( max(degree_all),min(degree_all),sum(degree_all) ) )
		
		print("\nAll nodes")
		
		print("eigenvector:%.4f degree:%.4f betweenness:%.4f closeness:%.4f" % ( np.mean(np.array(eigenvector_all)),np.mean(np.array(degree_all)),np.mean(np.array(betweenness_all)),np.mean(np.array(closeness_all)) ) )
		
		eigenvector_all_dict[snap_id] = np.mean(np.array(eigenvector_all))
		degree_all_dict[snap_id] = np.mean(np.array(degree_all))
		closeness_all_dict[snap_id] = np.mean(np.array(closeness_all))
		betweenness_all_dict[snap_id] = np.mean(np.array(betweenness_all))
				
		
		eigenvector_backbone = []
		degree_backbone = []
		betweenness_backbone = []
		closeness_backbone = []
		for node in nodes_in_backbone:
			eigenvector_backbone.append(eigenvector_centrality[node])
			degree_backbone.append(degree_centrality[node])
			betweenness_backbone.append(betweenness_centrality[node])
			closeness_backbone.append(closeness_centrality[node])
			
		print("Nodes in backbone")
		print("eigenvector:%.4f degree:%.4f betweenness:%.4f closeness:%.4f" % ( np.mean(np.array(eigenvector_backbone)),np.mean(np.array(degree_backbone)),np.mean(np.array(betweenness_backbone)),np.mean(np.array(closeness_backbone)) ) )
		
		
		eigenvector_backbone_dict[snap_id] = np.mean(np.array(eigenvector_backbone))
		degree_backbone_dict[snap_id] = np.mean(np.array(degree_backbone))
		closeness_backbone_dict[snap_id] = np.mean(np.array(closeness_backbone))
		betweenness_backbone_dict[snap_id] = np.mean(np.array(betweenness_backbone))
		
		
	print("\n\n")
	print("==================")
	print("Resultado:")
	print("==================")
	
	#assortativity_full_all_nodes_dict = {}
	assortativity_full_network_all_nodes = []
	print("\nAssortativity full all nodes")
	for snap_id in networks:
		print("%s\t%.4f" % (snap_id,assortativity_full_all_nodes_dict[snap_id]))
		assortativity_full_network_all_nodes.append(assortativity_full_all_nodes_dict[snap_id])
	
	#assortativity_full_nodes_in_top10_dict = {}
	assortativity_full_network_backbone = []
	print("\nAssortativity full top 10")
	for snap_id in networks:
		print("%s\t%.4f" % (snap_id,assortativity_full_nodes_in_top10_dict[snap_id]))
		
	
	
	#assortativity_full_nodes_in_top50_dict = {}
	print("\nAssortativity full top 50")
	for snap_id in networks:
		print("%s\t%.4f" % (snap_id,assortativity_full_nodes_in_top50_dict[snap_id]))
	
	#assortativity_full_nodes_in_backbone_dict = {}
	print("\nAssortativity full nodes in backbone")
	for snap_id in networks:
		print("%s\t%.4f" % (snap_id,assortativity_full_nodes_in_backbone_dict[snap_id]))
		assortativity_full_network_backbone.append(assortativity_full_nodes_in_backbone_dict[snap_id])
	
	#assortativity_backbone_all_nodes_dict = {}
	assortativity_backbone_all_nodes = []
	print("\nAssortativity backbone all nodes")
	for snap_id in networks:
		print("%s\t%.4f" % (snap_id,assortativity_backbone_all_nodes_dict[snap_id]))
		assortativity_backbone_all_nodes.append(assortativity_backbone_all_nodes_dict[snap_id])
	
	#assortativity_backbone_nodes_in_top10_dict = {}
	print("\nAssortativity backbone top 10")
	for snap_id in networks:
		print("%s\t%.4f" % (snap_id,assortativity_backbone_nodes_in_top10_dict[snap_id]))
		
	#assortativity_backbone_nodes_in_top50_dict = {}
	print("\nAssortativity backbone top 50")
	for snap_id in networks:
		print("%s\t%.4f" % (snap_id,assortativity_backbone_nodes_in_top50_dict[snap_id]))
		
	dblp_plot.plot_assortativity(assortativity_full_network_all_nodes,assortativity_full_network_backbone,assortativity_backbone_all_nodes)
	
	
	
	eigenvector_all_nodes = []
	egenvector_backbone_nodes = []
	
	degree_all_nodes = []
	degree_backbone_nodes = []
	
	closeness_all_nodes = []
	closeness_backbone_nodes = []
	
	betweenness_all_nodes = []
	betweenness_backbone_nodes = []
	
	print("\nEigenvector all")
	for snap_id in networks:
		print("%s\t%.4f" % (snap_id,eigenvector_all_dict[snap_id]))
		eigenvector_all_nodes.append(eigenvector_all_dict[snap_id])
		
	print("\nDegree all")
	for snap_id in networks:
		print("%s\t%.4f" % (snap_id,degree_all_dict[snap_id]))
		degree_all_nodes.append(degree_all_dict[snap_id])
		
	print("\nCloseness all")
	for snap_id in networks:
		print("%s\t%.4f" % (snap_id,closeness_all_dict[snap_id]))
		closeness_all_nodes.append(closeness_all_dict[snap_id])
		
	print("\nBetweenness all")
	for snap_id in networks:
		print("%s\t%.4f" % (snap_id,betweenness_all_dict[snap_id]))
		betweenness_all_nodes.append(betweenness_all_dict[snap_id])
	
	print("\nEigenvector backbone")
	for snap_id in networks:
		print("%s\t%.4f" % (snap_id,eigenvector_backbone_dict[snap_id]))
		egenvector_backbone_nodes.append(eigenvector_backbone_dict[snap_id])
		
	print("\nDegree backbone")
	for snap_id in networks:
		print("%s\t%.4f" % (snap_id,degree_backbone_dict[snap_id]))
		degree_backbone_nodes.append(degree_backbone_dict[snap_id])
		
	print("\nCloseness backbone")
	for snap_id in networks:
		print("%s\t%.4f" % (snap_id,closeness_backbone_dict[snap_id]))
		closeness_backbone_nodes.append(closeness_backbone_dict[snap_id])
		
	print("\nBetweenness backbone")
	for snap_id in networks:
		print("%s\t%.4f" % (snap_id,betweenness_backbone_dict[snap_id]))
		betweenness_backbone_nodes.append(betweenness_backbone_dict[snap_id])
		
		
	dblp_plot.plot_centrality(eigenvector_all_nodes,egenvector_backbone_nodes,'eigenvector')
	dblp_plot.plot_centrality(degree_all_nodes,degree_backbone_nodes,'degree')
	dblp_plot.plot_centrality(closeness_all_nodes,closeness_backbone_nodes,'closeness')
	dblp_plot.plot_centrality(betweenness_all_nodes,betweenness_backbone_nodes,'betweenness')
	
	#"""
	# Esse trecho eu executo uma vez só, para salvar o pickle. Nas seguintes é só carregar o pickle.
	with open('data/all_topological_ranks_ranked_simple.pickle', 'wb') as handle:
		pickle.dump(all_topological_ranks,handle)
	#"""
		
	return all_topological_ranks
	

	

