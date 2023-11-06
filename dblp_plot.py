import matplotlib.pyplot as plt
import numpy as np
import random
import networkx as nx
import os

import dblp_network_analysis
#import dblp_community_analysis

def ccdf(x, xlabel, ylabel, parameters):
	type_plot = 'frequency'
	
	if type_plot == 'frequency':
		x, y = sorted(x,reverse=True), np.arange(len(x))# / len(x)
	elif tye_plot == 'probability':
		x, y = sorted(x,reverse=True), np.arange(len(x)) / len(x)
	
	plt.plot(x, y,color = parameters["color"],label = parameters["label"], linewidth=1.8, alpha=0.8)
	plt.legend(loc="upper right",fontsize=16)
	plt.xlabel(xlabel,fontsize=16)
	plt.ylabel(ylabel,fontsize=16)
	plt.xticks(np.arange(0, max(x), 5))
	
	plt.tick_params(axis='both', which='major', labelsize=16)
	plt.tick_params(axis='both', which='minor', labelsize=12)
	
	if parameters["log"] == 1:
		plt.yscale('log')
		plt.xscale('log')
	plt.tight_layout()
	
	#plt.show()
	
def plot_edge_weights_distribution(networks,name,filenames):
	edge_weights = dblp_network_analysis.get_edge_weigths(networks)
	
	for snap_id in networks:
		graph = networks[snap_id]
		
		if name == 'full':
			color = 'tab:blue'
		elif name == 'backbone':
			color = 'tab:red'
			
		plt.xticks(np.arange(0, max(edge_weights[snap_id]), 5))
		plt.hist(edge_weights[snap_id], density=False, bins='auto', color=color,alpha=0.7,label=['%s - %s' % (filenames[snap_id],name)])
		plt.legend(prop={'size': 10})
		plt.xlabel('edge weight')
		plt.tight_layout()
		
		"""
		plt.title("%s" % (filenames[snap_id]),fontsize=12)	
		plt.savefig("results/dist_edge_weight/histogram_%s_snapshot%s.png" % (name,snap_id),bbox_inches='tight')
		plt.clf()
		plt.close()
		"""
	
		parameters = {}
		parameters['color'] = color
		parameters['label'] = "%s - %s" % (filenames[snap_id],name)
		parameters['log'] = 1
		#ccdf(edge_weights[snap_id], 'edge weight', 'P(>weight)',parameters)
		ccdf(edge_weights[snap_id], 'edge weight', 'freq. weight',parameters) 
		plt.title("%s" % (filenames[snap_id]),fontsize=12)
		plt.savefig("results/dist_edge_weight/ccdf_log_%s_snapshot%s.png" % (name,snap_id),bbox_inches='tight')
		plt.clf()
		plt.close()
		
		parameters['log'] = 0
		#ccdf(edge_weights[snap_id], 'edge weight', 'P(>weight)',parameters)
		ccdf(edge_weights[snap_id], 'edge weight', 'freq. weight',parameters)
		plt.title("%s" % (filenames[snap_id]),fontsize=12)
		plt.savefig("results/dist_edge_weight/ccdf_%s_snapshot%s.png" % (name,snap_id),bbox_inches='tight')
		plt.clf()
		plt.close()
		
		
def plot_edge_weights_distribution_both(networks,backbones,name,filenames):
	edge_weights = dblp_network_analysis.get_edge_weigths(networks)
	edge_weights_backbones = dblp_network_analysis.get_edge_weigths(backbones)
	
	for snap_id in networks:
		graph = networks[snap_id]
		backbone = backbones[snap_id]
		
		plt.clf()
		plt.close()
		
		#print("+++++ WEIGHT:")
		#print(edge_weights_backbones[snap_id])
		
		plt.xticks(np.arange(0, max(edge_weights[snap_id]), 5))
		plt.hist(edge_weights[snap_id], density=False, bins=30, color='tab:blue', alpha=0.7, label = ['%s - full' % filenames[snap_id]])
		plt.hist(edge_weights_backbones[snap_id], density=False, bins=30, color='tab:red', alpha=0.7, label=['%s - backbone' % filenames[snap_id]])
		plt.legend(prop={'size': 10})
		plt.xlabel('Edge weight')
		
		plt.tight_layout()
		plt.title("%s" % (filenames[snap_id]),fontsize=12)
		plt.savefig("results/dist_edge_weight/histogram_%s_snapshot%s.png" % (name,snap_id),bbox_inches='tight')
		plt.clf()
		plt.close()
	
		parameters = {} # aquiiii
		#plt.legend(prop={'size': 20})
		parameters['color'] = 'tab:blue'
		#parameters['label'] = "%s - full network" % filenames[snap_id]
		parameters['label'] = "Full network"
		parameters['log'] = 1
		#ccdf(edge_weights[snap_id], 'edge weight', 'P(>weight)',parameters)
		ccdf(edge_weights[snap_id], 'Edge weight', 'Edge weight frequency',parameters)
		
		parameters['color'] = 'tab:red'
		#parameters['label'] = "%s - backbone" % filenames[snap_id]
		parameters['label'] = "Backbone"
		#ccdf(edge_weights_backbones[snap_id], 'edge weight', 'P(>weight)',parameters)
		ccdf(edge_weights_backbones[snap_id], 'Edge weight', 'Edge weight frequency',parameters)
		#plt.title("%s" % (filenames[snap_id]),fontsize=12)
		plt.savefig("results/dist_edge_weight/ccdf_log_%s_snapshot%s.png" % (name,snap_id),bbox_inches='tight')
		plt.clf()
		plt.close()
		
		
		parameters['color'] = 'tab:blue'
		parameters['label'] = "%s - full network" % filenames[snap_id]
		parameters['log'] = 0
		#ccdf(edge_weights[snap_id], 'edge weight', 'P(>weight)',parameters)
		ccdf(edge_weights[snap_id], 'edge weight', 'freq. weight',parameters)
		
		parameters['color'] = 'tab:red'
		parameters['label'] = "%s - backbone" % filenames[snap_id]
		parameters['log'] = 0
		#ccdf(edge_weights_backbones[snap_id], 'edge weight', 'P(>weight)',parameters)
		ccdf(edge_weights_backbones[snap_id], 'edge weight', 'freq. weight',parameters)
		plt.title("%s" % (filenames[snap_id]),fontsize=12)
		plt.savefig("results/dist_edge_weight/ccdf_%s_snapshot%s.png" % (name,snap_id),bbox_inches='tight')
		plt.clf()
		plt.close()
		
		
def plot_size_communities_distribution(communities,name,filenames):
	
	if name == 'full':
		color = 'tab:blue'
	elif name == 'backbone':
		color = 'tab:red'
	for snap_id in communities:
		size_communities = dblp_community_analysis.get_size_communities(communities[snap_id].communities)
		parameters = {}
		parameters['color'] = color
		parameters['label'] = "%s - %s" % (filenames[snap_id],name)
		parameters['log'] = 1
		ccdf(size_communities, 'community size', 'P(>size)',parameters)
		plt.title("%s" % (filenames[snap_id]),fontsize=12)
		plt.legend(prop={'size': 10})
		#plt.xlabel('edge weight')
		plt.savefig("results/dist_comm_size/comm_size_log_%s_snapshot%s.png" % (name,snap_id),bbox_inches='tight')
		plt.clf()
		plt.close()
		
		
		plt.hist(size_communities, density=False, bins=20, color=parameters['color'], alpha=0.7, label = parameters['label'])
		plt.title("%s" % (filenames[snap_id]),fontsize=12)
		plt.legend(prop={'size': 10})
		plt.xlabel('community size')
		plt.ylabel('frequency')
		plt.savefig("results/dist_comm_size/comm_size_hist_%s_snapshot%s.png" % (name,snap_id),bbox_inches='tight')
		
		#parameters['log'] = 0
		#ccdf(edge_weights[snap_id], 'edge weight', 'P(>weight)',parameters)
		#plt.savefig("results/ccdf_%s_%s.png" % (snap_id,name))
		plt.clf()
		plt.close()
		
		

	
	
	
def plot_networks_communities(networks,communities,name,filenames):
	for snap_id in communities:
	
		community_color = []
		for community_index in range(len(communities[snap_id].communities)):
			color = "#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
			community_color.append(color)
		
		node_to_color = dict()
		for community_index in range(len(communities[snap_id].communities)):
			for node in communities[snap_id].communities[community_index]:
				node_to_color[node] = community_color[community_index]
			
		color_map = []
		for node in networks[snap_id]:
			color_map.append(node_to_color[node])
		
		plt.figure(figsize=(15,15))
		options = {"edgecolors": "tab:gray","alpha": 0.65}
		#pos = nx.kamada_kawai_layout(networks[snap_id])
		pos = nx.spring_layout(networks[snap_id],k=0.1, iterations=20)
		
		
		#degrees = [val * 60 for (node, val) in networks[snap_id].degree()]
		
		#weights=nx.get_node_attributes(networks[snap_id],'weight')
		#print(weights)
		#print(type(weights))
		
		weights = []
		for node in networks[snap_id]:
			weights.append(20 * networks[snap_id].degree(weight='weight')[node])
		
		#nx.draw(networks[snap_id], pos, node_color=color_map, node_size=degrees, width=0.4, **options)
		nodelist = networks[snap_id].nodes()
		nx.draw_networkx_nodes(networks[snap_id],pos,nodelist=nodelist,node_color=color_map, node_size=weights, alpha=0.7)
		
		edge_weight = list(nx.get_edge_attributes(networks[snap_id],'weight').values())
		widths = []
		for value in edge_weight:
			widths.append(1.0 * value)
		
		edgelist = networks[snap_id].edges()
		nx.draw_networkx_edges(networks[snap_id],pos,edgelist=edgelist,width=widths,edge_color='tab:gray',alpha=0.4)
		plt.box(False)
		plt.tight_layout()
		plt.title("%s" % (filenames[snap_id]),fontsize=12)
		plt.savefig("results/plot_network/plot_%s_snapshot%s.png" % (name,snap_id),bbox_inches='tight')
		plt.clf()
		plt.close()
		
def plot_networks_evolution(networks,communities,name,filenames,top_ranked):
	
	graph_all = nx.Graph()
	for snap_id in networks:
		graph = networks[snap_id]
		nodelist = graph.nodes()
		add_list = []
		for node in nodelist:
			###if graph_all.has_node(node) == False: #add node
			add_list.append(node)
		graph_all.add_nodes_from(add_list)
		
		for user_from, user_to, data in graph.edges(data=True):
			if graph_all.has_edge(user_from,user_to):
				graph_all[user_from][user_to]['weight']+=data['weight']
			else:
				graph_all.add_edge(user_from,user_to,weight=data['weight'])
				
	# OK, tenho um graph_all com todos os backbones empilhados
	# Agora quero as posições dos nós
	
	pos_graph_all = nx.spring_layout(graph_all,k=0.1, iterations=20)
	#pos_graph_all = nx.spring_layout(graph_all,scale=2)
	#print(pos_graph_all)
	
	
	
	for snap_id in networks:
		add_list = []
		for node in graph_all.nodes():
			if networks[snap_id].has_node(node) == False:
				add_list.append(node)
			networks[snap_id].add_nodes_from(add_list)
	
	# OK, agora é plotar cada uma
		
	for snap_id in communities:
	
		"""
		community_color = []
		for community_index in range(len(communities[snap_id].communities)):
			color = "#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
			community_color.append(color)
		
		node_to_color = dict()
		for community_index in range(len(communities[snap_id].communities)):
			for node in communities[snap_id].communities[community_index]:
				node_to_color[node] = community_color[community_index]
			
		color_map = []
		for node in networks[snap_id]:
			color_map.append(node_to_color[node])
		"""
		
		plt.figure(figsize=(10,10))
		options = {"edgecolors": "tab:gray","alpha": 0.65}
				
		weights = []
		labels = {}

		for node in graph_all:
			if networks[snap_id].has_node(node):
				weights.append(8 * networks[snap_id].degree(weight='weight')[node])
			else:
				weights.append(0)
			#print(type(node))
			if node in top_ranked.keys():
				labels[node] = top_ranked[node]
		
		nodelist = graph_all.nodes()
		nx.draw_networkx_nodes(graph_all,pos_graph_all,nodelist=nodelist,node_color='tab:blue', linewidths=0, node_size=weights, alpha=0.7)
		
		edge_weight = list(nx.get_edge_attributes(networks[snap_id],'weight').values())
		widths = []
		for value in edge_weight:
			widths.append(1.0 * value)
		
		edgelist = networks[snap_id].edges()
		nx.draw_networkx_edges(networks[snap_id],pos_graph_all,edgelist=edgelist,width=widths,edge_color='tab:gray',alpha=0.4)
		
		plt.box(False)
		plt.draw()
		
		
		
		nx.draw_networkx_labels(networks[snap_id],pos_graph_all,labels=labels,font_size=10,font_color='tab:red')
		
		plt.title("%s" % (filenames[snap_id]),fontsize=16)
		plt.tight_layout()
		plt.savefig("results/evolution/plot_evolution_%s_snapshot%s.png" % (name,snap_id))
		plt.clf()
		plt.close()
		
	
	
	
	
	return graph_all
	

def plot_ego_network_many(networks,backbones,author_index,top_ranked_full,filenames,num_top_ranked=None):

	if num_top_ranked == None: # nesse caso vamos considerar como top_ranked todos que foram passados
		top_ranked = top_ranked_full
		num_top_ranked = len(top_ranked_full)
	else:
		top_ranked = {}
		count = 0
		for author in top_ranked_full:
			top_ranked[author] = top_ranked_full[author]
			count+=1
			if count == num_top_ranked:
				break
	
	
	subgraphs_full = {}
	subgraph_all_full = nx.Graph()
	
	subgraphs_backbone = {}
	subgraph_all_backbone = nx.Graph()
	
	
	for snap_id in networks:
		
		graph_full = networks[snap_id]
		graph_backbone = backbones[snap_id]
	
		neighbors_full = []
		neighbors_backbone = []
		
		for author_id in top_ranked: # considera um author ego (tem que dar uma conferida se é assim que se separa um author_id mesmo)
			
			if graph_full.has_node(author_id):
				neighbors_full_author = graph_full.neighbors(author_id)
				for node in neighbors_full_author:
					if node not in neighbors_full:
						neighbors_full.append(node)
				if author_id not in neighbors_full:
					neighbors_full.append(author_id)
			
			if graph_backbone.has_node(author_id):	
				neighbors_backbone_author = graph_backbone.neighbors(author_id)
				
				for node in neighbors_backbone_author:
					if node not in neighbors_backbone:
						neighbors_backbone.append(node)
				if author_id not in neighbors_backbone:
					neighbors_backbone.append(author_id)
			
			
		# end. OK, colocou os vizinhos de todos top_ranked
		
		
		
		subgraphs_full[snap_id] = graph_full.subgraph(neighbors_full)
		subgraphs_backbone[snap_id] = graph_backbone.subgraph(neighbors_backbone)
		
		
		# Adding nodes
		subgraph_all_full.add_nodes_from(neighbors_full)
		subgraph_all_backbone.add_nodes_from(neighbors_backbone)
			
		# Adding edges full
		for user_from, user_to, data in subgraphs_full[snap_id].edges(data=True):
			if subgraph_all_full.has_edge(user_from,user_to):
				subgraph_all_full[user_from][user_to]['weight']+=data['weight']
			else:
				subgraph_all_full.add_edge(user_from,user_to,weight=data['weight'])
				
		# Adding edges backbone
		for user_from, user_to, data in subgraphs_backbone[snap_id].edges(data=True):
			if subgraph_all_backbone.has_edge(user_from,user_to):
				subgraph_all_backbone[user_from][user_to]['weight']+=data['weight']
			else:
				subgraph_all_backbone.add_edge(user_from,user_to,weight=data['weight'])
	#end	
		
	pos_subgraph_all_full = nx.spring_layout(subgraph_all_full, k=0.3, iterations=20)
	pos_subgraph_all_backbone = nx.spring_layout(subgraph_all_backbone, k=0.3, iterations=20)
	
	ego_path = "results/ego_top_ranked/"
	if os.path.exists(ego_path) == False:
		os.mkdir(ego_path)
	
	
	for snap_id in subgraphs_full:
		
		# Resolvendo as cores dos nós
		node_colors_full = []
		for node in subgraph_all_full:
			#print("node:",type(node))
			#print("author_id:",type(author_id))
			#exit()()
			if node in top_ranked:
				node_colors_full.append('tab:red')
			elif backbones[snap_id].has_node(node):
				node_colors_full.append('tab:blue')
			else:
				node_colors_full.append('tab:gray')
				
		node_colors_backbone = []
		for node in subgraph_all_backbone:
			#print("node:",type(node))
			#print("author_id:",type(author_id))
			#exit()()
			if node in top_ranked:
				node_colors_backbone.append('tab:red')
			elif backbones[snap_id].has_node(node):
				node_colors_backbone.append('tab:blue')
			else:
				node_colors_backbone.append('tab:gray')
				
		
		plt.figure(figsize=(14,14))
		options = {"edgecolors": "tab:gray","alpha": 0.65}
		
		# Resolvendo pesos e labels		
		weights_full = []
		labels_full = {}
		for node in subgraph_all_full:
			if subgraphs_full[snap_id].has_node(node):
				weights_full.append(8 * subgraphs_full[snap_id].degree(weight='weight')[node])
			else:
				weights_full.append(0)
			labels_full[node] = author_index[node]
		
		weights_backbone = []
		labels_backbone = {}
		for node in subgraph_all_backbone:
			if subgraphs_backbone[snap_id].has_node(node):
				weights_backbone.append(8 * subgraphs_backbone[snap_id].degree(weight='weight')[node])
			else:
				weights_backbone.append(0)
			labels_backbone[node] = author_index[node]
		
		
		# Resolve os nós
		nodelist_full = subgraph_all_full.nodes()
		nodelist_backbone = subgraph_all_backbone.nodes()
		
		
		# Resolve as arestas
		edge_weight_full = list(nx.get_edge_attributes(subgraphs_full[snap_id],'weight').values())
		widths_full = []
		for value in edge_weight_full:
			widths_full.append(1.0 * value)
			
		edge_weight_backbone = list(nx.get_edge_attributes(subgraphs_backbone[snap_id],'weight').values())
		widths_backbone = []
		for value in edge_weight_backbone:
			widths_backbone.append(1.0 * value)
		
		edgelist_full = subgraphs_full[snap_id].edges()
		edgelist_backbone = subgraphs_backbone[snap_id].edges()
		
		# Plota full
		nx.draw_networkx_nodes(subgraph_all_full,pos_subgraph_all_full,nodelist=nodelist_full,node_color=node_colors_full, linewidths=0, node_size=weights_full, alpha=0.7)
		nx.draw_networkx_edges(subgraphs_full[snap_id],pos_subgraph_all_full,edgelist=edgelist_full,width=widths_full,edge_color='tab:gray',alpha=0.4)
		###nx.draw_networkx_labels(subgraphs_full[snap_id],pos_subgraph_all_full,labels=labels_full,font_size=7,font_color='dimgray')
		plt.box(False)
		plt.draw()
		plt.title("Ego - Top %s hyperprolific - %s (full network)" % (num_top_ranked,filenames[snap_id]),fontsize=16)
		plt.tight_layout()
		image_filename = "results/ego_top_ranked/%s_full_snapshot%03d.png" % (num_top_ranked,snap_id)
		plt.savefig(image_filename)
		#plt.show()
		plt.clf()
		plt.close()
		
		# Plota backbone
		
		plt.figure(figsize=(14,14))
		options = {"edgecolors": "tab:gray","alpha": 0.65}
		nx.draw_networkx_nodes(subgraph_all_backbone,pos_subgraph_all_backbone,nodelist=nodelist_backbone,node_color=node_colors_backbone, linewidths=0, node_size=weights_backbone, alpha=0.7)
		nx.draw_networkx_edges(subgraphs_backbone[snap_id],pos_subgraph_all_backbone,edgelist=edgelist_backbone,width=widths_backbone,edge_color='tab:gray',alpha=0.4)
		###nx.draw_networkx_labels(subgraphs_backbone[snap_id],pos_subgraph_all_backbone,labels=labels_backbone,font_size=7,font_color='dimgray')
		plt.box(False)
		plt.draw()
		plt.title("Ego - Top %s hyperprolific - %s (backbone)" % (num_top_ranked,filenames[snap_id]),fontsize=16)
		plt.tight_layout()
		image_filename = "results/ego_top_ranked/%s_backbone_snapshot%03d.png" % (num_top_ranked,snap_id)
		plt.savefig(image_filename)
		#plt.show()
		plt.clf()
		plt.close()
		
		
		
	generate_ego_many_gif('full',num_top_ranked)
	generate_ego_many_gif('backbone',num_top_ranked)
			
				
			
			
		
	

def plot_ego_network(author_name,author_index,networks,backbones,filenames):
	author_id = list(author_index.keys())[list(author_index.values()).index(author_name)]#find id from name
	
	##print("author_id",type(author_id),author_id)
	
	### problema com o tipo desse id
	
	subgraphs_full = {}
	subgraph_all_full = nx.Graph()
	
	subgraphs_backbone = {}
	subgraph_all_backbone = nx.Graph()
	
	for snap_id in networks:
		
		
		graph_full = networks[snap_id]
		graph_backbone = backbones[snap_id]
		
		neighbors_full = []
		neighbors_backbone = []
		if graph_full.has_node(author_id):
			# pega co-autores na rede full
			neighbors_full = [n for n in graph_full.neighbors(author_id)]
			neighbors_full.append(author_id)
		
		if graph_backbone.has_node(author_id):
			# pega co-autores na rede backbone
			neighbors_backbone = [n for n in graph_backbone.neighbors(author_id)]
			neighbors_backbone.append(author_id)
		
		##print("neighbors[0]:",type(neighbors[0]),neighbors[0])
		
		#exit()
		
		#print(neighbors)
		#print(len(neighbors))
		#print(graph_full.degree(author_id))
		
		subgraphs_full[snap_id] = graph_full.subgraph(neighbors_full)
		subgraphs_backbone[snap_id] = graph_backbone.subgraph(neighbors_backbone)
		
		
		# Adding nodes
		subgraph_all_full.add_nodes_from(neighbors_full)
		#subgraph_all.add_node(author_id)
		
		subgraph_all_backbone.add_nodes_from(neighbors_backbone)
		
		
		# Adding edges full
		for user_from, user_to, data in subgraphs_full[snap_id].edges(data=True):
			if subgraph_all_full.has_edge(user_from,user_to):
				subgraph_all_full[user_from][user_to]['weight']+=data['weight']
			else:
				subgraph_all_full.add_edge(user_from,user_to,weight=data['weight'])
				
		# Adding edges backbone
		for user_from, user_to, data in subgraphs_backbone[snap_id].edges(data=True):
			if subgraph_all_backbone.has_edge(user_from,user_to):
				subgraph_all_backbone[user_from][user_to]['weight']+=data['weight']
			else:
				subgraph_all_backbone.add_edge(user_from,user_to,weight=data['weight'])
		
		
	#end
	
	# subgraph_all está acumulado
	print(nx.info(subgraph_all_full))
	print(nx.info(subgraph_all_backbone))
	
		
	
	pos_subgraph_all_full = nx.spring_layout(subgraph_all_full, k=0.3, iterations=20)
	pos_subgraph_all_backbone = nx.spring_layout(subgraph_all_backbone, k=0.3, iterations=20)
	
	ego_path = "results/ego/%s/" % (author_id)
	if os.path.exists(ego_path) == False:
		os.mkdir(ego_path)
	
	for snap_id in subgraphs_full:
		
		# Resolvendo as cores dos nós
		node_colors_full = []
		for node in subgraph_all_full:
			#print("node:",type(node))
			#print("author_id:",type(author_id))
			#exit()()
			if node == author_id:
				node_colors_full.append('tab:red')
			elif backbones[snap_id].has_node(node):
				node_colors_full.append('tab:blue')
			else:
				node_colors_full.append('tab:gray')
				
		node_colors_backbone = []
		for node in subgraph_all_backbone:
			#print("node:",type(node))
			#print("author_id:",type(author_id))
			#exit()()
			if node == author_id:
				node_colors_backbone.append('tab:red')
			elif backbones[snap_id].has_node(node):
				node_colors_backbone.append('tab:blue')
			else:
				node_colors_backbone.append('tab:gray')
				
		
		plt.figure(figsize=(10,10))
		options = {"edgecolors": "tab:gray","alpha": 0.65}
		
		# Resolvendo pesos e labels		
		weights_full = []
		labels_full = {}
		for node in subgraph_all_full:
			if subgraphs_full[snap_id].has_node(node):
				weights_full.append(8 * subgraphs_full[snap_id].degree(weight='weight')[node])
			else:
				weights_full.append(0)
			labels_full[node] = author_index[node]
		
		weights_backbone = []
		labels_backbone = {}
		for node in subgraph_all_backbone:
			if subgraphs_backbone[snap_id].has_node(node):
				weights_backbone.append(8 * subgraphs_backbone[snap_id].degree(weight='weight')[node])
			else:
				weights_backbone.append(0)
			labels_backbone[node] = author_index[node]
		
		
		# Resolve os nós
		nodelist_full = subgraph_all_full.nodes()
		nodelist_backbone = subgraph_all_backbone.nodes()
		
		
		# Resolve as arestas
		edge_weight_full = list(nx.get_edge_attributes(subgraphs_full[snap_id],'weight').values())
		widths_full = []
		for value in edge_weight_full:
			widths_full.append(1.0 * value)
			
		edge_weight_backbone = list(nx.get_edge_attributes(subgraphs_backbone[snap_id],'weight').values())
		widths_backbone = []
		for value in edge_weight_backbone:
			widths_backbone.append(1.0 * value)
		
		edgelist_full = subgraphs_full[snap_id].edges()
		edgelist_backbone = subgraphs_backbone[snap_id].edges()
		
		# Plota full
		nx.draw_networkx_nodes(subgraph_all_full,pos_subgraph_all_full,nodelist=nodelist_full,node_color=node_colors_full, linewidths=0, node_size=weights_full, alpha=0.7)
		nx.draw_networkx_edges(subgraphs_full[snap_id],pos_subgraph_all_full,edgelist=edgelist_full,width=widths_full,edge_color='tab:gray',alpha=0.4)
		nx.draw_networkx_labels(subgraphs_full[snap_id],pos_subgraph_all_full,labels=labels_full,font_size=7,font_color='dimgray')
		plt.box(False)
		plt.draw()
		plt.title("%s - %s (full network)" % (author_name,filenames[snap_id]),fontsize=16)
		plt.tight_layout()
		image_filename = "results/ego/%s/full_author_%s_snapshot%03d.png" % (author_id,author_id,snap_id)
		plt.savefig(image_filename)
		#plt.show()
		plt.clf()
		plt.close()
		
		# Plota backbone
		nx.draw_networkx_nodes(subgraph_all_backbone,pos_subgraph_all_backbone,nodelist=nodelist_backbone,node_color=node_colors_backbone, linewidths=0, node_size=weights_backbone, alpha=0.7)
		nx.draw_networkx_edges(subgraphs_backbone[snap_id],pos_subgraph_all_backbone,edgelist=edgelist_backbone,width=widths_backbone,edge_color='tab:gray',alpha=0.4)
		nx.draw_networkx_labels(subgraphs_backbone[snap_id],pos_subgraph_all_backbone,labels=labels_backbone,font_size=7,font_color='dimgray')
		plt.box(False)
		plt.draw()
		plt.title("%s - %s (backbone)" % (author_name,filenames[snap_id]),fontsize=16)
		plt.tight_layout()
		image_filename = "results/ego/%s/backbone_author_%s_snapshot%03d.png" % (author_id,author_id,snap_id)
		plt.savefig(image_filename)
		#plt.show()
		plt.clf()
		plt.close()
		
		
		
	generate_ego_gif(author_id,'full')
	generate_ego_gif(author_id,'backbone')


def generate_ego_gif(author_id,name):
	import glob
	from PIL import Image

	# filepaths
	fp_in = "results/ego/%s/%s_*.png" % (author_id,name)
	fp_out = "results/ego/%s/%s_author_%s.gif" % (author_id,name,author_id)

	# https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#gif
	imgs = (Image.open(f) for f in sorted(glob.glob(fp_in)))
	img = next(imgs)  # extract first image from iterator
	img.save(fp=fp_out, format='GIF', append_images=imgs,save_all=True, duration=800, loop=0)
	
def generate_ego_many_gif(name,num_top_ranked):
	import glob
	from PIL import Image

	# filepaths
	fp_in = "results/ego_top_ranked/%s_%s*.png" % (num_top_ranked,name)
	fp_out = "results/ego_top_ranked/%s_%s.gif" % (num_top_ranked,name)

	# https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#gif
	imgs = (Image.open(f) for f in sorted(glob.glob(fp_in)))
	img = next(imgs)  # extract first image from iterator
	img.save(fp=fp_out, format='GIF', append_images=imgs,save_all=True, duration=800, loop=0)
	
	
	
def plot_probability_presence_backbone(values_random,values_top_percent,values_top):
	# Plot presence in backbone
	#values_random = [0.18]
	#values_top_percent = [0.51,0.4,0.31]
	#values_top = [0.82,1,0.56,0.64,0.91,0.73,0.91,0.8,0.64,0.8,0.82,0.67,0.5,1,0.82,0.88,0.67,0.56,0.64,0.73,0.5,1,0.6,0.73,0.71,0.71,0.67,0.73,1,0.73,0.64,0.33,0.73,0.91,0.5,0.82,0.64,0.78,0,0.55,0.33,0.73,0.64,0.56,0.7,0.64,0.64,0.91,0.36,0.91]


	labels_random = ['Random']
	labels_top_percent = ['Top 10%','Top 25%', 'Top 50%']
	labels_top = ['Top 1','Top 2','Top 3','Top 4','Top 5','Top 6','Top 7','Top 8','Top 9','Top 10','Top 11','Top 12','Top 13','Top 14','Top 15','Top 16','Top 17','Top 18','Top 19','Top 20','Top 21','Top 22','Top 23','Top 24','Top 25','Top 26','Top 27','Top 28','Top 29','Top 30','Top 31','Top 32','Top 33','Top 34','Top 35','Top 36','Top 37','Top 38','Top 39','Top 40','Top 41','Top 42','Top 43','Top 44','Top 45','Top 46','Top 47','Top 48','Top 49','Top 50']

	pos_random = [0]
	pos_top_percent = [1,2,3]
	pos_top = [4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53]

	#fig, ax = plt.subplots()

	plt.figure(figsize=(20,6))
	barWidth = 0.60

	labels_all = labels_random+labels_top_percent+labels_top
	pos_all = pos_random+pos_top_percent+pos_top
	values_all = values_random+values_top_percent+values_top

	#pl1 = plt.bar(pos_random,values_random,color='tab:blue',alpha=0.7,width = barWidth)
	#pl2 = plt.bar(labels_top_percent,values_top_percent,color='tab:green',alpha=0.7,width = barWidth)#, labels=labels_top_percent)
	#pl3 = plt.bar(pos_top,values_top,color='tab:red',alpha=0.7,width = barWidth)

	color = ['tab:blue','tab:green','tab:green','tab:green','tab:red','tab:red','tab:red','tab:red','tab:red','tab:red','tab:red','tab:red','tab:red','tab:red','tab:red','tab:red','tab:red','tab:red','tab:red','tab:red','tab:red','tab:red','tab:red','tab:red','tab:red','tab:red','tab:red','tab:red','tab:red','tab:red','tab:red','tab:red','tab:red','tab:red','tab:red','tab:red','tab:red','tab:red','tab:red','tab:red','tab:red','tab:red','tab:red','tab:red','tab:red','tab:red','tab:red','tab:red','tab:red','tab:red','tab:red','tab:red','tab:red','tab:red']

	pl = plt.bar(labels_all,values_all,color=color,alpha=0.7,width = barWidth)
	
	plt.xlabel('Selected authors',fontsize=16)
	plt.ylabel('Probability of presence in backbone',fontsize=16)

	plt.xlim((-1, 54))

	#plt.xticks(labels_random+labels_top_percent+labels_top)

	#plt.set_xlabel("Number of hits in backbones")
	#plt.set_ylabel("Probability of presence in backbone")

	plt.tick_params(axis='both', which='major', labelsize=16)
	plt.tick_params(axis='both', which='minor', labelsize=12)

	plt.xticks(rotation='vertical')

	plt.savefig("results/probability_backbone/hits.png",bbox_inches='tight')

	plt.clf()
	plt.close()
#end

def plot_probability_hops(values_random,values_top_percent,values_hop):

	# Fig. 7
	#values_random = [0.18]
	#values_top_percent = [0.44,0.33,0.24]
	#values_top = [0.45,0.35,0.3,0.26]


	labels_random = ['Random']
	labels_top_percent = ['Top 10%','Top 25%', 'Top 50%']
	labels_hop = ['Hop 1\n(from top 10)','Hop 2\n(from top 10)','Hop 3\n(from top 10)','Hop 4\n(from top 10)']

	pos_random = [0]
	pos_top_percent = [1,2,3]
	pos_top = [4,5,6,7]

	#fig, ax = plt.subplots()

	plt.figure(figsize=(8,6))
	barWidth = 0.60


	pos_all = pos_random+pos_top_percent+pos_top
	labels_all = labels_random+labels_top_percent+labels_hop
	values_all = values_random+values_top_percent+values_hop
	colors = ['tab:blue','tab:green','tab:green','tab:green','tab:red','tab:red','tab:red','tab:red']

	#pl1 = plt.bar(pos_random,values_random,color='tab:blue',alpha=0.7,width = barWidth)
	#pl2 = plt.bar(pos_top_percent,values_top_percent,color='tab:green',alpha=0.7,width = barWidth)#, labels=labels_top_percent)
	#pl3 = plt.bar(pos_top,values_top,color='tab:red',alpha=0.7,width = barWidth)

	pl = plt.bar(labels_all,values_all,color=colors,alpha=0.7,width = barWidth)

	plt.xticks(rotation='vertical')


	#pl1.set_yticks(pos_random)
	#.set_yticks(pos_top_percent)
	#plt.xticks(pos_random:pos_top_percent:pos_top)

	#plt.bar_label(labels_top_percent)

	#plt.legend(prop={'size': 14})
	###plt.xlabel('Probability of presence in backbone',fontsize=16)
	plt.ylabel('Probability of presence in backbone',fontsize=16)

	##plt.xlim((-1, 54))

	#plt.xticks(labels_random+labels_top_percent+labels_top)

	#plt.set_xlabel("Number of hits in backbones")
	#plt.set_ylabel("Probability of presence in backbone")

	plt.tick_params(axis='both', which='major', labelsize=16)
	plt.tick_params(axis='both', which='minor', labelsize=12)

	plt.savefig("results/probability_backbone/hops.png",bbox_inches='tight')

	plt.clf()
	plt.close()
#end

def plot_assortativity(full_network_all_nodes,full_network_backbone,backbone_all_nodes):
	#full_network_all_nodes = [0.2354,0.2583,0.2456,0.2347,0.2186,0.2123,0.1572,0.1897,0.1732,0.1433,0.135]
	#full_network_backbone = [0.3483,0.3268,0.2704,0.3285,0.1693,0.1454,0.0646,0.1319,0.0847,0.0654,0.0674]
	#backbone_all_nodes = [0.6586,0.5689,0.5102,0.3056,0.2106,0.343,0.1904,0.4349,0.2927,0.2225,0.2193]

	years = [2010,2011,2012,2013,2014,2015,2016,2017,2018,2019,2020]

	plt.figure(figsize=(10,6))

	plt.plot(years, full_network_all_nodes,color = 'tab:blue',label = 'Full network (all nodes)', linewidth=1.8, alpha=0.7, marker='o')
	plt.plot(years, full_network_backbone,color = 'tab:red',label = 'Full network (nodes in backbone)', linewidth=1.8, alpha=0.7, marker='^')
	plt.plot(years, backbone_all_nodes,color = 'gold',label = 'Backbone (all nodes)', linewidth=1.8, alpha=0.7, marker='s')

	plt.legend(loc="upper right",fontsize=16)
	#plt.xlabel(xlabel,fontsize=16)
	plt.ylabel("Assortativity",fontsize=16)
	#plt.xticks(np.arange(0, max(x), 5))
	plt.tick_params(axis='both', which='major', labelsize=16)
	plt.tick_params(axis='both', which='minor', labelsize=12)

	plt.xticks(years)

	plt.savefig("results/network_topology/assortativity.png",bbox_inches='tight')


	plt.clf()
	plt.close()


#end


def plot_centrality(all_nodes,backbone_nodes,centrality):

	#all_nodes = [0.0032,0.0029,0.003,0.0033,0.0035,0.0039,0.0032,0.005,0.005,0.0047,0.0048]
	#backbone_nodes = [0.0185,0.0179,0.0131,0.0105,0.0114,0.0129,0.0078,0.0139,0.0127,0.0114,0.0116]

	years = [2010,2011,2012,2013,2014,2015,2016,2017,2018,2019,2020]

	plt.figure(figsize=(10,6))

	plt.plot(years, all_nodes,color = 'tab:blue',label = 'All nodes', linewidth=1.8, alpha=0.7, marker='o')
	plt.plot(years, backbone_nodes,color = 'tab:red',label = 'Nodes in backbone', linewidth=1.8, alpha=0.7, marker='^')

	plt.legend(loc="upper right",fontsize=16)
	#plt.xlabel(xlabel,fontsize=16)
	plt.ylabel("Average centrality",fontsize=16)
	#plt.xticks(np.arange(0, max(x), 5))
	plt.tick_params(axis='both', which='major', labelsize=16)
	plt.tick_params(axis='both', which='minor', labelsize=12)

	plt.xticks(years)

	plt.savefig("results/network_topology/%s.png" % centrality,bbox_inches='tight')
	
#end


def plot_ao_similarity(eigenvector,degree,closeness,betweenness):
	#eigenvector = [0.1683,0.187,0.1749,0.2189,0.2158,0.2635,0.2276,0.3259,0.3636,0.3137,0.3273]
	#degree = [0.1854,0.2469,0.2485,0.2837,0.3259,0.3742,0.4033,0.4545,0.5171,0.5151,0.5466]
	#closeness = [0.172,0.1859,0.2126,0.2259,0.2478,0.2888,0.2956,0.3466,0.4164,0.4333,0.4592]
	#betweenness = [0.1728,0.2205,0.2052,0.2424,0.2705,0.3208,0.3368,0.3731,0.4184,0.4552,0.476]

	years = [2010,2011,2012,2013,2014,2015,2016,2017,2018,2019,2020]

	plt.figure(figsize=(10,6))

	plt.plot(years,eigenvector,color = 'tab:blue',label = 'Eigenvector', linewidth=1.8, alpha=0.7, marker='o')
	plt.plot(years,degree,color = 'tab:red',label = 'Degree', linewidth=1.8, alpha=0.7, marker='^')
	plt.plot(years,closeness,color = 'gold',label = 'Closeness', linewidth=1.8, alpha=0.7, marker='s')
	plt.plot(years,betweenness,color = 'tab:green',label = 'Betweenness', linewidth=1.8, alpha=0.7, marker='P')

	plt.legend(fontsize=16)
	plt.xlabel("Snapshot",fontsize=16)
	plt.ylabel("Correlation hyperprolific X topological rank",fontsize=16)
	#plt.xticks(np.arange(0, max(x), 5))
	plt.tick_params(axis='both', which='major', labelsize=16)
	plt.tick_params(axis='both', which='minor', labelsize=12)

	plt.xticks(years)

	plt.savefig("results/correlation/rbo.png",bbox_inches='tight')


	plt.clf()
	plt.close()

#end

def plot_tlcc(top,mid,random,prop):

	#top = [0.9428571429,0.9642857143,0.9761904762,0.9833333333,0.9878787879,0.9909090909,1,1,1,1,1]
	#mid = [0.9856107606,0.9549937105,0.9518763181,0.9664206778,0.9634325446,0.9382372119,0.9024558013,0.865576781,0.7711403083,0.6910233191,0.7944613466]
	#random = [0.8285714286,0.8288624657,0.8383383833,0.865576781,0.841479058,0.716902451,0.585376736,0.4033755873,0.03614720195,-0.2727723628,-0.05797710357]
	
	max_top = max(top)
	max_mid = max(mid)
	max_random = max(random)
	max_concat = [max_top,max_mid,max_random]

	min_top = min(top)
	min_mid = min(mid)
	min_random = min(random)
	min_concat = [min_top,min_mid,min_random]
	
	horizontal = [0,0,0,0,0,0,0,0,0,0,0]
	vertical = [min(min_concat)-0.02,max(max_concat)+0.02]
	vert_zero = [0,0]

	offset = [-5,-4,-3,-2,-1,0,1,2,3,4,5]

	plt.figure(figsize=(10,6))

	plt.plot(offset,top,color = 'tab:blue',label = 'Top-ranked', linewidth=1.8, alpha=0.7, marker='o')
	plt.plot(offset,mid,color = 'tab:red',label = 'Mid-ranked', linewidth=1.8, alpha=0.7, marker='^')
	plt.plot(offset,random,color = 'gold',label = 'Random', linewidth=1.8, alpha=0.7, marker='s')
	plt.plot(offset,horizontal,color = 'black',linewidth=0.8, alpha=0.7)
	plt.plot(vert_zero,vertical,color = 'black',linewidth=0.8, alpha=0.7)


	plt.legend(fontsize=16)
	plt.xlabel("Offset",fontsize=16)
	plt.ylabel("TLCC (speed)",fontsize=16)
	#plt.xticks(np.arange(0, max(x), 5))
	plt.tick_params(axis='both', which='major', labelsize=16)
	plt.tick_params(axis='both', which='minor', labelsize=12)

	plt.xticks(offset)

	plt.savefig("results/correlation/tlcc_%s.png" % prop,bbox_inches='tight')


	plt.clf()
	plt.close()

	
	
