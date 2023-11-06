import csv
import networkx as nx
import itertools

#from cdlib import algorithms, evaluation

def csv_parser(prefix,filenames):
	network_content = {}
	
	num_snaps = len(filenames)
	for snap_id in range(num_snaps):
		csv_file = "%s/%s" % (prefix,filenames[snap_id])
		with open(csv_file, newline='') as csvfile:
			dblp_papers = csv.DictReader(csvfile, delimiter=';')
			for row in dblp_papers:
				
				#snap_id = row[1]
				#print(row)
				#snap_id = 0 # ajustar quando for trabalhar com mais de um snapshot
				if snap_id not in network_content:
					network_content[snap_id] = {}

				paper_id = int(row['paper'])
				if paper_id not in network_content[snap_id]:
					network_content[snap_id][paper_id] = set()

				author_id = row['author']
				network_content[snap_id][paper_id].add(int(author_id))
			
	return network_content
	
def csv_read_author_index(filename):

	csv_file = "%s" % (filename)
	author_index = dict()
	with open(csv_file, newline='') as csvfile:
		author_index_csv = csv.DictReader(csvfile, delimiter=';')
		for row in author_index_csv:
			author_id = int(row['id'])
			author_name = row['author']
			
			author_index[author_id] = author_name
			
	return author_index
	
def network_generator(network_content):
	full_networks = {}
	for snapshot in network_content:
		graph = nx.Graph(name='snapshot%d' % snapshot)

		for messageID in network_content[snapshot]:
			for pair in itertools.combinations(network_content[snapshot][messageID], 2):
				if graph.has_edge(pair[0], pair[1]):
					graph[pair[0]][pair[1]]['weight'] += 1
				else:
					graph.add_edge(pair[0], pair[1], weight=1)

		full_networks[snapshot] = graph
	return full_networks
	
def filter_authors(prefix,min_papers):
	
	authors_count = {}
	filenames = ['graph-2010','graph-2011','graph-2012','graph-2013','graph-2014','graph-2015','graph-2016','graph-2017','graph-2018','graph-2019','graph-2020','graph-2021']
	
	for filename in filenames:
		csv_file = "%s/%s" % (prefix,filename)
		#print(filename)
		with open(csv_file, newline='') as csvfile:
			dblp_papers = csv.DictReader(csvfile, delimiter=';')
			for row in dblp_papers:
				
				snap_id = 0 # ajustar quando for trabalhar com mais de um snapshot
				
				author_id = int(row['author'])
				if author_id not in authors_count:
					authors_count[author_id] = 0
					
				authors_count[author_id] += 1

	authors_ok_list = []
	authors_not_ok_list = []
	for author_id in authors_count:
		if authors_count[author_id] >= min_papers: # <=== Aqui está o filtro!!
			authors_ok_list.append(author_id)
		else:
			authors_not_ok_list.append(author_id)
		
			
	return [authors_ok_list,authors_not_ok_list]

def filter_null_degree(networks):
	
	for snap_id in networks:
		#remove = [node for node, degree in networks[snap_id].degree() if degree == 0]
		#networks[snap_id].remove_nodes_from(remove)
		graph = networks[snap_id]
		remove_list = []
		for (node, val) in graph.degree():
			if val == 0:
				#print(" ========= ACHOU ========")
				#exit()
				remove_list.append(node)
		graph.remove_nodes_from(remove_list)
			
	return networks
	
def write_author_names(author_index,communities,name):
	#for line in author_index:
	#	print(line,author_index[line])
	
	for snap_id in communities:
		with open('results/communities_info/communities_%s_snapshot%s.csv' % (name,snap_id), 'w', newline='') as csvfile:
			fieldnames = ['id', 'author','community']
			writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

			writer.writeheader()
			community_index = 0
			
			# falta ordenar as comunidades
			# Esse código funciona, mas acho que elas já vêm ordenadas, então deixei pra lá.
			#comm = 0
			#for community in communities[snap_id].communities:
			#	print(comm,len(communities[snap_id].communities[comm]))
			#	comm+=1
			#exit()
				
			for community in communities[snap_id].communities:
				for node in community:
					writer.writerow({'id': node, 'author': author_index[node], 'community': community_index})
					#print(node,author_index[str(node)],community_index)
				community_index+=1

def export_networks(networks):
	for snap_id in networks:
		graph = networks[snap_id]
		with open('networks/network_snapshot%s.csv' % (snap_id), 'w', newline='') as csvfile:
			spamwriter = csv.writer(csvfile, delimiter=',',)
			for user_from, user_to, data in graph.edges(data=True):
				data = [user_from,user_to,data['weight']]
				spamwriter.writerow(data)
				#print(user_from,',',user_to,',',data['weight'])
				
def export_gml(networks,type_graph):
	for snap_id in networks:
		graph = networks[snap_id]
		filename = 'gml/%s_%s.gml' % (type_graph,snap_id)
		nx.write_gml(graph, filename)
		
def export_gml_snapshots_is_backbone(networks,backbones):
	for snap_id in networks:
		graph = networks[snap_id]
		nx.set_node_attributes(graph, 0, "is_backbone")
		for node in graph:
			if backbones[snap_id].has_node(node):
				graph.nodes[node]["is_backbone"] = 1
		filename = 'gml/graph_%s.gml' % (snap_id)
		nx.write_gml(graph, filename)
		
def export_single_network_gml(full_networks,backbones):
	single_net=nx.Graph()
	weights_dict = {}
	for snap_id in full_networks:
		#print("\n\n\nSnapshot",snap_id,":::")
		graph = full_networks[snap_id]
		for user_from, user_to, data in graph.edges(data=True):
			#print((user_from,user_to),data['weight'],"snap",snap_id)
			if (user_from,user_to) not in weights_dict:
				weights_dict[(user_from,user_to)] = data['weight']
				#if user_from == 5168 and user_to == 95843:
				#	print("presente primeira vez. peso",weights_dict[(user_from,user_to)])
			else:
				weights_dict[(user_from,user_to)] += data['weight']
				#if user_from == 5168 and user_to == 95843:
				#	print("presente repetindo. peso",weights_dict[(user_from,user_to)])
			
	single_net.add_edges_from(weights_dict.keys())
	
	nx.set_edge_attributes(single_net, 0, "weight")
	
	for edge in weights_dict:
		single_net[edge[0]][edge[1]]['weight'] += weights_dict[edge]
		
		
	nx.set_node_attributes(single_net, 0, "backbone_hits")
	for snap_id in full_networks:
		nx.set_node_attributes(single_net, 0, "backbone%d" % snap_id)
		for node in single_net:
			if backbones[snap_id].has_node(node):
				single_net.nodes[node]["backbone%d" % snap_id] = 1
				single_net.nodes[node]["backbone_hits"] += 1
	
	"""
	print("\n\n\n\nBackbone hits+++")
	for node in single_net:
		print(node,single_net.nodes[node]["backbone_hits"])
	"""
	
	"""
	print("\n\n\n\nSingle net---")
	for user_from, user_to, data in single_net.edges(data=True):
		print((user_from,user_to),data['weight'])
	"""
	
	filename = 'gml/single_networkk.gml'# % (type_graph,snap_id)
	nx.write_gml(single_net, filename)
	
	return single_net
	
	
def export_single_network_gml_with_rank(full_networks,backbones,all_ranked_simple,all_features):
	print("estou dentro do export_single_network_gml_with_rank")
	
	single_net=nx.Graph()
	weights_dict = {}
	for snap_id in full_networks:
		#print("\n\n\nSnapshot",snap_id,":::")
		graph = full_networks[snap_id]
		for user_from, user_to, data in graph.edges(data=True):
			#print((user_from,user_to),data['weight'],"snap",snap_id)
			if (user_from,user_to) not in weights_dict:
				weights_dict[(user_from,user_to)] = data['weight']
				#if user_from == 5168 and user_to == 95843:
				#	print("presente primeira vez. peso",weights_dict[(user_from,user_to)])
			else:
				weights_dict[(user_from,user_to)] += data['weight']
				#if user_from == 5168 and user_to == 95843:
				#	print("presente repetindo. peso",weights_dict[(user_from,user_to)])
	single_net.add_edges_from(weights_dict.keys())
	
	nx.set_edge_attributes(single_net, 0, "weight")
	
	for edge in weights_dict:
		single_net[edge[0]][edge[1]]['weight'] += weights_dict[edge]
		
		
	nx.set_node_attributes(single_net, 0, "backbone_hits")
	for snap_id in full_networks:
		nx.set_node_attributes(single_net, 0, "backbone%d" % snap_id)
		for node in single_net:
			if backbones[snap_id].has_node(node):
				single_net.nodes[node]["backbone%d" % snap_id] = 1
				single_net.nodes[node]["backbone_hits"] += 1
				
	
	
	nx.set_node_attributes(single_net, 0, "rank")
	nx.set_node_attributes(single_net, 0, "speed_mean")
	nx.set_node_attributes(single_net, 0, "speed_std")
	nx.set_node_attributes(single_net, 0, "acceleration_mean")
	nx.set_node_attributes(single_net, 0, "acceleration_std")
	
	
	for node in single_net:
		try:
			single_net.nodes[node]["rank"] = all_ranked_simple[node]
			single_net.nodes[node]["speed_mean"] = all_features[node]["speed_mean"]
			single_net.nodes[node]["speed_std"] = all_features[node]["speed_std"]
			single_net.nodes[node]["acceleration_mean"] = all_features[node]["acceleration_mean"]
			single_net.nodes[node]["acceleration_std"] = all_features[node]["acceleration_std"]
		except KeyError:
			print("%d not found" % node)
		
		
		
	
	filename = 'gml/single_network_with_rank.gml'# % (type_graph,snap_id)
	nx.write_gml(single_net, filename)
				
	
#end
	
	
				
def get_authors_features(filename):
	csv_file = "%s" % (filename)
	
	authors_features = {}
	authors_features['speed.papers'] = {}
	authors_features['acceleration.papers'] = {}
	#AuthorId
	#Time
	
	time_list = []
	
	with open(csv_file, newline='') as csvfile:
		features_csv = csv.DictReader(csvfile, delimiter=',')
		
		headers = features_csv.fieldnames
		#print(headers)
		for row in features_csv:
		
			author_id = int(row["AuthorId"])
			time_window = int(row["Time"])
			speed = int(row["speed.papers"])
			acceleration = int(row["acceleration.papers"])
			
			if author_id not in authors_features['speed.papers']:
				authors_features['speed.papers'][author_id] = {}
				authors_features['acceleration.papers'][author_id] = {}
			if time_window not in time_list:
				time_list.append(time_window)
			
			authors_features['speed.papers'][author_id][time_window] = speed
			authors_features['acceleration.papers'][author_id][time_window] = acceleration
			
	#print("antes do sort:", time_list)
	
	time_list.sort()
	
	#print("depois do sort:", time_list)
	
	
	return [authors_features,time_list]
			
				
def analysis(backbones,communities):
	for snapshot in backbones:
		graph = backbones[snapshot]

		print("----------")
		print("Snapshot ID:", snapshot)
		print("Number of nodes:", graph.number_of_nodes())
		print("Number of edges:", graph.number_of_edges())
		print("Average degree:", "{:.2f}".format(sum([d for (n, d) in nx.degree(graph)]) / float(graph.number_of_nodes())))
		print("Average weigth:", "{:.2f}".format(sum(nx.get_edge_attributes(graph, 'weight').values()) / float(graph.number_of_edges())))
		print("Average clustering:", "{:.2f}".format(nx.average_clustering(graph)))
		#print("Louvain communities:", len(set(communities[snapshot].values())))
		print("Louvain communities:", len(communities[snapshot].communities))
		#print("Louvain communities modularity:", "{:.2f}".format(community_louvain.modularity(communities[snapshot], graph, weight='weight')))
		print("Louvain communities modularity: %.2f" % evaluation.newman_girvan_modularity(graph,communities[snapshot]).score ) 
				

