import random
from random import sample
#from BackboneExtractor import backbone_from_DisparityFilter
import backboning
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pickle

from dblp_network_analysis import n_neighbor

import dblp_plot

def backbone_extractor(full_networks,significance):
	backbones = {}
	for snapshot in full_networks:
		confidence_value = 1 - significance
		#print("snapshot (full dentro):", nx.info(full_networks[snapshot]))
		backbones[snapshot] = backbone_from_DisparityFilter(full_networks[snapshot], confidence_value)
		#print("snapshot (backbone dentro):", nx.info(backbones[snapshot]))
	return backbones
	

def backbone_from_DisparityFilter(g, confidence=0.90):
    temp = "temp.csv"
    df = nx.to_pandas_edgelist(g)
    df.rename(columns={"source": "src", "target": "trg"},  inplace=True)
    df.to_csv(temp, index = None,sep='\t')
    table, nnodes, nnedges = backboning.read(temp, "weight", triangular_input = True, consider_self_loops = True, undirected = True, drop_zeroes = True, sep = "\t")
    nc_table = backboning.disparity_filter(table, undirected = True, return_self_loops = False)
    nc_table = nc_table[nc_table['score'] >= confidence]
    nc_table['weight'] = nc_table['nij']
    G = nx.from_pandas_edgelist(nc_table, 'src', 'trg', ['weight'])
    return G


def calculate_probability_backbone(networks,backbones,single_full_network,top_ranked_full,all_ranked,filenames,num_top_ranked=None):

	if num_top_ranked == None: # nesse caso vamos considerar como top_ranked todos que foram passados
		top_ranked = top_ranked_full
		num_top_ranked = len(top_ranked_full)
	else:
		top_ranked = {}
		count = 0
		for author in all_ranked:
			top_ranked[author] = all_ranked[author]
			count+=1
			if count == num_top_ranked:
				break
	
	## Calculando a presença dos top nos backbones e a probabilidade dos random em estar em algum backbone
	hit_top_backbone = np.zeros(num_top_ranked,int)
	try_top_backbone = np.zeros(num_top_ranked,int)
	
	
	num_nodes_network = 0
	num_nodes_backbone = 0
	
	all_nodes_networks = set()
	for snap_id in networks:
		num_nodes_network += networks[snap_id].order()
		num_nodes_backbone += backbones[snap_id].order()
		
		all_nodes_networks.update( list( networks[snap_id].nodes() ) )
				
		for author_id in top_ranked:
			if networks[snap_id].has_node(author_id):
				try_top_backbone[top_ranked[author_id]-1]+=1
				if backbones[snap_id].has_node(author_id):
					hit_top_backbone[top_ranked[author_id]-1]+=1
				
				
		
	values_top = []
	for i in range(num_top_ranked):
		
		#prob = float(hit_top_backbone[i]) / float(len(filenames))
		prob = float(hit_top_backbone[i]) / float(try_top_backbone[i])
		values_top.append(prob)
		print("Probability #%s: %.2f" % (i+1,prob))
	
	
	num_random = 1000
	try_random = 0
	hit_random = 0	
	for snap_id in networks:
		random_nodes = sample(list(networks[snap_id].nodes()), num_random)
		for node in random_nodes:
			try_random += 1
			if backbones[snap_id].has_node(node):
				hit_random += 1
	prob_random = float(hit_random) / float(try_random)
	print("Probability random node (new): %.2f" % prob_random)
	values_random = [prob_random]
	
	top_10_percent = {}
	top_25_percent = {}
	top_50_percent = {}
	count = 0
	num_10_percent = int(len(all_ranked)*0.1)
	num_25_percent = int(len(all_ranked)*0.25)
	num_50_percent = int(len(all_ranked)*0.50)
	for author in all_ranked:
		if count < num_10_percent:
			top_10_percent[author] = all_ranked[author]
		if count < num_25_percent:
			top_25_percent[author] = all_ranked[author]
		if count < num_50_percent:
			top_50_percent[author] = all_ranked[author]
			
		count+=1
	
	try_10_percent = 0
	hit_10_percent = 0
	try_25_percent = 0
	hit_25_percent = 0
	try_50_percent = 0
	hit_50_percent = 0
	
	for snap_id in networks:
		nodes_10_percent = list(top_10_percent.keys())
		nodes_25_percent = list(top_25_percent.keys())
		nodes_50_percent = list(top_50_percent.keys())
		
		for node in nodes_10_percent:
			if networks[snap_id].has_node(node):
				try_10_percent+=1
			if backbones[snap_id].has_node(node):
				hit_10_percent+=1
		
		for node in nodes_25_percent:
			if networks[snap_id].has_node(node):
				try_25_percent+=1
			if backbones[snap_id].has_node(node):
				hit_25_percent+=1
				
		for node in nodes_50_percent:
			if networks[snap_id].has_node(node):
				try_50_percent+=1
			if backbones[snap_id].has_node(node):
				hit_50_percent+=1
				
	prob_10_percent = float(hit_10_percent) / float(try_10_percent)
	prob_25_percent = float(hit_25_percent) / float(try_25_percent)
	prob_50_percent = float(hit_50_percent) / float(try_50_percent)
	
	print("Probability 10 percent: %.2f" % prob_10_percent)
	print("Probability 25 percent: %.2f" % prob_25_percent)
	print("Probability 50 percent: %.2f" % prob_50_percent)
	
	values_top_percent = [prob_10_percent,prob_25_percent,prob_50_percent]
	
	dblp_plot.plot_probability_presence_backbone(values_random,values_top_percent,values_top)
	
		
	## OK. Agora vou calcular a probabilidade de hit no backbone dos aleatórios e dos top percent
	
	top_percent = 50
	
	nodes_percent = {}
	nodes_percent[10] = nodes_10_percent
	nodes_percent[25] = nodes_25_percent
	nodes_percent[50] = nodes_50_percent
	
	random_nodes = sample(all_nodes_networks,num_random)
	
	try_random = {}
	hit_random = {}
	prob_random = {}
	for node in random_nodes:
		try_random[node] = 0
		hit_random[node] = 0
		prob_random[node] = 0
	for snap_id in networks:
		for node in random_nodes:
			try_random[node] +=1
			if backbones[snap_id].has_node(node):
				hit_random[node]+=1
	
	
	try_top = {}
	hit_top = {}
	prob_top = {}
	for node in nodes_percent[top_percent]:
		try_top[node] = 0
		hit_top[node] = 0
		prob_top[node] = 0
		
	for snap_id in networks:
		for node in nodes_percent[top_percent]:
			try_top[node] +=1
			if backbones[snap_id].has_node(node):
				hit_top[node]+=1
				
	
				
	
	count_hit_random = np.zeros(len(filenames)+1,float)
	for node in random_nodes:
		prob_random[node] = float(hit_random[node]) / float(try_random[node])
		count_hit_random[hit_random[node]] += 1
		
	annotation_random = np.copy(count_hit_random)
	#print(annotation_random)
	#exit()
	for i in range(len(filenames)+1):
		count_hit_random[i] = count_hit_random[i] / num_random
		
	
		
	
	count_hit_top = np.zeros(len(filenames)+1,float)
	for node in nodes_percent[top_percent]:
		prob_top[node] = float(hit_top[node]) / float(try_top[node])
				
		count_hit_top[hit_top[node]] += 1
		
	annotation_top = np.copy(count_hit_top)
	for i in range(len(filenames)+1):
		count_hit_top[i] = count_hit_top[i] / len(nodes_percent[top_percent])
		
	labels = []
	for i in range(len(filenames)+1):
		labels.append(i)
	
	plt.figure(figsize=(10,6))
	barWidth = 0.28
	#br0 = np.arange(len(filenames)+1)
	#br1 = [x + barWidth for x in labels]
	br1 = labels
	br2 = [x + barWidth for x in br1]
	#br3 = [x + barWidth for x in br2]
	
	pl1 = plt.bar(br1,count_hit_top,color='tab:green',alpha=0.7,label='Top %s percent hyperprolific authors' % top_percent,width = barWidth)
	pl2 = plt.bar(br2,count_hit_random,color='tab:blue',alpha=0.7,label='Random authors',width = barWidth)
	#pl3 = plt.bar(br3,count_hit_random_50,color='tab:purple',alpha=0.7,label='Random authors ($\geq$50 publications)',width = barWidth)
	
	count = 0
	for bar in pl1:
		#text, position ANNOTATE
		print(bar.get_height())
		plt.annotate(int(annotation_top[count]), xy=(bar.get_x(), bar.get_height()+0.002),fontsize=8,color='tab:gray')
		count+=1
	
	count = 0
	for bar in pl2:
		#text, position ANNOTATE
		plt.annotate(int(annotation_random[count]), xy=(bar.get_x(), bar.get_height()+0.002),fontsize=8,color='tab:gray')
		count+=1
		
	plt.xticks([r + barWidth/2 for r in range(len(filenames)+1)],labels)
	
	
	#plt.xticks(np.arange(0, max(edge_weights[snap_id]), 5))
	###plt.hist(list(hit_random.values()), density=True, color='tab:blue', alpha=0.7, label = ['random'])
	###plt.hist(edge_weights_backbones[snap_id], density=True, bins=11, color='tab:red', alpha=0.7, label=['top50 hyperprolific'])
	plt.legend(prop={'size': 14})
	plt.xlabel('Number backbones hits',fontsize=16)
	plt.ylabel('Fraction of authors',fontsize=16)
	
	plt.tick_params(axis='both', which='major', labelsize=16)
	plt.tick_params(axis='both', which='minor', labelsize=12)
	
	plt.savefig("results/probability_backbone/presence_backbones_random_top%s_percent.png" % (top_percent),bbox_inches='tight')
	
	plt.clf()
	plt.close
	
		
	####
	#### ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
	# OK. Agora calculando a distribuição da presença dos vértices aleatórios nos backbones e tb dos top
	
	
	random_nodes = sample(all_nodes_networks,num_random)
	
	try_random = {}
	hit_random = {}
	prob_random = {}
	for node in random_nodes:
		try_random[node] = 0
		hit_random[node] = 0
		prob_random[node] = 0
		
	for snap_id in networks:
		for node in random_nodes:
			##if networks[snap_id].has_node(node):
			try_random[node] +=1  ### <<=== ALTERAR AQUIII
			if backbones[snap_id].has_node(node):
				hit_random[node]+=1
				
				
	try_top = {}
	hit_top = {}
	prob_top = {}
	for node in top_ranked:
		try_top[node] = 0
		hit_top[node] = 0
		prob_top[node] = 0
		
	for snap_id in networks:
		for node in top_ranked:
			if networks[snap_id].has_node(node):
				try_top[node] +=1
			if backbones[snap_id].has_node(node):
				hit_top[node] +=1
				
				
					
	
	count_hit_random = np.zeros(len(filenames)+1,float)
	for node in random_nodes:
		prob_random[node] = float(hit_random[node]) / float(try_random[node])
		count_hit_random[hit_random[node]] += 1
		
		
	
	
	annotation_random = np.copy(count_hit_random)
	for i in range(len(filenames)+1):
		count_hit_random[i] = count_hit_random[i] / num_random
	
	
		
	print(hit_top)
	count_hit_top = np.zeros(len(filenames)+1,float)
	for node in top_ranked:
		prob_top[node] = float(hit_top[node]) / float(try_top[node])
				
		count_hit_top[hit_top[node]] += 1
		
	annotation_top = np.copy(count_hit_top)
	
	print(annotation_top)
	
	#exit()
	for i in range(len(filenames)+1):
		count_hit_top[i] = count_hit_top[i] / num_top_ranked
		
	labels = []
	for i in range(len(filenames)+1):
		labels.append(i)
	
	plt.figure(figsize=(10,6))
	barWidth = 0.28
	#br0 = np.arange(len(filenames)+1)
	#br1 = [x + barWidth for x in labels]
	br1 = labels
	br2 = [x + barWidth for x in br1]
	###br3 = [x + barWidth for x in br2]
	
	pl1 = plt.bar(br1,count_hit_top,color='tab:red',alpha=0.7,label='Top%s hyperprolific authors' % num_top_ranked,width = barWidth)
	pl2 = plt.bar(br2,count_hit_random,color='tab:blue',alpha=0.7,label='Random authors',width = barWidth)
	###pl3 = plt.bar(br3,count_hit_random_50,color='tab:purple',alpha=0.7,label='Random authors ($\geq$50 publications)',width = barWidth)
	
	count = 0
	for bar in pl1:
		#text, position ANNOTATE
		print(bar.get_height())
		plt.annotate(int(annotation_top[count]), xy=(bar.get_x(), bar.get_height()+0.002),fontsize=10,color='black')
		count+=1
	
	count = 0
	for bar in pl2:
		#text, position ANNOTATE
		plt.annotate(int(annotation_random[count]), xy=(bar.get_x(), bar.get_height()+0.002),fontsize=10,color='black')
		count+=1
		
	###count = 0
	###for bar in pl3:
	###	#text, position ANNOTATE
	###	plt.annotate(int(annotation_random_50[count]), xy=(bar.get_x(), bar.get_height()+0.002),fontsize=10,color='black')
	###	count+=1
	
	plt.xticks([r + barWidth for r in range(len(filenames)+1)],labels)
	
	
	#plt.xticks(np.arange(0, max(edge_weights[snap_id]), 5))
	###plt.hist(list(hit_random.values()), density=True, color='tab:blue', alpha=0.7, label = ['random'])
	###plt.hist(edge_weights_backbones[snap_id], density=True, bins=11, color='tab:red', alpha=0.7, label=['top50 hyperprolific'])
	plt.legend(prop={'size': 14})
	plt.xlabel('Number of backbone hits',fontsize=16)
	plt.ylabel('Fraction of authors',fontsize=16)
	
	plt.tick_params(axis='both', which='major', labelsize=16)
	plt.tick_params(axis='both', which='minor', labelsize=12)
	
	plt.savefig("results/probability_backbone/presence_backbones_random_top%s.png" % (num_top_ranked),bbox_inches='tight')
	
	plt.clf()
	plt.close
	
	#### ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
	## OK. Agora calculando a probabilidade dos vizinhos hop 2, hop 3 e hop 4 dos top (ou dos top 10?) estarem no backbone
	set_hop1 = set()
	set_hop2 = set()
	set_hop3 = set()
	set_hop4 = set()
	
	try_hop1 = 0
	try_hop2 = 0
	try_hop3 = 0
	try_hop4 = 0
	
	hit_hop1 = 0
	hit_hop2 = 0
	hit_hop3 = 0
	hit_hop4 = 0
	
	for snap_id in networks:
		for ego_node in top_ranked:
			if networks[snap_id].has_node(ego_node):
				ego_hop1 = set( n_neighbor(networks[snap_id],ego_node,1) )
				set_hop1.update(ego_hop1)
				
				ego_hop2 = set( n_neighbor(networks[snap_id],ego_node,2) )
				ego_hop2 = ego_hop2.difference(ego_hop1)
				set_hop2.update(ego_hop2)
				
				ego_hop3 = set( n_neighbor(networks[snap_id],ego_node,3) )
				ego_hop3 = ego_hop3.difference(ego_hop2).difference(ego_hop1)
				set_hop3.update(ego_hop3)
				
				ego_hop4 = set( n_neighbor(networks[snap_id],ego_node,4) )
				ego_hop4 = ego_hop4.difference(ego_hop3).difference(ego_hop2).difference(ego_hop1)
				set_hop4.update(ego_hop4)
				
				#print("intersection:",len(ego_hop2),len(ego_hop3),len(ego_hop2.intersection(ego_hop3)) )
				
				#exit()
				
				
		
		for node in set_hop1:
			if networks[snap_id].has_node(node):
				try_hop1+=1
			if backbones[snap_id].has_node(node):
				hit_hop1+=1
		for node in set_hop2:
			if networks[snap_id].has_node(node):
				try_hop2+=1
			if backbones[snap_id].has_node(node):
				hit_hop2+=1
		for node in set_hop3:
			if networks[snap_id].has_node(node):
				try_hop3+=1
			if backbones[snap_id].has_node(node):
				hit_hop3+=1
		for node in set_hop4:
			if networks[snap_id].has_node(node):
				try_hop4+=1
			if backbones[snap_id].has_node(node):
				hit_hop4+=1
	
	prob_hop1 = float(hit_hop1) / float(try_hop1)			
	prob_hop2 = float(hit_hop2) / float(try_hop2)
	prob_hop3 = float(hit_hop3) / float(try_hop3)
	prob_hop4 = float(hit_hop4) / float(try_hop4)
	
	print("Probability of hop 1 in backbone: %.2f" % prob_hop1)
	print("Probability of hop 2 in backbone: %.2f" % prob_hop2)
	print("Probability of hop 3 in backbone: %.2f" % prob_hop3)
	print("Probability of hop 4 in backbone: %.2f" % prob_hop4)
	
	
	values_hop = [prob_hop1,prob_hop2,prob_hop3,prob_hop4]
	
	dblp_plot.plot_probability_hops(values_random,values_top_percent,values_hop)
	
	
	count = 0
	num_coauthors_to_plot = 5
	for ego_node in top_ranked:
		coauthors = set()
		#print("coauthors of ego_node %s" % ego_node)
		
		# Pegando os coautores do ego
		for snap_id in networks:
			if networks[snap_id].has_node(ego_node):
				#print("snapshot %s ok, vai processar" % snap_id)
				ego_neighbors = networks[snap_id].neighbors(ego_node)
				
				for coauthor in ego_neighbors:
					#print(coauthor)
					coauthors.add(coauthor)
				
		
		# OK, agora vou percorrer e verificar os hit nos backbones
		try_coauthors = {}
		hit_coauthors = {}
		prob_coauthors = {}
			
		for coauthor in coauthors:
			try_coauthors[coauthor] = 0
			hit_coauthors[coauthor] = 0
			prob_coauthors[coauthor] = 0
		
		for snap_id in networks:
			for coauthor in coauthors:
				if networks[snap_id].has_node(coauthor):
					try_coauthors[coauthor] +=1
				if backbones[snap_id].has_node(coauthor):
					hit_coauthors[coauthor]+=1
		
		
				
		# Agora eu vou contar a probabilidae em cada número de hits
		count_hit_coauthors = np.zeros(len(filenames)+1,float)
		for coauthor in coauthors:
			prob_coauthors[coauthor] = float(hit_coauthors[coauthor]) / float(try_coauthors[coauthor])
			count_hit_coauthors[hit_coauthors[coauthor]] += 1
			
		annotation_coauthor = np.copy(count_hit_coauthors)
		for i in range(len(filenames)+1):
			count_hit_coauthors[i] = count_hit_coauthors[i] / len(coauthors)#networks[snap_id].degree[str(ego_node)]
			
		labels = []
		for i in range(len(filenames)+1):
			labels.append(i)
		
		plt.figure(figsize=(6,6))
		barWidth = 0.40
		#br0 = np.arange(len(filenames)+1)
		#br1 = [x + barWidth for x in labels]
		br1 = labels
		#br2 = [x + barWidth for x in br1]
		
		pl = plt.bar(br1,count_hit_coauthors,color='tab:red',alpha=0.7,width = barWidth)
		#plt.bar(br2,count_hit_random,color='tab:blue',alpha=0.7,label='Random authors',width = barWidth)
		
		#plt.xticks([r + barWidth/2 for r in range(len(filenames)+1)],labels)
		plt.xticks([r for r in range(len(filenames)+1)],labels)
		
		
		#plt.xticks(np.arange(0, max(edge_weights[snap_id]), 5))
		###plt.hist(list(hit_random.values()), density=True, color='tab:blue', alpha=0.7, label = ['random'])
		###plt.hist(edge_weights_backbones[snap_id], density=True, bins=11, color='tab:red', alpha=0.7, label=['top50 hyperprolific'])
		#plt.legend(prop={'size': 14})
		plt.xlabel('Number of backbone hits',fontsize=16)
		plt.ylabel('Fraction of coauthors',fontsize=16)
		
		plt.tick_params(axis='both', which='major', labelsize=16)
		plt.tick_params(axis='both', which='minor', labelsize=12)
		
		count_bar = 0
		for bar in pl:
			#text, position ANNOTATE
			plt.annotate(int(annotation_coauthor[count_bar]), xy=(bar.get_x(), bar.get_height()+0.002),fontsize=12,color='black')
			count_bar+=1
		
		plt.savefig("results/probability_backbone/presence_backbones_%s_%s.png" % (count+1,ego_node),bbox_inches='tight')
		
		plt.clf()
		plt.close
				
		
		count+=1
		if count == num_coauthors_to_plot:
			break
#end

