import numpy as np
import pickle
import rbo
from scipy import stats

import dblp_plot
import dblp_aux_functions
import dblp_authors

def get_top_single_timeline_backbone_hit(backbones,filenames,authors_ranked):
	#print(all_ranked)
	
	
	num_authors_ranked = len(authors_ranked)
	
	
	
	prob_top_backbone = np.zeros(len(filenames),float)
	
	snap_id_index = 0
	for snap_id in backbones:
		#print("\n",snap_id)
		hit_top_backbone = 0
		for author_id in authors_ranked:
			#print(author_id)
			if backbones[snap_id].has_node(author_id):
				#print("hit!")
				hit_top_backbone+=1
				#print("degree = %s" % backbones[snap_id].degree[author_id])
			#else:
			#	print("miss...")
		#end
		
		prob_top_backbone[snap_id_index] = float(hit_top_backbone) / float(num_authors_ranked)
		
		snap_id_index+=1
	#end
	
	#for i in range(len(filenames)):
	#	print("Snap id: %s = %f" % (filenames[snap_id],prob_top_backbone[i] ) )

	return prob_top_backbone
#end

def get_top_single_timeline_feature(feature,authors_features,time_list,authors_ranked):
	
	num_authors_ranked = len(authors_ranked)
	
	
	###print('authors_ranked (get_top_single_timeline_feature):',authors_ranked)
	
	
	selected_authors_feature = np.zeros(len(time_list),float)
	selected_authors_feature_count = np.zeros(len(time_list),int)
	
	for author_id in authors_ranked:
		feature_author = authors_features[feature][author_id]
		
		#print(feature_author)
		
		for time_window in feature_author:
		
			#print(time_window)
			
			#exit()
			position = time_list.index(time_window)
			
			selected_authors_feature[position]+=feature_author[time_window]
			selected_authors_feature_count[position]+=1
		#end
	#end
	
	for i in range(len(selected_authors_feature)):
		selected_authors_feature[i] = selected_authors_feature[i] / selected_authors_feature_count[i]
	#end
	
	#print(selected_authors_feature)
	#print(selected_authors_feature_count)
	
	return selected_authors_feature
	
	
#end
		
	
	




def calculate_rank_correlations(all_ranked,all_topological_ranks_filename,networks):
	
	with open('data/%s' % all_topological_ranks_filename, 'rb') as handle:
		all_topological_ranks = pickle.load(handle)
		
	print("leu o pickle do topological rank")
	
	
	num_top = 1000
	
	rank_top_hyperprolific = []
	rank_hyperprolific = sorted(all_ranked.items(), key=lambda x: x[1])
	count = 0
	for element in rank_hyperprolific:
		rank_top_hyperprolific.append(element[0])
		count+=1
		if count == num_top:
			break
	
	
	ao_eigenvector = []
	ao_degree = []
	ao_closeness = []
	ao_betweenness = []
	
	for snap_id in networks:
		rank_top_eigenvector = []
		rank_top_degree = []
		rank_top_closeness = []
		rank_top_betweenness = []
		
		rank_hyperprolific_snap_id = []
		count = 1
		print(len(networks[snap_id]), len(list(all_ranked.keys())))
		for node in networks[snap_id]:
			author = list(all_ranked.keys())[list(all_ranked.values()).index(int(count))]
			##author = all_ranked.keys()[all_ranked.values().index(count)]
			rank_hyperprolific_snap_id.append(author)
			count+=1
			if count >= len(all_ranked) or count > len(networks[snap_id]):
				break
		
		
		rank_eigenvector = sorted(all_topological_ranks[snap_id]['eigenvector'].items(), key=lambda x: x[1], reverse=True)
		rank_degree = sorted(all_topological_ranks[snap_id]['degree'].items(), key=lambda x: x[1], reverse=True)
		rank_closeness = sorted(all_topological_ranks[snap_id]['closeness'].items(), key=lambda x: x[1], reverse=True)
		rank_betweenness = sorted(all_topological_ranks[snap_id]['betweenness'].items(), key=lambda x: x[1], reverse=True)
		
		count = 0
		rank_top_hyperprolific_snap_id = []
		for element in rank_hyperprolific_snap_id:
			rank_top_hyperprolific_snap_id.append(element)
			count+=1
			if count == num_top:
				break
		
		count = 0
		rank_eigenvector_snap_id = []
		for element in rank_eigenvector:
			if count < num_top:
				rank_top_eigenvector.append(element[0])
			count+=1
			if count < len(all_ranked):
				rank_eigenvector_snap_id.append(element[0])
		
		count = 0
		rank_degree_snap_id = []
		for element in rank_degree:
			if count < num_top:
				rank_top_degree.append(element[0])
			count+=1
			if count < len(all_ranked):
				rank_degree_snap_id.append(element[0])
		
		count = 0
		rank_closeness_snap_id = []
		for element in rank_closeness:
			if count < num_top:
				rank_top_closeness.append(element[0])
			count+=1
			if count < len(all_ranked):
				rank_closeness_snap_id.append(element[0])
		
		count = 0
		rank_betweenness_snap_id = []
		for element in rank_betweenness:
			if count < num_top:
				rank_top_betweenness.append(element[0])
			count+=1
			if count < len(all_ranked):
				rank_betweenness_snap_id.append(element[0])
				
		
		
		
		similarity_eigenvector = rbo.RankingSimilarity(rank_top_eigenvector,rank_top_hyperprolific).rbo()
		similarity_degree = rbo.RankingSimilarity(rank_top_degree,rank_top_hyperprolific).rbo()
		similarity_closeness = rbo.RankingSimilarity(rank_top_closeness,rank_top_hyperprolific).rbo()
		similarity_betweenness = rbo.RankingSimilarity(rank_top_betweenness,rank_top_hyperprolific).rbo()
		print('(snap_id: %d) AO similarity hyperprolific x eigenvector/degree/closeness/betweeness: %.4f\t%.4f\t%.4f\t%.4f' % (snap_id,similarity_eigenvector, similarity_degree, similarity_closeness, similarity_betweenness))
		
		ao_eigenvector.append(similarity_eigenvector)
		ao_degree.append(similarity_degree)
		ao_closeness.append(similarity_closeness)
		ao_betweenness.append(similarity_betweenness)
		
	dblp_plot.plot_ao_similarity(ao_eigenvector,ao_degree,ao_closeness,ao_betweenness)
		
		
		
#end



def calculate_dynamic_correlation(backbones,authors_features_filename,all_ranked_simple,filenames):

	authors_features,time_list = dblp_aux_functions.get_authors_features(authors_features_filename)
	
	pos_rank_start = 1
	pos_rank_finish = 100
	selected_authors_ranked_top = dblp_authors.get_selected_authors(all_ranked_simple,pos_rank_start,pos_rank_finish)
	
	pos_rank_start = 1001
	pos_rank_finish = 1100
	selected_authors_ranked_mid = dblp_authors.get_selected_authors(all_ranked_simple,pos_rank_start,pos_rank_finish)
	
	
	pos_rank_start = 0
	pos_rank_finish = 0
	selected_authors_ranked_random = dblp_authors.get_selected_authors(all_ranked_simple,pos_rank_start,pos_rank_finish)
	
	print(selected_authors_ranked_top)
	print(selected_authors_ranked_mid)
	print(selected_authors_ranked_random)
	
	
	print("Getting backbone probability of selected authors (top-ranked/mid-ranked/random)...")
	selected_authors_prob_backbone_top = get_top_single_timeline_backbone_hit(backbones,filenames,selected_authors_ranked_top)
	selected_authors_prob_backbone_mid = get_top_single_timeline_backbone_hit(backbones,filenames,selected_authors_ranked_mid)
	selected_authors_prob_backbone_random = get_top_single_timeline_backbone_hit(backbones,filenames,selected_authors_ranked_random)
	print("OK!\n")
	
	print("selected_authors_prob_backbone_top:",selected_authors_prob_backbone_top)
	print("selected_authors_prob_backbone_random:",selected_authors_prob_backbone_random)
	print("selected_authors_prob_backbone_mid:",selected_authors_prob_backbone_mid)
	
	
	print("Getting feature of selected authors: speed (top-ranked/mid-ranked/random)...")
	selected_authors_feature_speed_top = get_top_single_timeline_feature('speed.papers',authors_features,time_list,selected_authors_ranked_top)
	selected_authors_feature_speed_mid = get_top_single_timeline_feature('speed.papers',authors_features,time_list,selected_authors_ranked_mid)
	selected_authors_feature_speed_random = get_top_single_timeline_feature('speed.papers',authors_features,time_list,selected_authors_ranked_random)
	print("OK!\n")
	
	print("selected_authors_feature_speed_top:",selected_authors_feature_speed_top)
	print("selected_authors_feature_speed_mid:",selected_authors_feature_speed_mid)
	print("selected_authors_feature_speed_random:",selected_authors_feature_speed_random)
	
	
	speed_top = []
	speed_mid = []
	speed_random = []
	
	print("Calculating TLCC (Prob.Backbone leads to speed.papers (top-ranked)...")
	for i in range(-5,6,1):
		corr = crosscorr(selected_authors_prob_backbone_top,selected_authors_feature_speed_top,lag=i)
		print(i,'\t',corr)
		speed_top.append(corr)
	print("OK!\n")
	print("Calculating TLCC (Prob.Backbone leads to speed.papers (mid-ranked)...")
	for i in range(-5,6,1):
		corr = crosscorr(selected_authors_prob_backbone_mid,selected_authors_feature_speed_mid,lag=i)
		print(i,'\t',corr)
		speed_mid.append(corr)
	print("OK!\n")
	print("Calculating TLCC (Prob.Backbone leads to speed.papers (random)...")
	for i in range(-5,6,1):
		corr = crosscorr(selected_authors_prob_backbone_random,selected_authors_feature_speed_random,lag=i)
		print(i,'\t',corr)
		speed_random.append(corr)
	print("OK!\n")
	
	
	dblp_plot.plot_tlcc(speed_top,speed_mid,speed_random,'speed')
	
	
	
	print("Getting feature of selected authors: acceleration (top-ranked/mid-ranked)...")
	selected_authors_feature_acceleration_top = get_top_single_timeline_feature('acceleration.papers',authors_features,time_list,selected_authors_ranked_top)
	selected_authors_feature_acceleration_mid = get_top_single_timeline_feature('acceleration.papers',authors_features,time_list,selected_authors_ranked_mid)
	selected_authors_feature_acceleration_random = get_top_single_timeline_feature('acceleration.papers',authors_features,time_list,selected_authors_ranked_random)
	print("OK!\n")
	
	print("selected_authors_feature_acceleration_top:",selected_authors_feature_acceleration_top)
	print("selected_authors_feature_acceleration_mid:",selected_authors_feature_acceleration_mid)
	print("selected_authors_feature_acceleration_random:",selected_authors_feature_acceleration_random)
	
	
	
	
	acceleration_top = []
	acceleration_mid = []
	acceleration_random = []
	
	print("Calculating TLCC (Prob.Backbone leads to acceleration.papers (top-ranked)...")
	for i in range(-5,6,1):
		corr = crosscorr(selected_authors_prob_backbone_top,selected_authors_feature_acceleration_top,lag=i)
		print(i,corr)
		acceleration_top.append(corr)
		
	print("OK!\n")
	print("Calculating TLCC (Prob.Backbone leads to acceleration.papers (mid-ranked)...")
	for i in range(-5,6,1):
		corr = crosscorr(selected_authors_prob_backbone_mid,selected_authors_feature_acceleration_mid,lag=i)
		print(i,corr)
		acceleration_mid.append(corr)
		
	print("OK!\n")
	print("Calculating TLCC (Prob.Backbone leads to acceleration.papers (random)...")
	for i in range(-5,6,1):
		corr = crosscorr(selected_authors_prob_backbone_random,selected_authors_feature_acceleration_random,lag=i)
		print(i,corr)
		acceleration_random.append(corr)
	print("OK!\n")
	
	
	dblp_plot.plot_tlcc(acceleration_top,acceleration_mid,acceleration_random,'acceleration')
	
	
	




def crosscorr(datax, datay, lag=0, wrap=False):
	import pandas as pd
	
	d = {}
	d['datax'] = list(datax)
	d['datay'] = list(datay)
	
	df = pd.DataFrame(d)
	
	datax = df['datax']
	datay = df['datay']
	
		
	""" Lag-N cross correlation. 
	Shifted data filled with NaNs 
	
	Parameters
	----------
	lag : int, default 0
	datax, datay : pandas.Series objects of equal length
	Returns
	----------
	crosscorr : float
	"""
	if wrap:
		shiftedy = datay.shift(lag)
		shiftedy.iloc[:lag] = datay.iloc[-lag:].values
		return datax.corr(shiftedy,method='spearman')
	else: 
		return datax.corr(datay.shift(lag),method='spearman')
