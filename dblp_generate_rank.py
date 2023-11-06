import networkx as nx
import numpy as np
import pickle

#from scipy import *

import time

#import plot_communities as pc
###from netgraph import Graph

import scipy.stats as scs

import dblp_aux_functions


def generate_ranked_simple(authors_features_filename):

	
	authors_features,time_list = dblp_aux_functions.get_authors_features(authors_features_filename)
	
	speed_mean = []
	speed_std = []
	acceleration_mean = []
	acceleration_std = []
	authors_ids = []
	
	
	all_features = {}
	
	for author_id in authors_features['speed.papers']:
		author_speed_list = list(authors_features['speed.papers'][author_id].values())
		author_acceleration_list = list(authors_features['acceleration.papers'][author_id].values())
		#print(author_id,author_speed_list,np.mean(author_speed_list),np.std(author_speed_list),np.mean(author_acceleration_list),np.std(author_acceleration_list))
		
		speed_mean.append(np.mean(author_speed_list))
		speed_std.append(np.std(author_speed_list))
		acceleration_mean.append(np.mean(author_acceleration_list))
		acceleration_std.append(np.std(author_acceleration_list))
		authors_ids.append(author_id)
		
		all_features[author_id] = {}
		all_features[author_id]["speed_mean"] = np.mean(author_speed_list)
		all_features[author_id]["speed_std"] = np.std(author_speed_list)
		all_features[author_id]["acceleration_mean"] = np.mean(author_acceleration_list)
		all_features[author_id]["acceleration_std"] = np.std(author_acceleration_list)
		
		#if author_id == 100:
		#	break
			
	#print(speed_mean)
	#print(speed_std)
	#print(acceleration_mean)
	#print(acceleration_std)
	#print(authors_ids)
	
	speed_mean = np.array(speed_mean)
	speed_std = np.array(speed_std)
	acceleration_mean = np.array(acceleration_mean)
	acceleration_std = np.array(acceleration_std)
	
	sorted_speed_mean = np.argsort(speed_mean)
	sorted_speed_std = np.argsort(speed_std)
	sorted_acceleration_mean = np.argsort(acceleration_mean)
	sorted_acceleration_std = np.argsort(acceleration_std)
	
	#print("sorted_speed_mean",sorted_speed_mean)
	#print("sorted_speed_std",sorted_speed_std)
	#print("sorted_acceleration_mean",sorted_acceleration_mean)
	#print("sorted_acceleration_std",sorted_acceleration_std)
	
	borda = np.zeros(len(sorted_speed_mean),int)
	
	for i in range(len(sorted_speed_mean)):
		pos_speed_mean = np.where(sorted_speed_mean == i)[0][0]
		pos_speed_std = np.where(sorted_speed_std == i)[0][0]
		pos_acceleration_mean = np.where(sorted_acceleration_mean == i)[0][0]
		pos_acceleration_std = np.where(sorted_acceleration_std == i)[0][0]
		
		#print(i,"is in position",pos_speed_mean,authors_ids[i])
		#print(i,"is in position",pos_speed_std,authors_ids[i])
		#print(i,"is in position",pos_acceleration_mean,authors_ids[i])
		#print(i,"is in position",pos_acceleration_std,authors_ids[i])
		
		borda[i] = pos_speed_mean + pos_speed_std + pos_acceleration_mean + pos_acceleration_std
		
		#print("\tborda of",i,"is",borda[i],authors_ids[i])
		
	
	sorted_borda = np.argsort(borda)
	#print(sorted_borda)
	
	
	rank_borda = []
	all_ranked_simple = {}
	top_ranked_simple = {}
	for i in range(len(sorted_borda)-1,-1,-1):
		rank_borda.append(authors_ids[sorted_borda[i]])
		
		all_ranked_simple[authors_ids[sorted_borda[i]]] = len(sorted_borda)-i
		
	count = 0
	for author_id in all_ranked_simple:
		top_ranked_simple[author_id] = all_ranked_simple[author_id]
		count+=1
		if count == 50:
			break	
	
	#print(rank_borda)
	#print(all_ranked_simple)
	#print("time_list:",time_list)
	
	
	"""
	for author_id in top_ranked_simple:
		print(author_id,top_ranked_simple[author_id],all_features[author_id])
		
	exit()
	"""
	
	return all_ranked_simple,top_ranked_simple,all_features
