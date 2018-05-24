#! /usr/bin/python2
#-*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import random, pickle, argparse, json, os, urllib2, sys, re
from collections import OrderedDict
from operator import itemgetter
from os import listdir
from os.path import isfile, join
from csv import reader


def query_from(q, f):
	q = q+'&from='+str(f)
	response = urllib2.urlopen(q)
	data = json.load(response)
	subject_ids = np.array(range(len(data['hits'])), dtype=np.object)
	for (i, hit) in enumerate(data['hits']):
		subject_ids[i] = hit['id']
	return subject_ids

def query(q):
	response = urllib2.urlopen(q)
	data = json.load(response)
	nb_requests = 1 + data['total'] / 1000
	if nb_requests > 10: # maximum number of pages due to API pagination restrection
		nb_requests = 10
	subject_ids = query_from(q, 0)
	for i in range(nb_requests)[1:]:
		f = i * 1000
		next_request = query_from(q, f)
		subject_ids = np.hstack((subject_ids, next_request))
	return subject_ids.tolist()

def find_intersection(list_a, list_b):
	return list(set(list_a) & set(list_b))

def list_union(list_a, list_b):
	return list(set(list_a) | set(list_b))

def list_xor(list_a, list_b):
	return list(set(list_a) ^ set(list_b))

def term2url(string):
	string = string.split(' ')
	res = ''
	for s in string:
		res = res + s + '%20'
	res = res[:-3]
	#res = res + '%22'
	return res

def babel_synset(synset):
	q = 'https://api.istex.fr/document/?q=(('
	for syn in synset:
		syn = term2url(syn)
		q = q + 'title:' + syn + '%20OR%20abstract:' + syn + '%20OR%20'
	q = q[:-8]
	q = q + ')%20AND%20(qualityIndicators.abstractWordCount:[35%20500]%20AND%20qualityIndicators.pdfPageCount:[3%2060]%20AND%20publicationDate:[1990%202016]%20AND%20language:(%22eng%22%20OR%20%22unknown%22)%20AND%20genre:(%22research_article%22%20OR%20%22conference[eBooks]%22%20OR%20%22article%22%20)%20))&size=1000&output=id'
	return q

def babel_subj_keyword(topic):
	q = 'https://api.istex.fr/document/?q=(('
	topic = term2url(topic)
	q = q+ 'subject.value:' + topic + '%20OR%20keywords.teeft:' + topic  + '%20OR%20categories.wos:' + topic
	q = q + ')%20AND%20(qualityIndicators.abstractWordCount:[35%20500]%20AND%20qualityIndicators.pdfPageCount:[3%2060]%20AND%20publicationDate:[1990%202016]%20AND%20language:(%22eng%22%20OR%20%22unknown%22)%20AND%20genre:(%22research_article%22%20OR%20%22conference[eBooks]%22%20OR%20%22article%22%20)%20))&size=1000&output=id'
	return q
 
def babel_title_abst(topic):
	q = 'https://api.istex.fr/document/?q=(('
	topic = term2url(topic)
	q = q+ 'title:' + topic + '%20OR%20abstract:' + topic
	q = q + ')%20AND%20(qualityIndicators.abstractWordCount:[35%20500]%20AND%20qualityIndicators.pdfPageCount:[3%2060]%20AND%20publicationDate:[1990%202016]%20AND%20language:(%22eng%22%20OR%20%22unknown%22)%20AND%20genre:(%22research_article%22%20OR%20%22conference[eBooks]%22%20OR%20%22article%22%20)%20))&size=1000&output=id'
	return q
 
def babelnet_syn_get_input(topic, synset, istex_ids_pool):
	results = query(babel_synset(synset))
	_gs = query(babel_subj_keyword(topic))
	if istex_ids_pool is not None:
		results = find_intersection(results, istex_ids_pool)
	_abst_title = query(babel_title_abst(topic))
	test_set = _inter = {x for x in _gs if x not in _abst_title}
	if istex_ids_pool is not None:
		test_set = find_intersection(test_set, istex_ids_pool)
	results = list(results)
	test = list(test_set)
	return results, test

def top_thresh(ordered_dict_pickle, thresh):
	ranked_all = pickle.load(open(ordered_dict_pickle, 'rb'))
	ranked_all_np = np.array(ranked_all.items())
	ranked_all_df = pd.DataFrame(data=ranked_all_np, index=None, columns=['istex_id', 'score'])
	ranked_all_df['score'] = ranked_all_df[['score']].astype(float)
	return ranked_all_df[ranked_all_df['score'] > thresh]

def top_thresh_lst(res_lst_pickle, thresh, top=100000):
	ranked_all = pickle.load(open(res_lst_pickle, 'rb'))
	if type(ranked_all) is OrderedDict:
		ranked_all = ranked_all.items()[:top]
	ranked_all_np = np.array(ranked_all)
	ranked_all_df = pd.DataFrame(data=ranked_all_np, index=None, columns=['istex_id', 'score'])
	ranked_all_df['score'] = ranked_all_df[['score']].astype(float)
	return ranked_all_df[ranked_all_df['score'] > thresh]

def babelnet_eval_PR(topic, synset, istex_ids_pool):
	babelnet_results, test = babelnet_syn_get_input(topic, synset, istex_ids_pool)
	print 'babelnet results size of the topic "' + topic + '":', len(babelnet_results) 
	print 'ground truth size', len(test)
	babel_test_intersection = find_intersection(test,babelnet_results)
	babel_test_intersection_size = len(babel_test_intersection)
	print 'intersection with the ground truth:', babel_test_intersection_size
	precision = babel_test_intersection_size / float(len(babelnet_results))
	recall = babel_test_intersection_size / float(len(test))
	if babel_test_intersection_size is not 0:
		F1 = 2 * (precision * recall) / (precision + recall)
	else:
		F1 = 0.0 
	print "F1", F1
	print 'babel precision: ', precision
	print 'babel recall: ', recall

#Evaluate 3SH results list at treshold
def eval_all_at_thresh_lst(res_pickle, topic, synset, thresh=0.75):
	babelnet_eval_PR(topic, synset, svd_index_keys)
	_, test = babelnet_syn_get_input(topic, synset)
	t = len(test)
	top_res = top_thresh_lst(res_pickle, thresh)
	n = len(top_res)
	if n > 10000:
		thresh = thresh + 0.1
		top_res = top_thresh_lst(res_pickle, thresh)
		if len(top_res) > 10000:
			thresh = thresh + 0.05
			top_res = top_thresh_lst(res_pickle, thresh)
	elif n < 1000:
		thresh = thresh - 0.1
		top_res = top_thresh_lst(res_pickle, thresh)
		if len(top_res) > 1000:
			thresh = thresh - 0.05
			top_res = top_thresh_lst(res_pickle, thresh)
	n = len(top_res)
	res = list(top_res['istex_id'])
	recall = len(find_intersection(test,res))/float(t)
	precision = len(find_intersection(test,res))/float(n)
	F1 = 2 * (precision * recall) / (precision + recall)
	print "length of s3h results: ", n, "length of test set", t
	print "F1: ", F1
	print "precision s3h: ", precision, "recall s3h: ", recall

#Evaluate fusion df
def eval_all(fusion_df, topic, synset, res_pickle, thresh=0.75):
	eval_all_at_thresh_lst(res_pickle, topic, synset, thresh)
	babelnet_res, test = babelnet_syn_get_input(topic, synset)
	n = len(babelnet_res)
	t = len(test)

	n = 2 * len(babelnet_res)
	print "Evaluation of AE2TS:"
	res = fusion_df["istex_id"].tolist()[:n]
	matches = len(find_intersection(test,res))
	recall = matches / float(t)
	precision = matches / float(n)
	if matches is not 0:
		F1 = 2 * (precision * recall) / (precision + recall)
	else:
		F1 = 0
	print "F1: ", F1 
	print "precision fusion: ", precision, "recall fusion: ",recall

def load_svd_index(jsonfname, verbose=1):
	inv_index = json.load(open(jsonfname,'rb'))
	if verbose:
		print 'original inversed_index'
		print inv_index.items()[:3]
	inversed_index = dict()
	for (k, v) in inv_index.items():
		key = k.split('_')[1]
		inversed_index[key] = v
	if verbose:
		print 'processed inversed_index'
		print inversed_index.items()[:3]
	return inversed_index

def gen_benchmark(topic_synsets_fname, istex_ids_pool, verbose):
	topic_synsets = pd.DataFrame.from_csv(topic_synsets_fname, header=0,
	                      sep=';', index_col=0, encoding=None,
	                      tupleize_cols=False,
	                      infer_datetime_format=False)
	topics = topic_synsets["topic"].tolist()
	synsets = topic_synsets["synset"].tolist()
	benchmark = np.array(range(len(topics)), dtype=np.object)

	for (i, topic) in enumerate(topics):
		topic_dataset = {}
		synset = synsets[i]
		topic_dataset["topic"] = topic
		bab = synset[1:-1]
		bab.replace('&', 'and')
		synset = bab.split(',')
		print "generating data for topic "+str(i)+": "+topic
		try:
			synset_es_results, test = babelnet_syn_get_input(topic, synset, istex_ids_pool)
			topic_dataset["synset_es_results"] = synset_es_results
			topic_dataset["test"] = test
			benchmark[i] = topic_dataset
		except:
			print "error with the istex request,, skipping topic: "+topic
	if verbose:
		print "sample of second topic's first 5 synset_es_results"
		print benchmark[1]['synset_es_results'][:5]
	return benchmark

def get_sslda_istex_ids(path='data/CSV_merged/', verbose=1):
	files = [join(path, f) for f in listdir(path) if isfile(join(path, f))]
	# Merge the istex_id(s) and export the unique ones in a merged CSV file
	merged = np.array([], dtype=np.object)
	for f in files:
		t = pd.read_csv(f,header=None, names=['istex_id'])
		t = np.array(t['istex_id'])
		merged = np.hstack((merged,t))

	unique_merged = np.unique(merged)
	if verbose:
		print "number of merged documents:", len(merged)
		print "number of unique documents:", len(unique_merged)
	return unique_merged.tolist()

def attach_topic_name(topic):
	topic_tokens = topic.split(" ")
	topic = ''
	for token in topic_tokens:
		topic+= token+"_"
	topic = topic[:-1]
	return topic

def gen_stst(topic, synset, synset_es_results, s3h_res, a=2):
	if type(s3h_res) is OrderedDict:
		s3h_res = s3h_res.items()
	topic_s3h_top100k_results = s3h_res[:100000]
	fus = np.array(range(100000))
	fus = fus * len(synset_es_results)
	for i, s3h in enumerate(topic_s3h_top100k_results):
		for j, bab in enumerate(synset_es_results):
			if s3h[0] == bab:
				fus[i] = (i + j) / 2
	fusion_df = pd.DataFrame(data=topic_s3h_top100k_results, columns=["istex_id", "3sh_score"])
	fusion_df["fus_rank"] = fus
	fusion_res = fusion_df.sort_values("fus_rank")

	n = a * len(synset_es_results)
	istex_ids = fusion_res["istex_id"].tolist()[:n]
	scores = np.zeros(n)
	for i in range(n):
		scores[i] = (n-i) / float(n)
	stst_res = pd.DataFrame(columns=["istex_id","score"])
	stst_res["istex_id"] = istex_ids
	stst_res["score"] = scores
	return stst_res

def gen_test_lst(test_lst):
	n = len(test_lst)
	scores = np.ones(n)
	test_lst_score_ones = pd.DataFrame(columns=["istex_id","score"])
	test_lst_score_ones["istex_id"] = test_lst
	test_lst_score_ones["score"] = scores
	return test_lst_score_ones

def gen_baseline_lst(baseline_lst):
	n = len(baseline_lst)
	scores = np.zeros(n)
	for i in range(n):
		scores[i] = (n-i) / float(n)
	baseline_lst_score = pd.DataFrame(columns=["istex_id","score"])
	baseline_lst_score["istex_id"] = baseline_lst
	baseline_lst_score["score"] = scores
	return baseline_lst_score

def gen_per_article(per_topic_dir):
	# This function was developed by fabien.rico@univ-lyon1.fr in March, 2018
	if not os.path.isdir(per_topic_dir):
		print "{} n'est pas un répertoire !".format(per_topic_dir)
		exit(1)

	fichs = os.listdir(per_topic_dir)
	print ("{} topics à traiter".format(len(fichs)))
	articles = {}
	numart = 0
	for fich in fichs:
		numart = numart+1
		print("{} : Traite '{}'".format(numart, fich))
		nomtop = fich
		nomtop = re.sub("^res__", "", nomtop)
		nomf = os.path.join(per_topic_dir, fich)
		all_articles = pickle.load(open(nomf, 'rb'))
		all_articles = np.array(all_articles).tolist()
		print("{} articles à traiter".format(len(all_articles)))
		nb = 0
		for art in all_articles:
			nb = nb+1
			if nb % 1000 == 1 and False:
				print("nb={}".format(nb))
			ident = art[0]
			score = float(art[1])
			if ident not in articles:
				articles[ident] = {}
			if score not in articles[ident]:
				articles[ident][score] = []
			articles[ident][score].append(nomtop)

	# tri de chaque articles
	print ("{} articles à trier".format(len(articles)))

	articles_tries = {}
	nb = 0
	for art in articles:
		nb = nb+1
		if nb % 10000 == 1 and False:
			print("nb={}".format(nb))
		articles_tries[art] = []
		score = articles[art].keys()
		score.sort()
		for s in score:
			articles_tries[art].append({"score":s, "topics":articles[art][s]})

	return articles_tries

def slda_res_prep(slda_topic_ids_fname, slda_res_fname):
	topic_ids_df = pd.read_csv(slda_topic_ids_fname, sep=';', header=None)
	topic_ids_dict = {}
	for (value, key) in np.array(topic_ids_df, dtype=np.str).tolist():
		topic_ids_dict[key] = value
	slda = pd.read_csv(slda_res_fname, sep=';', index_col='Doc_index')
	istex_ids = slda['Istex_Id']
	Topics = np.array(slda['Topics'])
	lda_out_res_dict = {}
	for (istex_id, Topics) in np.array(slda).tolist():
		topic_ids = Topics.split('_')
		n_t_ids = len(topic_ids)
		predicts = np.array(range(n_t_ids), dtype=np.object)
		for (i, topic_id) in enumerate(topic_ids):
			topic = [topic_ids_dict[topic_id]]
			predict = {}
			predict['score'] = 1.0
			predict['topics'] = topic
			predicts[i] = predict
		lda_out_res_dict[istex_id] = predicts.tolist()

	return lda_out_res_dict

def eval_res(per_articl_res_output, per_articl_test_output):
	#get intersection of istex_ids of both results and test
	results_istex_ids = per_articl_res_output.keys()
	test_istex_ids = per_articl_test_output.keys()
	intersection_lst = find_intersection(results_istex_ids, test_istex_ids)

	#build a dataframe of y_predict and y_test for intersection_lst
	evaluation = pd.DataFrame(data=intersection_lst, columns=["istex_id"])
	m = len(intersection_lst)
	y_predict = np.zeros(m, dtype=np.object)
	y_test = np.zeros(m, dtype=np.object)
	article_meta = np.zeros(m, dtype=np.object)
	article_pdf = np.zeros(m, dtype=np.object)
	is_predict_in_test = np.zeros(m, dtype=np.int)
	jaccard = np.zeros(m, dtype=np.float16)
	precision = np.zeros(m, dtype=np.float16)
	recall = np.zeros(m, dtype=np.float16)
	f1 = np.zeros(m, dtype=np.float16)
	hamming = np.zeros(m, dtype=np.float16)
	predict_label_cardinality = np.zeros(m, np.int)
	test_label_cardinality = np.zeros(m, np.int)
	predict_label_density = np.zeros(m, np.int)
	test_label_density = np.zeros(m, np.int)

	for (i, istex_id) in enumerate(intersection_lst):
		#y_predict
		topics = []
		for topic_dict in per_articl_res_output[istex_id]:
			topics.append(topic_dict['topics'][0])
		y_predict[i] = topics
		#y_test
		topics = []
		for topic_dict in per_articl_test_output[istex_id]:
			topics.append(topic_dict['topics'][0])
		y_test[i] = topics
		#is_predict_in_test?
		intersection_len = len(find_intersection(y_predict[i], y_test[i])) * 1.0	
		if intersection_len > 0:
			is_predict_in_test[i] = 1
		else: is_predict_in_test[i] = 0
		#jaccard
		union_len = len(list_union(y_predict[i], y_test[i]))
		jaccard[i] = intersection_len / union_len
		#precision, recall and f1
		precision[i] = intersection_len / len(y_predict[i])
		recall[i] = intersection_len / len(y_test[i])
		f1[i] = 2 * precision[i] * recall[i] / (precision[i] + recall[i])
		#hamming (35 is the number of labels)
		hamming[i] = len(list_xor(y_predict[i], y_test[i])) / 35.0
		#predict_label_cardinality
		predict_label_cardinality[i] = len(y_predict[i])
		#test_label_cardinality
		test_label_cardinality[i] = len(y_test[i])
		#predict_label_density
		#predict_label_density[i] = len(y_predict[i]) / 35.0
		#test_label_density
		#test_label_density[i] = len(y_test[i]) / 35.0
		#link to the article
		article_meta[i] = 'https://api.istex.fr/document/'+istex_id
		article_pdf[i] = article_meta[i]+'/fulltext/pdf'
	evaluation['y_predict'] = y_predict
	evaluation['y_test'] = y_test
	evaluation['is_predict_in_test'] = is_predict_in_test
	evaluation['jaccard'] = jaccard
	evaluation['precision'] = precision
	evaluation['recall'] = recall
	evaluation['f1'] = f1
	evaluation['hamming'] = hamming
	evaluation['predict_label_cardinality'] = predict_label_cardinality
	evaluation['test_label_cardinality'] = test_label_cardinality
	#evaluation['predict_label_density'] = predict_label_density
	#evaluation['test_label_density'] = test_label_density
	evaluation['article_meta'] = article_meta
	evaluation['article_pdf'] = article_pdf

	return evaluation

def uu_combine_eval(per_articl_slda_output, per_articl_res_output, per_articl_test_output):
	slda_istex_ids = per_articl_slda_output.keys()
	results_istex_ids = per_articl_res_output.keys()
	u_istex_ids = list_union(results_istex_ids,slda_istex_ids)
	n_test = per_articl_test_output.keys()
	intersection_lst = find_intersection(u_istex_ids, n_test)

	#build a dataframe of y_predict and y_test for intersection_lst
	evaluation = pd.DataFrame(data=intersection_lst, columns=["istex_id"])
	m = len(intersection_lst)
	y_predict = np.zeros(m, dtype=np.object)
	y_test = np.zeros(m, dtype=np.object)
	article_meta = np.zeros(m, dtype=np.object)
	article_pdf = np.zeros(m, dtype=np.object)
	is_predict_in_test = np.zeros(m, dtype=np.int)
	jaccard = np.zeros(m, dtype=np.float16)
	precision = np.zeros(m, dtype=np.float16)
	recall = np.zeros(m, dtype=np.float16)
	f1 = np.zeros(m, dtype=np.float16)
	hamming = np.zeros(m, dtype=np.float16)
	predict_label_cardinality = np.zeros(m, np.int)
	test_label_cardinality = np.zeros(m, np.int)
	#predict_label_density = np.zeros(m, np.int)
	#test_label_density = np.zeros(m, np.int)

	for (i, istex_id) in enumerate(intersection_lst):
		#y_predict
		topics = []
		if istex_id not in slda_istex_ids:
			for topic_dict in per_articl_res_output[istex_id]:
				topics.append(topic_dict['topics'][0])
		elif istex_id not in results_istex_ids:
			for topic_dict in per_articl_slda_output[istex_id]:
				topics.append(topic_dict['topics'][0])
		else: #in both
			for topic_dict in per_articl_slda_output[istex_id]:
				topics.append(topic_dict['topics'][0])
			for topic_dict in per_articl_slda_output[istex_id]:
				topics.append(topic_dict['topics'][0])
			topics = set(topics) #to get unique topics
			topics = list(topics)
		y_predict[i] = topics
		#y_test
		topics = []
		for topic_dict in per_articl_test_output[istex_id]:
			topics.append(topic_dict['topics'][0])
		y_test[i] = topics
		#is_predict_in_test?
		intersection_len = len(find_intersection(y_predict[i], y_test[i])) * 1.0
		if intersection_len > 0:
			is_predict_in_test[i] = 1
		else: is_predict_in_test[i] = 0
		#jaccard
		union_len = len(list_union(y_predict[i], y_test[i]))
		jaccard[i] = intersection_len / union_len
		#precision, recall and f1
		precision[i] = intersection_len / len(y_predict[i])
		recall[i] = intersection_len / len(y_test[i])
		f1[i] = 2 * precision[i] * recall[i] / (precision[i] + recall[i])
		#hamming (35 is the number of labels)
		hamming[i] = len(list_xor(y_predict[i], y_test[i])) / 35.0
		#predict_label_cardinality
		predict_label_cardinality[i] = len(y_predict[i])
		#test_label_cardinality
		test_label_cardinality[i] = len(y_test[i])
		#predict_label_density
		#predict_label_density[i] = len(y_predict[i]) / 35.0
		#test_label_density
		#test_label_density[i] = len(y_test[i]) / 35.0
		#link to the article
		article_meta[i] = 'https://api.istex.fr/document/'+istex_id
		article_pdf[i] = article_meta[i]+'/fulltext/pdf'
	evaluation['y_predict'] = y_predict
	evaluation['y_test'] = y_test
	evaluation['is_predict_in_test'] = is_predict_in_test
	evaluation['jaccard'] = jaccard
	evaluation['precision'] = precision
	evaluation['recall'] = recall
	evaluation['f1'] = f1
	evaluation['hamming'] = hamming
	evaluation['predict_label_cardinality'] = predict_label_cardinality
	evaluation['test_label_cardinality'] = test_label_cardinality
	#evaluation['predict_label_density'] = predict_label_density
	#evaluation['test_label_density'] = test_label_density
	evaluation['article_meta'] = article_meta
	evaluation['article_pdf'] = article_pdf

	return evaluation

def un_combine_eval(per_articl_slda_output, per_articl_res_output, per_articl_test_output):
	slda_istex_ids = per_articl_slda_output.keys()
	results_istex_ids = per_articl_res_output.keys()
	u_istex_ids = list_union(results_istex_ids,slda_istex_ids)
	n_test = per_articl_test_output.keys()
	intersection_lst = find_intersection(u_istex_ids, n_test)

	#build a dataframe of y_predict and y_test for intersection_lst
	evaluation = pd.DataFrame(data=intersection_lst, columns=["istex_id"])
	m = len(intersection_lst)
	y_predict = np.zeros(m, dtype=np.object)
	y_test = np.zeros(m, dtype=np.object)
	article_meta = np.zeros(m, dtype=np.object)
	article_pdf = np.zeros(m, dtype=np.object)
	is_predict_in_test = np.zeros(m, dtype=np.int)
	jaccard = np.zeros(m, dtype=np.float16)
	precision = np.zeros(m, dtype=np.float16)
	recall = np.zeros(m, dtype=np.float16)
	f1 = np.zeros(m, dtype=np.float16)
	hamming = np.zeros(m, dtype=np.float16)
	predict_label_cardinality = np.zeros(m, np.int)
	test_label_cardinality = np.zeros(m, np.int)
	#predict_label_density = np.zeros(m, np.int)
	#test_label_density = np.zeros(m, np.int)

	for (i, istex_id) in enumerate(intersection_lst):
		#y_predict
		topics = []
		if istex_id not in slda_istex_ids:
			for topic_dict in per_articl_res_output[istex_id]:
				topics.append(topic_dict['topics'][0])
		elif istex_id not in results_istex_ids:
			for topic_dict in per_articl_slda_output[istex_id]:
				topics.append(topic_dict['topics'][0])
		else: #in both
			topics_slda = []
			topics_res = []
			for topic_dict in per_articl_slda_output[istex_id]:
				topics_slda.append(topic_dict['topics'][0])
			for topic_dict in per_articl_slda_output[istex_id]:
				topics_res.append(topic_dict['topics'][0])
			topics = find_intersection(topics_slda, topics_res)
		y_predict[i] = topics
		#y_test
		topics = []
		for topic_dict in per_articl_test_output[istex_id]:
			topics.append(topic_dict['topics'][0])
		y_test[i] = topics
		#is_predict_in_test?
		intersection_len = len(find_intersection(y_predict[i], y_test[i])) * 1.0
		if intersection_len > 0:
			is_predict_in_test[i] = 1
		else: is_predict_in_test[i] = 0
		#jaccard
		union_len = len(list_union(y_predict[i], y_test[i]))
		jaccard[i] = intersection_len / union_len
		#precision, recall and f1
		precision[i] = intersection_len / len(y_predict[i])
		recall[i] = intersection_len / len(y_test[i])
		f1[i] = 2 * precision[i] * recall[i] / (precision[i] + recall[i])
		#hamming (35 is the number of labels)
		hamming[i] = len(list_xor(y_predict[i], y_test[i])) / 35.0
		#predict_label_cardinality
		predict_label_cardinality[i] = len(y_predict[i])
		#test_label_cardinality
		test_label_cardinality[i] = len(y_test[i])
		#predict_label_density
		#predict_label_density[i] = len(y_predict[i]) / 35.0
		#test_label_density
		#test_label_density[i] = len(y_test[i]) / 35.0
		#link to the article
		article_meta[i] = 'https://api.istex.fr/document/'+istex_id
		article_pdf[i] = article_meta[i]+'/fulltext/pdf'
	evaluation['y_predict'] = y_predict
	evaluation['y_test'] = y_test
	evaluation['is_predict_in_test'] = is_predict_in_test
	evaluation['jaccard'] = jaccard
	evaluation['precision'] = precision
	evaluation['recall'] = recall
	evaluation['f1'] = f1
	evaluation['hamming'] = hamming
	evaluation['predict_label_cardinality'] = predict_label_cardinality
	evaluation['test_label_cardinality'] = test_label_cardinality
	#evaluation['predict_label_density'] = predict_label_density
	#evaluation['test_label_density'] = test_label_density
	evaluation['article_meta'] = article_meta
	evaluation['article_pdf'] = article_pdf

	return evaluation


def eval_slda(per_articl_res_output, per_articl_test_output, per_articl_common_output):
	#get intersection of istex_ids of both results and test
	results_istex_ids = per_articl_res_output.keys()
	test_istex_ids = per_articl_test_output.keys()
	intersection_lst = find_intersection(results_istex_ids, test_istex_ids)
	common_istex_ids = per_articl_common_output.keys()
	intersection_lst = find_intersection(intersection_lst, common_istex_ids)

	#build a dataframe of y_predict and y_test for intersection_lst
	evaluation = pd.DataFrame(data=intersection_lst, columns=["istex_id"])
	m = len(intersection_lst)
	y_predict = np.zeros(m, dtype=np.object)
	y_test = np.zeros(m, dtype=np.object)
	article_meta = np.zeros(m, dtype=np.object)
	article_pdf = np.zeros(m, dtype=np.object)
	is_predict_in_test = np.zeros(m, dtype=np.int)
	jaccard = np.zeros(m, dtype=np.float16)
	precision = np.zeros(m, dtype=np.float16)
	recall = np.zeros(m, dtype=np.float16)
	f1 = np.zeros(m, dtype=np.float16)
	hamming = np.zeros(m, dtype=np.float16)
	predict_label_cardinality = np.zeros(m, np.int)
	test_label_cardinality = np.zeros(m, np.int)
	#predict_label_density = np.zeros(m, np.int)
	#test_label_density = np.zeros(m, np.int)

	for (i, istex_id) in enumerate(intersection_lst):
		#y_predict
		topics = []
		for topic_dict in per_articl_res_output[istex_id]:
			topics.append(topic_dict['topics'][0])
		y_predict[i] = topics
		#y_test
		topics = []
		for topic_dict in per_articl_test_output[istex_id]:
			topics.append(topic_dict['topics'][0])
		y_test[i] = topics
		#is_predict_in_test?
		intersection_len = len(find_intersection(y_predict[i], y_test[i])) * 1.0	
		if intersection_len > 0:
			is_predict_in_test[i] = 1
		else: is_predict_in_test[i] = 0
		#jaccard
		union_len = len(list_union(y_predict[i], y_test[i]))
		jaccard[i] = intersection_len / union_len
		#precision, recall and f1
		precision[i] = intersection_len / len(y_predict[i])
		recall[i] = intersection_len / len(y_test[i])
		f1[i] = 2 * precision[i] * recall[i] / (precision[i] + recall[i])
		#hamming (35 is the number of labels)
		hamming[i] = len(list_xor(y_predict[i], y_test[i])) / 35.0
		#predict_label_cardinality
		predict_label_cardinality[i] = len(y_predict[i])
		#test_label_cardinality
		test_label_cardinality[i] = len(y_test[i])
		#predict_label_density
		#predict_label_density[i] = len(y_predict[i]) / 35.0
		#test_label_density
		#test_label_density[i] = len(y_test[i]) / 35.0
		#link to the article
		article_meta[i] = 'https://api.istex.fr/document/'+istex_id
		article_pdf[i] = article_meta[i]+'/fulltext/pdf'
	evaluation['y_predict'] = y_predict
	evaluation['y_test'] = y_test
	evaluation['is_predict_in_test'] = is_predict_in_test
	evaluation['jaccard'] = jaccard
	evaluation['precision'] = precision
	evaluation['recall'] = recall
	evaluation['f1'] = f1
	evaluation['hamming'] = hamming
	evaluation['predict_label_cardinality'] = predict_label_cardinality
	evaluation['test_label_cardinality'] = test_label_cardinality
	#evaluation['predict_label_density'] = predict_label_density
	#evaluation['test_label_density'] = test_label_density
	evaluation['article_meta'] = article_meta
	evaluation['article_pdf'] = article_pdf

	return evaluation

def print_eval(evaluation):
	m = len(evaluation)
	is_predict_in_test = evaluation['is_predict_in_test']
	jaccard = evaluation['jaccard']
	precision = evaluation['precision']
	recall = evaluation['recall']
	f1 = evaluation['f1']
	hamming = evaluation['hamming']
	predict_label_cardinality = evaluation['predict_label_cardinality']
	test_label_cardinality = evaluation['test_label_cardinality']
	#predict_label_density = evaluation['predict_label_density']
	#test_label_density = evaluation['test_label_density']
	print "number of compared articles with the test set based on intersection: ", str(m)
	print "intersection accuracy score = ",str(is_predict_in_test.sum()/ float(m))
	print "mean Jaccard index = ",str(jaccard.sum()/ float(m))
	print "mean precision = ",str(precision.sum()/ float(m))
	print "mean recall = ",str(recall.sum()/ float(m))
	print "mean f1 = ",str(f1.sum()/ float(m))
	print "hamming loss = ",str(hamming.sum()/ float(m))
	print "predict_label_cardinality = ",str(predict_label_cardinality.sum()/ float(m))
	print "test_label_cardinality = ",str(test_label_cardinality.sum()/ float(m))
	#print "predict_label_density = ",str(predict_label_density.sum()/ float(m))
	#print "test_label_density = ",str(test_label_density.sum()/ float(m))

def decorate_slda_res(slda_topic_ids_fname, slda_istex_ids_fname, slda_score_res_fname):
	#get topic id dictionary
	topic_ids_df = pd.read_csv(slda_topic_ids_fname, sep=';', header=None)
	topic_ids_dict = {}
	for (value, key) in np.array(topic_ids_df, dtype=np.str).tolist():
		topic_ids_dict[key] = value

	#get an order list of istex_ids
	istex_ids_df = pd.read_csv(slda_istex_ids_fname, sep=';')
	istex_ids = istex_ids_df['Istex_Id'].tolist()

	# read sLDA results
	csv_reader = reader(open(slda_score_res_fname,'r'))
	r = 0
	lda_out_res_dict = {}
	for row in csv_reader:
		elements = str(row).split(';')[1:-1]
		m = len(elements)
		istex_id = istex_ids[r]
		topics = []
		for i in np.arange(0,m,2):
			topics.append(topic_ids_dict[str(i)])
		scores = []
		for i in np.arange(1,m+1,2):
			scores.append(elements[i])
		predicts = []
		for (i, topic) in enumerate(topics):
			if float(scores[i]) > 0.1:
				predict = {}
				predict['score'] = scores[i]
				predict['topics'] = [topic]
				predicts.append(predict)
			else:
				continue
		lda_out_res_dict[istex_id] = predicts
		r+= 1
	return lda_out_res_dict


if __name__ == "__main__" :
	parser = argparse.ArgumentParser()
	parser.add_argument("--verbose", default=1, type=int)
	parser.add_argument("--istex_benchmark_fname", default='data/istex_benchmark.p', type=str)
	parser.add_argument("--istex_lda_csv_merged_fname", default='data/CSV_merged/', type=str)
	parser.add_argument("--sslda", default=1, type=int)
	parser.add_argument("--sslda_istex_ids_fname", default='data/sslda_istex_ids.p', type=str)
	parser.add_argument("--svd_index_keys_fname", default='data/svd_index_keys.p', type=str)
	parser.add_argument("--svd_index_fname", default='data/svd_inversed_index.json', type=str)
	parser.add_argument("--topic_synsets_fname", default="data/35_topic_synsets.csv", type=str)
	parser.add_argument("--test_dir", default="test", type=str)
	parser.add_argument("--baseline_dir", default="baseline", type=str)
	parser.add_argument("--fusion_list_multiple", default=2, type=int)
	parser.add_argument("--out_dir", default="results", type=str)
	parser.add_argument("--per_article_dir", default="per_article_eval", type=str)
	parser.add_argument("--per_article_results_fname", default="per_article_results.p", type=str)
	parser.add_argument("--per_article_test_fname", default="per_article_test.p", type=str)
	parser.add_argument("--per_article_baseline_fname", default="per_article_baseline.p", type=str)
	parser.add_argument("--per_article_baseline_eval_fname", default="baseline_eval.csv", type=str)
	parser.add_argument("--per_article_eval_fname", default="evaluation.csv", type=str)
	parser.add_argument("--slda_res_fname", default="data/sLDA_p30/Docs_idsCom33TopicsAllPrior.csv", type=str)
	parser.add_argument("--slda_topic_ids_fname", default="data/sLDA_p30/TopicsIds.csv", type=str)
	parser.add_argument("--per_article_slda_eval_fname", default="slda_p30_eval.csv", type=str)
	parser.add_argument("--slda_istex_ids_fname", default="data/sLDA_p30/Docs_idsCom33TopicsAllPrior.csv", type=str)
	parser.add_argument("--slda_score_res_fname", default="data/sLDA_p30/resultsComs33AllPrior.csv", type=str)

	args = parser.parse_args()
	verbose = args.verbose
	istex_lda_csv_merged_fname = args.istex_lda_csv_merged_fname
	sslda = args.sslda
	sslda_istex_ids_fname = args.sslda_istex_ids_fname
	svd_index_keys_fname = args.svd_index_keys_fname
	istex_benchmark_fname = args.istex_benchmark_fname
	svd_index_fname = args.svd_index_fname
	topic_synsets_fname = args.topic_synsets_fname
	test_dir = args.test_dir
	if not os.path.exists(test_dir):
		os.makedirs(test_dir)
	baseline_dir = args.baseline_dir
	if not os.path.exists(baseline_dir):
		os.makedirs(baseline_dir)
	a = args.fusion_list_multiple	
	out_dir = args.out_dir
	if not os.path.exists(out_dir):
		os.makedirs(out_dir)
	per_article_dir = args.per_article_dir
	if not os.path.exists(per_article_dir):
		os.makedirs(per_article_dir)
	per_article_results_fname = args.per_article_results_fname
	per_article_test_fname = args.per_article_test_fname
	per_article_baseline_fname = args.per_article_baseline_fname
	per_article_baseline_eval_fname = args.per_article_baseline_eval_fname
	per_article_eval_fname = args.per_article_eval_fname
	slda_res_fname = args.slda_res_fname
	slda_topic_ids_fname =  args.slda_topic_ids_fname
	per_article_slda_eval_fname = args.per_article_slda_eval_fname
	slda_istex_ids_fname = args.slda_istex_ids_fname
	slda_score_res_fname =  args.slda_score_res_fname

	#load benchmark
	if not os.path.exists(istex_benchmark_fname):	
		if sslda:
			if not os.path.exists(sslda_istex_ids_fname):
				sslda_istex_ids = get_sslda_istex_ids(istex_lda_csv_merged_fname, verbose=verbose)
				pickle.dump(sslda_istex_ids, open(sslda_istex_ids_fname,'wb'))
			else:
				sslda_istex_ids = pickle.load(open(sslda_istex_ids_fname,'rb'))
		if not os.path.exists(svd_index_keys_fname):
			if svd_index_fname is not None:
				inversed_index = load_svd_index(svd_index_fname, verbose=verbose)
				svd_index_keys = inversed_index.keys()
				pickle.dump(svd_index_keys, open(svd_index_keys_fname,'wb'))
			elif sslda:
				svd_index_keys = sslda_istex_ids
			else:
				svd_index_keys = None
		else:
			svd_index_keys = pickle.load(open(svd_index_keys_fname,'rb'))

		if sslda:
			istex_ids_pool = find_intersection(sslda_istex_ids, svd_index_keys)
		else:
			istex_ids_pool = svd_index_keys

		benchmark = gen_benchmark(topic_synsets_fname, istex_ids_pool, verbose=verbose)
		pickle.dump(benchmark, open(istex_benchmark_fname,'wb'))
		if verbose:
			print "bechmark was generated and stored in: "+istex_benchmark_fname
	else:
		benchmark = pickle.load(open(istex_benchmark_fname,'rb'))
	if verbose:
		print "benchmark is successfully load"

	#generate ground truth folder for evaluation
	for i in range(len(benchmark)):
		topic = benchmark[i]['topic']
		topic = attach_topic_name(topic)
		output_fname = 'res__'+topic
		output_fname = os.path.join(test_dir,output_fname)
		if not os.path.exists(output_fname):
			test_lst = benchmark[i]['test']
			test_lst_score_ones = gen_test_lst(test_lst)
			pickle.dump(test_lst_score_ones, open(output_fname,'wb'))
			if verbose:
				print "test set of topic number "+str(i)+": "+topic+" is successfully written"	
		else:
			if verbose:
				print "++test file of the topic number "+str(i)+": "+topic+" is already there"	

	#generate baseline folder for evaluation
	for i in range(len(benchmark)):
		topic = benchmark[i]['topic']
		topic = attach_topic_name(topic)
		output_fname = 'res__'+topic
		output_fname = os.path.join(baseline_dir,output_fname)
		if not os.path.exists(output_fname):
			baseline_lst = benchmark[i]['synset_es_results']
			baseline_lst_score = gen_baseline_lst(baseline_lst)
			pickle.dump(baseline_lst_score, open(output_fname,'wb'))
			if verbose:
				print "test set of topic number "+str(i)+": "+topic+" is successfully written"	
		else:
			if verbose:
				print "++test file of the topic number "+str(i)+": "+topic+" is already there"	

	#loop on topics to generate fusion by loading top10K 3SH results
	# and mean_rank it with synset_es_results
	topic_synsets = pd.DataFrame.from_csv(topic_synsets_fname, header=0,
	                      sep=';', index_col=0, encoding=None,
	                      tupleize_cols=False,
	                      infer_datetime_format=False)
	topics = topic_synsets["topic"].tolist()
	synsets = topic_synsets["synset"].tolist()
	for (i, topic) in enumerate(topics):
		topic = attach_topic_name(topic)
		output_fname = 'res__'+topic
		output_fname = os.path.join(out_dir,output_fname)
		if not os.path.exists(output_fname):
			synset = synsets[i]
			synset_es_results = benchmark[i]['synset_es_results']
			s3h_res_pickle = 'data/s3h_results_per_topic/res__'+topic
			if os.path.exists(s3h_res_pickle):
				s3h_res = pickle.load(open(s3h_res_pickle,'rb'))
				stst_res = gen_stst(topic, synset, synset_es_results, s3h_res, a)
				pickle.dump(stst_res, open(output_fname,'wb'))
			else:
				print "--did not find s3h result of the topic number "+str(i)+": "+topic
		else:
			if verbose:
				print "++result of the topic number "+str(i)+": "+topic+" is already there"

	#generate per article results and per article test set for evaluation
	#for results
	per_articl_res_output_fname = os.path.join(per_article_dir, per_article_results_fname)
	if not os.path.exists(per_articl_res_output_fname):
		per_articl_res_output = gen_per_article(out_dir)
		pickle.dump(per_articl_res_output, open(per_articl_res_output_fname,'wb'))
		if verbose:
			print "per article results file was successfully written as: "+per_articl_res_output_fname
	else:
		per_articl_res_output = pickle.load(open(per_articl_res_output_fname,'rb'))
		if verbose:
			print "per article results file already exists and was successfully loaded"
	#for test
	per_articl_test_output_fname = os.path.join(per_article_dir, per_article_test_fname)
	if not os.path.exists(per_articl_test_output_fname):
		per_articl_test_output = gen_per_article(test_dir)
		pickle.dump(per_articl_test_output, open(per_articl_test_output_fname, 'wb'))
		if verbose:
			print "per article test file was successfully written as: "+per_articl_test_output_fname
	else:
		per_articl_test_output = pickle.load(open(per_articl_test_output_fname, 'rb'))
		if verbose:
			print "per article test file already exists and was successfully loaded"
	#for baseline (synset_es_results)
	per_articl_basline_output_fname = os.path.join(per_article_dir, per_article_baseline_fname)
	if not os.path.exists(per_articl_basline_output_fname):
		per_articl_basline_output = gen_per_article(baseline_dir)
		pickle.dump(per_articl_basline_output, open(per_articl_basline_output_fname, 'wb'))
		if verbose:
			print "per article baseline file was successfully written as: "+per_articl_test_output_fname
	else:
		per_articl_basline_output = pickle.load(open(per_articl_basline_output_fname, 'rb'))
		if verbose:
			print "per article baseline file already exists and was successfully loaded"

	#for slda
	per_articl_slda_test = slda_res_prep(slda_topic_ids_fname, slda_res_fname)
	per_articl_slda_output = decorate_slda_res(slda_topic_ids_fname, slda_istex_ids_fname, slda_score_res_fname)
	if verbose:
		print "per article slda file was successfully loaded"

#################################################### Evaluation #############

	#for U,U
#	print "uu:"
#	evaluation = uu_combine_eval(per_articl_slda_output, per_articl_res_output, per_articl_test_output)
#	if verbose:
#		print_eval(evaluation)
#	uu_evaluation_out_fname = os.path.join(per_article_dir, 'uu_eval.csv')
#	evaluation.to_csv(uu_evaluation_out_fname, sep=',', float_format=None,
#	 columns=None, header=True, index=True, index_label='index')

	#for u,n
#	print "un:"
#	evaluation = un_combine_eval(per_articl_slda_output, per_articl_res_output, per_articl_test_output)
#	if verbose:
#		print_eval(evaluation)
#	un_evaluation_out_fname = os.path.join(per_article_dir, 'un_eval.csv')
#	evaluation.to_csv(un_evaluation_out_fname, sep=',', float_format=None,
#	 columns=None, header=True, index=True, index_label='index')

	# slda evaluation
	print "intersection with test and Fusion method results: sLDA method"
	evaluation = eval_slda(per_articl_slda_output, per_articl_test_output, per_articl_res_output)
	slda_evaluation_out_fname = os.path.join(per_article_dir, per_article_slda_eval_fname)
	evaluation.to_csv(slda_evaluation_out_fname, sep=',', float_format=None,
	 columns=None, header=True, index=True, index_label='index')
	if verbose:
		print_eval(evaluation)

	# slda evaluation
	print "intersection only with sLDA test: sLDA method"
	evaluation = eval_res(per_articl_slda_output, per_articl_slda_test)
	slda_evaluation_out_fname = os.path.join(per_article_dir, per_article_slda_eval_fname)
	evaluation.to_csv(slda_evaluation_out_fname, sep=',', float_format=None,
	 columns=None, header=True, index=True, index_label='index')
	if verbose:
		print_eval(evaluation)

	# slda evaluation
	print "intersection only with out test: sLDA method"
	evaluation = eval_res(per_articl_slda_output, per_articl_test_output)
	slda_evaluation_out_fname = os.path.join(per_article_dir, per_article_slda_eval_fname)
	evaluation.to_csv(slda_evaluation_out_fname, sep=',', float_format=None,
	 columns=None, header=True, index=True, index_label='index')
	if verbose:
		print_eval(evaluation)

	print "intersection with test: Synset method"
	# baseline evaluation (synset_es_results)
	evaluation = eval_res(per_articl_basline_output, per_articl_test_output)
	baseline_evaluation_out_fname = os.path.join(per_article_dir, per_article_baseline_eval_fname)
	evaluation.to_csv(baseline_evaluation_out_fname, sep=',', float_format=None,
	 columns=None, header=True, index=True, index_label='index')
	if verbose:
		print_eval(evaluation)

	# our method evaluation
	print "intersection with sLDA: Fusion n="+str(a)
	evaluation = eval_slda(per_articl_res_output, per_articl_test_output, per_articl_slda_output)
	res_evaluation_out_fname = os.path.join(per_article_dir, per_article_eval_fname)
	evaluation.to_csv(res_evaluation_out_fname+'common_sLDA', sep=',', float_format=None,
	 columns=None, header=True, index=True, index_label='index')
	if verbose:
		print_eval(evaluation)

	# our method evaluation
	print "intersection only with test-set: Fusion n="+str(a)
	evaluation = eval_res(per_articl_res_output, per_articl_test_output)
	res_evaluation_out_fname = os.path.join(per_article_dir, per_article_eval_fname)
	evaluation.to_csv(res_evaluation_out_fname, sep=',', float_format=None,
	 columns=None, header=True, index=True, index_label='index')
	if verbose:
		print_eval(evaluation)
