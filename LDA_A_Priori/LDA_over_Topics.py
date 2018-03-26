mport json
import pandas as pd
import numpy as np
from os import listdir
from os.path import isfile, join
import re
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
from gensim import corpora
from gensim.models.ldamodel import LdaModel

from nltk.corpus import stopwords


def UnicodAndTokenize(line, stop_w) :
	line2 = ""
	stop = stopwords.words(stop_w)
	for w in line.split() :
		try :
			w_en=w.encode('utf-8','ignore').decode('utf-8')
			#print w_en, (w_en not in stop), (w_en not in [u'?',u'.',u'!', u":", u";"])
			if (w_en not in stop) and (w not in ['()', ')',',','?','.','!', ":", ";"]) :
				w=re.sub('[?.!:;,()]','',w)
				line2+=(w)+" "
				#print(w)
		except :
			if w[-1] in ['?','.','!'] :
				line2 += w[-1] + " "
	line = line2.rstrip() # remove last space if it exists

	lst = line.lower().split()
	lst = [ i for i in lst if not i.isdigit()]
	return lst

def findTopic(file_name):
    topic_name = file_name.split( "_")[0]
    return topic_name


def import_docs() :
	path="../IstexTopicsList/JSON_MergedTest"
	documents_list = []
	stop_word='english'
	count=0
	topics=[] # contain the label of the community and the id of the community

	documents_topic=dict()

	for f in listdir(path):
		f_opened=open(join(path, f))
		data=json.load(f_opened)
		f_opened.close()
		topic_f = findTopic(f)
		if topic_f not in topics :
			topics.append(topic_f)
		for doc in data :
			if doc["istex_id"] not in documents_topic :
				documents_topic[doc["istex_id"]]=[topic_f]
				line = doc["title"]+". " + doc["abstract"]
				documents_list.append(UnicodAndTokenize(line, stop_word))
				count+=1
			if topics.index(topic_f) not in documents_topic[doc["istex_id"]] :
				documents_topic[doc["istex_id"]].append(topics.index(topic_f))
	docs_ids = open("Docs_idsCom.csv", "w")
	count_doc=0
	docs_ids.write("Doc_index;Istex_Id;Topics\n")
	for d in documents_topic :
		t=""
		for i in documents_topic[d] :
			t+=str(i)+"_"
		if len(t) > 1 :
			t=t[:-1]
		docs_ids.write("%d;%s;%s\n"%(count_doc, d, t))
		count_doc+=1
	docs_ids.close()  
	return documents_list 

documents_list=import_docs()
dictionary=corpora.Dictionary(documents_list)
# Converting list of documents (corpus) into Document Term Matrix using the dictionary

doc_term_matrix = [dictionary.doc2bow(doc) for doc in documents_list]

corpora.MmCorpus.serialize('corpus.mm', doc_term_matrix)
nbTopics=33
nbPasses=5
# LDA model by Gensim with prior parameter (eta)
# TO BE DONE

####################
ldamodel = LdaModel(doc_term_matrix, num_topics=nbTopics, id2word = dictionary, passes=nbPasses)
# to have the topic of a document : ldamodel[doc]
topics = ldamodel.get_document_topics(doc_term_matrix, per_word_topics=True)


all_topics = [(doc_topics, word_topics, word_phis) for doc_topics, word_topics, word_phis in topics]
doc_topics, word_topics, word_phis = all_topics[1]
f_res=open("resultsComs.csv", "w")

count=0
for d in all_topics:
	for i in range(len(d[0])):
		f_res.write("%d;%d;%f;"%(count, d[0][i][0], d[0][i][1]))
	f_res.write("\n")
	count+=1
f_res.close()


