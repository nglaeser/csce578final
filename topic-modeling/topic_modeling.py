#Copyright (c) 2019 Nicolas Quan

import os
import re
from collections import defaultdict
from operator import itemgetter
from gensim import corpora, matutils
from gensim.models import LsiModel, TfidfModel
from nltk.tokenize import RegexpTokenizer
from gensim.models.coherencemodel import CoherenceModel
from sklearn.cluster import KMeans
import json
from pprint import pprint
from datetime import datetime, timezone, timedelta
import math
from collections import defaultdict
import numpy as np
from math import sqrt

# list of punc to exclude from analysis
end_punc = ['.', '!','?']

# create nickname set
nicknames = set(['Also', 'When', 'Can', 'Vu', 'Liz' ,'CDC', 'The', 'We', 'It',
 'If', 'That', 'This', 'You', 'Oh', 'Like', 'What', 'Let', 'Are'])

try:
    project_dir = os.environ['PROJECT_PATH']
except:
    print("Please make sure you have set your PROJECT_PATH variable as described in the readme.")
    exit(0)

try:
    stopword_file = os.path.join(project_dir, 'utilities/stopwords.txt')
    stopwords = set(line.strip() for line in open(stopword_file))
    stopwords.update(['www', 'http', 'https', 'com'])
except:
    print("Stopword file " + str(stopword_file) + "not found")
    exit(0)

transcript_filename = input("Enter the name of your transcript file: ")
try:
    transcript_filepath = os.path.join(project_dir, transcript_filename)
    with open(transcript_filepath) as f:
        data = json.load(f)
except:
    print("Could not open the file " + str(transcript_filename) + " in " + os.path.split(transcript_filepath)[:-1][0])
    exit(0)

fakename = defaultdict(str)

fakename_file = os.path.join(project_dir, 'utilities/fakenames.txt')
try:
    f = open(fakename_file, 'r')
except:
    print("Could not find fakenames.txt in the utilities directory of your project path.")
    exit(0)

try:
    for line in f:
        tokens = line.split()
        fakename[tokens[0]] = tokens[1]
except:
    print("The file utilities/fakenames.txt is improperly formatted.")
    exit(0)
 
# last UTC message day time in UTC Epoch = 1555214400
last_day = 1555214400
day_length = 86400

# function that calculates sentiment for VADER and TextBlob
def get_data(day):
	
	# start time and end time for each day
	start_time = day - 86400
	end_time = day
	#print(end_time)
	# count num of messages
	message_counter = 0
	# create id to name dict     
	id_to_name = defaultdict(str)
	
	# create list of sentences
	sentences = []
	
	# add sentences for time period
	for item in data:
   		if item['text'] != None and (start_time < item['created_at'] < end_time): 
   			user_id = item['user_id']
   			words = item['text']
   			
   			# ignore system and calendar messages
   			if user_id == 'system' or user_id == 'calendar':
   				continue
   			# take out stopwords
   			for w in words:
   				if w in stopwords:
   					# skip
   					continue
   			# potentially add user to id_to_name dict
   			if user_id not in id_to_name:
   				id_to_name[user_id] = item['name']
   			
   			sentences.append(str(item['text']))
   			
   			# add nickname to set if not in already
   			nickname = item['name']
   			nickname_split = nickname.split(' ')
   			for nick in nickname_split:
   				if len(nick) >= 4:
   					nicknames.add(nick)

   			message_counter+=1
   			#print("%s: %s" % (fake_name[item['user_id']], item['text']))	
	sentences = ''.join(sentences)
	return sentences
	
# method to convert strings into tokens
def preprocess(docs):
	tokenized_docs = []
	tokenizer = RegexpTokenizer(r'\w+')
	for doc in docs:
		tokens = tokenizer.tokenize(doc)
		pre_final_tokens = [token for token in tokens if not token in stopwords]
		final_tokens = [token for token in pre_final_tokens if token[0].isupper()]
		tokenized_docs.append(final_tokens)
	return tokenized_docs

# method to create corpus to be used for lsi
def create_corpus(final_docs):
	# create term dictionary
	dict = corpora.Dictionary(final_docs)
	# create doc-term matrix using term dictionary
	doc_term_matrix = []
	for doc in final_docs:
		doc_term_matrix.append(dict.doc2bow(doc))
	# convert doc_term matrix to tfidf
	tfidf = TfidfModel(doc_term_matrix)
	tfidf_doc_term_matrix = tfidf[doc_term_matrix]
	return dict, tfidf_doc_term_matrix

# method to create lsi models and find most coherent number of topics
def get_coherence_values(dict, doc_term_matrix, final_docs, stop, start, step):
	lsi_models = []
	coherence_values = []
	for num_topics in range(start, stop, step):
		# create lsi models
		lsi_model = LsiModel(doc_term_matrix, num_topics=num_topics, id2word=dict)
		lsi_models.append(lsi_model)
		# get coherence values for each model
		coherence_model = CoherenceModel(model=lsi_model, texts=final_docs, dictionary=dict, coherence='c_v')
		coherence_values.append(coherence_model.get_coherence())
	print("Coherence values", coherence_values)
	return lsi_models, coherence_values
    
# set time period to run on
# start_date = datetime(2015, 8, 16, 4, 0)
# stop_date = datetime(2019, 4, 14, 4, 0)

def run_on_time_period(start, stop):
	# create lists to hold data
	start_date = start
	stop_date = stop
	date_list = []
	raw_docs = []

	# run the data getter
	while start_date <= stop_date:
		timestamp = start_date.replace(tzinfo=timezone.utc).timestamp()
		doc = get_data(timestamp)
		raw_docs.append(doc)
		real_date = start_date - timedelta(days=1)
		date_list.append(real_date.date())
		start_date+= timedelta(days=1)

	# make list of docs without name

	for i in range(len(raw_docs)):
		for name in nicknames:
			if name in raw_docs[i]:
				raw_docs[i] = raw_docs[i].replace(name, '')
			
	final_docs = preprocess(raw_docs)
	dict, doc_term_matrix = create_corpus(final_docs)
	# lsi_models, coherence_values = get_coherence_values(dict, doc_term_matrix, final_docs, 10, 1, 2)
	lsi_model = LsiModel(doc_term_matrix, num_topics=10, id2word=dict)
	counter = 1
	print(lsi_model.print_topics(num_topics=5, num_words=5))

# find topics for each month
cc = 1
while cc < 13:
	print('Month: '+str(cc))
	start_date = datetime(2018, cc, 1, 4, 0)
	stop_date = datetime(2018, cc, 28, 4, 0)
	run_on_time_period(start_date, stop_date)
	cc+=1


