#!/usr/bin/env python
# coding: utf-8

import os
import json
from pprint import pprint
from datetime import datetime
import math

from collections import defaultdict
import numpy as np

punctlist = ['.', '!', '?', ',', ';', ':', '"', "'", '(', ')', '--', '-', '...', '/', '\'', '@',
            '&', '>', '<', '=', '’', '^', '…', '_']
# split on punctuation and get rid of it

########### USEFUL FUNCTIONS ########################
def dot(arr1, arr2):
    if len(arr1) != len(arr2):
        print("vector dimensions don't match")
        return -1
    
    dot = 0
    for i in range(len(arr1)):
        dot += arr1[i]*arr2[i]
    return dot
        
def norm(arr):
    norm = 0
    for i in range(len(arr)):
        norm += math.pow(arr[i], 2)
    return math.sqrt(float(norm))
#####################################################

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

id_to_name = defaultdict(str)
doc_wordcounts = defaultdict(lambda: defaultdict(int)) # each document has a dict of word counts


for item in data:
    if item['text'] != None: 
        
        user_id = item['user_id']
        words = item['text']
        
        # ignore system and calendar messages
        if user_id == 'system' or user_id == 'calendar':
            continue
            
        # clean out punctuation
        for punct in punctlist:
            words = words.replace(punct, ' ')
        words = words.lower().split()
        
        # take out stopwords
        for w in words:
            if w in stopwords:
                # skip
                continue
            # add the words to the doc's word counts
            doc_wordcounts[user_id][w] += 1
        
        # potentially add user to id_to_name dict
        if user_id not in id_to_name:
            id_to_name[user_id] = item['name']
        
# now turn these raw counts into frequencies (normalize by doc length)
for doc in doc_wordcounts:
    word_counts = doc_wordcounts[doc]
    length = 0
    for word in word_counts:
        length += word_counts[word]
    for word in word_counts:
        word_counts[word] /= length

# now get idf for each term
dfs_t = defaultdict(int)
idfs_t = defaultdict(float)

termlist = set()
for doc in doc_wordcounts:
    termlist.update(doc_wordcounts[doc].keys())
    
for term in termlist:
    for doc in doc_wordcounts:
        if term in doc_wordcounts[doc]:
            dfs_t[term] += 1

num_docs = len(doc_wordcounts)
for term in termlist:
    idfs_t[term] = math.log(num_docs/float(dfs_t[term]))

tf_idfs = defaultdict(lambda: defaultdict(float))
for doc in doc_wordcounts:
    for term in doc_wordcounts[doc]:
        tf_idfs[doc][term] = doc_wordcounts[doc][term] * idfs_t[term]

# now, in each document, sort terms by tf-idf
# put top 10 into dict of vectors
vectors = defaultdict(list)

top_tfidf_words = set()
for doc in tf_idfs:

    # top ten words per doc by tf-idf
    for key_value_pair in sorted(tf_idfs[doc].items(),
           key=lambda k_v: k_v[1],
           reverse=True)[:10]:
        
        word = key_value_pair[0]
        tf_idf = key_value_pair[1]
        
        top_tfidf_words.add(word)
    
top_tfidf_words = list(top_tfidf_words)

for doc in tf_idfs:
    vec = []
    for word in top_tfidf_words:
        vec.append(tf_idfs[doc][word])
    vectors[doc] = vec


# make cluster matrix
matrix = np.zeros((num_docs, num_docs), dtype=float)
i = 0
j = 0

for doc1 in vectors:
    j=0
    if i>37:
        break
    for doc2 in vectors:
        if j>37:
            break
        
        # populate with cos(alpha)
        v1 = vectors[doc1]
        v2 = vectors[doc2]
        
        num = (dot(v1, v2))
        denom = norm(v1)*norm(v2)
        if denom == 0: # one of the vectors is populated with 0s?
            # then put the docs as far apart as possible (cos = 1)
            matrix[i][j] = 1
            continue
        matrix[i][j] = num/denom
        j += 1
    i += 1

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

doc_indices = list(vectors.keys())
similarities = defaultdict(float)

# only traverse half the matrix (since it's symmetric)
for i in range(len(matrix)):
    for j in range(i, len(matrix)):
        if i == j:
            continue
        doc_pair = (fakename[doc_indices[i]], fakename[doc_indices[j]])
        similarities[doc_pair] = matrix[i][j]

# print top ten similar docs
print("***** Top ten similar documents by tfidf *****")
for key_value_pair in sorted(similarities.items(),
           key=lambda k_v: k_v[1],
           reverse=True)[:10]:
    doc_pair = key_value_pair[0]
    cosine = key_value_pair[1]
    print("%s:\t\t%s" % (doc_pair, cosine))
