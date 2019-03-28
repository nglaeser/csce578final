#!/usr/bin/env python
# coding: utf-8

# In[202]:


import json
from pprint import pprint
from datetime import datetime
import math

from collections import defaultdict
import numpy as np


# In[203]:


punctlist = ['.', '!', '?', ',', ';', ':', '"', "'", '(', ')', '--', '-', '...', '/', '\'', '@',
            '&', '>', '<', '=', '’', '^', '…', '_']
# split on punctuation and get rid of it

stopwords = set(line.strip() for line in open('../stopwords.txt'))
stopwords.update(['www', 'http', 'https', 'com'])


# In[204]:


with open('transcript-15528094.json') as f:
    data = json.load(f)

#pprint(data[0])


# In[205]:


id_to_name = defaultdict(str)
doc_wordcounts = defaultdict(lambda: defaultdict(int)) # each document has a dict of word counts


# In[206]:


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
        
        #print("%s: %s" % (item['name'], item['text']))
        
#pprint(doc_wordcounts.keys())
#pprint(doc_wordcounts['21235813'])
#pprint(id_to_name)


# In[208]:


# now turn these raw counts into frequencies (normalize by doc length)
for doc in doc_wordcounts:
    word_counts = doc_wordcounts[doc]
    length = 0
    for word in word_counts:
        length += word_counts[word]
    for word in word_counts:
        word_counts[word] /= length


# In[209]:


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
#print(num_docs)
for term in termlist:
    idfs_t[term] = math.log(num_docs/float(dfs_t[term]))


# In[210]:


tf_idfs = defaultdict(lambda: defaultdict(float))
for doc in doc_wordcounts:
    for term in doc_wordcounts[doc]:
        tf_idfs[doc][term] = doc_wordcounts[doc][term] * idfs_t[term]


# In[221]:


# now, in each document, sort terms by tf-idf
# put top 10 into dict of vectors
vectors = defaultdict(list)

top_tfidf_words = set()
for doc in tf_idfs:
    #print('\n**** user %s **** ' % id_to_name[doc])
    
    # top ten words per doc by tf-idf
    for key_value_pair in sorted(tf_idfs[doc].items(),
           key=lambda k_v: k_v[1],
           reverse=True)[:10]:
        
        word = key_value_pair[0]
        tf_idf = key_value_pair[1]
        
        top_tfidf_words.add(word)
        
        #print("%s:\t%f" % (word, tf_idf))
    
top_tfidf_words = list(top_tfidf_words)
#print(len(top_tfidf_words))

for doc in tf_idfs:
    vec = []
    for word in top_tfidf_words:
        vec.append(tf_idfs[doc][word])
    vectors[doc] = vec
#print(len(vectors.keys()))


# In[213]:


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


# In[214]:


# make cluster matrix
matrix = np.zeros((num_docs, num_docs), dtype=float)
i = 0
j = 0

# pprint(vectors.keys())
# for doc in vectors:
#     print(doc)
# # check that they're in the same order

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


# In[184]:


#pprint(matrix)


# In[215]:


fake_name = defaultdict(str)
fake_name = {'10820250': 'Jack',
             '13207027': 'John',
             '14137183': 'Peter',
             '18559389': 'Anna',
             '18559680': 'Ethan',
             '19458971': 'Alex',
             '20963086': 'Kylie',
             '21235741': 'Jim',
             '21235813': 'Nathan',
             '21241679': 'Aaron',
             '21349729': 'Sarah',
             '21755924': 'Bob',
             '22207515': 'Claire',
             '22276941': 'Sally',
             '23354087': 'Mike',
             '24927292': 'Tim',
             '26498711': 'Nina',
             '26508082': 'Jacob',
             '26514517': 'Lisa',
             '26601370': 'Mitch',
             '270093': 'Alan',
             '270280': 'tom',
             '27380802': 'Jill',
             '27546073': 'Walter',
             '28068527': 'Simon',
             '28222063': 'Annie',
             '28880161': 'Chris',
             '29652195': 'Nancy',
             '29666163': 'Michael',
             '29747722': 'Jason',
             '29750591': 'Trey',
             '29775461': 'Nicole',
             '29775466': 'Liza',
             '29818898': 'Thomas',
             '29846726': 'Isaac',
             '45033570': 'Patrick',
             '46185459': 'Zo',
             '51427894': 'Natalie'}


# In[218]:


doc_indices = list(vectors.keys())
similarities = defaultdict(float)

# only traverse half the matrix (since it's symmetric)
for i in range(len(matrix)):
    for j in range(i, len(matrix)):
        if i == j:
            continue
        doc_pair = (fake_name[doc_indices[i]], fake_name[doc_indices[j]])
        similarities[doc_pair] = matrix[i][j]
#         if matrix[i][j] > 0.5:
#             print("%s and %s have a cosine of %s" % (doc_indices[i], doc_indices[j], matrix[i][j]))


# In[219]:


# print top ten similar docs
for key_value_pair in sorted(similarities.items(),
           key=lambda k_v: k_v[1],
           reverse=True)[:10]:
    doc_pair = key_value_pair[0]
    cosine = key_value_pair[1]
    print("%s:\t\t%s" % (doc_pair, cosine))


# In[ ]:




