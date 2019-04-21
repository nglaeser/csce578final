#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python

import json, os, glob, subprocess
from pprint import pprint
import re

from collections import defaultdict
import nltk


# In[2]:


# variables
id_to_name = defaultdict(str)
doc_sentences = defaultdict(lambda: []) # each user (key) has a dict of sentences (value)


# In[3]:


# set up environment

try:
    project_dir = os.environ['PROJECT_PATH']
except:
    print("Please make sure you have set your PROJECT_PATH variable as described in the readme.")
    exit(0)


# In[4]:


######## read_sentences.ipynb ########

# read data from transcript
transcript_filename = input("Enter the name of your transcript file: ")

try:
    transcript_filepath = os.path.join(project_dir, transcript_filename)
    with open(transcript_filepath) as f:
        data = json.load(f)
except:
    print("Could not open the file " + str(transcript_filename) + " in " + os.path.split(transcript_filepath)[:-1][0])
    exit(0)

for item in data:
    if item['text'] != None:
        
        user_id = item['user_id']
        message = item['text']
        
        # ignore system and calendar messages
        if user_id == 'system' or user_id == 'calendar':
            continue
        
        # Split message if it contains newlines
        sents = message.split('\n')
        
        # separate message (or lines) into sentences
        sents = [item for s in sents for item in re.split('(?<=[.!?]) +',s)]
        # regex from https://stackoverflow.com/questions/14622835/split-string-on-or-keeping-the-punctuation-mark
        
        # add sentences to dict
        for sentence in sents:
            if len(sentence) == 0:
                # skip any empty sentences
                continue
            doc_sentences[user_id].append(sentence)
        
        # potentially add user to id_to_name dict
        if user_id not in id_to_name:
            id_to_name[user_id] = item['name']


# In[5]:


# print each user's sentences to a different file
# each sentence on a new line

os.makedirs('sentences', exist_ok=True) # make the sentences directory if it doesn't already exist

for user in doc_sentences:
    filename = str(user) + '_sentences.txt'
    # store them in their own directory
    filepath = os.path.join('sentences', filename)
    
    outfile = open(filepath, "w")
    
    sentences = doc_sentences[user]
    for sentence in sentences:
        outfile.write(sentence + "\n")
    outfile.close()


# In[8]:


####### parse_sentences.ipynb #########

# now do sentence complexity analysis

# set parser environment vars
os.environ['STANFORD_PARSER'] = '~/Downloads/stanford-parser-full-2018-10-17'
os.environ['STANFORD_MODELS'] = '~/Downloads/stanford-parser-full-2018-10-17'
os.environ['CLASSPATH'] = '~/Downloads/stanford-parser-full-2018-10-17/*'

# start up CoreNLP server

# start it as a background process:
#command = "java -mx4g -cp \"*\" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -preload tokenize,ssplit,pos,lemma,ner,parse,depparse -status_port 9000 -port 9000 -timeout 15000 &"
#subprocess.Popen(command.split())

# now connect to it
parser = nltk.parse.corenlp.CoreNLPParser(url='http://localhost:9000')


# In[9]:


# keep track of tree height and count
counts = defaultdict(lambda: [0,0]) # keep a tuple: total, count

# to store averages
average_depth = defaultdict(float)


# In[ ]:


data_path = glob.glob('sentences/*_sentences.txt')
parsed_path = 'parsed'
os.makedirs(parsed_path, exist_ok=True) # make the parsed directory if it doesn't already exist

for file in data_path:
    
    curr_file = open(file, "r")
    filename = os.path.split(file)[-1:][0]
    user = filename.split("_")[0]
    
    # write parsed sentences to a file
    # TODO: add functionality to read from file, or read only new stuff from the file
    parsed_file = os.path.join(parsed_path, str(user) + '_parsed.txt')
    outfile = open(parsed_file, 'w')

    for line in curr_file:
        
        # skip lines that contain unicode
        if any(ord(c) >= 128 for c in line):
            continue
        
        #try:
        parsed = parser.parse(line.split())
        print(parsed)
            # from answer by alvas on https://stackoverflow.com/questions/13883277/stanford-parser-and-nltk
            # also here: https://github.com/nltk/nltk/wiki/Stanford-CoreNLP-API-in-NLTK
        #except:
            #print("Could not parse this line: " + str(line))
            #print("Skipping...")
        #    continue

        outfile.write(parsed)
        
        tree = next(parsed)
        height = tree.height()
        
        # update height total
        counts[user][0] += height
        # update count
        counts[user][1] += 1
    
    if(counts[user][1] == 0):
        # avoid division by zero
        continue
    average_depth[user] = counts[user][0]/counts[user][1]


# In[9]:


# dict of names for each id
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


# In[10]:


######## results.ipynb ###########

results_file = open("results.txt", 'w')
results_latex = open("results_latex.txt", 'w')

# print sorted list in human-readable and latex table format
for key_value_pair in sorted(average_depth.items(),
           key=lambda k_v: k_v[1],
           reverse=True):
        
        user = key_value_pair[0]
        average = key_value_pair[1]
        
        results_latex.write("{} & {:.3f} \\\\".format(fake_name[user], average))
        results_file.write(fake_name[user] + "\t" + str(average) + "\n")

print("Processed " + str(len(average_depth)) + "users. Exiting...")

results_latex.close()
results_file.close()

