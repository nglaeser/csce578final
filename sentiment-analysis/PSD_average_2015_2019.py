# Copyright Nick Quan 2019

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from matplotlib import pyplot as plt
### using some import code for GroupMe by Noemi ###
import json, os
from pprint import pprint
from datetime import datetime, timezone, timedelta
import math
from collections import defaultdict
import numpy as np
from math import sqrt

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
def sentiment_for_day(day):
	
	# start time and end time for each day
	start_time = day - 86400
	end_time = day
	
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
   			
   			sentences.append(item['text'])
   			message_counter+=1
   			#print("%s: %s" % (fakename[item['user_id']], item['text']))

	# Run NLTK VADER
	# get polarity scores for each message
	VADER_scores = []
	analyzer = SentimentIntensityAnalyzer()
	#print('<<<NLTK VADER Sentiment Scores>>>')
	for sentence in sentences:
		senti = analyzer.polarity_scores(sentence)['compound']
		VADER_scores.append(senti)

	# Run TextBlob
	# get polarity scores for each message
	TextBlob_scores = []
	#print('<<<TextBlob Sentiment Scores>>>')
	for sentence in sentences:
		textBlob_sentence = TextBlob(sentence)
		senti = textBlob_sentence.sentiment.polarity
		TextBlob_scores.append(senti)
		
	
	return VADER_scores, TextBlob_scores
	 
# run sentiment on each day
start_date = datetime(2015, 8, 17, 4, 0)
stop_date = datetime(2019, 4, 14, 4, 0)
# 14 is end
VADER_list = []
TextBlob_list = []
date_list = []

# get data for each day
while start_date <= stop_date:
	timestamp = start_date.replace(tzinfo=timezone.utc).timestamp()
	a,b = sentiment_for_day(timestamp)
	real_date = start_date - timedelta(days=1)
	date_list.append(real_date.date())
	VADER_list.append(np.mean(a))
	TextBlob_list.append(np.mean(b))
	start_date+= timedelta(days=1)

# run Fourier transform and convert to Power Spectral Density graph to look for cycles
frate = 365
Pfft = np.fft.fft(VADER_list)
print(np.abs(Pfft))
Pfft[0] = 0  
freqs = np.fft.fftfreq(len(Pfft), 1. / frate)
abs = np.abs(Pfft)
abs_squared = abs**2
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(freqs, abs_squared)
ax1.set_xlim(0)
plt.show()

# running inverse Fourier transform

# combined = [list(a) for a in zip(freqs, Pfft)]	
# save_list = [0, 0.33302919708029194, 0.6660583941605839, 1.3321167883211678, 1.6651459854014596, 3.9963503649635035,
# 52.285583941605836, 99.242700729927, 104.23813868613138]
# for comb in combined:
# 	if comb[0] not in save_list:
# 		comb[1] = 0
# 			
# Pfft = [x[1] for x in combined]
# inverse_pfft = np.fft.ifft(Pfft)
# print(inverse_pfft)
# 
# fig = plt.figure()
# ax1 = fig.add_subplot(111)
# ax1.plot(date_list, inverse_pfft)
# ax1.set_xticklabels(date_list)
# plt.xticks(date_list)
# plt.xticks(rotation=90)
# plt.title('IFFT Spyral Mood 3 years')
# plt.xlabel('Date')
# plt.ylabel('Positivity Number')
# plt.legend(loc='upper center')
# plt.tight_layout()
# plt.show()
