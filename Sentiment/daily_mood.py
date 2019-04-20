# Copyright Nick Quan 2019

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from matplotlib import pyplot as plt
### using some import code for GroupMe by Noemi ###
import json
from pprint import pprint
from datetime import datetime, timezone, timedelta
import math
from collections import defaultdict
import numpy as np
from math import sqrt

# create stopwords set
stopwords = set(line.strip() for line in open('stopwords.txt'))
stopwords.update(['www', 'http', 'https', 'com'])

# open transcript
with open('transcript-15528094.json') as f:
    data = json.load(f)

# create fake names for each person
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
             '270280': 'Tom',
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
   			#print("%s: %s" % (fake_name[item['user_id']], item['text']))

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
	 
# set time period to run sentiment on
start_date = datetime(2015, 8, 16, 4, 0)
stop_date = datetime(2019, 4, 14, 4, 0)
# 14 is end
VADER_list = []
TextBlob_list = []
date_list = []

# run sentiment on specified period
while start_date <= stop_date:
	timestamp = start_date.replace(tzinfo=timezone.utc).timestamp()
	a,b = sentiment_for_day(timestamp)
	real_date = start_date - timedelta(days=1)
	date_list.append(real_date.date())

	VADER_list.append(np.mean(a))
	TextBlob_list.append(np.mean(b))
	start_date+= timedelta(days=1)

# plot data
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(date_list, VADER_list, label='NLTK VADER')
ax1.plot(date_list, TextBlob_list, label='TextBlob')
ax1.set_xticklabels(date_list)
#plt.xticks(date_list)
#plt.xticks(rotation=90)
plt.title('Spyral Mood (Average Daily Score) By Day (8/16/2015 through 4/13/2019)')
#plt.xlabel('Date')
plt.ylabel('Positivity Number')
plt.legend(loc='upper center')
plt.tight_layout()
plt.show()