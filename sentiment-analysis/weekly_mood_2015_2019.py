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
	 
# set time period to run sentiment on
start_date = datetime(2015, 8, 17, 4, 0)
stop_date = datetime(2019, 4, 14, 4, 0)

# calc num of days the period is
the_time = stop_date - start_date
num_of_days = the_time.days

# 14 is end
VADER_list = []
TextBlob_list = []
date_list = []

# monday is [0] and sunday is [6]
w1 = [[],[]]
w2 = [[],[]]
w3 = [[],[]]
w4 = [[],[]]
w5 = [[],[]]
w6 = [[],[]]
w7 = [[],[]]
weekday = [w1,w2,w3,w4,w5,w6,w7]

# run sentiment on specified period
while start_date <= stop_date:
	timestamp = start_date.replace(tzinfo=timezone.utc).timestamp()
	a,b = sentiment_for_day(timestamp)
	real_date = start_date - timedelta(days=1)
	date_list.append(real_date.date())
	
	# record score per day of week + counter of messages per weekday
	
	#this is monday
	if real_date.weekday() == 0:
		weekday[0][0].extend(a)
		weekday[0][1].extend(b)
	#this is tuesday..etc
	if real_date.weekday() == 1:
		weekday[1][0].extend(a)
		weekday[1][1].extend(b)
	if real_date.weekday() == 2:
		weekday[2][0].extend(a)
		weekday[2][1].extend(b)
	if real_date.weekday() == 3:
		weekday[3][0].extend(a)
		weekday[3][1].extend(b)
	if real_date.weekday() == 4:
		weekday[4][0].extend(a)
		weekday[4][1].extend(b)
	if real_date.weekday() == 5:
		weekday[5][0].extend(a)
		weekday[5][1].extend(b)
	if real_date.weekday() == 6:
		weekday[6][0].extend(a)
		weekday[6][1].extend(b)
	
	VADER_list.append(np.mean(a))
	TextBlob_list.append(np.mean(b))
	start_date+= timedelta(days=1)

# num of days counter
num_days = [0,1,2,3,4,5,6]
# what day correlates to what day from counter
days_week = ['','Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday',
'Friday', 'Saturday']
# get vader scores for each day of week, starting with Sunday
# normalize by num messages in day of week
weekday_VADER = [np.mean(weekday[6][0]), np.mean(weekday[0][0]),
np.mean(weekday[1][0]), np.mean(weekday[2][0]), np.mean(weekday[3][0]),
np.mean(weekday[4][0]), np.mean(weekday[5][0])]
# get blob scores for each day of week, starting with sunday
# normalize by num messages day of week 
weekday_blob = [np.mean(weekday[6][1]), np.mean(weekday[0][1]),
np.mean(weekday[1][1]), np.mean(weekday[2][1]), np.mean(weekday[3][1]),
np.mean(weekday[4][1]), np.mean(weekday[5][1])]

# plot mood per day
fig = plt.figure()
ax1 = fig.add_subplot(111)

VADER_error = []
err = (np.std(weekday[6][0])/(sqrt(num_of_days/7)))*1.96
VADER_error.append(err)
for x in range(0,6):
	err = (np.std(weekday[x][0])/(sqrt(num_of_days/7)))*1.96
	VADER_error.append(err)
blob_error = []
err = (np.std(weekday[6][1])/(sqrt(num_of_days/7)))*1.96
blob_error.append(err)
for x in range(0,6):
	err = (np.std(weekday[x][1])/(sqrt(num_of_days/7)))*1.96
	blob_error.append(err)

ax1.plot(num_days, weekday_VADER, label='NLTK VADER')
ax1.plot(num_days, weekday_blob, label='TextBlob')
ax1.errorbar(num_days, weekday_VADER, yerr=VADER_error)
ax1.errorbar(num_days, weekday_blob, yerr=blob_error)
ax1.set_xticklabels(days_week)
plt.xticks(rotation=90)
plt.title('Spyral Mood (Average Daily Score) By Day of Week (8/16/2015 through 4/13/2019)')
plt.xlabel('Day of Week')
plt.ylabel('Positivity Score')
plt.legend(loc='upper left')
plt.margins(x=0.01)
plt.tight_layout()
plt.show()
