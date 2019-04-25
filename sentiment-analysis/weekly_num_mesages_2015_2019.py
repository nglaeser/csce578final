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
def num_messages(day):
	
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
   			message_counter+=1
   			#print("%s: %s" % (fake_name[item['user_id']], item['text']))
		
	return message_counter
	 
# set time period to run sentiment on
start_date = datetime(2015, 8, 16, 4, 0)
stop_date = datetime(2019, 4, 14, 4, 0)

# calc num of days the period is
the_time = stop_date - start_date
num_of_days = the_time.days

date_list = []

# monday is [0] and sunday is [6]

weekday = [0,0,0,0,0,0,0]

# run sentiment on specified period
while start_date <= stop_date:
	timestamp = start_date.replace(tzinfo=timezone.utc).timestamp()
	a = num_messages(timestamp)
	real_date = start_date - timedelta(days=1)
	date_list.append(real_date.date())
	
	# record score per day of week + counter of messages per weekday
	
	#this is sunday
	if real_date.weekday() == 6:
		weekday[0]+=a
	#this is monday..etc
	if real_date.weekday() == 0:
		weekday[1]+=a
	if real_date.weekday() == 1:
		weekday[2]+=a
	if real_date.weekday() == 2:
		weekday[3]+=a
	if real_date.weekday() == 3:
		weekday[4]+=a
	if real_date.weekday() == 4:
		weekday[5]+=a
	# this is saturday
	if real_date.weekday() == 5:
		weekday[6]+=a
	start_date+= timedelta(days=1)

# num of days counter
num_days = [0,1,2,3,4,5,6]
# what day correlates to what day from counter
days_week = ['','Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday',
'Friday', 'Saturday']

# plot mood per day
fig = plt.figure()
ax1 = fig.add_subplot(111)

print(weekday[0])
ax1.plot(num_days, weekday)
ax1.set_xticklabels(days_week)
plt.xticks(rotation=90)
plt.title('Number of Messages by Day of Week (8/16/2015 through 4/13/2019)')
plt.xlabel('Day of Week')
plt.ylabel('Number of Messages')
plt.margins(x=0.01)
plt.tight_layout()
plt.show()