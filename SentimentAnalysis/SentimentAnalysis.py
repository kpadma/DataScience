# coding: utf-8

# ## Tweet Collection

# In[ ]:

# Import necessary modules
import json
import time
from twitter import Twitter, OAuth, TwitterHTTPError, TwitterStream

# Variables that contains the user credentials to access Twitter API
ACCESS_TOKEN = ''
ACCESS_SECRET = ''
CONSUMER_KEY = ''
CONSUMER_SECRET = ''

oauth = OAuth(ACCESS_TOKEN, ACCESS_SECRET, CONSUMER_KEY, CONSUMER_SECRET)

# Initiate the connection to Twitter Streaming API
twitter_stream = TwitterStream(auth=oauth)

# Get tweets of particular keywords following through Twitter
iterator = twitter_stream.statuses.filter(track="TakeTheKnee,TakeAKnee, boycottnfl", language = "en")

# Print each tweet in the stream to the screen
tweet_count = 20000
for tweet in iterator:
    tweet_count -= 1
    # Convert back to the JSON format to print/score
    print json.dumps(tweet)

    # Pause every 5000 tweets
    if tweet_count == 15000 or tweet_count == 10000 or tweet_count == 5000:
        time.sleep(10)
    # Check if 20000 tweets are collected
    if tweet_count <= 0:
        print("Successful Tweet Collection of:", tweet_count)
        break


# ## Code for Sentiment Analysis

# In[1]:

# Import necessary modules
import json
from collections import defaultdict
import nltk
import re
import string
from nltk.corpus import stopwords
from textblob import TextBlob
import csv


# In[2]:

# Create a list of states 
statelist = ["Alabama","AL","Alaska","AK","Arizona","AZ","Arkansas","AR","California","CA","Colorado","CO","Connecticut","CT","Delaware","DE","Florida","FL","Georgia","GA","Hawaii","HI","Idaho","ID","Illinois","IL","Indiana","IN","Iowa","IA","Kansas","KS","Kentucky","KY","Louisiana","LA","Maine","ME","Maryland","MD","Massachusetts","MA","Michigan","MI","Minnesota","MN","Mississippi","MS","Missouri","MO","Montana","MT","Nebraska","NE","Nevada","NV","New Hampshire","NH","New Jersey","NJ","New Mexico","NM","New York","NY","North Carolina","NC","North Dakota","ND","Ohio","OH","Oklahoma","OK","Oregon","OR","Pennsylvania","PA","Rhode Island","RI","South Carolina","SC","South Dakota","SD","Tennessee","TN","Texas","TX","Utah","UT","Vermont","VT","Virginia","VA","Washington","WA","West Virginia","WV","Wisconsin","WI","Wyoming","WY","USA","United States", "United States of America"]
statelist = [x.upper() for x in statelist]
print statelist


# In[3]:

# Create a dictionary of states and their abbrevations
StatesDict = {"Alabama":"AL","Alaska":"AK","Arizona":"AZ","Arkansas":"AR","California":"CA","Colorado":"CO","Connecticut":"CT","Delaware":"DE","Florida":"FL","Georgia":"GA","Hawaii":"HI","Idaho":"ID","Illinois":"IL","Indiana":"IN","Iowa":"IA","Kansas":"KS","Kentucky":"KY","Louisiana":"LA","Maine":"ME","Maryland":"MD","Massachusetts":"MA","Michigan":"MI","Minnesota":"MN","Mississippi":"MS","Missouri":"MO","Montana":"MT","Nebraska":"NE","Nevada":"NV","New Hampshire":"NH","New Jersey":"NJ","New Mexico":"NM","New York":"NY","North Carolina":"NC","North Dakota":"ND","Ohio":"OH","Oklahoma":"OK","Oregon":"OR","Pennsylvania":"PA","Rhode Island":"RI","South Carolina":"SC","South Dakota":"SD","Tennessee":"TN","Texas":"TX","Utah":"UT","Vermont":"VT","Virginia":"VA","Washington":"WA","West Virginia":"WV","Wisconsin":"WI","Wyoming":"WY"}


# In[4]:

# Open the tweet file 
infile = open("20k_tweets.txt","r")

# Temporary variable declarations
uniqueLocation = []
temp = []
masterDict = defaultdict(list)
count = 0

# Create a dictionary of tweets based on location
for l in infile:
    t = json.loads(l.strip())
    if 'text' in t:
        if t['user']['location'] != None and t['user']['location'] not in uniqueLocation:
            uniqueLocation.append(t['user']['location'])
        masterDict[t['user']['location']].append(t['text'])

print masterDict


# In[5]:

# Function that cleans the key (location) of the tweets
def preprocess(k):
    tmp = []
    if k == None:
        k == "Unknown"
    else:
        txt = k.upper().split()
        tx = ",".join(t for t in txt if t in statelist)
        if tx != "":
            return tx


# In[6]:

# Calculate sentiment scores for each location
intermediateDict = defaultdict(list)
for key, val in masterDict.iteritems():
    txt = []
    snt = []
    s = []
    k = preprocess(key)
    for i in range(len(val)):  
        txt = TextBlob(val[i])
        snt.append(txt.sentiment.polarity)
        s = round(reduce(lambda x,y: x+y, snt),2)
    intermediateDict[k].append(s)
print intermediateDict


# In[7]:

# Aggregate the sentiment scores for each location
interimDict = {}
for k, v in intermediateDict.iteritems():
    v = reduce(lambda x,y: x + y, v)
    interimDict[k] = v
print interimDict


# In[8]:

# Create a dictionary of US states and a list of sentiment scores for each state
USdict = defaultdict(list)
for i,t in interimDict.iteritems():
    if i != None:
        spl = i.split(",")
        for j in spl:
            solstuff = j
            reqsol = solstuff.encode("ascii", "ignore")
            if len(reqsol) > 2:
                result = reqsol[:1].upper() + reqsol[1:].lower()
                for k,v in StatesDict.iteritems():
                    key = str(k)
                    if result in key:
                        #print v,t
                        USdict[key].append(t)
print USdict


# In[9]:

# Aggregate the sentiment score for each US state
dictUSonly = {}
for k,v in USdict.iteritems():
    v = reduce(lambda x,y: x + y, v)
    dictUSonly[k] = v
print dictUSonly


# In[10]:

# Write output of the dictionary to a csv file
with open('SentimentScoreOfUSstates.csv', 'wb') as csv_file:
    writer = csv.writer(csv_file)
    for key, value in dictUSonly.items():
        writer.writerow([key, value])


# ## Code for Topic Modeling

## Reference: https://medium.com/@aneesha/topic-modeling-with-scikit-learn-e80d33668730

# In[11]:

# Import necessary modules
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
import re, string
import nltk
import numpy as np  
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import glob
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity
from scipy.cluster.hierarchy import ward, dendrogram
from sklearn.manifold import MDS
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.decomposition import NMF
import string


# In[12]:

# Open the tweet file and extract only the tweets into a list
infile = open("20k_tweets.txt","r")
tweets_texts = []
for l in infile:
    t = json.loads(l.strip())
    tt = t['text']
    tt = tt.encode("ascii","ignore")
    tweets_texts.append(tt)
print tweets_texts


# In[13]:

# Function that cleans the tweets and generates tokens
def process_tweet_text(tweet):
    if tweet.startswith('@null'):
        return "[Tweet not available]"
    tweet = re.sub(r'RT','',tweet)
    tweet = re.sub(r"(@[A-Za-z0-9]+)","",tweet)
    tweet = re.sub(r'\$\w*','',tweet) # Remove tickers
    tweet = re.sub(r'https?:\/\/.*\/\w*','',tweet) # Remove hyperlinks    
    tweet = re.sub(r'['+string.punctuation+']+', ' ',tweet) # Remove puncutations like 's
    twtok = TweetTokenizer(strip_handles=True, reduce_len=True)
    tokens = twtok.tokenize(tweet)
    return tokens


# In[14]:

# Preprocess tweets
tweetList = []
for st in tweets_texts:
    tweetList.append(process_tweet_text(st))
print tweetList


# In[15]:

# Make a flat list of corpus
tweetList = [item for sublist in tweetList for item in sublist]
print tweetList


# In[16]:

# Create a DTM
no_features = 1000

tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, max_features=no_features, stop_words='english')
tfidf = tfidf_vectorizer.fit_transform(tweetList)
tfidf_feature_names = tfidf_vectorizer.get_feature_names()
print tfidf_feature_names


# In[17]:

# Create a NMF object
no_topics = 10
nmf = NMF(n_components=no_topics, random_state=1).fit(tfidf)


# In[18]:

# Function that displays the topics
def display_topics(model, feature_names, no_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print "Topic %d:" % (topic_idx)
        print " ".join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]])
# Number of words for each topic
no_top_words = 10
display_topics(nmf, tfidf_feature_names, no_top_words)


