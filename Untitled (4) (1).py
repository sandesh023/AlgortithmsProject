#!/usr/bin/env python
# coding: utf-8

# In[12]:


import tweepy
from tweepy import Stream
from tweepy import OAuthHandler
from tweepy.streaming import StreamListener
import time
import argparse
import string
import config
import json
from tweepy import OAuthHandler
import re
import sys 
from pprint import pprint
from textblob import TextBlob
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import bigrams 
import operator 
from collections import Counter
from collections import defaultdict
import matplotlib.pyplot as plt
import math
import numpy as np


consumer_key = "AFu8JMZUNJIPDRB46NHnPDtZr"
consumer_secret = "P9NXobuwIzdHL3VkRSb4lqVdHdAxJ3xyAv76nPHWsWESRvd0hp"
access_token_key = "1098553588526137344-1oiL71I0IojM7abF3cZJUO2pqCNIoS"
access_token_secret = "JMvxsHNLRhsFzj7MPqcWZ9415IhLuyS9uiTKD03acv2sH"
 
auth = OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token_key, access_token_secret)
 
api = tweepy.API(auth)
user = api.me()

com = defaultdict(lambda : defaultdict(int))

emoticons_str = r"""
     (?:
        [:=;] # Eyes
        [oO\-]? # Nose (optional)
        [D\)\]\(\]/\\OpP] # Mouth
    )"""
 
regex_str = [
    emoticons_str,
    r'<[^>]+>', # HTML tags
    r'(?:@[\w_]+)', # @-mentions
    r"(?:\#+[\w_]+[\w\'_\-]*[\w_]+)", # hash-tags
    r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&amp;+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+', # URLs
 
    r'(?:(?:\d+,?)+(?:\.?\d+)?)', # numbers
    r"(?:[a-z][a-z'\-_]+[a-z])", # words with - and '
    r'(?:[\w_]+)', # other words
    r'(?:\S)' # anything else
]
    
tokens_re = re.compile(r'('+'|'.join(regex_str)+')', re.VERBOSE | re.IGNORECASE)
emoticon_re = re.compile(r'^'+emoticons_str+'$', re.VERBOSE | re.IGNORECASE)
 
def tokenize(s):
    return tokens_re.findall(s)
 
def preprocess(s, lowercase=False):
    tokens = tokenize(s)
    if lowercase:
        tokens = [token if emoticon_re.search(token) else token.lower() for token in tokens]
    return tokens
 
#def process_or_store(tweet):
#    print(json.dumps(tweet))

#for tweet in tweepy.Cursor(api.search,q=['modi','bjp,narendramodi'],count=10,lang="en").items(10):
#   process_or_store(tweet._json)
with open('mytweets.json','a') as f:
    for tweet in tweepy.Cursor(api.search,q=["Pulwama","Pulwama Attack","PulwamaAttack",'pulwama','pulwamaattack'],count=100000,lang="en").items(100000):
       t=tweet._json
       f.write( json.dumps(t) + '\n' )

with open('mytweets.json', 'r') as f:
    line = f.readline() 
    tweet = json.loads(line) 
    #print(json.dumps(tweet['text'], indent=4))
    
with open('mytweets.json', 'r') as f:
    for line in f:
        tweet = json.loads(line)
        #print(tweet['text'])  #print tweets
        tokens = preprocess(tweet['text'])

fname = 'mytweets.json'




with open(fname, 'r') as f:
    count_all = Counter()
    for line in f:
        tweet = json.loads(line)
        punctuation = list(string.punctuation)
        terms_all = [term for term in preprocess(tweet['text'])]
        stop = stopwords.words('english') + punctuation + ['rt', 'via','RT',"..."]
        terms_stop = [term for term in preprocess(tweet['text']) if term not in stop]
        terms_single = set(terms_all)
        terms_hash = [term for term in preprocess(tweet['text']) if term.startswith('#')]
        terms_only = [term for term in preprocess(tweet['text']) if term not in stop and not term.startswith(('#', '@'))] 
        terms_bigram = bigrams(terms_stop)
        l=[terms_stop,terms_single,terms_hash,terms_only,terms_bigram]
        for item in l:
            count_all.update(item)
    print("\nTop 5 most occuring words: ",count_all.most_common(5))
    for line in f: 
        tweet = json.loads(line)
        terms_only = [term for term in preprocess(tweet['text']) if term not in stop and not term.startswith(('#', '@'))]
 
    for i in range(len(terms_only)-1):            
        for j in range(i+1, len(terms_only)):
            w1, w2 = sorted([terms_only[i], terms_only[j]])                
            if w1 != w2:
                com[w1][w2] += 1
    com_max = []
    
    for t1 in com:
        t1_max_terms = sorted(com[t1].items(), key=operator.itemgetter(1), reverse=True)[:5]
        for t2, t2_count in t1_max_terms:
            com_max.append(((t1, t2), t2_count))
    terms_max = sorted(com_max, key=operator.itemgetter(1), reverse=True)
    print("\nTop 5 most cooccuring terms: ",terms_max[:5])
    search_word = ['attack','pulwamaattack','pulwama attack','Attack','Pulwama','PulwamaAttack','Pulwama Attack']
    count_search = Counter()
    for line in f:
        tweet = json.loads(line)
        terms_only = [term for term in preprocess(tweet['text']) if term not in stop and not term.startswith(('#', '@'))]
        for s_w in search_word: 
            if s_w in terms_only:
                count_search.update(terms_only)
    #print("Co-occurrence for %s:" % search_word)
    #print(count_search.most_common(20))
    p_t = {}
    p_t_com = defaultdict(lambda : defaultdict(int))
 
    for term, n in count_all.items():
        p_t[term] = n / 100000
        for t2 in com[term]:
            p_t_com[term][t2] = com[term][t2] / 100000
    positive_vocab = ['no war','peace','talk','negotiate','no war','political solution','save']
    negative_vocab = ['war','retaliate','destroy','attack','army solution','kill']
    pmi = defaultdict(lambda : defaultdict(int))
    for t1 in p_t:
        for t2 in com[t1]:
            denom = p_t[t1] * p_t[t2]
            pmi[t1][t2] = math.log2(p_t_com[t1][t2] / denom)
 
    semantic_orientation = {}
    for term, n in p_t.items():
        positive_assoc = sum(pmi[term][tx] for tx in positive_vocab)
        negative_assoc = sum(pmi[term][tx] for tx in negative_vocab)
        semantic_orientation[term] = positive_assoc - negative_assoc
    semantic_sorted = sorted(semantic_orientation.items(), key=operator.itemgetter(1),reverse=True)
    top_pos = semantic_sorted[:10]
    top_neg = semantic_sorted[-10:]
    pos_term=[]
    pos_occ=[]
    neg_term=[]
    neg_occ=[]
    for term in top_pos:
        pos_term=pos_term+[term[0]]
        pos_occ=pos_occ+[term[1]]
    for term in top_neg:
        neg_term=neg_term+[term[0]]
        neg_occ=neg_occ+[term[1]]
    np_p_occ=np.array(pos_occ)
    np_n_occ=np.array(neg_occ)
    print("\nTop 10 positive terms: ",pos_term)
    print("\nTop 10 negative terms: ",neg_term)
    plt.hist(pos_occ, bins=10, alpha=0.5, label='No War')
    plt.xticks(pos_occ,pos_term)
    plt.xlabel("Emotion")
    plt.ylabel("Frequency")
    plt.legend()
    plt.figure()
    plt.hist(neg_occ, bins=10, alpha=0.5, label='War')
    plt.xticks(neg_occ,neg_term)
    plt.xlabel("Emotion")
    plt.ylabel("Frequency")
    plt.legend()
    plt.figure()
    plt.hist(pos_occ, bins=10, alpha=0.5, label='No War')
    plt.hist(neg_occ, bins=10, alpha=0.5, label='War')
    plt.xlabel("Emotion")
    plt.ylabel("Frequency")
    plt.legend()
    


# In[ ]:




