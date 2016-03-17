import os
import sys
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import treetaggerwrapper
import time
from sklearn.neighbors import DistanceMetric
from sklearn.metrics.pairwise import cosine_similarity
from lexical_features import punctuation
from lexical_features import vocabulary
from lexical_features import phrases

#Script Structure
path = '/home/yassine/EMSE 2015-2016/Projet Recherche/Author-Verification-/corpus-english-sample'
filenames = []
    
for root, dirs, files in os.walk(path):
    for directory in sorted(dirs):
        #print directory[-2:]  #corpus
        for r, d, f in os.walk(path+'/'+directory):
            
            print 'processing problem: '+directory
            
            #get documents for the problem: e.g known01.txt , known02.txt ... and unknown.txt
            for name in sorted(f):
                if  not "tags" in name and not "first" in name and not "last" in name:
                    filenames.append(path+'/'+directory+'/'+name) #populating filenames array
              
        ### TASK 1 ###
        #Compute the documents representations using the paths stored in the 'filenames' array
        
            #char 8-gram tfidf feature
            print 'calculting 8-gram tfidf for all documents...'
            tfidf = TfidfVectorizer(input='filename',use_idf=True,analyzer='char',ngram_range=(8,8))
            eight_char = tfidf.fit_transform(filenames).toarray() 
            
            #Lexical features: create 3 features vector: punct, vocab, phrase
            print 'Calculating lexical features for all documents...'
            i = 0
            for doc in filenames:
                    f = open(doc,"r")
                    train = f.read()
                    tmp1 = punctuation(train)
                    tmp2 = vocabulary(train)
                    tmp3 = phrases(train)
                    if i == 0: 
                            punct = tmp1
                            vocab = tmp2
                            phrase = tmp3
                            i += 1
                    else: 
                            punct = np.concatenate((punct,tmp1))
                            vocab = np.concatenate((vocab,tmp2))
                            phrase = np.concatenate((phrase,tmp3))

        ### TASK 2 ###
        #compute the distance between known and unknown documents for each feature
            
            #distance between known and unknown doc on 8-char feature
            print 'calculating distances for 8-gram feature...'
            eight_charM = cosine_similarity(eight_char)
            
            print 'calculating distances for lexical features...'
            punctM = cosine_similarity(punct)
            dist = DistanceMetric.get_metric('euclidean')
            vocabM = dist.pairwise(vocab)
            phraseM = np.corrcoef(phrase)
        
        ### TASK 3 ###
        #strore the new vector of distance as a new observation for the specific problem (in 'directory')
            
            #take the mean of the all the distances between known documents and the unknown doc
            print eight_charM[(len(eight_charM)-1),:(len(eight_charM)-1)].mean()
            print punctM[(len(punctM)-1),:(len(punctM)-1)].mean()
            print vocabM[(len(vocabM)-1),:(len(vocabM)-1)].mean()
            print phraseM[(len(phraseM)-1),:(len(phraseM)-1)].mean()

            #empty 'filenames' array
            filenames = []
    
