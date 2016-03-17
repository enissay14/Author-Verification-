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
                    
            #Compute the documents representations using the paths stored in the 'filenames' array
            
            #char 8-gram tfidf feature
            print 'calculting 8-gram tfidf for all documents...'
            tfidf = TfidfVectorizer(input='filename',use_idf=True,analyzer='char',ngram_range=(8,8))
            eight_char = tfidf.fit_transform(filenames).toarray() 

            #distance between known and unknown doc on 8-char feature
            print 'calculating distances for 8-gram feature...'
            eight_charM = cosine_similarity(eight_char)
            print eight_charM
            
            
            
            #compute the distance between known and unknown documents for each feature
            
            #strore the new vector of distance as a new observation for the specific problem (in 'directory')
