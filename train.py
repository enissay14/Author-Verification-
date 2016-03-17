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

#convert text to tags using TreeTagger wrapper for Python
def text_to_tags(text):
    tagger = treetaggerwrapper.TreeTagger(TAGLANG='en',TAGDIR='/home/yassine/EMSE 2015-2016/Projet Recherche/tree-tagger-linux-3.2')
    tags = treetaggerwrapper.make_tags(tagger.tag_text(unicode(text,encoding='utf-8')))
    pos_tags = []
    for pos in tags:
        pos_tags.append(pos[1])
    return " ".join(pos_tags)


path = '/home/yassine/EMSE 2015-2016/Projet Recherche/Author-Verification-/corpus-english-sample'
filenames = []
filenames_tags = []
filenames_first = []
filenames_last = []

start_time = time.time()
for root, dirs, files in os.walk(path):
    for directory in sorted(dirs):
        #print directory[-2:]  #corpus
        for r, d, f in os.walk(path+'/'+directory):
            
            print '~~~~~~//// Processing Problem: '+directory+' \\\\\\~~~~~~'
            
            #get documents paths for the problem: e.g known01.txt , known02.txt ... and unknown.txt
            print 'Getting documents paths...'
            for name in sorted(f):
                if  not "tags" in name and not "first" in name and not "last" in name:
                    filenames.append(path+'/'+directory+'/'+name)                                           #populating filenames array
                if "tags" in name:
                    filenames_tags.append(path+'/'+directory+'/'+name)   #populating entire text in tags filenames array
                if "first" in name:
                    filenames_first.append(path+'/'+directory+'/'+name) #populating first tags filenames array
                if "last" in name:
                    filenames_last.append(path+'/'+directory+'/'+name)   #populating last tags filenames array
                    
              
        ### TASK 1 ###
        #Compute the documents representations using the paths stored in the 'filenames' array
        
            #char 8-gram tfidf feature
            print '[8-gram] Calculting tfidf for all documents...'
            tfidf = TfidfVectorizer(input='filename',use_idf=True,analyzer='char',ngram_range=(8,8))
            eight_char = tfidf.fit_transform(filenames).toarray() 
            
            #Lexical features: create 3 features vector: punct, vocab, phrase
            print '[Lexical Features] Calculating punct/vocab/phrases stats for all documents...'
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

            #contructing pos frequency and 4-pos grams
            print "[Part-Of-Speech]  Calculting n-POS tfidf for all documents... "
            freq_four_pos = TfidfVectorizer(input='filename',use_idf=False,analyzer='word',ngram_range=(4,4))
            four_pos = freq_four_pos.fit_transform(filenames_tags).toarray()

            #first and last pos tag freqency in each sentence
            print "[Part-Of-Speech]  Calculting first & last POS tag frequencies for all documents... "
            freq_pos = TfidfVectorizer(input='filename',use_idf=False,analyzer='word',ngram_range=(1,1))
            first_pos = freq_pos.fit_transform(filenames_first).toarray()
            last_pos = freq_pos.fit_transform(filenames_last).toarray()

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
            
            print 'calculating distances for Part-Of-Speach features...'
            four_posM = cosine_similarity(first_pos)
            first_posM = cosine_similarity(first_pos)
            last_posM = cosine_similarity(last_pos)
            
        ### TASK 3 ###
        #strore the new vector of distance as a new observation for the specific problem (in 'directory')
            
            #take the mean of the all the distances between known documents and the unknown doc
            print eight_charM[(len(eight_charM)-1),:(len(eight_charM)-1)].mean()
            print punctM[(len(punctM)-1),:(len(punctM)-1)].mean()
            print vocabM[(len(vocabM)-1),:(len(vocabM)-1)].mean()
            print phraseM[(len(phraseM)-1),:(len(phraseM)-1)].mean()
            print four_posM[(len(four_posM)-1),:(len(four_posM)-1)].mean()
            print first_posM[(len(first_posM)-1),:(len(first_posM)-1)].mean()
            print last_posM[(len(last_posM)-1),:(len(last_posM)-1)].mean()

            #empty 'filenames' arrays
            filenames = []
            filenames_tags = []
            filenames_first = []
            filenames_last = []
    
