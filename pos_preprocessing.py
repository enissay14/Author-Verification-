import os
import sys
import re
import numpy as np
import treetaggerwrapper
import time

#convert text to tags using TreeTagger wrapper for Python
def text_to_tags(text):
    tagger = treetaggerwrapper.TreeTagger(TAGLANG='en',TAGDIR='/home/yassine/EMSE 2015-2016/Projet Recherche/tree-tagger-linux-3.2')
    tags = treetaggerwrapper.make_tags(tagger.tag_text(unicode(text,encoding='utf-8')))
    pos_tags = []
    for pos in tags:
        pos_tags.append(pos[1])
    return " ".join(pos_tags)


path = '/home/yassine/EMSE 2015-2016/Projet Recherche/Author-Verification-/corpus-english-sample'

start_time = time.time()
for root, dirs, files in os.walk(path):
    for directory in sorted(dirs):
        #print directory[-2:]  #corpus
        for r, d, f in os.walk(path+'/'+directory):
            
            print 'Processing problem: '+directory
            
            #get documents for the problem: e.g known01.txt , known02.txt ... and unknown.txt
            for name in sorted(f):
                #POS Processing
                first_tags = []
                last_tags = []
                text_tags = []
                
                if not "tags" in name and not "first" in name and not "last" in name:
                    
                    print "[Part-Of-Speech] Processing file: "+ directory +"/"+ name + "... %f seconds elapsed" % (time.time() - start_time) 
                   
                   #converting text to pos tags sentence by sentence to keep first and last tags seperately
                    f = open(path+'/'+directory+'/'+name,"r")
                    train = f.read()
                    sentences = train.replace('\n','')
                    sentences = train.replace('.','\n').splitlines()
                    for s in sentences:
                        s = text_to_tags(s)
                        first_tags.append(" ".join(s.split()[0:1])) #add first pos tags of each sentence to 'first_tags'
                        last_tags.append(" ".join(s.split()[-1:]))  #add last pos tags of each sentence to 'last_tags'
                        text_tags.append(s)       #add all pos tags of each sentence to 'lext_tags'
                    
                    f_tags = open(path+'/'+directory+'/'+ re.sub('\.txt$', '', name)+'_tags.txt',"w") #write entire text in tags to file
                    f_tags.write(" ".join(text_tags))
                    f_tags.close()   
                    
                    f_first = open(path+'/'+directory+'/'+ re.sub('\.txt$', '', name)+'_first.txt',"w") #write first tags to file
                    f_first.write(" ".join(first_tags))
                    f_first.close() 
                    
                    f_last = open(path+'/'+directory+'/'+ re.sub('\.txt$', '', name)+'_last.txt',"w") #write last tags to file
                    f_last.write(" ".join(last_tags))
                    f_last.close() 
                    
              

