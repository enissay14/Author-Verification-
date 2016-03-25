from sklearn import tree
from sklearn.externals import joblib
from sklearn import cross_validation
import numpy as np
import sys

if sys.argv[1] == '-cv':
    crossval = sys.argv[2]
else: 
    print "Missing parameter -cv numberOfCV"
    sys.exit()

print '~~~~~~//// Decision Tree Classfier \\\\\\~~~~~~'
print 'Reading Mtrain and Target file...'
#print np.column_stack((Mtrain,target))

#f_Mtrain = open('Mtrain',"r") 
Mtrain = np.load('Mtrain.npy')

#f_target = open('Target',"r") 
target = np.load('Target.npy')

if int(0.05*len(Mtrain)) == 0:
        min_samples = 1
else:
        min_samples = int(len(Mtrain)*0.05)

print 'Fitting a DecisionTree with min_samples = '+ str(min_samples)
clf = tree.DecisionTreeClassifier(min_samples_leaf=min_samples)
fi = clf.fit(Mtrain, target)

print 'feature importance:'
print fi.feature_importances_ 
joblib.dump(fi, 'DT_model.pkl',compress=9)

print 'Scores of '+crossval+' CV folds and mean:'
scores = cross_validation.cross_val_score(clf, Mtrain, np.array(target), cv=int(crossval))
print scores
print 'Accurancy : '
print scores.mean()