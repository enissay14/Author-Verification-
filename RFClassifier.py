from sklearn.ensemble import RandomForestClassifier 
from sklearn.externals import joblib
from sklearn import cross_validation
import numpy as np
import sys

if sys.argv[1] == '-cv':
    crossval = sys.argv[2]
else: 
    print "Missing parameter -cv numberOfCV"
    sys.exit()
    
if sys.argv[3] == '-nt':
    trees = sys.argv[4]
else: 
    print "Missing parameter -nt numberOfTrees"
    sys.exit()

		
print '~~~~~~//// Random Forest Classfier \\\\\\~~~~~~'

print 'Reading Mtrain and Target file...'
Mtrain = np.load('Mtrain.npy')
target = np.load('Target.npy')

print 'Fitting a Random Forest Classifier with '+trees+' trees'
forest = RandomForestClassifier(n_estimators = int(trees))
fi = forest.fit(Mtrain, target)

print 'feature importance:'
print fi.feature_importances_ 
joblib.dump(fi, 'RF_model.pkl',compress=9)

print 'Scores of '+ crossval +' folds and mean:'
scores = cross_validation.cross_val_score(forest, Mtrain, np.array(target), cv = int(crossval))
print scores
print 'Accurancy : '
print scores.mean()