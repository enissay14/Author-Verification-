from sklearn.ensemble import RandomForestClassifier 
from sklearn.externals import joblib
from sklearn import cross_validation
import numpy as np
import sys

import numpy as np

from time import time
from operator import itemgetter
from scipy.stats import randint as sp_randint

from sklearn.grid_search import GridSearchCV
from sklearn.datasets import load_digits

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

# Utility function to report best scores
def report(grid_scores, n_top=3):
    top_scores = sorted(grid_scores, key=itemgetter(1), reverse=True)[:n_top]
    for i, score in enumerate(top_scores):
        print("Model with rank: {0}".format(i + 1))
        print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
              score.mean_validation_score,
              np.std(score.cv_validation_scores)))
        print("Parameters: {0}".format(score.parameters))
        print("")

# use a full grid over all parameters
param_grid = {"max_depth": [3, None],
              "max_features": [1, 3, 7],
              "min_samples_split": [1, 3, 10],
              "min_samples_leaf": [1, 3, 10],
              "bootstrap": [True, False],
              "criterion": ["gini", "entropy"]}

print 'Fitting a Random Forest Classifier with '+trees+' trees'
forest = RandomForestClassifier(n_estimators = int(trees))

# run grid search
grid_search = GridSearchCV(forest, param_grid=param_grid)
start = time()
grid_search.fit(Mtrain, target)

print("GridSearchCV took %.2f seconds for %d candidate parameter settings."
      % (time() - start, len(grid_search.grid_scores_)))
report(grid_search.grid_scores_)
