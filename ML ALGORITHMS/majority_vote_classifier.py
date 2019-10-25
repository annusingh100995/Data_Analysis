from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.preprocessing import LabelEncoder
#from sklearn.externals import six
from sklearn.base import clone
from sklearn.pipeline import _name_estimators
import numpy as np
import operator

"""Implementing a weighted majority vote classifier"""

class MajorityVoteClassifier(BaseEstimator, ClassifierMixin):
    """ Majority Vote Ensemble Classifier
    Parameters 
        
        Classifiers: array-like, shape = [n_classifiers]
        Different classifiers for the ensemble

        vote: str, {'classlabel', 'probability'}
        If 'classlable' the prediction is based on the argmax of the class labels.
        Else id 'probability' the argmax is the sum of probabilities is used to predict the class label

        weights: array-like, shape = [n_classifiers]
        optional , default : None
        If a list of 'int' or 'float' values are provided the classifiers are weighted by importance
        """
    
    def __init__(self, classifiers, vote='classlabel', weights=None):
        self.classifiers = classifiers
        self.named_classifiers = {key:value for key, value in _name_estimators(classifiers)}
        self.vote = vote
        self.weights = weights
    def fit(self,X,y):
        """ 
            Fit classifiers
            
            Parameters:
            X: {array-like, sparse matrix}, shape = [n_samples, n_features]
            Matrix of trainign samples

            y: array-like, shape = [n_sampels]
            Vector os target class labels.

            Returns
            self: object  
        """
        #Use LabelEncoder to ensure that class labels start with 0, which si important for np.argmax call in self.predict
        self.lablenc_ = LabelEncoder()
        self.lablenc_.fit(y)
        self.classes_ = self.lablenc_.classes_
        self.classifiers_ = []
        for clf in self.classifiers:
            fitted_clf = clone(clf).fit(X, self.lablenc_.transform(y))
            self.classifiers_.append(fitted_clf)
        return self 
    def predict(self,X):

        """ Predict class labes for X.
        
            Parameters 
            X: [array-like, sparse matrix]
            shape = [n_samples, n_features]
            matrix of training samples

            Returns
            maj_vote: arrray-like shape =[n_samples]
            Predicted class labesl

        """
        if self.vote == 'probability':
            maj_vote = np.argmax(self.predict_proba(X), axis=1)

        else: # 'classlabel vote'
            predictions = np.asarray([clf.predict(X) for clf in self.classifiers_]).T
            maj_vote = np.apply_along_axis(lambda x: np.argmax(np.bincount(x, weights= self.weights)), axis=1, arr=predictions)
            maj_vote = self.lablenc_.inveres_transform(maj_vote)
            return maj_vote

    def predict_proba(self, X):
        """ Predict class probabilities for X
            Parameters: 
            X : array-like, sparse matrix, shape = [n_samples,n_features]
            Training vectors, where n_samples is the number of samples and n_features is the number of features
            Returns:
            avg_proba: array-like, shape = [n_samples, n_classes]
            Weighted average probability for each class per sample.
            """            
        probas = np.array([clf.predict_proba(X) for clf in self.classifiers])
        avg_proba = np.average(probas, axis=0, weights=self.weights)
        return avg_proba
    """
    def get_params(self, deep=True):
         Get classifers parameters names for GridSearch
        if not deep:
            return super(MajorityVoteClassifier,self).get_params(deep=False)
        else:
            out = self.named_classifiers.copy()
            for name, step in (self.named_classifiers).items():
                for key , value in (step.get_params(deep=True).items()):
                        out['%s__%s' %(name, key)] = value
            return out
    """

# Using majority vote principle to make predictions

from sklearn import datasets 
from sklearn.model_selection  import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import VotingClassifier

# Loading the iris data
iris = datasets.load_iris()
X, y = iris.data[50:, [1,2]], iris.target[50:]
le = LabelEncoder()
y = le.fit_transform(y)

#Splitting the data

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.5, random_state=1, stratify=y)

# Importig the relevant modules

from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline 
import numpy as np

# Defining the classifiers, with all the parameters set here
clf1 = LogisticRegression(penalty='l2', C=0.001, random_state=1, solver='lbfgs')
clf2 = DecisionTreeClassifier(max_depth=1,criterion='entropy', random_state=0)
clf3 = KNeighborsClassifier(n_neighbors=1, p=2, metric='minkowski')

# Creating a pipeline, scalling follwowed by the classifier
pipe1 = Pipeline([['sc',StandardScaler()], ['clf', clf1]])
pipe3 = Pipeline([['sc',StandardScaler()], ['clf', clf3]])

# Naming the class labels
clf_labels = ['Logistic regression', 'Decision Tres', 'KNN']
print('10- Fold Cross Validation:\n')

# Training the data,and pinting the classification accuracy
for clf, label in zip([pipe1,clf2,pipe3], clf_labels):
    scores = cross_val_score(estimator=clf, X = X_train, y = y_train, cv=10 , scoring = 'roc_auc')
    print("ROC AUC: %0.2f (+/- % 0.2f) [%s]" %(scores.mean(), scores.std(), label))


my_clf = MajorityVoteClassifier(classifiers =[pipe1,clf2,pipe3])
clf_labels +=['Majority Vote']
all_clf = [pipe1, clf2, pipe3, my_clf]
for clf, label in zip([pipe1,clf2,pipe3,my_clf], clf_labels):
    scores = cross_val_score(estimator=clf, X = X_train, y = y_train, cv=10 , scoring = 'roc_auc')
    print("ROC AUC: %0.2f (+/- % 0.2f) [%s]" %(scores.mean(), scores.std(), label))

