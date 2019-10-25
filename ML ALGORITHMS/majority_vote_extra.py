
from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
import numpy as np
import operator

class EnsembleClassifier(BaseEstimator, ClassifierMixin):
    """
    Ensemble classifier for scikit-learn estimators.

    Parameters
    ----------

    clf : `iterable`
      A list of scikit-learn classifier objects.
    weights : `list` (default: `None`)
      If `None`, the majority rule voting will be applied to the predicted class labels.
        If a list of weights (`float` or `int`) is provided, the averaged raw probabilities (via `predict_proba`)
        will be used to determine the most confident class label.

    """
    def __init__(self, clfs, weights=None):
        self.clfs = clfs
        self.weights = weights

    def fit(self, X, y):
        """
        Fit the scikit-learn estimators.

        Parameters
        ----------

        X : numpy array, shape = [n_samples, n_features]
            Training data
        y : list or numpy array, shape = [n_samples]
            Class labels

        """
        for clf in self.clfs:
            clf.fit(X, y)

    def predict(self, X):
        """
        Parameters
        ----------

        X : numpy array, shape = [n_samples, n_features]

        Returns
        ----------

        maj : list or numpy array, shape = [n_samples]
            Predicted class labels by majority rule

        """

        self.classes_ = np.asarray([clf.predict(X) for clf in self.clfs])
        if self.weights:
            avg = self.predict_proba(X)

            maj = np.apply_along_axis(lambda x: max(enumerate(x), key=operator.itemgetter(1))[0], axis=1, arr=avg)

        else:
            maj = np.asarray([np.argmax(np.bincount(self.classes_[:,c])) for c in range(self.classes_.shape[1])])

        return maj

    def predict_proba(self, X):

        """
        Parameters
        ----------

        X : numpy array, shape = [n_samples, n_features]

        Returns
        ----------

        avg : list or numpy array, shape = [n_samples, n_probabilities]
            Weighted average probability for each class per sample.

        """
        self.probas_ = [clf.predict_proba(X) for clf in self.clfs]
        avg = np.average(self.probas_, axis=0, weights=self.weights)

        return avg

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
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier


# Defining the classifiers, with all the parameters set here
clf1 = LogisticRegression(penalty='l2', C=0.001, random_state=1, solver='lbfgs')
clf2 = DecisionTreeClassifier(max_depth=1,criterion='entropy', random_state=0)
clf3 = KNeighborsClassifier(n_neighbors=1, p=2, metric='minkowski')
clf4 = RandomForestClassifier()
clf5 = GaussianNB()

# Creating a pipeline, scalling follwowed by the classifier
pipe1 = Pipeline([['sc',StandardScaler()], ['clf', clf1]])
pipe3 = Pipeline([['sc',StandardScaler()], ['clf', clf3]])

# Naming the class labels
clf_labels = ['Logistic regression', 'Decision Tres', 'KNN', 'Random Forest', 'naive Bayes']
print('10- Fold Cross Validation:\n')


for clf, label in zip([clf1, clf2, clf3, clf4, clf5], ['Logistic Regression', 'Decision Tree', 'KNN ', 'Random Forest', 'naive Bayes']):
    
    scores = cross_val_score(clf, X, y, cv=5, scoring='roc_auc')
    print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))


eclf = EnsembleClassifier(clfs=[clf1, clf2, clf3,clf4, clf5])
#eclf = EnsembleClassifier(clfs=[clf1, clf2, clf3])


for clf, label in zip([clf1, clf2, clf3,clf4,clf5, eclf], ['Logistic Regression', 'Decision Tree', 'KNN','Random Forest','Gaussian' 'Ensemble']):

    scores = cross_val_score(clf, X, y, cv=5, scoring='roc_auc')
    print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))
