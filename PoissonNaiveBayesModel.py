import numpy as np
from scipy.stats import poisson
from scipy.special import logsumexp
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

"""defines a custom PoissonNB class (since scikit-learn does not have a built in Poisson Naive Bayes Classifier"""
class PoissonNB(BaseEstimator, ClassifierMixin):
    def __init__(self, alpha=1e-6):
        self.alpha = alpha
    def fit(self, X, y):
        X, y = check_X_y(X, y, accept_sparse=False)
        self.classes_ = unique_labels(y)
        
        n_classes = len(self.classes_)
        n_features = X.shape[1]
        
        self.class_prior_ = np.zeros(n_classes)
        self.feature_lambda_ = np.zeros((n_classes, n_features))
        
        for idx, c in enumerate(self.classes_):
            X_c = X[y==c]
            self.class_prior_[idx] = X_c.shape[0] / X.shape[0]
        
            self.feature_lambda_[idx,:] = np.mean(X_c, axis=0) + self.alpha    
        return self
    
    def predict_proba(self,X):
        check_is_fitted(self)
        X = check_array(X)
        
        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        log_probas = np.zeros((n_samples, n_classes))
        
        for idx in range(n_classes):
            prior = np.log(self.class_prior_[idx])
            lambdas = self.feature_lambda_[idx,:]
            log_likelihood = np.sum(poisson.logpmf(X, lambdas), axis=1)
            log_probas[:, idx] = prior + log_likelihood
            
        log_denominator = logsumexp(log_probas, axis=1, keepdims=True)
        normalized_log_probas = log_probas - log_denominator
        normalized_probas = np.exp(normalized_log_probas)
        return normalized_probas

    def predict(self, X):
        log_probas = self.predict_proba(X)
        return self.classes_[np.argmax(log_probas, axis=1)]
    
    def normalized_entropy(self, X):
        normalized_probas = self.predict_proba(X)
        normalized_entropies = []
        for trial in normalized_probas:
            entropy = 0
            for probability in trial:
                if probability > 0:
                    entropy += -probability * np.log2(probability)
            normalized_entropy = entropy / 3
            normalized_entropies.append(normalized_entropy)
        return normalized_entropies
        
        
