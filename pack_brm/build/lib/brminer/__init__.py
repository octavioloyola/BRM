import numpy as np
import math
import os
import random
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

class BRM(BaseEstimator):
    def __init__(self, classifier_count=100, bootstrap_sample_percent=100, use_bootstrap_sample_count=False, bootstrap_sample_count=0, use_past_even_queue=False, max_event_count=3, alpha=0.5, user_threshold=95):
        self.classifier_count = classifier_count
        self.bootstrap_sample_percent = bootstrap_sample_percent
        self.use_bootstrap_sample_count = use_bootstrap_sample_count
        self.bootstrap_sample_count = bootstrap_sample_count
        self.use_past_even_queue = use_past_even_queue
        self.max_event_count = max_event_count
        self.alpha = alpha
        self.user_threshold = user_threshold
        
    def _evaluate(self, current_similarity):
        if (current_similarity < 0):
            current_similarity = 0

        if (self.use_past_even_queue == False):
            return -1+2*current_similarity
        
        result_similarity = (self.alpha * self._similarity_sum / self.max_event_count + (1 - self.alpha) * current_similarity)
        if (result_similarity < 0):
            result_similarity = 0

        self._similarity_sum += current_similarity

        if (len(self._past_events) == self.max_event_count):
            self._similarity_sum -= self._past_events.pop(0)

        self._past_events.append(current_similarity)

        if (self._similarity_sum < 0):
            self._similarity_sum = 0

        return -1+2*result_similarity

    def score_samples(self, X):
        X = np.array(X)
        X_test = self._scaler.transform(X)

        result = []
        batch_size = 100
        for i in range(min(len(X_test), batch_size), len(X_test) + batch_size, batch_size):
            current_X_test = X_test[[j for j in range(max(0, i-batch_size), min(i, len(X_test)))]]

            current_similarity = np.average([np.exp(-np.power(np.amin(euclidean_distances(current_X_test, self._centers[i]), axis=1)/self._max_dissimilarity, 2)/(self._sd[i])) for i in range(len(self._centers))], axis=0)
        
            result = result + [j for j in list(map(self._evaluate, current_similarity))]

        return result
        

    def predict(self, X):
        if (len(X.shape) < 2):
            raise ValueError('Reshape your data')

        if (X.shape[1] != self.n_features_in_):
            raise ValueError('Reshape your data')

        if not self._is_threshold_Computed:            
            x_pred_classif = self.score_samples(self._X_train)            
            x_pred_classif.sort()
            self._inner_threshold = x_pred_classif[(100-self.user_threshold)*len(x_pred_classif)//100]
            self._is_threshold_Computed = True

        y_pred_classif = self.score_samples(X)
        return [-1 if s <= self._inner_threshold else 1 for s in y_pred_classif]
        

    def fit(self, X, y = None):
        # Check that X and y have correct shape
        if y is not None:
            X_train, y_train = check_X_y(X, y)
        else:
             X_train = check_array(X)
                
        self._similarity_sum = 0
        self._is_threshold_Computed = False

        self.n_features_in_ = X_train.shape[1]

        if self.n_features_in_ < 1:
            raise ValueError('Unable to instantiate the train dataset - Empty vector')     
        
        self._scaler = MinMaxScaler()
        X_train = pd.DataFrame(X_train)
        X_train = pd.DataFrame(self._scaler.fit_transform(X_train[X_train.columns]), index=X_train.index, columns=X_train.columns)


        self._max_dissimilarity = math.sqrt(self.n_features_in_)
        self._sd = np.empty(0)
        sampleSize = int(self.bootstrap_sample_count) if (self.use_bootstrap_sample_count) else int(0.01 * self.bootstrap_sample_percent * len(X_train));
        self._centers = np.empty((0, sampleSize, self.n_features_in_))

        list_instances = X_train.values.tolist()
        for i in range(0, self.classifier_count):            
            centers = random.choices(list_instances, k=sampleSize)
            self._centers = np.insert(self._centers, i, centers, axis=0)
            self._sd = np.insert(self._sd, i, 2*(np.mean(euclidean_distances(centers, centers))/self._max_dissimilarity)**2)

        return self
