import pickle
import math
import numpy as np   # We recommend to use numpy arrays
from os.path import isfile
import os
import sys
import random
from sklearn.ensemble import GradientBoostingClassifier
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances
import time
import itertools
from multiprocessing import Pool, Semaphore, cpu_count, Process
import matplotlib.pyplot as plt

class BRM:
    def __init__(self):
        self.ClassifierCount = 100
        self.BootstrapSamplePercent = 100
        self.UseBootstrapSampleCount = False
        self.BootstrapSampleCount = 0
        self.UsePastEvenQueue = True
        self._sd = []
        self._centers = []
        self._pastEvents = []
        self._similaritySum = 0
        self._maxEventCount = 3
        self._alpha = 0.5
        self.distance = 0
        self._numberOfFeatures = 0
        self._maxDissimilarity = 0
        self.sampleSize = 0
    

    def evaluate(self, current_similarity):
        if (current_similarity < 0):
            current_similarity = 0

        if (self.UsePastEvenQueue == False):
            return 1 - current_similarity
            
        result_similarity = (self._alpha * self._similaritySum / self._maxEventCount + (1 - self._alpha) * current_similarity)
        if (result_similarity < 0):
            result_similarity = 0

        self._similaritySum += current_similarity

        if (len(self._pastEvents) == self._maxEventCount):
            self._similaritySum -= self._pastEvents.pop(0)

        self._pastEvents.append(current_similarity)

        if (self._similaritySum < 0):
            self._similaritySum = 0

        return 1 - result_similarity

    def score_samples(self, X_test):
        X_test = pd.DataFrame(self._scaler.transform(X_test[X_test.columns]), index=X_test.index, columns=X_test.columns)  

        if (X_test.shape[1] != self._numberOfFeatures):
            raise Exception('Unable to compare objects: Invalid instance model')

        current_similarity = np.sum(np.reshape(np.array([np.exp(-np.power(np.amin(euclidean_distances(X_test, self._centers[i]), axis=1)/self._maxDissimilarity, 2)/(self._sd[i])) for i in range(len(self._centers))]), (len(X_test),-1)), axis=1)/self.ClassifierCount
        return list(map(self.evaluate, current_similarity))
        

    def fit(self, X_train, y_train):
        self._numberOfFeatures = X_train.shape[1]
        if self._numberOfFeatures < 1:
            raise Exception('Unable to instantiate the train dataset - Empty vector')        
        
        self._scaler = MinMaxScaler()
        X_train =  pd.DataFrame(self._scaler.fit_transform(X_train[X_train.columns]), index=X_train.index, columns=X_train.columns)


        self._maxDissimilarity = math.sqrt(self._numberOfFeatures)
        self._sd = np.empty(0)
        self.sampleSize = int(self.BootstrapSampleCount) if (self.UseBootstrapSampleCount) else int(0.01 * self.BootstrapSamplePercent * len(X_train));
        self._centers = np.empty((0, self.sampleSize, self._numberOfFeatures))

        list_instances = X_train.values.tolist()
        for i in range(0, self.ClassifierCount):            
            centers = random.choices(list_instances, k=self.sampleSize)
            self._centers = np.insert(self._centers, i, centers, axis=0)
            self._sd = np.insert(self._sd, i, 2*(np.mean(euclidean_distances(centers, centers))/self._maxDissimilarity)**2)


# Function importing Dataset 
def importdata(trainFile, testFile): 
    train = pd.read_csv(trainFile, sep= ',', header = None) 
    test = pd.read_csv(testFile, sep= ',', header = None) 
    return train, test     


def splitdataset(train, test): 
    ohe = OneHotEncoder(sparse=True)
    objInTrain = len(train)

    allData = pd.concat([train, test], ignore_index=True, sort =False, axis=0)
    AllDataWihoutClass = allData.iloc[:, :-1]
    AllDataWihoutClassOnlyNominals = AllDataWihoutClass.select_dtypes(include=['object'])
    AllDataWihoutClassNoNominals = AllDataWihoutClass.select_dtypes(exclude=['object'])

    encAllDataWihoutClassNominals = ohe.fit_transform(AllDataWihoutClassOnlyNominals)
    encAllDataWihoutClassNominalsToPanda = pd.DataFrame(encAllDataWihoutClassNominals.toarray())
    
    if AllDataWihoutClassOnlyNominals.shape[1] > 0:
        codAllDataAgain = pd.concat([encAllDataWihoutClassNominalsToPanda, AllDataWihoutClassNoNominals], ignore_index=True, sort =False, axis=1)
    else:
        codAllDataAgain = AllDataWihoutClass

    # Seperating the target variable 
    X_train = codAllDataAgain[:objInTrain]
    y_train = train.values[:, -1]

    X_test = codAllDataAgain[objInTrain:]
    y_test = test.values[:, -1]
    
    mm_scaler = MinMaxScaler()
    X_train_minmax = pd.DataFrame(mm_scaler.fit_transform(X_train[X_train.columns]), index=X_train.index, columns=X_train.columns)
    X_test_minmax = pd.DataFrame(mm_scaler.transform(X_test[X_test.columns]), index=X_test.index, columns=X_test.columns)
    
    std_scaler = StandardScaler()
    X_train_std = pd.DataFrame(std_scaler.fit_transform(X_train[X_train.columns]), index=X_train.index, columns=X_train.columns)
    X_test_std = pd.DataFrame(std_scaler.transform(X_test[X_test.columns]), index=X_test.index, columns=X_test.columns)
    
    X_train_minmax_std = pd.DataFrame(std_scaler.fit_transform(X_train_minmax[X_train_minmax.columns]), index=X_train_minmax.index, columns=X_train_minmax.columns)
    X_test_minmax_std = pd.DataFrame(std_scaler.transform(X_test_minmax[X_test_minmax.columns]), index=X_test_minmax.index, columns=X_test_minmax.columns)
    
    return X_train, X_test, y_train, y_test, X_train_minmax, X_test_minmax, X_train_std, X_test_std, X_train_minmax_std, X_test_minmax_std


def main():
    trainFile = 'D:/Jc/Documents/Universidad/8vo/ExtraOct/BRM/data/abalone-tra_num.csv'
    testFile = 'D:/Jc/Documents/Universidad/8vo/ExtraOct/BRM/data/abalone-tst_num.csv'

    # Loading data 
    train, test = importdata(trainFile, testFile)

    X_train, X_test, y_train, y_test, X_train_minmax, X_test_minmax, X_train_std, X_test_std, X_train_minmax_std, X_test_minmax_std = splitdataset(train, test) 

    # Here i will initializate the Class of BRM, and send its parans and stuff of the intreface of sklearn
    classifier = BRM() 

    # train
    start = time.time()

    classifier.fit(X_train, y_train)

    end = time.time()
    elapsed = int(round((end - start)*1000))
    print("Time consummed for training: ", elapsed ,"ms")

    #classify
    start = time.time()

    y_pred_classif = classifier.score_samples(X_test)

    end = time.time()
    elapsed = int(round((end - start)*1000))
    print("Time consummed for classifying: ", elapsed ,"ms")

if __name__ == '__main__':
    main()