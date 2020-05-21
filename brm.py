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
import time
import itertools

class Model:
    def __init__(self):
        self.ClassifierCount = 100
        self.BootstrapSamplePercent = 1
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
        self._maxLessMin = []
        self._minFeaturesValues = []
        self._maxFeaturesValues = []
        self._numberOfFeatures = 0
        self._maxDissimilarity = 0
        self.sampleSize = 0

    # this clasify method was fully generated according to the implementation of c# (it's very slow)
    def classify(self, instance):

        current_similarity = 0
    
        for i in range(self.ClassifierCount):
            min_distance = float("inf")
            for center in self._centers[i]:
                d = self.compare(instance, center)

                if (d < min_distance):
                    min_distance = d

            current_similarity += (min_distance > 0) and math.exp(-(min_distance * min_distance) / (2 * self._sd[i] * self._sd[i])) or 1
        
        current_similarity /= self.ClassifierCount

        if (current_similarity < 0):
                current_similarity = 0;

        if (self.UsePastEvenQueue == False):
            return 1-current_similarity

        result_similarity = (self._alpha * self._similaritySum / self._maxEventCount + (1 - self._alpha) * current_similarity);
        if (result_similarity < 0):
            result_similarity = 0;

        self._similaritySum += current_similarity;

        if (len(self._pastEvents) == self._maxEventCount):
            self._similaritySum -= self._pastEvents.pop(0)

        self._pastEvents.append(current_similarity);

        if (self._similaritySum < 0):
            self._similaritySum = 0

        return 1 - result_similarity

    
    # upgrade of the implememtation of c# USING THIS ONE
    def classifyv2(self, instance):
        current_similarity =0.0

        for i in range(0, self.ClassifierCount):
            # compare
            d = np.amin(np.sqrt(np.sum(np.power(self._centers[i] - instance, 2), axis=1))/self._maxDissimilarity)
            
            if d > 0:
                current_similarity += math.exp(-(d * d )/(2*self._sd[i]*self._sd[i]))
            else:
                current_similarity += 1

        current_similarity /= self.ClassifierCount
        
        if (current_similarity < 0):
            current_similarity = 0

        if (self.UsePastEvenQueue == False):
            return 1-current_similarity

        result_similarity = (self._alpha * self._similaritySum / self._maxEventCount + (1 - self._alpha) * current_similarity);
        if (result_similarity < 0):
            result_similarity = 0

        self._similaritySum += current_similarity;

        if (len(self._pastEvents) == self._maxEventCount):
            self._similaritySum -= self._pastEvents.pop(0)

        self._pastEvents.append(current_similarity)

        if (self._similaritySum < 0):
            self._similaritySum = 0

        return 1 - result_similarity


    def score_samples(self, X_test):
        list_instances = X_test.values.tolist()
        currentSimilarity = 0
        num_test_samples = X_test.shape[0]
        test_numberOfFeatures = X_test.shape[1] 

        if X_test.ndim > 1: 
            test_numberOfFeatures = X_test.shape[1]

        if (test_numberOfFeatures != self._numberOfFeatures):
            raise Exception('Unable to compare objects: Invalid instance model')

        y_labels = []
        
        for instance in list_instances:
            y_labels.append(self.classifyv2(instance)) 
        
        return y_labels


    def fit(self, X_train, y_train):
        self._numberOfFeatures = X_train.shape[1]
        self.sampleSize = (self.UseBootstrapSampleCount) and int(self.BootstrapSampleCount) or int(self.BootstrapSamplePercent * len(X_train) / 100);

        self._sd = np.empty(0)
        self._centers = np.empty((0, self.sampleSize, self._numberOfFeatures))
        self._distance = self.euclidean_dissimilarity(X_train)

    # this compare method was generated according to the implementation of c#
    def compare(self, source, compareTo):
        try:
            suma = 0
            for feature_index in range(self._numberOfFeatures):
                # exists
                if(source[feature_index] and compareTo[feature_index]):
                    # numeric
                    if( isinstance(source[feature_index], int) or isinstance(source[feature_index], float) ):

                        if self._maxLessMin[feature_index] > 0:
                            componentDiff = abs(source[feature_index] - compareTo[feature_index]) / self._maxLessMin[feature_index]
                            suma += (componentDiff > 1) and 1 or math.pow(componentDiff, 2)

                    elif int(source[feature_index]) != int(compareTo[feature_index]):
                        suma += 1
                else:
                    suma += 1

            return math.sqrt(suma) / self._maxDissimilarity
        except:
            raise Exception('Unable to compare objects: Invalid instance model')



    # this compare method was generated based in the implemnetation of c# but with some assumptions, USING THIS ONE
    def comparev2(self,source, compareTo):
        componentDiff = (abs( np.subtract(source, compareTo) ) / self._maxLessMin)
        num = sum((element > 1)and 1 or math.pow(element, 2) for element in componentDiff)
        sumCat = np.count_nonzero(source == compareTo)
        return math.sqrt(sumCat+num)/ self._maxDissimilarity


        
    def compute_beta(self, centers):
        suma = 0.0 
        suma2= 0.0
        count = 0

        for i in range(0, len(centers)-1):
            for j in range(i+1, len(centers)):
                suma += self.comparev2(centers[i], centers[j])
                #suma += self.compare(centers[i], centers[j])
                count += 1

        return suma / count
    
    def euclidean_dissimilarity(self, instances):
        if self._numberOfFeatures < 1:
            raise Exception('Unable to instantiate the train dataset - Empty vector')

        self._maxFeaturesValues = np.amax(instances, axis=0)
        self._minFeaturesValues = np.amin(instances, axis=0)
        self._maxLessMin = np.subtract(self._maxFeaturesValues, self._minFeaturesValues)
        self._maxDissimilarity = math.sqrt(self._numberOfFeatures)

        list_instances = instances.values.tolist()
        #list_instances = instances.to_numpy()
        for i in range(0, self.ClassifierCount):
            centers = random.choices(list_instances, k=self.sampleSize)
            #centers = list_instances[np.random.randint(list_instances.shape[0], size=self.sampleSize), :]
            self._centers = np.insert(self._centers, i, centers, axis=0)
            self._sd = np.insert(self._sd, i, self.compute_beta(self._centers[i]))


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

        


trainFile = 'D:/Jc/Documents/Universidad/8vo/ExtraOct/BRM/data/abalone-tra_num.csv'
testFile = 'D:/Jc/Documents/Universidad/8vo/ExtraOct/BRM/data/abalone-tst_num.csv'


# Loading data 
train, test = importdata(trainFile, testFile)

X_train, X_test, y_train, y_test, X_train_minmax, X_test_minmax, X_train_std, X_test_std, X_train_minmax_std, X_test_minmax_std = splitdataset(train, test) 

# Here i will initializate the Class of BRM, and send its parans and stuff of the intreface of sklearn
classifier = Model() 

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
print(y_pred_classif)