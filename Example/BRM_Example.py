# Example Created by Octavio Loyola-González and Miguel Angel Medina-Pérez

import pandas as pd
import brminer
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Function importing Dataset 
def importdata(trainFile, testFile): 
    train = pd.read_csv(trainFile, sep= ',', header = None) 
    test = pd.read_csv(testFile, sep= ',', header = None) 
    return train, test    

# Function to split target from data 
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

# Function to make predictions 
def prediction(X_test, clf_object):  
    y_pred = clf_object.score_samples(X_test) 
    return y_pred 

def result_of_Class(y_test, y_pred, saveFile):       
    np.savetxt(saveFile, y_pred, fmt='%.4f')

trainFile = 'abalone-17_vs_7-8-9-10-5-2tra.dat.csv'
testFile = 'abalone-17_vs_7-8-9-10-5-2tst.dat.csv'

# Loading data 
train, test = importdata(trainFile, testFile)

# Training
X_train, X_test, y_train, y_test, X_train_minmax, X_test_minmax, X_train_std, X_test_std, X_train_minmax_std, X_test_minmax_std = splitdataset(train, test) 

clf_classif1 = brminer.BRM()

clf_classif1.fit(X_train, y_train) 

y_pred = clf_classif1.score_samples(X_test)
auc = roc_auc_score(y_test,  y_pred)
print(f'Testing AUC: {auc if auc > .5 else 1 - auc}')
