import matplotlib.pyplot as plt
%matplotlib inline
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.cluster import KMeans, FeatureAgglomeration
from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.decomposition import PCA, FastICA
from scipy.stats import kurtosis, skew
from sklearn.random_projection import GaussianRandomProjection
from sklearn.neural_network import MLPClassifier
import time

from warnings import filterwarnings
filterwarnings('ignore')

# Read the dataset
url1 = "https://www.openml.org/data/get_csv/44/dataset_44_spambase.arff"
url2 = "https://archive.ics.uci.edu/ml/machine-learning-databases/letter-recognition/letter-recognition.data"
spamdata_origin = pd.read_csv(url1)
letterdata_origin = pd.read_csv(url2, header= None)

dataset_name = 'spam'

def sample_preprocess(frac_now):
    #split attribute and class
    if dataset_name == 'spam':
        data = spamdata_origin.sample(frac=frac_now, random_state = 1)
        X = data.drop('class', axis=1).astype(float)
        y = data['class']
    
    elif dataset_name == 'letter':
        data = letterdata_origin.sample(frac=frac_now, random_state = 1)
        X = data.drop(0, axis=1).astype(float)
        y = data[0]
        y = y.map(ord) - ord('A') + 1
    else:
        print('Invalid Dataset Name')
        return
    
    print('dataset:', dataset_name)
    print('data percentage: ' + str(frac_now * 100) + '%')
    
    #split test and train data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)
    
    #scale the data
    scaler = MinMaxScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)  
    X_test = scaler.transform(X_test)

    print('total:', len(X))
    print('training data:', len(X_train))
    print('testing data:', len(X_test))
    print('number of attributes:', len(X_train[0]))
    print('number of classes:', len(np.unique(y)))
    
    return X_train, X_test, y_train, y_test

def drawMultiple(data1, data2, title, x_label, y_label, label1s, label2s, r):
    color = ['r','g','y','c']
    plt.figure(figsize=(15,10))
    plt.axhline(ori_train_score, linestyle='solid', color='b', alpha=0.8, label = 'original train score')
    plt.axhline(ori_test_score, linestyle='dashed', color='b', alpha=0.8, label = 'original test score')
    for i in range(0, len(data1)):
        plt.plot(r, data1[i], color[i], label=label1s[i])
    for i in range(0, len(data2)):
        plt.plot(r, data2[i], color[i], linestyle='dashed', label=label2s[i])
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend(loc='best')
    plt.show()
    
def drawTime(data, title, x_label, y_label, labels, r):
    color = ['r','g','y','c']
    plt.figure(figsize=(15,10))
    plt.axhline(ori_time, linestyle='solid', color='b', alpha=0.8, label = 'original training time')
    for i in range(0, len(data)):
        plt.plot(r, data[i], color[i], label=labels[i])
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend(loc='best')
    plt.show()

def Reduction(X_train, alg_name):
    train_scores=[]
    test_scores=[]
    times=[]
    print('algorithm:', alg_name)
    for component in range(1, len(X_train[0])+1):
        if component % 10 == 0:
            print(component)
        if alg_name == 'pca':
            alg = PCA(n_components=component, random_state=1)
        elif alg_name == 'ica':
            alg = FastICA(random_state=1, n_components=component)
        elif alg_name == 'rp':
            alg = GaussianRandomProjection(n_components=component, random_state=1)
        elif alg_name == 'fa':
            alg = FeatureAgglomeration(n_clusters=component)
        else:
            break
        
        X_train_reduced=alg.fit_transform(X_train)
        X_test_reduced=alg.transform(X_test)
        
        start_time = time.time()
        train_score, test_score = NN(X_train_reduced, X_test_reduced)
        times.append((time.time() - start_time))
        
        train_scores.append(train_score)
        test_scores.append(test_score)

    return train_scores, test_scores, times

def NN(X_train_reduced, X_test_reduced):
    mlp = MLPClassifier(hidden_layer_sizes=(10,10), max_iter=200)
    mlp.fit(X_train_reduced, y_train)
    train_score = mlp.score(X_train_reduced, y_train)
    test_score = mlp.score(X_test_reduced, y_test)
    
    return train_score, test_score

X_train, X_test, y_train, y_test = sample_preprocess(1)

#baseline
NN(X_train, X_test)
start_time = time.time()
ori_train_score, ori_test_score = NN(X_train, X_test)
ori_time = time.time() - start_time

alg_names = ['pca', 'ica', 'rp', 'fa']
label1s = ['PCA train score', 'ICA train score', 'RP train score', 'FA train score']
label2s = ['PCA train score', 'ICA test score', 'RP test score', 'FA test score']
label3s = ['PCA', 'ICA', 'RP', 'FA']

data_result1=[]
data_result2=[]
times_all=[]
for alg_name in alg_names:
    train_scores, test_scores, times = Reduction(X_train, alg_name)
    data_result1.append(train_scores)
    data_result2.append(test_scores)
    times_all.append(times)
drawMultiple(data_result1, data_result2, 'NN Accuracy over Reduction Algorithms (Spam dataset)', 'Components', 'Accuracy', label1s, label2s, range(1, len(X_train[0])+1))
drawTime(times_all, 'NN Training Time over Reduction Algorithms (Spam dataset)', 'Components', 'Time(s)', label3s, range(1, len(X_train[0])+1))
