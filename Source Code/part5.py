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

def draw(data, title, x_label, y_label, labels, r):
    color = ['r','g','y','c']
    plt.figure(figsize=(10,6))
    plt.axhline(ori_train_score, linestyle='dashed', color='r', alpha=0.8, label = 'original training score')
    plt.axhline(ori_test_score, linestyle='dashed', color='g', alpha=0.8, label = 'original testing score')
    for i in range(0, len(data)):
        plt.plot(r, data[i], color[i], label=labels[i])
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend(loc='best')
    plt.show()
    
def drawTime(data, title, x_label, y_label, labels, r):
    color = ['r','g','y','c']
    plt.figure(figsize=(10,6))
    plt.axhline(ori_time, linestyle='solid', color='b', alpha=0.8, label = 'original training time')
    for i in range(0, len(data)):
        plt.plot(r, data[i], color[i], label=labels[i])
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend(loc='best')
    plt.show()

def NN(X_train_clustered, X_test_clustered):
    mlp = MLPClassifier(hidden_layer_sizes=(10,10), max_iter=200)
    mlp.fit(X_train_clustered, y_train)
    train_score = mlp.score(X_train_clustered, y_train)
    test_score = mlp.score(X_test_clustered, y_test)
    
    return train_score, test_score

X_train, X_test, y_train, y_test = sample_preprocess(1)

# baseline
start_time = time.time()
ori_train_score, ori_test_score = NN(X_train, X_test)
ori_time = time.time() - start_time

train_score_total = []
test_score_total = []
time_reduction = []
time_training = []
time_total = []

# K-means
r = range(2, 11)
for clusters in r:
    print(str(clusters) + '/10')
    start_time = time.time()
    kmeans = KMeans(n_clusters=clusters, random_state=1)
    kmeans.fit(X_train)
    X_train_clustered = kmeans.predict(X_train)
    time_reduction.append(time.time()-start_time)
    X_test_clustered = kmeans.predict(X_test)
    X_train_clustered = np.reshape(X_train_clustered,(-1, 1))
    X_test_clustered = np.reshape(X_test_clustered,(-1, 1))
    
    start_time = time.time()
    train_score, test_score = NN(X_train_clustered, X_test_clustered)
    time_training.append(time.time()-start_time)
    
    time_total.append(time_reduction[-1]+time_training[-1])
    train_score_total.append(train_score) 
    test_score_total.append(test_score)

draw([train_score_total, test_score_total], 'KNN Accuracy over K-Means Clusters (Spam dataset)', 'k', 'accuracy', ['clustered training score', 'clustered testing score'], r)
drawTime([time_reduction, time_training, time_total], 'Time of KNN over K-Means Clusters (Spam dataset)', 'k', 'time (s)', ['K-Means reduction time', 'NN training time', 'total time'], r)

# EM
train_score_total = []
test_score_total = []
time_reduction = []
time_training = []
time_total = []

r = range(2, 11)
for components in r:
    print(str(components) + '/10')
    start_time = time.time()
    gmm = GaussianMixture(n_components=components, covariance_type='full', random_state=1)
    gmm.fit(X_train)
    X_train_clustered = gmm.predict(X_train)
    time_reduction.append(time.time()-start_time)
    X_test_clustered = gmm.predict(X_test)
    X_train_clustered = np.reshape(X_train_clustered,(-1, 1))
    X_test_clustered = np.reshape(X_test_clustered,(-1, 1))
    
    start_time = time.time()
    train_score, test_score = NN(X_train_clustered, X_test_clustered)
    time_training.append(time.time()-start_time)
    
    time_total.append(time_reduction[-1]+time_training[-1])
    train_score_total.append(train_score) 
    test_score_total.append(test_score)

draw([train_score_total, test_score_total], 'KNN Accuracy over EM Clusters (Spam dataset)', 'Components', 'accuracy', ['clustered training score', 'clustered testing score'], r)
drawTime([time_reduction, time_training, time_total], 'Time of KNN over EM Clusters (Spam dataset)', 'Components', 'time (s)', ['EM reduction time', 'NN training time', 'total time'], r)
