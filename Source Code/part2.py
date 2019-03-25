import matplotlib.pyplot as plt
%matplotlib inline
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score 
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA, FastICA
from scipy.stats import kurtosis, skew
from sklearn.random_projection import GaussianRandomProjection
from sklearn.neighbors import KNeighborsClassifier

# Read the dataset
url1 = "https://www.openml.org/data/get_csv/44/dataset_44_spambase.arff"
url2 = "https://archive.ics.uci.edu/ml/machine-learning-databases/letter-recognition/letter-recognition.data"
spamdata_origin = pd.read_csv(url1)
letterdata_origin = pd.read_csv(url2, header= None)

### choose dataset
#dataset_name = 'letter'
dataset_name = 'spam'
###

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

def drawMultiple(data, title, x_label, y_label, labels, r):
    color = ['r', 'g']
    plt.figure(figsize=(10,5))
    plt.axhline(ori_train_score, linestyle='dashed', color='red', alpha=0.8, label = 'original train score')
    plt.axhline(ori_test_score, linestyle='dashed', color='green', alpha=0.8, label = 'original test score')
    for i in range(0, len(data)):
        plt.plot(r, data[i], color[i], label=labels[i])
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend(loc='best')
    plt.show()

X_train, X_test, y_train, y_test = sample_preprocess(1)

#PCA
pca=PCA(n_components=len(X_train[0]), random_state=1)
newData=pca.fit_transform(X_train)
# print(pca.n_components)
# print(pca.explained_variance_ratio_)
# print(pca.components_)
data = pca.explained_variance_ratio_
sum = 0
threshold=0.9
cap_idx=0
for cap_idx in range(len(data)):
    if sum > threshold:
        break
    else:
        sum += data[cap_idx]

x = range(1, len(data) + 1)
plt.figure(figsize=(10,5))
plt.xlabel('Principal Components')
plt.ylabel('Variance Ratios')
if dataset_name=='spam':
    plt.title('PCA Component Analysis(Spam dataset)')
elif dataset_name=='letter':
    plt.title('PCA Component Analysis(Letter Recognition dataset)')
plt.bar(x, height= data, width=0.8, color = 'blue', label='>'+ str(threshold*100) +'%')
plt.bar(range(1,cap_idx + 1), height= data[0:cap_idx], width=0.8, color='red', label='<'+ str(threshold*100) +'%')
plt.legend(loc='best')
plt.show()

#ICA
ica=FastICA(random_state=1)
X_train_reduced=ica.fit_transform(X_train)
abs_kurtosis = np.abs(kurtosis(X_train_reduced))
threshold = max(abs_kurtosis) * 0.1

_, ax = plt.subplots(1,1,figsize=(10,5))
plt.xlabel('Independent Components')
plt.ylabel('Kurtosis')
if dataset_name=='spam':
    plt.title('ICA Component Analysis(Spam dataset)')
elif dataset_name=='letter':
    plt.title('ICA Component Analysis(Letter Recognition dataset)')
#plt.title('ICA Component Analysis(Letter Recognition dataset), threshold: ' + str(round(threshold,2)))
plt.bar(range(1,len(X_train[0])+1), height=abs_kurtosis, width=0.8, color='blue', label='below threshold')
indices = np.where(abs_kurtosis>=threshold)
plt.bar([x + 1 for x in indices][0], height=abs_kurtosis[indices], width=0.8, color='red', label = 'beyond threshold')
plt.axhline(threshold, linestyle='dashed', color='black', alpha=0.8)
yt = ax.get_yticks() 
yt = np.append(yt,threshold)
ax.set_yticks(yt)
plt.legend(loc='best')
plt.show()

#KNN baseline
knn = KNeighborsClassifier(n_neighbors=3)  
knn.fit(X_train, y_train)
ori_train_score = knn.score(X_train, y_train)
ori_test_score = knn.score(X_test, y_test)

#RP
train_scores=[]
test_scores=[]
for component in range(1, len(X_train[0])+1):
    grp = GaussianRandomProjection(n_components=component, random_state=1)
    X_train_reduced = grp.fit_transform(X_train)
    X_test_reduced = grp.transform(X_test)

    knn = KNeighborsClassifier(n_neighbors=3)  
    knn.fit(X_train_reduced, y_train)
    train_scores.append(knn.score(X_train_reduced, y_train))
    test_scores.append(knn.score(X_test_reduced, y_test))
if dataset_name=='spam':
    drawMultiple([train_scores, test_scores], 'KNN Accuracy over Randomized Projected Components (Spam dataset)', 'Number of Projected Components', 'Accuracy', ['projected train score','projected test score'], range(1, len(X_train[0])+1))
elif dataset_name=='letter':
    drawMultiple([train_scores, test_scores], 'KNN Accuracy over Randomized Projected Components (Letter Recognition dataset)', 'Number of Projected Components', 'Accuracy', ['projected train score','projected test score'], range(1, len(X_train[0])+1))

#FA
train_scores=[]
test_scores=[]
for component in range(1, len(X_train[0])+1):
    fa = FeatureAgglomeration(n_clusters=component)
    X_train_reduced = fa.fit_transform(X_train)
    X_test_reduced = fa.transform(X_test)

    knn = KNeighborsClassifier(n_neighbors=3)  
    knn.fit(X_train_reduced, y_train)
    train_scores.append(knn.score(X_train_reduced, y_train))
    test_scores.append(knn.score(X_test_reduced, y_test))
if dataset_name=='spam':
    drawMultiple([train_scores, test_scores], 'KNN Accuracy over Feature Agglomeration Components (Spam dataset)', 'Number of Agglomerated Components', 'Accuracy', ['Agglomerated train score','Agglomerated test score'], range(1, len(X_train[0])+1))
elif dataset_name=='letter':
    drawMultiple([train_scores, test_scores], 'KNN Accuracy over Feature Agglomeration Components (Letter Recognition dataset)', 'Number of Agglomerated Components', 'Accuracy', ['Agglomerated train score','Agglomerated test score'], range(1, len(X_train[0])+1))
