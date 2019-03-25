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
import warnings
warnings.simplefilter(action='ignore', category=RuntimeWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

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
    color = ['bo-', 'ro-', 'go-']
    for i in range(0, len(data)):
        plt.plot(r, data[i], color[i], label=labels[i])
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend(loc='best')
    plt.show()

X_train, X_test, y_train, y_test = sample_preprocess(1)

#k-means
Sum_of_squared_distances=[]
Adjusted_rand_index=[]
Mutual_info_score=[]
Adjusted_mutual_info_score=[]
Normalized_mutual_info_score=[]
Homogeneity_score=[]
Completeness_score=[]
V_measure_score=[]
Fowlkes_mallows_score=[]
Silhouette_coefficient=[]
Calinski_harabaz_score=[]
Davies_bouldin_score=[]
AIC=[]
BIC=[]

r = range(0)
total = 0
if dataset_name == 'letter':
    r = range(2, 31)
    total = 30
elif dataset_name == 'spam':
    r = range(2, 11)
    total = 10

for clusters in r:
    print(str(clusters) + '/' + str(total))
    kmeans = KMeans(n_clusters=clusters, random_state=1)
    kmeans.fit(X_train)
    labels = kmeans.predict(X_train)
    Sum_of_squared_distances.append(kmeans.inertia_)
    Adjusted_rand_index.append(metrics.adjusted_rand_score(y_train, labels))
    Mutual_info_score.append(metrics.mutual_info_score(y_train, labels))
    Adjusted_mutual_info_score.append(metrics.adjusted_mutual_info_score(y_train, labels))
    Normalized_mutual_info_score.append(metrics.normalized_mutual_info_score(y_train, labels))
    Homogeneity_score.append(metrics.homogeneity_score(y_train, labels))
    Completeness_score.append(metrics.completeness_score(y_train, labels))
    V_measure_score.append(metrics.v_measure_score(y_train, labels))
    Fowlkes_mallows_score.append(metrics.fowlkes_mallows_score(y_train, labels))
    
    Silhouette_coefficient.append(metrics.silhouette_score(X_train, labels, metric='euclidean'))
    Calinski_harabaz_score.append(metrics.calinski_harabaz_score(X_train, labels))
    Davies_bouldin_score.append(metrics.davies_bouldin_score(X_train, labels))

drawMultiple(data=[Sum_of_squared_distances], title='Elbow Method For Optimal k', x_label='k', y_label='Sum Of Squared Distances',labels=['Sum Of Squared Distances'], r=r)
drawMultiple(data=[Adjusted_rand_index], title='Adjusted Rand Index over k', x_label='k', y_label='Adjusted Rand Score',labels=['Adjusted Rand Score'], r=r)
drawMultiple(data=[Mutual_info_score, Adjusted_mutual_info_score, Normalized_mutual_info_score], title='Mutual Information over k', x_label='k', y_label='Mutual Info Score', labels=['NMI','MI','AMI'], r=r)
drawMultiple(data=[Homogeneity_score, Completeness_score, V_measure_score], title='Homogeneity, Completeness, V-measure over k', x_label='k', y_label='Score', labels=['v-measure', 'homogeneity','completeness'], r=r)
drawMultiple(data=[Fowlkes_mallows_score], title='Fowlkes Mallows Scores over k', x_label='k', y_label='Fowlkes Mallows Score',labels=['Fowlkes Mallows Score'], r=r)
drawMultiple(data=[Silhouette_coefficient], title='Silhouette Coefficient over k', x_label='k', y_label='Silhouette Coefficient',labels=['Silhouette Coefficient'], r=r)
drawMultiple(data=[Calinski_harabaz_score], title='Calinski Harabaz Scores over k', x_label='k', y_label='Calinski Harabaz Score',labels=['Calinski Harabaz Score'], r=r)
drawMultiple(data=[Davies_bouldin_score], title='Davies Bouldin Scores over k', x_label='k', y_label='Davies Bouldin Score',labels=['Davies Bouldin Score'], r=r)

#EM
Log_likelihood=[]
Adjusted_rand_index=[]
Mutual_info_score=[]
Adjusted_mutual_info_score=[]
Normalized_mutual_info_score=[]
Homogeneity_score=[]
Completeness_score=[]
V_measure_score=[]
Fowlkes_mallows_score=[]
Silhouette_coefficient=[]
Calinski_harabaz_score=[]
Davies_bouldin_score=[]
AIC=[]
BIC=[]

r = range(0)
total = 0
if dataset_name == 'letter':
    r = range(2, 31)
    total = 30
elif dataset_name == 'spam':
    r = range(2, 11)
    total = 10

for components in r:
    print(str(components) + '/' + str(total))
    gmm = GaussianMixture(n_components=components, covariance_type='full', random_state=1)
    gmm.fit(X_train)
    labels = gmm.predict(X_train)
    Log_likelihood.append(gmm.score(X_train))
    Adjusted_rand_index.append(metrics.adjusted_rand_score(y_train, labels))
    Mutual_info_score.append(metrics.mutual_info_score(y_train, labels))
    Adjusted_mutual_info_score.append(metrics.adjusted_mutual_info_score(y_train, labels))
    Normalized_mutual_info_score.append(metrics.normalized_mutual_info_score(y_train, labels))
    Homogeneity_score.append(metrics.homogeneity_score(y_train, labels))
    Completeness_score.append(metrics.completeness_score(y_train, labels))
    V_measure_score.append(metrics.v_measure_score(y_train, labels))
    Fowlkes_mallows_score.append(metrics.fowlkes_mallows_score(y_train, labels))
    
    Silhouette_coefficient.append(metrics.silhouette_score(X_train, labels, metric='euclidean'))
    Calinski_harabaz_score.append(metrics.calinski_harabaz_score(X_train, labels))
    Davies_bouldin_score.append(metrics.davies_bouldin_score(X_train, labels))
    AIC.append(gmm.aic(X_train))
    BIC.append(gmm.bic(X_train))

drawMultiple(data=[Log_likelihood], title='Elbow Method For Optimal Components', x_label='Components', y_label='Log likelihood', labels=['Log likelihood'], r=r)
drawMultiple(data=[Adjusted_rand_index], title='Adjusted Rand Index over Components', x_label='Components', y_label='Adjusted Rand Score', labels=['Adjusted Rand Score'], r=r)
drawMultiple(data=[Mutual_info_score, Adjusted_mutual_info_score, Normalized_mutual_info_score], title='Mutual Information over Components', x_label='Components', y_label='Mutual Info Score', labels=['NMI','MI','AMI'], r=r)
drawMultiple(data=[Homogeneity_score, Completeness_score, V_measure_score], title='Homogeneity, Completeness, V-measure over Components', x_label='Components', y_label='Score', labels=['v-measure','homogeneity','completeness'], r=r)
drawMultiple(data=[Fowlkes_mallows_score], title='Fowlkes Mallows Scores over Components', x_label='Components', y_label='Fowlkes Mallows Score', labels=['Fowlkes Mallows Score'], r=r)
drawMultiple(data=[Silhouette_coefficient], title='Silhouette Coefficient over Components', x_label='Components', y_label='Silhouette Coefficient', labels=['Silhouette Coefficient'], r=r)
drawMultiple(data=[Calinski_harabaz_score], title='Calinski Harabaz Scores over Components', x_label='Components', y_label='Calinski Harabaz Score', labels=['Calinski Harabaz Score'], r=r)
drawMultiple(data=[Davies_bouldin_score], title='Davies Bouldin Scores over Components', x_label='Components', y_label='Davies Bouldin Score', labels=['Davies Bouldin Score'], r=r)
drawMultiple(data=[AIC], title='Akaike Information Criterion over Components', x_label='Components', y_label='AIC Score', labels=['AIC Score'], r=r)
drawMultiple(data=[BIC], title='Bayesian Information Criterion over Components', x_label='Components', y_label='BIC Score', labels=['BIC Score'], r=r)
