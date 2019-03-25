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
    color = ['bo-', 'ro-', 'go-', 'yo-', 'co-']
    for i in range(0, len(data)):
        plt.plot(r, data[i], color[i], label=labels[i])
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend(loc='best')
    plt.show()

def PCA_reduced(X_train):
    pca=PCA(n_components=len(X_train[0]), random_state=1)
    X_train_reduced=pca.fit_transform(X_train)
    #print(pca.n_components)
    # print(pca.explained_variance_ratio_)
    # print(pca.components_)
    data = pca.explained_variance_ratio_
    sum = 0
    pca_threshold=0.9
    cap_idx=0
    for cap_idx in range(len(data)):
        if sum > pca_threshold:
            break
        else:
            sum += data[cap_idx]
    #print(cap_idx, sum)
    x = range(1, len(data) + 1)
    plt.figure(figsize=(10,5))
    plt.xlabel('Principal Components')
    plt.ylabel('Variance Ratios')
    if dataset_name=='spam':
        plt.title('PCA Component Analysis(Spam dataset)')
    elif dataset_name=='letter':
        plt.title('PCA Component Analysis(Letter Recognition dataset)')
    plt.bar(x, height= data, width=0.8, color = 'blue', label='>'+ str(pca_threshold*100) +'%')
    plt.bar(range(1,cap_idx + 1), height= data[0:cap_idx], width=0.8, color='red', label='<'+ str(pca_threshold*100) +'%')
    plt.legend(loc='best')
    #plt.xticks(x, x)
    plt.show()

    return X_train_reduced[:,range(cap_idx)]

def ICA_reduced(X_train):
    c=16
    ica=FastICA(random_state=1, n_components=c)
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
    plt.bar(range(1,c+1), height=abs_kurtosis, width=0.8, color='blue', label='below threshold')
    #len(X_train[0])+1
    indices = np.where(abs_kurtosis>=threshold)
    plt.bar([x + 1 for x in indices][0], height=abs_kurtosis[indices], width=0.8, color='red', label = 'beyond threshold')
    plt.axhline(threshold, linestyle='dashed', color='black', alpha=0.8)
    yt = ax.get_yticks() 
    yt = np.append(yt,threshold)
    ax.set_yticks(yt)
    plt.legend(loc='best')
    plt.show()
    
    return X_train_reduced[:,indices[0]]

def RP_reduced(X_train):
    grp = GaussianRandomProjection(n_components=12, random_state=1)
    X_train_reduced = grp.fit_transform(X_train)
    return X_train_reduced

def FA_reduced(X_train):
    fa = FeatureAgglomeration(n_clusters=10)
    X_train_reduced = fa.fit_transform(X_train)
    return X_train_reduced

X_train, X_test, y_train, y_test = sample_preprocess(1)
X_train_PCA = PCA_reduced(X_train)
X_train_ICA = ICA_reduced(X_train)
X_train_RP = RP_reduced(X_train)
X_train_FA = FA_reduced(X_train)

#k-means
X_trains = [X_train, X_train_PCA, X_train_ICA, X_train_RP, X_train_FA]
arr1,arr2,arr3,arr4,arr5,arr6,arr7,arr8,arr9,arr10,arr11,arr12=[],[],[],[],[],[],[],[],[],[],[],[]
alg = ['Origin', 'PCA', 'ICA', 'RP', 'FA']

for i in range(len(X_trains)):
    print('K-means on', alg[i])
    X_train_now = X_trains[i]

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
        kmeans.fit(X_train_now)
        labels = kmeans.predict(X_train_now)
        Sum_of_squared_distances.append(kmeans.inertia_)
        Adjusted_rand_index.append(metrics.adjusted_rand_score(y_train, labels))
        Mutual_info_score.append(metrics.mutual_info_score(y_train, labels))
        Adjusted_mutual_info_score.append(metrics.adjusted_mutual_info_score(y_train, labels))
        Normalized_mutual_info_score.append(metrics.normalized_mutual_info_score(y_train, labels))
        Homogeneity_score.append(metrics.homogeneity_score(y_train, labels))
        Completeness_score.append(metrics.completeness_score(y_train, labels))
        V_measure_score.append(metrics.v_measure_score(y_train, labels))
        Fowlkes_mallows_score.append(metrics.fowlkes_mallows_score(y_train, labels))

        Silhouette_coefficient.append(metrics.silhouette_score(X_train_now, labels, metric='euclidean'))
        Calinski_harabaz_score.append(metrics.calinski_harabaz_score(X_train_now, labels))
        Davies_bouldin_score.append(metrics.davies_bouldin_score(X_train_now, labels))

    arr1.append(Sum_of_squared_distances)
    arr2.append(Adjusted_rand_index)
    arr3.append(Mutual_info_score)
    arr4.append(Adjusted_mutual_info_score)
    arr5.append(Normalized_mutual_info_score)
    arr6.append(Homogeneity_score)
    arr7.append(Completeness_score)
    arr8.append(V_measure_score)
    arr9.append(Fowlkes_mallows_score)
    arr10.append(Silhouette_coefficient)
    arr11.append(Calinski_harabaz_score)
    arr12.append(Davies_bouldin_score)

drawMultiple(data=arr1, title='Elbow Method For Optimal k', x_label='k', y_label='Sum Of Squared Distances',labels=['origin','PCA','ICA','RP','FA'], r=r)
drawMultiple(data=arr2, title='Adjusted Rand Index over k', x_label='k', y_label='Adjusted Rand Score',labels=['origin','PCA','ICA','RP','FA'], r=r)
#drawMultiple(data=[Mutual_info_score, Adjusted_mutual_info_score, Normalized_mutual_info_score], title='Mutual Information over k', x_label='k', y_label='Mutual Info Score', labels=['NMI','MI','AMI'], r=r)
#drawMultiple(data=[Homogeneity_score, Completeness_score, V_measure_score], title='Homogeneity, Completeness, V-measure over k', x_label='k', y_label='Score', labels=['v-measure', 'homogeneity','completeness'], r=r)
drawMultiple(data=arr9, title='Fowlkes Mallows Scores over k', x_label='k', y_label='Fowlkes Mallows Score',labels=['origin','PCA','ICA','RP','FA'], r=r)
drawMultiple(data=arr10, title='Silhouette Coefficient over k', x_label='k', y_label='Silhouette Coefficient',labels=['origin','PCA','ICA','RP','FA'], r=r)
drawMultiple(data=arr11, title='Calinski Harabaz Scores over k', x_label='k', y_label='Calinski Harabaz Score',labels=['origin','PCA','ICA','RP','FA'], r=r)
drawMultiple(data=arr12, title='Davies Bouldin Scores over k', x_label='k', y_label='Davies Bouldin Score',labels=['origin','PCA','ICA','RP','FA'], r=r)

drawMultiple(data=arr5, title='Normalized Mutual Info over k', x_label='k', y_label='Normalized Mutual Info',labels=['origin','PCA','ICA','RP','FA'], r=r)
drawMultiple(data=arr8, title='V-Scores over k', x_label='k', y_label='V-Score',labels=['origin','PCA','ICA','RP','FA'], r=r)

#EM
X_trains = [X_train, X_train_PCA, X_train_ICA, X_train_RP, X_train_FA]
arr1,arr2,arr3,arr4,arr5,arr6,arr7,arr8,arr9,arr10,arr11,arr12,arr13,arr14=[],[],[],[],[],[],[],[],[],[],[],[],[],[]
alg = ['Origin', 'PCA', 'ICA', 'RP', 'FA']

for i in range(len(X_trains)):
    print('EM on', alg[i])
    X_train_now = X_trains[i]
    
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
        gmm.fit(X_train_now)
        labels = gmm.predict(X_train_now)
        Log_likelihood.append(gmm.score(X_train_now))
        Adjusted_rand_index.append(metrics.adjusted_rand_score(y_train, labels))
        Mutual_info_score.append(metrics.mutual_info_score(y_train, labels))
        Adjusted_mutual_info_score.append(metrics.adjusted_mutual_info_score(y_train, labels))
        Normalized_mutual_info_score.append(metrics.normalized_mutual_info_score(y_train, labels))
        Homogeneity_score.append(metrics.homogeneity_score(y_train, labels))
        Completeness_score.append(metrics.completeness_score(y_train, labels))
        V_measure_score.append(metrics.v_measure_score(y_train, labels))
        Fowlkes_mallows_score.append(metrics.fowlkes_mallows_score(y_train, labels))

        Silhouette_coefficient.append(metrics.silhouette_score(X_train_now, labels, metric='euclidean'))
        Calinski_harabaz_score.append(metrics.calinski_harabaz_score(X_train_now, labels))
        Davies_bouldin_score.append(metrics.davies_bouldin_score(X_train_now, labels))
        AIC.append(gmm.aic(X_train_now))
        BIC.append(gmm.bic(X_train_now))
        
    arr1.append(Log_likelihood)
    arr2.append(Adjusted_rand_index)
    arr3.append(Mutual_info_score)
    arr4.append(Adjusted_mutual_info_score)
    arr5.append(Normalized_mutual_info_score)
    arr6.append(Homogeneity_score)
    arr7.append(Completeness_score)
    arr8.append(V_measure_score)
    arr9.append(Fowlkes_mallows_score)
    arr10.append(Silhouette_coefficient)
    arr11.append(Calinski_harabaz_score)
    arr12.append(Davies_bouldin_score)
    arr13.append(AIC)
    arr14.append(BIC)

drawMultiple(data=arr1, title='Elbow Method For Optimal Components', x_label='Components', y_label='Log_likelihood',labels=['origin','PCA','ICA','RP','FA'], r=r)
drawMultiple(data=arr2, title='Adjusted Rand Index over Components', x_label='Components', y_label='Adjusted Rand Score',labels=['origin','PCA','ICA','RP','FA'], r=r)
#drawMultiple(data=[Mutual_info_score, Adjusted_mutual_info_score, Normalized_mutual_info_score], title='Mutual Information over Components', x_label='Components', y_label='Mutual Info Score', labels=['NMI','MI','AMI'], r=r)
#drawMultiple(data=[Homogeneity_score, Completeness_score, V_measure_score], title='Homogeneity, Completeness, V-measure over Components', x_label='Components', y_label='Score', labels=['v-measure', 'homogeneity','completeness'], r=r)
drawMultiple(data=arr9, title='Fowlkes Mallows Scores over Components', x_label='Components', y_label='Fowlkes Mallows Score',labels=['origin','PCA','ICA','RP','FA'], r=r)
drawMultiple(data=arr10, title='Silhouette Coefficient over Components', x_label='Components', y_label='Silhouette Coefficient',labels=['origin','PCA','ICA','RP','FA'], r=r)
drawMultiple(data=arr11, title='Calinski Harabaz Scores over Components', x_label='Components', y_label='Calinski Harabaz Score',labels=['origin','PCA','ICA','RP','FA'], r=r)
drawMultiple(data=arr12, title='Davies Bouldin Scores over Components', x_label='Components', y_label='Davies Bouldin Score',labels=['origin','PCA','ICA','RP','FA'], r=r)

drawMultiple(data=arr5, title='Normalized Mutual Info over Components', x_label='Components', y_label='Normalized Mutual Info',labels=['origin','PCA','ICA','RP','FA'], r=r)
drawMultiple(data=arr8, title='V-Scores over Components', x_label='Components', y_label='V-Score',labels=['origin','PCA','ICA','RP','FA'], r=r)
drawMultiple(data=arr13, title='Akaike Information Criterion over Components', x_label='Components', y_label='AIC Score', labels=['origin','PCA','ICA','RP','FA'], r=r)
drawMultiple(data=arr14, title='Bayesian Information Criterion over Components', x_label='Components', y_label='BIC Score', labels=['origin','PCA','ICA','RP','FA'], r=r)
