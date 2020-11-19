import pandas as pd
import numpy as np
from IPython.display import display
import matplotlib.pyplot as plt
from pandas import DataFrame
from pandas import concat
from scipy.signal import find_peaks as find_peaks
from scipy.fftpack import rfft
from sklearn.decomposition import PCA
from scipy.optimize import curve_fit
import pickle
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans, DBSCAN
from scipy.spatial import distance

COLUMNS = np.array(['Slope_minmax', 'PeakVal1_error', 'PeakVal2_error', 'PeakHt1_error', 'PeakHt2_error', 'Min1_window', 'Min2_window', 'Max1_window', 'Max2_window', 'Var1_window', 'Var2_window', 'Mean1_window', 'Mean2_window', 'Sig_coef1','Sig_coef2','Sig_coef3','Sig_coef4', 'Max_fft', 'Min_fft', 'Mean_fft', 'Var_fft'])

def sigmoid(x, L, x0, k, b):
    y = L / (1 + np.exp(-k * (x - x0))) + b
    return (y)

def CalcFeatureSet1(cgmNorm_np, cgmSeries_np):
    maxs = np.argmax(cgmNorm_np, axis=1)
    mins = [np.argmin(cgmNorm_np[i, maxs[i]:])+maxs[i] for i in range(len(maxs))]
    slopes = []
    time_diffs = []
    for i in range(len(maxs)):
        slope = (cgmNorm_np[i][maxs[i]]-cgmNorm_np[i][mins[i]])/(cgmSeries_np[maxs[i]]-cgmSeries_np[mins[i]])
        time_diffs.append(cgmSeries_np[maxs[i]]-cgmSeries_np[mins[i]])
        slopes.append(slope)
    slopes = np.nan_to_num(slopes)
    time_diffs = np.nan_to_num(time_diffs)
    reg_window_size = 4
    reg_errors = []
    peak_values = []
    peak_heights = []
    peak_time_diffs = []
    peak_times = []
    for j in range(len(cgmNorm_np)):
        errors = np.array([])
        for i in range(len(cgmNorm_np[j])-reg_window_size):
            times = cgmSeries_np[i:i+reg_window_size-1]
            if np.isnan(times).any():
                errors = np.append(errors, -1)
                continue
            coeffs = np.polyfit(times, cgmNorm_np[j][i:i+reg_window_size-1], 1)
            poly = np.poly1d(coeffs)
            error = poly(cgmSeries_np[i+reg_window_size])-cgmNorm_np[j][i+reg_window_size];
            errors = np.append(errors, error)
        peaks, height_dict = find_peaks(errors, height = 0)
        heights = height_dict['peak_heights']
        sorted_args = heights.argsort()
        peaks = peaks[sorted_args]
        peaks = peaks[-2:]
        heights = heights[sorted_args]
        heights = heights[-2:]
        values = cgmNorm_np[j][peaks+reg_window_size-1]
        times1 = cgmSeries_np[peaks+reg_window_size]
        times2 = cgmSeries_np[peaks+reg_window_size-1]
        reg_errors.append(errors)
        while(len(values) < 2):
            values = np.append(values, 0)
            heights = np.append(heights, 0)
            times1 = np.append(times, 0)
            times2 = np.append(times2, 0)
        peak_values.append(values)
        peak_heights.append(heights)
        peak_time_diffs.append(times1)
        peak_times.append(times2)
    reg_errors = np.array(reg_errors)
    matrix = []
    for i in range(0, len(cgmNorm_np)):
        matrix_row = np.array([])
        matrix_row = np.append(matrix_row, slopes[i])
        matrix_row = np.append(matrix_row, peak_values[i])
        matrix_row = np.append(matrix_row, peak_heights[i])
        matrix.append(matrix_row)
    matrix = np.array(matrix)
    return matrix

def CalcFeatureSet2(cgmNorm_np, cgmSeries_np):
    window_mins = []
    window_maxs = []
    window_means = []
    window_vars = []
    for i in range(0, len(cgmNorm_np)):
        window_input = DataFrame(cgmNorm_np[i][::-1])
        width=5
        shifted=window_input.shift(width - 1)
        window=shifted.rolling(window=width)
        dataframe=concat([window.var(), window.min(),  window.mean(), window.max() ], axis=1)
        dataframe.columns = ['var', 'min', 'mean', 'max']
        window_features = dataframe.nlargest(2,'var')
        window_values = window_features.values
        window_mins.append([window_values[0][1], window_values[1][1]])
        window_maxs.append([window_values[0][3], window_values[1][3]])
        window_vars.append([window_values[0][0], window_values[1][0]])
        window_means.append([window_values[0][2], window_values[1][2]])  
    matrix = []
    for i in range(0, len(cgmNorm_np)):
        matrix_row = np.array([])
        matrix_row = np.append(matrix_row, window_mins[i])
        matrix_row = np.append(matrix_row, window_maxs[i])
        matrix_row = np.append(matrix_row, window_vars[i])
        matrix_row = np.append(matrix_row, window_means[i])
        matrix.append(matrix_row)
    matrix = np.array(matrix)
    return matrix

def CalcFeatureSet3(cgmNorm_np, cgmSeries_np):
    n_series = []
    n_datenum = []
    sig1 = []
    sig2 = []
    sig3 = []
    sig4 = []
    for i in range(0, len(cgmNorm_np)):
        idx = np.isfinite(cgmSeries_np) & np.isfinite(cgmNorm_np[i])
        n_series.append(cgmNorm_np[i][idx])  
        n_datenum.append(cgmSeries_np[idx])
    for i in range(0,len(cgmNorm_np)):
        if(len(n_series[i]) !=0 ):
            try:
                p0 = [max(n_series[i]), np.median(n_datenum[i]),250,min(n_series[i])] 
                popt, pcov = curve_fit(sigmoid, n_datenum[i], n_series[i],p0,method='trf')
            except: 
                popt=[0,0,0,0]
            sig1.append(popt[0])
            sig2.append(popt[1])
            sig3.append(popt[2])
            sig4.append(popt[3])    
    matrix = []
    for i in range(0, len(cgmNorm_np)):
        matrix_row = np.array([])
        matrix_row = np.append(matrix_row, sig1[i])
        matrix_row = np.append(matrix_row, sig2[i])
        matrix_row = np.append(matrix_row, sig3[i])
        matrix_row = np.append(matrix_row, sig4[i])
        matrix.append(matrix_row)
    matrix = np.array(matrix)
    return matrix

def CalcFeatureSet4(cgmNorm_np, cgmSeries_np):
    Feature_vector=[]
    for i in range(0, len(cgmNorm_np)):
        fastfouriertransform=rfft(cgmNorm_np[i])
        fft_max=np.nanmax(fastfouriertransform)
        s=np.where(fastfouriertransform == fft_max)
        fft_min=np.nanmin(fastfouriertransform)
        s=np.where(fastfouriertransform == fft_min)
        fft_mean=np.nanmean(fastfouriertransform)
        fft_variance=np.nanvar(fastfouriertransform)
        Feature_vector.append(np.array([fft_max,fft_min,fft_mean,fft_variance]))
    matrix = np.array(Feature_vector)
    return matrix

def MergedFeatures(cgmNorm_np, cgmSeries_np):
    feature_set_1 = CalcFeatureSet1(cgmNorm_np, cgmSeries_np)
    feature_set_2 = CalcFeatureSet2(cgmNorm_np, cgmSeries_np)
    feature_set_3 = CalcFeatureSet3(cgmNorm_np, cgmSeries_np)
    feature_set_4 = CalcFeatureSet4(cgmNorm_np, cgmSeries_np)
    features = np.concatenate((feature_set_1, feature_set_2), axis=1)
    features = np.concatenate((features, feature_set_3), axis=1)
    features = np.concatenate((features, feature_set_4), axis=1)
    features = np.nan_to_num(features)
    return features

def GenerateDF(features, columns):
    feature_df = pd.DataFrame(features, columns=columns)
    return feature_df

def NormalizeDF(feature_df, columns, max_scale):
    for i in columns:
        feature_df[i] = feature_df[i]/max_scale[i]
    return feature_df

file_no = [1, 2, 3, 4, 5]
cgmData = pd.read_csv("mealData" + str(1) + '.csv', names=list(range(50)))
cgmLabels = pd.read_csv("mealAmountData" + str(1) + '.csv', names=['labels'])
cgmLabels = cgmLabels.iloc[:cgmData.shape[0]]
for i in file_no[1:]:
    cgm = pd.read_csv("mealData" + str(i) + '.csv', names=list(range(50)))
    cgm2 = pd.read_csv("mealAmountData" + str(i) + '.csv', names=['labels'])
    cgmData = cgmData.append(cgm)
    cgmLabels = cgmLabels.append(cgm2)
    cgmLabels = cgmLabels.iloc[:cgmData.shape[0]]
cgmData = cgmData.dropna(axis='columns', how='all')
cgmData = cgmData.mask(cgmData.eq(-1)).ffill(axis=1)
cgmData = cgmData.mask(cgmData.eq(-1)).bfill(axis=1)
zero_entries = cgmData.isna().any(axis=1)
cgmData = cgmData[zero_entries == False]
cgmLabels = cgmLabels[zero_entries == False]
cgmValues_np = cgmData.values
cgmNorm_np = cgmValues_np/400.0
length = len(cgmNorm_np[0])
cgmSeries_np = [0.0831*(length-i-1) for i in range(0, length)]
cgmSeries_np = np.array(cgmSeries_np)
features = MergedFeatures(cgmNorm_np, cgmSeries_np)
features_df = GenerateDF(features, COLUMNS)
max_scale = features_df.abs().max(axis=0)
max_scale.to_pickle('DataScale.pickle')
normal_df = NormalizeDF(features_df, COLUMNS, max_scale)
display(normal_df)
normal_df.to_csv('MealFeatures.csv', index=False)
print(cgmLabels)

PCA_filename = 'PCA.pickle'
def GeneratePCA(data, PCA_filename):
    pca = PCA(n_components=5)
    pca.fit(data)
    components = pca.components_
    x = [i for i in range(0, len(components[0]))]
    for i in range(0, 5):
        plt.figure()
        plt.bar(x, components[i])
        plt.xticks(np.arange(len(components[0])), x, rotation=90)
        plt.show()
    PCA_file = open(PCA_filename, 'wb')
    pickle.dump(pca, PCA_file)
    PCA_file.close()

def Transform(data, PCA_filename):
    PCA_file = open(PCA_filename, 'rb')
    pca = pickle.load(PCA_file)
    PCA_file.close()
    return pca.transform(data)

mealData = pd.read_csv('MealFeatures.csv')
GeneratePCA(mealData, PCA_filename)
mealTransform = Transform(mealData, PCA_filename)
cgmLabels2 = cgmLabels[:]
mealLabels = cgmLabels2.to_numpy()
for i, e in enumerate(mealLabels):
    if e == 0:
        mealLabels[i] = 1
    else:
        if mealLabels[i] % 20 <= 0:
            mealLabels[i] = mealLabels[i]/20+1
        else:
            mealLabels[i] = mealLabels[i]/20+2
mealDataWithLabels = np.concatenate((mealTransform, mealLabels), axis=1)
dataset = mealDataWithLabels

def metric(a, b):
    d = distance.euclidean(a, b)
    return d

def dbscan_sse(dbscan,X):
    m = dbscan.eps
    s = 0
    for j, x_n in enumerate(X):
        for i, x_c in enumerate(dbscan.components_):
            if metric(x_n, x_c) < m:
                m = metric(x_n, x_c)
        s += m
    return s

def cluster_prediction(predictions, g_truth, type='None'):
    if type == 'db':
        new_predictions = np.zeros((len(predictions)))
        new_predictions.fill(-1)
    else:
        new_predictions = np.zeros((len(predictions)))
    for cluster in range(0,6):
        v = np.zeros((6))
        for i, l in enumerate(predictions):
            if l == cluster:
                v[g_truth[i]] += 1
        print(v, sum(v))
        new_cluster = int(np.argmax(v))
        for i, l in enumerate(predictions):
            if l == cluster:
                new_predictions[i] = int(new_cluster)
    return new_predictions

def cluster_prediction_test_dbscan(dataset, predictions, knn, dbscan, ground_truth=None):
    new_predictions = np.zeros((len(predictions)))
    new_predictions.fill(-1)
    for cluster in range(0,6):
        v = np.zeros((6))
        for i, l in enumerate(predictions):
            if l == cluster:
                x = [dataset[i]]
                g_truth = knn.predict(x)
                v[g_truth] +=1
        print(v, sum(v))
        new_cluster = int(np.argmax(v))
        for i, l in enumerate(predictions):
            if l == cluster:
                new_predictions[i]=int(new_cluster)
    return new_predictions

def cluster_prediction_test_kmeans(dataset, predictions, knn, kmeans, ground_truth=None):
    new_predictions = np.zeros((len(predictions)))
    for cluster in range(0,6):
        v = np.zeros((6))
        for i, l in enumerate(predictions):
            if l == cluster:
                x = [dataset[i]]
                g_truth = knn.predict(x)
                v[g_truth] +=1
        print(v, sum(v))
        new_cluster = int(np.argmax(v))
        for i, l in enumerate(predictions):
            if l == cluster:
                new_predictions[i] = int(new_cluster)
    return new_predictions

kf = KFold(n_splits=3, shuffle=True, random_state=0)
for train_index, test_index in kf.split(dataset[:, :-1]):
    X_train_knn, X_test_knn = dataset[:, :-1][train_index], dataset[:, :-1][test_index]
    Y_train_knn, Y_test_knn = dataset[:, -1].astype(int)[train_index], dataset[:, -1].astype(int)[test_index]
    knn = KNeighborsClassifier(n_neighbors=10)
    knn.fit(X_train_knn, Y_train_knn)
    dbs = DBSCAN(eps=0.2, min_samples=2)
    dbs.fit(X_train_knn)
    mn = dbs.eps
    y_n = np.ones(shape=len(X_test_knn), dtype=int)*-1
    for j, x_n in enumerate(X_test_knn):
        for i, x_c in enumerate(dbs.components_):
            if metric(x_n, x_c) < dbs.eps:
                y_n[j] = dbs.labels_[dbs.core_sample_indices_[i]]
                break
    dbslabels = y_n
    dbslabels = cluster_prediction_test_dbscan(X_test_knn, dbslabels[:], knn, dbs, Y_test_knn)
    dbsse = dbscan_sse(dbs, X_train_knn)
    km = KMeans(n_clusters=6, random_state=0)
    km.fit(X_train_knn)
    kmlabels=km.predict(X_test_knn)
    kmlabels = cluster_prediction_test_kmeans(X_test_knn, kmlabels[:], knn, km, Y_test_knn)
    center_dbscan = 0
    center_kmeans = 0
    noise = 0
    for i in range(len(dbslabels)):
        if dbslabels[i] == -1:
            noise += 1
        center_dbscan += dbslabels[i] == Y_test_knn[i]
    for i in range(len(kmlabels)):
        center_kmeans += kmlabels[i] == Y_test_knn[i]
    print('DBSCAN Validation SSE = ', dbsse)
    print('DBScan Validation Accuracy = ', center_dbscan/(len(dbslabels)-noise))
    print('KMeans Validation SSE = ', km.inertia_)
    print('KMeans Validation Accuracy = ', center_kmeans/len(kmlabels))

knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(dataset[:, :-1], dataset[:, -1].astype(int))
save_file = open('KNN.pickle', 'wb')
pickle.dump(knn, save_file)
save_file.close()

dbs = DBSCAN(eps=0.2, min_samples=2)
dbslabels = dbs.fit_predict(dataset[:, :-1])
dbslabels = cluster_prediction_test_dbscan(dataset[:, :-1], dbslabels[:], knn, dbs, dataset[:, -1].astype(int))
dbsse = dbscan_sse(dbs, dataset[:, :-1])
save_file = open('DBSCAN.pickle', 'wb')
pickle.dump(dbs, save_file)
save_file.close()

km = KMeans(n_clusters=6, random_state=0)
kmlabels = km.fit_predict(dataset[:, :-1])
kmlabels = cluster_prediction_test_kmeans(dataset[:, :-1], kmlabels[:], knn, km, dataset[:, -1].astype(int))
save_file = open('KMeans.pickle', 'wb')
pickle.dump(km, save_file)
save_file.close()

center_kmeans = 0
center_dbscan = 0
noise = 0
for i in range(len(dbslabels)):
    if dbslabels[i] == -1:
        noise += 1
    center_dbscan += dbslabels[i] == int(dataset[i][-1])
for i in range(len(kmlabels)):
    center_kmeans += kmlabels[i] == int(dataset[i][-1])
print('DBSCAN labels = ', dbslabels)
print('KMeans labels = ', kmlabels)
print('DBSCAN Training SSE = ', dbsse)
print('DBSCAN Training Accuracy = ', center_dbscan/(len(dbslabels)-noise))
print('KMeans Training SSE = ', km.inertia_)
print('KMeans Training Accuracy = ', center_kmeans/len(kmlabels))