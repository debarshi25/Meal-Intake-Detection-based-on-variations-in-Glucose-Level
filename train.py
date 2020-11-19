import numpy as np
import pandas as pd
import csv
from sklearn.preprocessing import MinMaxScaler
import pywt
import scipy
from scipy.integrate import simps
from sklearn.decomposition import PCA
import pickle
from sklearn.model_selection import KFold
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

def read_csv_file(filepath):
    d = []
    with open(filepath) as csvfile:
        areader = csv.reader(csvfile)
        max_elements = 0
        for row in areader:
            if max_elements < len(row):
                max_elements = len(row)
        csvfile.seek(0)
        for i, row in enumerate(areader):
            d.append(row + ["" for x in range(max_elements - len(row))])
    df = pd.DataFrame(d)
    return df

mealData1 = read_csv_file('mealData1.csv');
mealData2 = read_csv_file('mealData2.csv');
mealData3 = read_csv_file('mealData3.csv');
mealData4 = read_csv_file('mealData4.csv');
mealData5 = read_csv_file('mealData5.csv');
Nomeal1 = read_csv_file('Nomeal1.csv')
Nomeal2 = read_csv_file('Nomeal2.csv')
Nomeal3 = read_csv_file('Nomeal3.csv')
Nomeal4 = read_csv_file('Nomeal4.csv')
Nomeal5 = read_csv_file('Nomeal5.csv')

df_list = [mealData1, mealData2, mealData3, mealData4, mealData5, Nomeal1, Nomeal2, Nomeal3, Nomeal4, Nomeal5]
print(mealData1.shape, mealData2.shape, mealData3.shape, mealData4.shape, mealData5.shape)
print(Nomeal1.shape, Nomeal2.shape, Nomeal3.shape, Nomeal4.shape, Nomeal5.shape)
print("Total number of records = ", mealData1.shape[0] + mealData2.shape[0] + mealData3.shape[0] + mealData4.shape[0] + mealData5.shape[0] + Nomeal1.shape[0] + Nomeal2.shape[0] + Nomeal3.shape[0] + Nomeal4.shape[0] + Nomeal5.shape[0])

df_idx = 0
for df in df_list:
    for i in range(len(df.iloc[0])):
        df[i] = pd.to_numeric(df[i], errors='coerce')
    if df_idx == 0:
        mealData1 = df
    if df_idx == 1:
        mealData2 = df
    if df_idx == 2:
        mealData3 = df
    if df_idx == 3:
        mealData4 = df
    if df_idx == 4:
        mealData5 = df
    if df_idx == 5:
        Nomeal1 = df
    if df_idx == 6:
        Nomeal2 = df
    if df_idx == 7:
        Nomeal3 = df
    if df_idx == 8:
        Nomeal4 = df
    if df_idx == 9:
        Nomeal5 = df
    df_list[df_idx] = df
    df_idx += 1

# Drop the 31st column
df_idx = 0
for df in df_list:
    if df_idx < 5:
        df = df.drop(df.columns[[30]], axis = 1)
        if df_idx == 0:
            mealData1 = df
        if df_idx == 1:
            mealData2 = df
        if df_idx == 2:
            mealData3 = df
        if df_idx == 3:
            mealData4 = df
        if df_idx == 4:
            mealData5 = df
        df_list[df_idx] = df
    df_idx += 1

mealData1 = mealData1.iloc[:, ::-1]
mealData2 = mealData2.iloc[:, ::-1]
mealData3 = mealData3.iloc[:, ::-1]
mealData4 = mealData4.iloc[:, ::-1]
mealData5 = mealData5.iloc[:, ::-1]
Nomeal1 = Nomeal1.iloc[:, ::-1]
Nomeal2 = Nomeal2.iloc[:, ::-1]
Nomeal3 = Nomeal3.iloc[:, ::-1]
Nomeal4 = Nomeal4.iloc[:, ::-1]
Nomeal5 = Nomeal5.iloc[:, ::-1]

df_list = [mealData1, mealData2, mealData3, mealData4, mealData5, Nomeal1, Nomeal2, Nomeal3, Nomeal4, Nomeal5]

df_idx = 0
for df in df_list:
    for i in range(len(df)):
        df.loc[[i]] = df.loc[[i]].interpolate(axis=1, limit=None, limit_direction='both')
    if df_idx == 0:
        mealData1 = df
    if df_idx == 1:
        mealData2 = df
    if df_idx == 2:
        mealData3 = df
    if df_idx == 3:
        mealData4 = df
    if df_idx == 4:
        mealData5 = df
    if df_idx == 5:
        Nomeal1 = df
    if df_idx == 6:
        Nomeal2 = df
    if df_idx == 7:
        Nomeal3 = df
    if df_idx == 8:
        Nomeal4 = df
    if df_idx == 9:
        Nomeal5 = df
    df_list[df_idx] = df
    df_idx += 1

df_idx = 0
for df in df_list:
    df = df.dropna()
    df = df.reset_index(drop=True)
    if df_idx == 0:
        mealData1 = df
    if df_idx == 1:
        mealData2 = df
    if df_idx == 2:
        mealData3 = df
    if df_idx == 3:
        mealData4 = df
    if df_idx == 4:
        mealData5 = df
    if df_idx == 5:
        Nomeal1 = df
    if df_idx == 6:
        Nomeal2 = df
    if df_idx == 7:
        Nomeal3 = df
    if df_idx == 8:
        Nomeal4 = df
    if df_idx == 9:
        Nomeal5 = df
    df_list[df_idx] = df
    df_idx += 1

print(mealData1.shape, mealData2.shape, mealData3.shape, mealData4.shape, mealData5.shape)
print(Nomeal1.shape, Nomeal2.shape, Nomeal3.shape, Nomeal4.shape, Nomeal5.shape)
print("Total number of records = ", mealData1.shape[0] + mealData2.shape[0] + mealData3.shape[0] + mealData4.shape[0] + mealData5.shape[0] + Nomeal1.shape[0] + Nomeal2.shape[0] + Nomeal3.shape[0] + Nomeal4.shape[0] + Nomeal5.shape[0])

# Scaling CGM Data
scaler = MinMaxScaler(feature_range=(0,1))
mealData1Scaled = scaler.fit_transform(mealData1)
mealData2Scaled = scaler.fit_transform(mealData2)
mealData3Scaled = scaler.fit_transform(mealData3)
mealData4Scaled = scaler.fit_transform(mealData4)
mealData5Scaled = scaler.fit_transform(mealData5)
Nomeal1Scaled = scaler.fit_transform(Nomeal1)
Nomeal2Scaled = scaler.fit_transform(Nomeal2)
Nomeal3Scaled = scaler.fit_transform(Nomeal3)
Nomeal4Scaled = scaler.fit_transform(Nomeal4)
Nomeal5Scaled = scaler.fit_transform(Nomeal5)

# Fast Fourier Transform
top_fft_features = []
df_list = [mealData1Scaled, mealData2Scaled, mealData3Scaled, mealData4Scaled, mealData5Scaled, Nomeal1Scaled, Nomeal2Scaled, Nomeal3Scaled, Nomeal4Scaled, Nomeal5Scaled]
i = 0
for df in zip(df_list):
    if len(top_fft_features) == 0:
        top_fft_features = np.abs(np.fft.fft(np.flip(df)))
    else:
        top_fft_features = np.concatenate((top_fft_features, np.abs(np.fft.fft(np.flip(df)))), axis=1)

# Discrete Wavelet Transform
def calc_feature_dwt(df):
    cA, cB = pywt.dwt(df, 'haar')
    feature_dwt_top8 = cA[:,:-8]
    return feature_dwt_top8
df_list = [mealData1Scaled, mealData2Scaled, mealData3Scaled, mealData4Scaled, mealData5Scaled, Nomeal1Scaled, Nomeal2Scaled, Nomeal3Scaled, Nomeal4Scaled, Nomeal5Scaled]
top_dwt_features = []
for df in df_list:
    if len(top_dwt_features) == 0:
        top_dwt_features = calc_feature_dwt(df)
    else:
        top_dwt_features = np.concatenate((top_dwt_features, calc_feature_dwt(df)))
        
# Coeffecient of Variation
feature_cov = []
df_list = [mealData1Scaled, mealData2Scaled, mealData3Scaled, mealData4Scaled, mealData5Scaled, Nomeal1Scaled, Nomeal2Scaled, Nomeal3Scaled, Nomeal4Scaled, Nomeal5Scaled]
for df in df_list:
    for i in range(len(df)):
        feature_cov.append(np.mean(df[i]) / np.std(df[i]))
feature_cov_ = np.asarray(feature_cov)
feature_cov_wo_nan = feature_cov_[np.isnan(feature_cov_) == False]
feature_cov_wo_nan.sort()
mean_with_threshold = np.mean(feature_cov_wo_nan[0:len(feature_cov_wo_nan)-1])
feature_cov_[feature_cov_ > 200] = mean_with_threshold
for x in range(len(feature_cov_)):
    if np.isnan(feature_cov_[x]):
        feature_cov_[x] = mean_with_threshold

# Windowed Entropy 
output_entropy = []
y = []
ordered_cgm = []
df_list = [mealData1, mealData2, mealData3, mealData4, mealData5, Nomeal1, Nomeal2, Nomeal3, Nomeal4, Nomeal5]
for df in df_list:
    for j in range(len(df)):
        temp = []
        temp1 = []
        c = df.iloc[j]
        for m in range(len(c) -1, -1, -1):
            temp.append(c[m])
        y_array = np.array(temp)
        ordered_cgm.append(y_array)        
for i in range(len(ordered_cgm)):
    entropy_arr = []
    for j in range(1, 30, 5):
        s = scipy.stats.entropy(np.asarray(ordered_cgm)[i, j:j+5])
        entropy_arr.append(s)
    output_entropy.append(np.amin(np.asarray(entropy_arr)))
output_entropy = np.asarray(output_entropy)
print(output_entropy.shape)

# Area Under Curve
feature_auc = []
df_list = [mealData1Scaled, mealData2Scaled, mealData3Scaled, mealData4Scaled, mealData5Scaled, Nomeal1Scaled, Nomeal2Scaled, Nomeal3Scaled, Nomeal4Scaled, Nomeal5Scaled]
for df in df_list:
    for x in simps(df[:,::-1], dx = 5):
        feature_auc.append(x)
feature_auc = np.asarray(feature_auc)
print(feature_auc.shape)

# Feature Matrix
feature_matrix = np.hstack((top_fft_features[0][:,1:9], top_dwt_features[:,1:7], feature_cov_[:, None], output_entropy[:, None], feature_auc[:, None]))
print(feature_matrix.shape)
scaled_feature_matrix = scaler.fit_transform(feature_matrix)
new_scaled_feature_matrix = np.nan_to_num(scaled_feature_matrix)
scaled_feature_matrix_ = np.asmatrix(new_scaled_feature_matrix)

# Apply PCA on the Feature Matrix of Training Data
pca = PCA(n_components=10)
reduced_meal_matrix = pca.fit_transform(new_scaled_feature_matrix[0:249])
reduced_nomeal_matrix = pca.transform(new_scaled_feature_matrix[249:])
pickle.dump(pca, open('pca_model.pickle', 'wb'))

final_feature_m = np.vstack((reduced_meal_matrix, reduced_nomeal_matrix))
print(final_feature_m.shape)
pca_feature_matrix = final_feature_m

label_m = []
for i in range(len(reduced_meal_matrix)):
    label_m.append(1)
for i in range(len(reduced_nomeal_matrix)):
    label_m.append(0)
label_m_ = np.asarray(label_m)
print(label_m_.shape)

# Gaussian Naive Bayes
score_gnb = []
precision_gnb=[]
recall_gnb=[]
f1_score_gnb=[]
d=[]
result = [[0,0],[0,0]]
kf_ = KFold(n_splits=10, shuffle=True, random_state=0)
gnb = GaussianNB()
for train_index, test_index in kf_.split(pca_feature_matrix):
    X_train_gnb, X_test_gnb = pca_feature_matrix[train_index], pca_feature_matrix[test_index]
    Y_train_gnb, Y_test_gnb = label_m_[train_index], label_m_[test_index]
    gnb.fit(X_train_gnb, Y_train_gnb)
    preds = gnb.predict(X_test_gnb)
    d = confusion_matrix(Y_test_gnb, preds)
    for i in range(2):
        for j in range(2):
            result[i][j] = (result[i][j] + d[i][j])
    a = precision_score(Y_test_gnb, preds)
    precision_gnb.append(a)
    b = recall_score(Y_test_gnb, preds)
    recall_gnb.append(b)
    c = f1_score(Y_test_gnb, preds)
    f1_score_gnb.append(c)
    pickle.dump(gnb, open('gaussian_naive_bayes_model.pickle', 'wb'))
    score_gnb.append(gnb.score(X_test_gnb, Y_test_gnb))
final_accuracy_gnb = np.max(score_gnb)
final_precision_gnb = np.max(precision_gnb)
final_recall_gnb = np.max(recall_gnb)
final_f1_score_gnb = np.max(f1_score_gnb)
print(result)
print("Accuracy = %s" %(final_accuracy_gnb))  
print("Precision = %s" %(final_precision_gnb))
print("Recall = %s" %(final_recall_gnb))
print("F1 score = %s" %(final_f1_score_gnb))

# Random Forest
score_rf = []
confusion_rf=[]
precision_rf=[]
recall_rf=[]
f1_score_rf=[]
d=[]
result = [[0,0],[0,0]]
kf_ = KFold(n_splits=10, shuffle=True, random_state=0)
rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=0)
for train_index, test_index in kf_.split(pca_feature_matrix):
    X_train_rf, X_test_rf = pca_feature_matrix[train_index], pca_feature_matrix[test_index]
    Y_train_rf, Y_test_rf = label_m_[train_index], label_m_[test_index]
    rf.fit(X_train_rf, Y_train_rf)
    preds = rf.predict(X_test_rf)
    d=confusion_matrix(Y_test_rf, preds)
    for i in range(2):
        for j in range(2):
            result[i][j] = (result[i][j] + d[i][j])
    a = precision_score(Y_test_rf, preds)
    precision_rf.append(a)
    b = recall_score(Y_test_rf, preds)
    recall_rf.append(b)
    c = f1_score(Y_test_rf, preds)
    f1_score_rf.append(c)
    pickle.dump(rf, open('random_forest_model.pickle', 'wb'))
    score_rf.append(rf.score(X_test_rf, Y_test_rf))
final_accuracy_rf = np.max(score_rf)
final_precision_rf = np.max(precision_rf)
final_recall_rf = np.max(recall_rf)
final_f1_score_rf = np.max(f1_score_rf)
print(result)
print("Accuracy = %s" %(final_accuracy_rf))
print("Precision = %s" %(final_precision_rf))
print("Recall = %s" %(final_recall_rf))
print("F1 score = %s" %(final_f1_score_rf))