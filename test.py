import numpy as np
import pandas as pd
import csv
from sklearn.preprocessing import MinMaxScaler
import pywt
import scipy
from scipy.integrate import simps
import pickle
import os

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

def convert_to_int(df_list):
    for df in df_list:
        for i in range(len(df.iloc[0])):
            df[i] = pd.to_numeric(df[i], errors='coerce')
    return df_list

def drop_last_column(df_list):
    preprocessed_df_list = []
    for df in df_list:
        if df.shape[1] >= 31:
            df = df.drop(df.columns[[30]], axis = 1)
        preprocessed_df_list.append(df)
    return preprocessed_df_list

def reverse_df(df):
    df = df.iloc[:, ::-1]
    return df

def interpolate_df(df_list):
    preprocessed_df_list = []
    for df in df_list:
        for i in range(len(df)):
            df.loc[[i]] = df.loc[[i]].interpolate(axis=1, limit=None, limit_direction='both')
        preprocessed_df_list.append(df)
    return preprocessed_df_list

def df_drop_na(df_list):
    preprocessed_df_list = []
    for df in df_list:
        df = df.dropna()
        df = df.reset_index(drop=True)
        preprocessed_df_list.append(df)
    return preprocessed_df_list

def calculate_fft(df_list):
    if len(df_list) == 1:
        return [np.abs(np.fft.fft(df_list[0]))]
    else:
        top_fft_features = []
        for df in zip(df_list):
            if len(top_fft_features) == 0:
                top_fft_features = np.abs(np.fft.fft(np.flip(df)))
            else:
                top_fft_features = np.concatenate((top_fft_features, np.abs(np.fft.fft(np.flip(df)))), axis=1)
        return top_fft_features

def calc_feature_dwt(df):
    cA, cB = pywt.dwt(df, 'haar')
    feature_dwt_top8 = cA[:, :-8]
    return feature_dwt_top8

def discrete_wavelet_transform(df_list):
    top_dwt_features = []
    if len(df_list) == 1:
        return calc_feature_dwt(df_list[0])
    else:
        for df in df_list:
            if len(top_dwt_features) == 0:
                top_dwt_features = calc_feature_dwt(df)
            else:
                top_dwt_features = np.concatenate((top_dwt_features, calc_feature_dwt(df)))

def coefficient_of_variation(df_list):
    feature_cov = []
    for df in df_list:
        for i in range(len(df)):
            feature_cov.append(np.mean(df[i]) / np.std(df[i]))
    feature_cov_ = np.asarray(feature_cov)
    feature_cov_wo_nan = feature_cov_[np.isnan(feature_cov_) == False]
    feature_cov_wo_nan.sort()
    mean_with_threshold = np.mean(feature_cov_wo_nan[0:len(feature_cov_wo_nan) - 1])
    feature_cov_[feature_cov_ > 200] = mean_with_threshold
    for x in range(len(feature_cov_)):
        if np.isnan(feature_cov_[x]):
            feature_cov_[x] = mean_with_threshold
    return feature_cov_

def windowed_entropy(df_list):
    output_entropy = []
    ordered_cgm = []
    for df in df_list:
        for j in range(len(df)):
            temp = []
            c = df.iloc[j]
            for m in range(len(c) -1, -1, -1):
                temp.append(c[m])
            y_array = np.array(temp)
            ordered_cgm.append(y_array)
    for i in range(len(ordered_cgm)):
        entropy_arr = []
        for j in range(1, 30, 5):
            s = scipy.stats.entropy(np.asarray(ordered_cgm)[i, j:j + 5])
            entropy_arr.append(s)
        output_entropy.append(np.amin(np.asarray(entropy_arr)))
    output_entropy = np.asarray(output_entropy)
    return output_entropy

def area_under_curve(df_list):
    feature_auc = []
    for df in df_list:
        for x in simps(df[:, ::-1], dx=5):
            feature_auc.append(x)
    feature_auc = np.asarray(feature_auc)
    return feature_auc

def construct_feature_matrix(top_fft_features, top_dwt_features, feature_cov_, output_entropy, feature_auc):
    feature_matrix = np.hstack((top_fft_features[0][:, 1:9], top_dwt_features[:, 1:7], feature_cov_[:, None], output_entropy[:, None], feature_auc[:, None]))
    return feature_matrix

def main():
    fpath = input("Enter the dataset absolute path: ")
    if not os.path.isfile(fpath):
        print("File does not exist")
        return -1

    testData = read_csv_file(fpath)
    pre_processed_test_data = convert_to_int([testData])
    pre_processed_test_data = drop_last_column(pre_processed_test_data)
    pre_processed_test_data = interpolate_df(pre_processed_test_data)
    pre_processed_test_data = df_drop_na(pre_processed_test_data)
    pre_processed_test_data = reverse_df(pre_processed_test_data[0])
    
    entropy_test = windowed_entropy([pre_processed_test_data])

    scaler = MinMaxScaler(feature_range=(0, 1))
    pre_processed_test_data = scaler.fit_transform(pre_processed_test_data)

    fft_test = calculate_fft([pre_processed_test_data])
    dwt_test = discrete_wavelet_transform([pre_processed_test_data])
    coef_var_test = coefficient_of_variation([pre_processed_test_data])
    feature_auc = area_under_curve([pre_processed_test_data])

    test_f_matrix = construct_feature_matrix(fft_test, dwt_test, coef_var_test, entropy_test, feature_auc)
    scaled_test_f_matrix = scaler.fit_transform(test_f_matrix)
    scaled_test_f_matrix_ = np.nan_to_num(scaled_test_f_matrix)
    final_test_matrix = np.asmatrix(scaled_test_f_matrix_)

    loaded_model = pickle.load(open('pca_model.pickle', 'rb'))
    reduced_test_matrix = loaded_model.transform(final_test_matrix)

    gnb_clf = pickle.load(open('gaussian_naive_bayes_model.pickle', 'rb'))
    rf_clf = pickle.load(open('random_forest_model.pickle', 'rb'))
    y_pred_results = {}
    y_pred = gnb_clf.predict(reduced_test_matrix)
    y_pred_results["gaussian_naive_bayes"] = y_pred
    y_pred = rf_clf.predict(reduced_test_matrix)
    y_pred_results["random_forest"] = y_pred
    df_result = pd.DataFrame().from_dict(y_pred_results)
    print(df_result)
    df_result.to_csv("predictions.csv", encoding='utf-8')

if __name__ == '__main__':
    main()