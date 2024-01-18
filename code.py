# silence warnings, not all iterations converge and it's annoying to see in the terminal
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.manifold import TSNE
from itertools import combinations
import matplotlib.pyplot as plt
import seaborn as sns


file_path = 'flags/flag.data'

column_names = [
    'name', 'landmass', 'zone', 'area', 'population', 'language', 'religion',
    'bars', 'stripes', 'colours', 'red', 'green', 'blue', 'gold', 'white', 'black',
    'orange', 'mainhue', 'circles', 'crosses', 'saltires', 'quarters', 'sunstars',
    'crescent', 'triangle', 'icon', 'animate', 'text', 'topleft', 'botright'
]


# Load the dataset into a DataFrame
df = pd.read_csv(file_path, names=column_names, header=None)
df = df.drop(['name', 'topleft', 'botright'], axis=1)

def tryEverySubset(df, subset_size):
    label_encoder = LabelEncoder()
    min_max_scaler = MinMaxScaler()
    y_train = label_encoder.fit_transform(df['mainhue'])
    df = df.drop(['mainhue'], axis=1)
    results = []
    all_combinations = list(combinations(list(df.columns.values), subset_size))
    
    for i, subset in enumerate(all_combinations):
        # print(subset) 
        if(i%300 == 0):
            print('iteration', i, 'of', len(all_combinations))
        subset_df = df[list(subset)]
        X_train = min_max_scaler.fit_transform(subset_df)
        
        # LR = LogisticRegression(max_iter=1000, verbose=0, multi_class='multinomial', penalty=None)
        # LR.fit(X_train, y_train)
        # train_predictionsLR = LR.predict(X_train)
        # train_accuracyLR = accuracy_score(y_train, train_predictionsLR)
        # results.append((train_accuracyLR, list(subset)))
        
        # # SVM
        # SVM = svm.SVC()
        # SVM.fit(X_train, y_train)
        # train_predictionsSVM = SVM.predict(X_train)
        # train_accuracySVM= accuracy_score(y_train, train_predictionsSVM)
        # results.append((train_accuracySVM, list(subset)))
        
        # # Random forest
        # RF = RandomForestClassifier()
        # RF.fit(X_train, y_train)
        # train_predictionsRF = RF.predict(X_train)
        # train_accuracyRF= accuracy_score(y_train, train_predictionsRF)
        # results.append((train_accuracyRF, list(subset)))
        
        # Decision tree
        DT = DecisionTreeClassifier()
        DT.fit(X_train, y_train)
        train_predictionsDT = DT.predict(X_train)
        train_accuracyDT= accuracy_score(y_train, train_predictionsDT)
        results.append((train_accuracyDT, list(subset)))
        
    results.sort(key=lambda x: x[0], reverse=True)
    return results[:3]


def visualizeUsingTSNE(df):
    # Extract features and target
    min_max_scaler = MinMaxScaler()
    feat_cols = (df.drop(['mainhue'], axis=1)).columns.values
    # feat_cols = ['landmass', 'zone', 'area', 'population']
    X = min_max_scaler.fit_transform(df[feat_cols])
    y = df['mainhue'].values

    # Perform t-SNE
    tsne = TSNE(n_components=2, verbose=1, perplexity=30, n_iter=300, learning_rate=200)
    tsne_results = tsne.fit_transform(X)
    
    df_subset = pd.DataFrame()
    df_subset['tsne-2d-one'] = tsne_results[:,0]
    df_subset['tsne-2d-two'] = tsne_results[:,1]
    df_subset['mainhue'] = y

    # Plot the results with each point colored by its corresponding 'mainhue' value
    plt.figure(figsize=(16,10))
    unique_colors = df_subset['mainhue'].unique()

    for color in unique_colors:
        # Select only data rows with the current color
        subset = df_subset[df_subset['mainhue'] == color]
        plt.scatter(subset['tsne-2d-one'], subset['tsne-2d-two'], label=color, color=color, s=50, edgecolors='black', linewidth=0.5)

    plt.legend()

    plt.savefig("matplotlib.png")  #savefig (show doesn't work on wsl)
    
def manuallyChooseFeatures(df, features):
    label_encoder = LabelEncoder()
    min_max_scaler = MinMaxScaler()
    y_train = label_encoder.fit_transform(df['mainhue'])
    df = df.drop(['mainhue'], axis=1)
    X_train = min_max_scaler.fit_transform(df)
    
    DT = DecisionTreeClassifier()
    DT.fit(X_train, y_train)
    train_predictionsDT = DT.predict(X_train)
    train_accuracyDT= accuracy_score(y_train, train_predictionsDT)
    
    return train_accuracyDT
    
manFeats = []
# append manually chosen features

for feats in manFeats:
    res = manuallyChooseFeatures(df, feats)
    print(f'For features {feats}, accuracy {res}')
    
    
# numToOrder = {
#     0: '',
#     1: 'second',
#     2: 'third',
# }

    
# for i in range(1, 5):
#     res = tryEverySubset(df, i)
#     for j, r in enumerate(res):
#         print(f'Number of features: {i}, {numToOrder[j]} best attributes {r[1]} accuracy {r[0]}')
        

visualizeUsingTSNE(df)
