import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler


def preprocess_dataset(dataset_path):
    # Your code starts here (Multiple lines)
    global feature_name

    player_df = pd.read_csv(dataset_path)
    numcols = ['Overall', 'Crossing', 'Finishing', 'ShortPassing', 'Dribbling', 'LongPassing', 'BallControl',
               'Acceleration', 'SprintSpeed', 'Agility', 'Stamina', 'Volleys', 'FKAccuracy', 'Reactions', 'Balance',
               'ShotPower', 'Strength', 'LongShots', 'Aggression', 'Interceptions']
    catcols = ['Preferred Foot', 'Position', 'Body Type', 'Nationality', 'Weak Foot']

    player_df = player_df[numcols + catcols]

    traindf = pd.concat([player_df[numcols], pd.get_dummies(player_df[catcols])], axis=1)
    features = traindf.columns
    traindf = traindf.dropna()

    traindf = pd.DataFrame(traindf, columns=features)

    y = traindf['Overall'] >= 87
    X = traindf.copy()
    del X['Overall']

    y.reset_index(drop=True, inplace=True)

    feature_name = list(X.columns)
    num_feats = 30

    return X, y, num_feats


def get_corr_support(selected_features):
    return [True if feature in selected_features else False for feature in feature_name]


def get_scaled_data(X):
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    return pd.DataFrame(X_scaled, columns=X.columns)


def cor_selector(X, y, num_feats):
    pearson_cor = pd.concat([get_scaled_data(X), y], axis=1).corr()["Overall"].to_dict()
    del pearson_cor["Overall"]

    sorted_features_with_values = sorted(pearson_cor.items(), key=lambda x: x[1], reverse=True)[:num_feats]
    selected_features = [sor[0] for sor in sorted_features_with_values]
    return get_corr_support(selected_features), selected_features


def chi_squared_selector(X, y, num_feats):
    chi2_features = SelectKBest(chi2, k=num_feats)
    chi2_features.fit(get_scaled_data(X), y)
    chi2_features.transform(X)
    selected_features = list(chi2_features.get_feature_names_out())
    return get_corr_support(selected_features), selected_features


def rfe_selector(X, y, num_feats):
    # Your code goes here (Multiple lines)

    estimator = LogisticRegression()
    selector = RFE(estimator, n_features_to_select=num_feats, step=10, verbose=True)
    selector = selector.fit(get_scaled_data(X), y)
    # Your code ends here
    return selector.support_, list(selector.get_feature_names_out())


def embedded_log_reg_selector(X, y, num_feats):
    # Your code goes here (Multiple lines)
    estimator = LogisticRegression()

    model = SelectFromModel(estimator, max_features=num_feats)
    model.fit(get_scaled_data(X), y)

    # Your code ends here
    return get_corr_support(model.get_feature_names_out()), list(model.get_feature_names_out())


def embedded_rf_selector(X, y, num_feats):
    estimator = RandomForestClassifier()

    model = SelectFromModel(estimator, max_features=num_feats)
    model.fit(get_scaled_data(X), y)

    # Your code ends here
    return get_corr_support(model.get_feature_names_out()), list(model.get_feature_names_out())


def embedded_lgbm_selector(X, y, num_feats):
    estimator = LGBMClassifier()

    model = SelectFromModel(estimator, max_features=num_feats)
    model.fit(get_scaled_data(X), y)

    # Your code ends here
    return get_corr_support(model.get_feature_names_out()), list(model.get_feature_names_out())


def autoFeatureSelector(dataset_path, methods=[]):
    # Parameters
    # data - dataset to be analyzed (csv file)
    # methods - various feature selection methods we outlined before, use them all here (list)

    # preprocessing
    X, y, num_feats = preprocess_dataset(dataset_path)
    cor_support, chi_support, rfe_support, embedded_lr_support, embedded_rf_support, embedded_lgbm_support = None, None, None, None, None, None
    # Run every method we outlined above from the methods list and collect returned best features from every method
    if 'pearson' in methods:
        cor_support, cor_feature = cor_selector(X, y, num_feats)
    if 'chi-square' in methods:
        chi_support, chi_feature = chi_squared_selector(X, y, num_feats)
    if 'rfe' in methods:
        rfe_support, rfe_feature = rfe_selector(X, y, num_feats)
    if 'log-reg' in methods:
        embedded_lr_support, embedded_lr_feature = embedded_log_reg_selector(X, y, num_feats)
    if 'rf' in methods:
        embedded_rf_support, embedded_rf_feature = embedded_rf_selector(X, y, num_feats)
    if 'lgbm' in methods:
        embedded_lgbm_support, embedded_lgbm_feature = embedded_lgbm_selector(X, y, num_feats)

    # Combine all the above feature list and count the maximum set of features that got selected by all methods
    feature_selection_df = pd.DataFrame(
        {'Feature': feature_name, 'Pearson': cor_support, 'Chi-2': chi_support, 'RFE': rfe_support,
         'Logistics': embedded_lr_support,
         'Random Forest': embedded_rf_support, 'LightGBM': embedded_lgbm_support})
    # count the selected times for each feature
    feature_selection_df['Total'] = np.sum(feature_selection_df, axis=1)
    # display the top 100
    feature_selection_df = feature_selection_df.sort_values(['Total', 'Feature'], ascending=False)
    feature_selection_df.index = range(1, len(feature_selection_df) + 1)

    return feature_selection_df.head(num_feats)["Feature"].to_list()


if __name__ == '__main__':
    best_features = autoFeatureSelector(dataset_path="data/fifa19.csv",
                                        methods=['pearson', 'chi-square', 'rfe', 'log-reg', 'rf', 'lgbm'])
    print("Selected best features by multiple methods - ", best_features)
