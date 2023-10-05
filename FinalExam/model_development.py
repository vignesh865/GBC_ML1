import pprint
from sklearn.experimental import enable_iterative_imputer
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from sklearn.naive_bayes import GaussianNB
from tqdm import tqdm
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.feature_selection import RFE
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC


def read_data(url):
    return pd.read_csv("https://raw.githubusercontent.com/subashgandyer/datasets/main/great_customers.csv")


def get_cleaned_data(url):
    data = pd.read_csv(url)
    data.drop(["user_id"], inplace=True, axis=1)
    cat_data = data.select_dtypes("object")
    encoded = pd.get_dummies(cat_data)
    data.drop(cat_data.columns, inplace=True, axis=1)
    new_data = pd.concat([data, encoded], axis=1)
    X = new_data.drop(["great_customer_class"], axis=1)
    y = new_data["great_customer_class"]
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        stratify=y,
                                                        test_size=0.3,
                                                        random_state=42)

    imputer = IterativeImputer(max_iter=10, random_state=0)
    imputer.fit(X_train, y)
    X_train_transformed = imputer.transform(X_train)
    X_test_transformed = imputer.transform(X_test)

    scaler = MinMaxScaler()
    scaler.fit(X_train_transformed, y_train)
    X_train_scaled = scaler.transform(X_train_transformed)
    X_test_scaled = scaler.transform(X_test_transformed)

    return pd.DataFrame(X_train_scaled, columns=X.columns), pd.DataFrame(X_test_scaled,
                                                                         columns=X.columns), y_train, y_test


def get_corr_support(selected_features):
    return [True if feature in selected_features else False for feature in feature_name]


def cor_selector(X, y, num_feats):
    pearson_cor = pd.concat([X, y], axis=1).corr()["great_customer_class"].to_dict()
    del pearson_cor["great_customer_class"]

    sorted_features_with_values = sorted(pearson_cor.items(), key=lambda x: x[1], reverse=True)[:num_feats]
    selected_features = [sor[0] for sor in sorted_features_with_values]
    return get_corr_support(selected_features), selected_features


def chi_squared_selector(X, y, num_feats):
    chi2_features = SelectKBest(chi2, k=num_feats)
    chi2_features.fit(X, y)
    chi2_features.transform(X)
    selected_features = list(chi2_features.get_feature_names_out())
    return get_corr_support(selected_features), selected_features


def rfe_selector(X, y, num_feats):
    # Your code goes here (Multiple lines)

    estimator = LogisticRegression()
    selector = RFE(estimator, n_features_to_select=num_feats, step=10, verbose=True)
    selector = selector.fit(X, y)
    # Your code ends here
    return selector.support_, list(selector.get_feature_names_out())


def embedded_log_reg_selector(X, y, num_feats):
    # Your code goes here (Multiple lines)
    estimator = LogisticRegression()

    model = SelectFromModel(estimator, max_features=num_feats)
    model.fit(X, y)

    # Your code ends here
    return get_corr_support(model.get_feature_names_out()), list(model.get_feature_names_out())


def embedded_rf_selector(X, y, num_feats):
    estimator = RandomForestClassifier()

    model = SelectFromModel(estimator, max_features=num_feats)
    model.fit(X, y)

    # Your code ends here
    return get_corr_support(model.get_feature_names_out()), list(model.get_feature_names_out())


def embedded_lgbm_selector(X, y, num_feats):
    estimator = LGBMClassifier()

    model = SelectFromModel(estimator, max_features=num_feats)
    model.fit(X, y)

    # Your code ends here
    return get_corr_support(model.get_feature_names_out()), list(model.get_feature_names_out())


def autoFeatureSelector(X, y, num_feats, methods=[]):
    # Parameters
    # data - dataset to be analyzed (csv file)
    # methods - various feature selection methods we outlined before, use them all here (list)

    # preprocessing
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

    best_features = feature_selection_df.head(num_feats)["Feature"].to_list()
    return X_train_scaled[best_features], X_test_scaled[best_features], best_features


def build_and_predict(X_train_selected, X_test_selected, y_train, y_test):
    algorithms = [LogisticRegression(), KNeighborsClassifier(n_neighbors=3, n_jobs=-1),
                  RandomForestClassifier(n_estimators=1000, max_depth=48, n_jobs=-1), SVC(),
                  GaussianNB()]

    result = {}
    fitted_models = {}
    cm = {}
    precision = {}
    recall = {}
    #  X_train_selected, X_test_selected
    for model in tqdm(algorithms):
        model.fit(X_train_selected, y_train)
        pred = model.predict(X_test_selected)

        result[type(model).__name__] = pred
        fitted_models[type(model).__name__] = model
        cm[type(model).__name__] = confusion_matrix(y_test, pred)

        precision[type(model).__name__] = precision_score(y_test, pred, average='binary')
        recall[type(model).__name__] = recall_score(y_test, pred, average='binary')

    return fitted_models, result, cm, precision, recall


def ensemble(X_train_selected, X_test_selected, y_train, y_test):
    estimators = [('lr', LogisticRegression()), ('knn', KNeighborsClassifier(n_neighbors=3, n_jobs=-1)),
                  ('rf', RandomForestClassifier(n_estimators=1000, max_depth=48, n_jobs=-1)),
                  ('svc', SVC(probability=True)),
                  ('naive', GaussianNB())]
    eclf2 = VotingClassifier(estimators=estimators, voting='soft')
    eclf2.fit(X_train_selected, y_train)

    pred = eclf2.predict(X_test_selected)

    return confusion_matrix(y_test, pred), precision_score(y_test, pred, average='binary'), recall_score(y_test, pred,
                                                                                                         average='binary')


if __name__ == '__main__':
    X_train_scaled, X_test_scaled, y_train, y_test = get_cleaned_data(
        "https://raw.githubusercontent.com/subashgandyer/datasets/main/great_customers.csv")
    print(X_train_scaled.shape)
    print(X_test_scaled.shape)
    print(y_train.shape)
    print(y_test.shape)

    # global feature_name
    feature_name = X_train_scaled.columns

    X_train_selected, X_test_selected, best_features = autoFeatureSelector(X_train_scaled, y_train, num_feats=20,
                                                                           methods=['pearson', 'chi-square', 'rfe',
                                                                                    'log-reg', 'rf', 'lgbm'])

    print("Selected best features by multiple methods - ")
    pprint.pprint(best_features)

    fitted_models, result, cm, pr, rc = build_and_predict(X_train_selected, X_test_selected, y_train, y_test)

    print("Confusion matrix of each model - ")
    pprint.pprint(cm)
    print("Precision score of each model - ")
    pprint.pprint(pr)
    print("Recall score of each model - ")

    pprint.pprint(rc)

    ensemble_model_cm, epr, erc = ensemble(X_train_selected, X_test_selected, y_train, y_test)

    print("\n Ensemble model confusion matrix - ")
    pprint.pprint(ensemble_model_cm)

    print("Precision score of ensemble model - ")

    pprint.pprint(epr)
    print("Recall score of ensemble model - ")

    pprint.pprint(erc)
