import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
from sklearn.impute import SimpleImputer
import os

# classifiers
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from datetime import datetime

# nltk
import nltk

from eventclf.feature_extraction import adapt_features, get_group_of_features, \
    get_train_test_features

nltk.download('stopwords')
from nltk.corpus import stopwords

stop_words = stopwords.words("english")

hash_tags = ['creamfields2016', 'creamfields', 'creamfield'] + ['vfestival',
                                                                'V21st',
                                                                'vfest2016',
                                                                'vfest',
                                                                'v festival',
                                                                'v fest']

events = ['Creamfields', 'vfestival']
events_date = [[datetime(2016, 8, 25), datetime(2016, 8, 26),
                datetime(2016, 8, 27), datetime(2016, 8, 28)]
    , [datetime(2016, 8, 20), datetime(2016, 8, 21)]]

# stop_words = []

random_state = 0
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

# types of features

textual_feats = ['tweet.text']
social_feats = ['au.followers_count', 'au.friends_count', 'ratio_ff']
temporal_feats = ['days_dif']
media_feats = ['tweet.instagram', 'tweet.foursquare', 'tweet.youtube',
               'tweet.facebook', 'tweet.photos']
all_feats = ['tweet.text', 'num.user_mentions', 'num.urls', 'num.hashtags',
             'num.emoticons'
    , 'au.followers_count', 'au.friends_count', 'ratio_ff', 'tweet.instagram'
    , 'tweet.foursquare', 'tweet.youtube', 'tweet.facebook', 'tweet.photos',
             'days_dif']

# Create our imputer to replace missing values with the mean e.g.
imp = SimpleImputer(missing_values=np.nan, strategy='mean')


# import pandas as pd
def classifaction_report_df(report):
    report_data = []
    lines = report.split('\n')
    for line in lines[2:4]:  # -3]:
        row = {}
        row_data = line.split('      ')
        row['class'] = row_data[1]
        row['precision'] = float(row_data[2])
        row['recall'] = float(row_data[3])
        row['f1_score'] = float(row_data[4])
        row['support'] = float(row_data[5])
        report_data.append(row)
    dataframe = pd.DataFrame.from_dict(report_data)
    dataframe[dataframe['class'].str.contains('1') == True]
    return dataframe.to_dict()


def get_dic_models(allalg=True):
    dic_models = {}
    if allalg:
        dic_models['LogisticRegression_Comb_w2vGN_bow'] = LogisticRegression(
            random_state=random_state)
        dic_models['svm.LinearSVC_Comb_w2vGN_bow'] = svm.LinearSVC(
            random_state=random_state)
        dic_models[
            'RandomForestClassifier_Comb_w2vGN_bow'] = RandomForestClassifier(
            random_state=random_state)
        dic_models[
            'GradientBoostingClassifier_Comb_w2vGN_bow'] = GradientBoostingClassifier(
            random_state=random_state, n_estimators=300)
    else:
        dic_models[
            'GradientBoostingClassifier_Comb_w2vGN_bow_100_3_None'] = GradientBoostingClassifier(
            random_state=random_state, n_estimators=100, max_depth=3)
    return dic_models


def save_log(X_test, y_test, y_prend, dic_key, event, task, group, k):
    location = os.path.dirname(os.path.realpath(__file__))

    debbug_result = X_test.copy()

    debbug_result_right = debbug_result[y_test == y_prend]
    file_log = dic_key + '_' + event + '_task(' + str(
        task) + ')_' + group + '_right_' + str(k) + '.csv'
    file_log = os.path.join(location, 'results', file_log)
    with open(file_log, 'a') as f:
        debbug_result_right.to_csv(f, sep='\t', encoding='utf-8')

    debbug_result_wrong = debbug_result[y_test != y_prend]
    file_log = dic_key + '_' + event + '_' + str(
        task) + '_' + group + '_wrong' + str(k) + '.csv'
    file_log = os.path.join(location, 'results', file_log)
    with open(file_log, 'a') as f:
        debbug_result_wrong.to_csv(file_log, sep='\t', encoding='utf-8')


def get_models_evaluation(model, idx_event, task, group_feat, vect_textual,
                          vect_w2v):
    event = events[idx_event]

    location = os.path.dirname(os.path.realpath(__file__))
    train_event_file = event + '_task%s_sample_ground_truth' % (
        task) + '.csv'
    train_event_file = os.path.join(location, 'data', train_event_file)

    with open(train_event_file, 'rb') as f:
        tw = pd.read_csv(f, sep='\t', encoding='utf-8')
    tw = adapt_features(tw, events_date[idx_event])
    tw = tw.reset_index(drop=True)

    # features
    features = get_group_of_features(group_feat)

    X = tw[features]
    y = tw['label_num']

    tuple_modeltask = ()
    cv_accuracy = []
    cv_dim = []
    cv_precision = []
    cv_recall = []
    cv_fscore = []

    for k, (train, test) in enumerate(kf.split(X, y)):
        X_train, X_test, vect = get_train_test_features(X, train, test,
                                                        features, vect_textual,
                                                        vect_w2v)
        y_train = y.iloc[train]
        y_test = y.iloc[test]

        # train model
        model.fit(X_train, y_train)

        # predic values
        y_prend = model.predict(X_test)
        cv_accuracy.append(metrics.accuracy_score(y_test, y_prend))

        # precision & recall
        dic_report = classifaction_report_df(
            metrics.classification_report(y_test, y_prend))
        cv_precision.append(dic_report['precision'][1])
        cv_recall.append(dic_report['recall'][1])
        cv_fscore.append(dic_report['f1_score'][1])
        cv_dim.append(X_train.shape[1])

    return np.mean(cv_accuracy), tuple_modeltask, np.mean(
        cv_precision), np.mean(cv_recall), np.mean(cv_fscore), np.mean(cv_dim)
