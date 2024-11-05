# auxiliar functions
from ast import literal_eval
import re
import numpy as np
from sklearn.impute import SimpleImputer
from eventclf.preprocessing import rich_analyzer_w2v

emotre = re.compile(
    r'(:\w+\:|\<[\/\\]?3|[\(\)\\\D|\*\$][\-\^]?[\:\;\=]|[\:\;\=B8][\-\^]?[3DOPp\@\$\*\\\)\(\/\|])(?=\s|[\!\.\?]|$)')

# types of features
textual_feats = ['tweet.text']
social_feats = ['au.followers_count', 'au.friends_count', 'ratio_ff']
temporal_feats = ['days_dif']
media_feats = ['tweet.instagram', 'tweet.foursquare', 'tweet.youtube',
               'tweet.facebook', 'tweet.photos']
all_feats = ['tweet.text', 'num.user_mentions', 'num.urls', 'num.hashtags',
             'num.emoticons', 'au.followers_count', 'au.friends_count',
             'ratio_ff', 'tweet.instagram', 'tweet.foursquare', 'tweet.youtube',
             'tweet.facebook', 'tweet.photos',
             'days_dif']

# Create our imputer to replace missing values with the mean e.g.
imp = SimpleImputer(missing_values=np.nan, strategy='mean')


def get_train_test_features(X, train, test, features, vect_textual, vect_w2v):
    X_train = X.iloc[train]
    X_test = X.iloc[test]

    # textual and other features
    if 'tweet.text' in features and len(features) > 1:
        X_train_not_text = np.matrix(
            X_train[filter(lambda x: x != 'tweet.text', features)])

        X_train_text_tex = vect_textual.fit_transform(X_train['tweet.text'])
        X_train_text_tex = X_train_text_tex.toarray()

        X_train_text_w2v = map(lambda x: rich_analyzer_w2v(x),
                               X_train['tweet.text'])
        X_train_text_w2v = vect_w2v.transform(X_train_text_w2v)

        X_train = np.concatenate(
            (X_train_text_w2v, X_train_text_tex, X_train_not_text), axis=1)

        X_test_not_text = np.matrix(
            X_test[filter(lambda x: x != 'tweet.text', features)])

        X_test_text_w2v = map(lambda x: rich_analyzer_w2v(x),
                              X_test['tweet.text'])
        X_test_text_w2v = vect_w2v.transform(X_test_text_w2v)

        X_test_text_tex = vect_textual.transform(X_test['tweet.text'])
        X_test_text_tex = X_test_text_tex.toarray()

        X_test = np.concatenate(
            (X_test_text_w2v, X_test_text_tex, X_test_not_text), axis=1)

    # only textual features
    elif 'tweet.text' in features and len(features) == 1:

        X_train_text_tex = vect_textual.fit_transform(X_train['tweet.text'])
        X_train_text_tex = X_train_text_tex.toarray()

        X_train_text_w2v = map(lambda x: rich_analyzer_w2v(x),
                               X_train['tweet.text'])
        X_train_text_w2v = vect_w2v.transform(X_train_text_w2v)

        X_train = np.concatenate((X_train_text_w2v, X_train_text_tex), axis=1)

        X_test_not_text = np.matrix(
            X_test[filter(lambda x: x != 'tweet.text', features)])

        X_test_text_w2v = map(lambda x: rich_analyzer_w2v(x),
                              X_test['tweet.text'])
        X_test_text_w2v = vect_w2v.transform(X_test_text_w2v)

        X_test_text_tex = vect_textual.transform(X_test['tweet.text'])
        X_test_text_tex = X_test_text_tex.toarray()

        X_test = np.concatenate((X_test_text_w2v, X_test_text_tex), axis=1)

    # no textual features
    elif 'tweet.text' not in features and len(features) > 1:
        X_train_not_text = np.matrix(X_train[features])
        X_train = X_train_not_text

        X_test_not_text = np.matrix(
            X_test[filter(lambda x: x != 'tweet.text', features)])
        X_test = X_test_not_text

    # not textual and only one features: basically tempora feature
    elif 'tweet.text' not in features and len(features) == 1:
        X_train_not_text = np.matrix(X_train)
        X_train = X_train_not_text  # .T

        X_test_not_text = np.matrix(X_test)
        X_test = X_test_not_text  # .T

    imp_trained = imp.fit(X_train)  # fit with training set
    X_train = imp_trained.transform(X_train)
    X_test = imp_trained.transform(X_test)
    return X_train, X_test, vect_textual  # , y_train, y_test, vect


def remove_features_list(listA, listB):
    return filter(lambda x: x not in listB, listA)


def has_text_features(group):
    if group in ['all', 'text', '-media', '-social', '-temporal']:
        return True
    else:
        return False


def get_group_of_features(group):
    result = []
    if group == 'all':
        result = list(all_feats)

    elif group == 'text':
        result = list(textual_feats)

    elif group == '-text':
        result = list(remove_features_list(all_feats, textual_feats))

    elif group == 'media':
        result = list(media_feats)

    elif group == '-media':
        result = list(remove_features_list(all_feats, media_feats))

    elif group == 'temporal':
        result = list(temporal_feats)

    elif group == '-temporal':
        result = list(remove_features_list(all_feats, temporal_feats))

    elif group == 'social':
        result = list(social_feats)

    elif group == '-social':
        result = list(remove_features_list(all_feats, social_feats))

    return result


def adapt_features(df, dates):
    # filtering spam
    df = df[df['label'].astype(str).str.contains('xxx') == False]
    df = df[df['label'].astype(str).str.contains('acc') == False]
    df = df[df['label'].astype(str).str.contains('-') == False]

    # convert label to number
    df['label_num'] = df['label'].apply(
        lambda x: int(1) if x == 'yes' else int(0))

    df['tweet.videos'] = df['entities.media'].apply(
        lambda x: 1 if 'video' in str(x) else 0)
    df['tweet.photos'] = df['entities.media'].apply(
        lambda x: 1 if 'photo' in str(x) else 0)
    df['tweet.instagram'] = df['entities.urls'].apply(
        lambda x: 1 if 'instagram' in str(x.encode('utf-8')) else 0)

    df['tweet.youtube'] = df['entities.urls'].apply(
        lambda x: 1 if 'youtube' in str(x.encode('utf-8')) else 0)
    df['tweet.foursquare'] = df['text'].apply(
        lambda x: 1 if 'checked' in x else 0)
    df['tweet.facebook'] = df['entities.urls'].apply(
        lambda x: 1 if 'facebook.com' in str(x.encode('utf-8')) else 0)
    df['tweet.snapchat'] = df['text'].apply(lambda x: 1 if 'snap' in x else 0)

    df['num.user_mentions'] = df['entities.user_mentions'].apply(
        lambda x: str(x.encode('utf-8')).count('id_str'))
    df['num.urls'] = df['entities.urls'].apply(lambda x: len(literal_eval(x)))
    df['num.hashtags'] = df['entities.hashtags'].apply(
        lambda x: len(literal_eval(x)))
    df['ratio_ff'] = (df[u'au.followers_count'] + df[u'au.friends_count']) * 0.5
    df['num.emoticons'] = df['text'].apply(lambda x: len(emotre.findall(x)))
    df = df.rename(columns={'text': 'tweet.text'})

    # convert days to number
    from dateutil import parser
    df['days_dif'] = df['created_at_tr'].apply(
        lambda x: min(abs((parser.parse(x) - min(dates)).days),
                      abs((parser.parse(x) - max(dates)).days)))
    return df
