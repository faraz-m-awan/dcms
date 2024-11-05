from dateutil import parser
from sklearn.impute import SimpleImputer
from ast import literal_eval

emotre = re.compile(
    r'(:\w+\:|\<[\/\\]?3|[\(\)\\\D|\*\$][\-\^]?[\:\;\=]|[\:\;\=B8][\-\^]?[3DOPp\@\$\*\\\)\(\/\|])(?=\s|[\!\.\?]|$)')

all_feats = ['tweet.text', 'num.user_mentions', 'num.urls', 'num.hashtags',
             'num.emoticons', 'au.followers_count', 'au.friends_count',
             'ratio_ff', 'tweet.instagram', 'tweet.foursquare', 'tweet.youtube',
             'tweet.facebook', 'tweet.photos',
             'days_dif']

def adapt_features(df:pd.DataFrame, dates):
    # filtering spam

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

    df['num.urls'] = df['entities.urls'].apply(lambda x: len(literal_eval(x)))
    df['num.hashtags'] = df['entities.hashtags'].apply(
        lambda x: len(literal_eval(x)))
    df['ratio_ff'] = (df[u'au.followers_count'] + df[u'au.friends_count']) * 0.5
    df['num.emoticons'] = df['text'].apply(lambda x: len(emotre.findall(x)))
    df = df.rename(columns={'text': 'tweet.text'})

    # convert days to number

    df['days_dif'] = df['created_at_tr'].apply(
        lambda x: min(abs((parser.parse(x) - min(dates)).days),
                      abs((parser.parse(x) - max(dates)).days)))
    return df