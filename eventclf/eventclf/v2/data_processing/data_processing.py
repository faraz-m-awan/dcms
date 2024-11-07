import re
from ast import literal_eval
from datetime import datetime
from typing import List

import numpy as np
import pandas as pd
from dateutil import parser
from sklearn.impute import SimpleImputer

emotre = re.compile(
    r"(:\w+\:|\<[\/\\]?3|[\(\)\\\D|\*\$][\-\^]?[\:\;\=]|[\:\;\=B8][\-\^]?[3DOPp\@\$\*\\\)\(\/\|])(?=\s|[\!\.\?]|$)"
)

all_feats = [
    "tweet.text",
    "num.user_mentions",
    "num.urls",
    "num.hashtags",
    "num.emoticons",
    "au.followers_count",
    "au.friends_count",
    "ratio_ff",
    "tweet.instagram",
    "tweet.foursquare",
    "tweet.youtube",
    "tweet.facebook",
    "tweet.photos",
    "days_dif",
]

imp = SimpleImputer(missing_values=np.nan, strategy="mean")


def _clean_text(text: str, remove_words: List[str]) -> str:
    """cleans social media post of user hashes and unwanted words

    Parameters
    ----------
    text : str
        text to clean
    remove_words : List[str]
        words to remove

    Returns
    -------
    str
        cleaned string
    """
    cleaned_text = re.sub("\@\S{40}", "@UserHandle ", text)  # noqa W605
    for word in remove_words:
        cleaned_text = cleaned_text.replace(word, "event")
    return cleaned_text


def clean_data(
    df: pd.DataFrame,
    dates: List[tuple[datetime]],
    remove_words: List[str],
    features: List[str] = all_feats,
) -> pd.DataFrame:
    """cleans and imputes data from pulsar

    Parameters
    ----------
    df : pd.DataFrame
        dataframe to clean
    dates : List[tuple[datetime]]
        date range for event
    remove_words : List[str]
        list of words to remove from tweet to prevent over fitting
    features : List[str], optional
        features in data to keep, by default all_feats

    Returns
    -------
    pd.DataFrame
        cleaned data
    """
    # filtering spam

    df["tweet.videos"] = df["entities.media"].apply(
        lambda x: 1 if "video" in str(x) else 0
    )
    df["tweet.photos"] = df["entities.media"].apply(
        lambda x: 1 if "photo" in str(x) else 0
    )
    df["tweet.instagram"] = df["entities.urls"].apply(
        lambda x: 1 if "instagram" in str(x.encode("utf-8")) else 0
    )

    df["tweet.youtube"] = df["entities.urls"].apply(
        lambda x: 1 if "youtube" in str(x.encode("utf-8")) else 0
    )
    df["tweet.foursquare"] = df["text"].apply(lambda x: 1 if "checked" in x else 0)
    df["tweet.facebook"] = df["entities.urls"].apply(
        lambda x: 1 if "facebook.com" in str(x.encode("utf-8")) else 0
    )
    df["tweet.snapchat"] = df["text"].apply(lambda x: 1 if "snap" in x else 0)

    df["num.urls"] = df["entities.urls"].apply(lambda x: len(literal_eval(x)))
    df["num.hashtags"] = df["entities.hashtags"].apply(lambda x: len(literal_eval(x)))
    df["ratio_ff"] = (df["au.followers_count"] + df["au.friends_count"]) * 0.5
    df["num.emoticons"] = df["text"].apply(lambda x: len(emotre.findall(x)))
    df = df.rename(columns={"text": "tweet.text"})

    # convert days to number

    df["days_dif"] = df["created_at_tr"].apply(
        lambda x: min(
            abs((parser.parse(x) - min(dates)).days),
            abs((parser.parse(x) - max(dates)).days),
        )
    )
    df = df[features]
    df["tweet.text"] = df["tweet.text"].apply(lambda x: _clean_text(x, remove_words))
    imp_trained = imp.fit(df.drop(columns="tweet.text"))
    imp_out = imp_trained.transform(df.drop(columns="tweet.text"))
    imp_out = pd.DataFrame(imp_out, columns=features[1:])
    return pd.concat([df["tweet.text"], imp_out], axis=1)
