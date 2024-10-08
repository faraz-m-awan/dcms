import re
from typing import Optional

import pandas as pd
from utils import obfuscate_text

COLUMN_MAP = {
    "content": "tweet.text",
    "links": "entities.urls",
    "hashtags": "entities.hashtags",
    "userFollowersCount": "au.followers_count",
    "userFriendsCount": "au.friends_count",
    "publishedAt": "created_at_tr",
}


def clean_data(data: pd.DataFrame, whitelist: Optional[pd.Dataframe]) -> pd.DataFrame:
    if whitelist:
        whitelist["handle"] = whitelist["handle"].apply(lambda x: re.sub(" ", "", x))
        whitelist["hash"] = whitelist["handle"].apply(lambda x: obfuscate_text(x))
        for _, row in whitelist.iterrows():
            data["content"] = data["content"].str.replace(row["hash"], row["handle"])
    data = data.rename(columns=COLUMN_MAP)
    data["entities.media"] = [
        "photo" if len(row["images"]) > 2 else "" for _, row in data.iterrows()
    ]
    data["entities.media"] = [
        f"{row['entities.media']} video"
        if len(row["videos"]) > 2
        else row["entities.media"]
        for _, row in data.iterrows()
    ]
    data["images"] = data["images"].apply(
        lambda x: eval(x)[0]["url"] if len(x) > 2 else ""
    )
    return data
