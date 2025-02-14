from typing import Dict, List, Optional, Union

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

MANUAL_SITE_MATCH = {
    "University of Oxford Botanic Garden": "Oxford Botanical Gardens and Arboretum",
    "Natural History Museum": "Natural History Museum (South Kensington)",
    "RAF Museum": "The Royal Air Force Museum London",
    "Imperial War Museum North": "IWM North",
    "Natural History Museum": "Natural History Museum (South Kensington)",
}
ATTRIBUTES = [
    "site_name",
    "year",
    "estimated_actual_footfall_sum",
    "estimated_actual_footfall_interpolated_sum",
    "bayesian_visitation_approximation",
    "entry",
    "location_type",
    "site_area",
    "pop_density",
    "unique_users",
    "visitors_day",
    "data_freq",
    "gt_total_visits",
]


def checkMultipleMatches(row: pd.Series) -> bool:
    """
    Description:
        This function checks if any site in a shape file has multiple matches in ALVA sites.
        Name of some sites in ALVA data changed in different years. This function captures
        that multiple matches

    Parameters:
        row (pd.Series): Function takes pandas' series (row) of the data.

    Retruns:
        Returns 'True' or 'False' depending upon the availability of multiple matches for a site

    """

    all_matches = row[
        "all_matched"
    ]  # Fetching all the sites that were labeled as potential matches sorted with respect to match score.
    all_matches.pop(
        0
    )  # Removing the first match as it's the best match. And we want to see if there's any other potential match other than the best match.
    best_match_words = row["best_match"].split(
        " "
    )  # Tokenizing the best matched site name
    found = False
    for (
        word
    ) in best_match_words:  # Iterating through each word in best matched site name
        if word not in [
            "Castle",
            "House",
            "Museum",
            "Garden",
            "Park",
            "Estate",
            "Hall",
            "Abbey",
        ]:  # Sites could be labeled as potential match based on some common words such as Castle, House, Museum etc. Checking if the matched word is not common word and it's present in the name of one or more sites in the all matched sites, it returns 'True'
            for site in all_matches:
                if word in site:
                    found = True
                    return found

    return found


def get_matched_sites(
    gt_sites: List[str],
    target_sites: List[str],
    manual_site_matches: Optional[Dict[str, str]] = MANUAL_SITE_MATCH,
) -> pd.DataFrame:
    """Matches names of events/sites bases on cosign distance and tfidf,
    for matching site names in ground truth/ ALVA data and huq data

    Parameters
    ----------
    gt_sites : List[str]
        list of site names from ground truth data
    target_sites : List[str]
        list of site names from HUQ data
    manual_site_matches : Optional[Dict[str,str]], optional
        dictionary of names to match manually, by default MANUAL_SITE_MATCH

    Returns
    -------
    pd.DataFrame
        dataframe of site names of huq data in one column and the best matching name in the ground truth data
    """
    all_sites = list(gt_sites) + list(target_sites)
    vectorizer = TfidfVectorizer().fit(all_sites)
    tfidf_matrix = vectorizer.transform(all_sites)
    tfidf_gt_sites = tfidf_matrix[: len(gt_sites)]
    tfidf_target_sites = tfidf_matrix[len(gt_sites) :]
    matching_sites = []
    for i, site in enumerate(target_sites):
        cosine_similarities = cosine_similarity(
            tfidf_target_sites[i], tfidf_gt_sites
        ).flatten()
        top_5_indices = cosine_similarities.argsort()[-10:][::-1]
        matched_sites = [gt_sites[idx] for idx in top_5_indices]
        scores = [cosine_similarities[idx] for idx in top_5_indices]
        best_match = matched_sites[0]
        best_score = scores[0]
        matching_sites.append([site, best_match, best_score, matched_sites, scores])
    match_df = pd.DataFrame(
        matching_sites,
        columns=[
            "sf_site",
            "best_match",
            "best_score",
            "all_matched",
            "matching_score",
        ],
    )
    match_df["multiple_matches"] = match_df.apply(checkMultipleMatches, axis=1)
    if manual_site_matches:
        match_df["best_match"] = (
            match_df["sf_site"]
            .map(manual_site_matches)
            .combine_first(match_df["best_match"])
        )
    # match_df = match_df[["sf_site", "best_match"]]
    return match_df


def process_data(
    huq_dfs: Union[pd.DataFrame, List[pd.DataFrame]],
    match_df: pd.DataFrame,
    gt_df: pd.DataFrame,
    one_hot_columns: Optional[List[str]] = [
        "entry",
        "location_type",
        "data_freq",
        "year",
    ],
    attributes: Optional[List[str]] = ATTRIBUTES,
) -> pd.DataFrame:
    """Combines and processes all data ready to be placed in the model:
    steps
    1. If multiple huq dfs concatinates them
    2. merges match df to allow for matching to other dfs
    3. merges other match dfs
    4. optional: selects attributes for model
    5. optional: one hot encodes variables

    Parameters
    ----------
    huq_dfs : Union[pd.DataFrame,List[pd.DataFrame]]
        either one dataframe, if a list of dataframes
        then concantenates them
    match_df : pd.DataFrame
        dataframe that contains name matches for events/ sites
    gt_df : pd.DataFrame
        ground true data frame
    one_hot_columns : Optional[List[str]], optional
        columns to one hot encode, by default ['entry', 'location_type', 'data_freq','year']
    attributes : Optional[List[str]], optional
        attributes to select for model, by default ATTRIBUTES

    Returns
    -------
    pd.DataFrame
        full processed dataframe for model.
    """

    # merge all huq dfs
    if isinstance(huq_dfs, list):
        huq_df = pd.concat(huq_dfs).reset_index(drop=True)
    else:
        huq_df = huq_dfs

    # merge match data
    huq_df = (
        huq_df.merge(match_df, left_on="site_name", right_on="sf_site", how="left")
        .dropna(subset=["best_match"])
        .drop(columns=["sf_site"])
        .reset_index(drop=True)
    )  # Adding the best match name from ALVA.

    # merge ground truth data
    huq_df = (
        huq_df.merge(
            gt_df,
            left_on=["best_match", "year"],
            right_on=["site_name", "year"],
            how="left",
        )
        .dropna(subset=["gt_total_visits"])
        .reset_index(drop=True)
    )  # Adding ground truth visitation numbers from ground truth data and dropping all those records where ground truth data is not available.
    huq_df = huq_df.drop(
        columns=["site_name_y"]
    )  # Dropping the 'y' prefix column added by merge operation because of the same column name on both merging dfs.
    huq_df = huq_df.rename(columns={"site_name_x": "site_name"})
    huq_df = huq_df.dropna(
        subset=[
            "gt_total_visits",
            "estimated_actual_footfall_interpolated_sum",
            "estimated_actual_footfall_sum",
        ]
    )  # Final validation dropna operation to ensure we have estimated and ground truth data available for all the records

    # get only attributes
    if attributes:
        huq_df = huq_df[attributes]

    # one hot encode
    if one_hot_columns:
        huq_df = pd.get_dummies(huq_df, columns=one_hot_columns)

    return huq_df
