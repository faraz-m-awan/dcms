from pathlib import PosixPath
from typing import Dict, List, Optional, Union

import pandas as pd

RENAME_SITES = {
    "Nymans": "Nymans Gardens",
    "Baddesley": "Baddesley Clinton",
    "Stowe": "Stowe Gardens and Park",
    "Stowe Gardens & Park": "Stowe Gardens and Park",
    "Monument to the Great Fire of London, The": "The Monument to The Great Fire of London",
    "National Gallery, The": "The National Gallery",
    "National Gallery The": "The National Gallery",
    "Clifford's Tower York": "Clifford's Tower, York",
    "Brunel's SS Great Britain": "SS Great Britain",
    "Kelvingrove Museum": "Kelvingrove Art Gallery and Museum",
    "Kelvingrove Art Gallery & Museum": "Kelvingrove Art Gallery and Museum",
    "Wimpole": "Wimpole Estate",
    "Beamish": "Beamish Museum",
    "Beamish, The Living Museum of the North": "Beamish Museum",
    "Burghley": "Burghley House",
    "Dunster": "Dunster Castle",
    "Hardwick": "Hardwick Hall",
    "Packwood": "Packwood House",
    "Shugborough": "Shugborough Estate",
    "Charlecote": "Charlecote Park",
    "Dyrham": "Park",
    "Polesden": "Polesden Lacey",
    "Oxford Botanic Garden and Arboretum": "Oxford Botanical Gardens and Arboretum",
    "Oxford Botanic Gardens and Arboretum": "Oxford Botanical Gardens and Arboretum",
    "RAF Museum Cosford": "The Royal Air Force Museum Midlands",
    "RAF Museum London": "The Royal Air Force Museum London",
    "The British Museum": "British Museum",
    "Giants Causeway": "Giant's Causeway",
    "Forestry England Westonbirt, The National Arboretum": "Westonbirt, The National Arboretum",
    "Warkworth Castle and Hermitage": "Warkworth Castle",
    "Tower Bridge": "Tower Bridge Exhibition",
    "Museum of Science and Industry": "Science and Industry Museum",
    "Salisbury Cathedral and Magna Carta": "Salisbury Cathedral",
    "Royal Botanic Gardens Kew": "Royal Botanic Gardens, Kew",
    "Royal Academy": "Royal Academy of Arts",
    "Burns Birthplace Museum": "Robert Burns Birthplace Museum",
}

SITES_EXCLUDED_GT = [
    "Longleat",
    "Scottish National Gallery of Modern Art",
    "Scottish National Gallery",
    "Dunfermline Palace",
    "Fountains Abbey Estate",
    "Fountains Abbey and Studley Royal Water Garden",
    "National Railway Museum",
    "Cliveden",
    "Croom",
    "Culzean Castle & Country Park",
    "Dunfermline Abbey",
    "Dunfermline Palace",
    "Knowsley Safari and Knowsley Hall",
    "Dalby Forest",
    "Kenwood",
    "Osborne",
    "Shugborough Estate",
    "Stourhead",
    "Stowe Gardens and Park",
    "Tyntesfield",
    "Wallington",
    "SS Great Britain",
    "Wimpole Estate",
    "National Museum Royal Navy",
    "Ulster Folk & Transport Museum",
]

# Define the replacement dictionary
DEFAULT_ENTRY_REPL_DICT = {
    2019: {"F": 1, "C": 2, "F/C": 3},
    2020: {"F": 1, "C": 2, "F/C": 3},
    2021: {"1": 1, "2": 3, "3": 3, "4": 2, "1/2/3/4": 3, "01-Apr": 2},
    2022: {"1": 1, "2": 3, "3": 3, "4": 3, "5": 2, "1234": 3},
    2023: {
        "1": 1,
        "2": 3,
        "3": 3,
        "4": 3,
        "5": 3,
        "6": 2,
        "Note v": 3,
        "Note ac": 3,
        "Note an": 3,
        "Note y": 3,
        "1 (Note t)": 3,
        "Other": 2,
    },
}


def clean_alva_data(
    alva_data_path: Union[str, PosixPath],
    rename_sites: Optional[Dict[str, str]] = RENAME_SITES,
    excluded_sites: Optional[List[str]] = SITES_EXCLUDED_GT,
    entry_replace_dict: Dict[int, Dict[str, int]] = DEFAULT_ENTRY_REPL_DICT,
) -> pd.DataFrame:
    """Processes and cleans ALVA ground truth data

    Parameters
    ----------
    alva_data_path : Union[str,PosixPath]
        path to alva data
    rename_sites : Optional[Dict[str,str]], optional
        some sites have inconsistent naming across names, dict for names standardisation, by default None
    excluded_sites : Optional[List[str]], optional
        sites to drop from ALVA data, by default None
    entry_replace_dict: Dict[int, Dict[str, int]]
        entry values replacement

    Returns
    -------
    pd.DataFrame
        Dataframe of processed alva data
    """
    gt_df = pd.read_csv(alva_data_path)
    gt_df = gt_df.rename(
        columns={
            "Site": "site_name",
            "Total visits": "gt_total_visits",
            "Charge /free": "entry",
            "In/ outdoor": "location_type",
            "% +/-": "pct_changed",
            "Area": "region",
            "Group": "group",
            "Note": "note",
            "Year": "year",
        }
    )
    # Some of the sites in ALVA data don't have same name (format) as the name of the sites in shape file. So, for consistency renaming the sites in ALVA data.
    if rename_sites:
        gt_df["site_name"] = gt_df["site_name"].replace(rename_sites)
    gt_df["gt_total_visits"] = (
        gt_df["gt_total_visits"].str.replace(",", "").astype(int)
    )  # Visitation numbers in ALVA data are available in thousand seperator format, which makes it an object (string) type column. Reformatting it into Int type
    gt_df["pct_changed"] = gt_df["pct_changed"].str.replace(
        "%", ""
    )  # Percentage change is an object type colum with values like (10.2%). Removing the % sign to convert it into float format
    gt_df["pct_changed"] = gt_df["pct_changed"].str.replace(
        "", "0"
    )  # Percentage change value is not available for some of the years for some sites. Some records have empty string instead of NA. Replacing empty string with '0'
    gt_df["pct_changed"] = pd.to_numeric(
        gt_df["pct_changed"], errors="coerce"
    )  # Converting the object type to float type
    gt_df["pct_changed"] = gt_df["pct_changed"].fillna(
        "0"
    )  # Filling NA values with '0'
    gt_df["pct_changed_calculated"] = (
        gt_df.sort_values(by=["site_name", "year"])
        .groupby(["site_name"])["gt_total_visits"]
        .pct_change()
        * 100
    )

    # replace_dict_2019 = {"F": 1, "C": 2, "F/C": 3}
    # replace_dict_2020 = {"F": 1, "C": 2, "F/C": 3}
    # replace_dict_2021 = {"1": 1, "2": 3, "3": 3, "4": 2, "1/2/3/4": 3, "01-Apr": 2}
    # replace_dict_2022 = {"1": 1, "2": 3, "3": 3, "4": 3, "5": 2, "1234": 3}
    # replace_dict_2023 = {
    #     "1": 1,
    #     "2": 3,
    #     "3": 3,
    #     "4": 3,
    #     "5": 3,
    #     "6": 2,
    #     "Note v": 3,
    #     "Note ac": 3,
    #     "Note an": 3,
    #     "Note y": 3,
    #     "1 (Note t)": 3,
    #     "Other": 2,
    # }

    # gt_df.loc[gt_df["year"] == 2019, "entry"] = gt_df.loc[
    #     gt_df["year"] == 2019, "entry"
    # ].replace(replace_dict_2019)
    # gt_df.loc[gt_df["year"] == 2020, "entry"] = gt_df.loc[
    #     gt_df["year"] == 2020, "entry"
    # ].replace(replace_dict_2020)
    # gt_df.loc[gt_df["year"] == 2021, "entry"] = gt_df.loc[
    #     gt_df["year"] == 2021, "entry"
    # ].replace(replace_dict_2021)
    # gt_df.loc[gt_df["year"] == 2022, "entry"] = gt_df.loc[
    #     gt_df["year"] == 2022, "entry"
    # ].replace(replace_dict_2022)
    # gt_df.loc[gt_df["year"] == 2023, "entry"] = gt_df.loc[
    #     gt_df["year"] == 2023, "entry"
    # ].replace(replace_dict_2023)

    for year, replace_dict in entry_replace_dict.items():
        gt_df.loc[gt_df["year"] == year, "entry"] = gt_df.loc[
            gt_df["year"] == year, "entry"
        ].replace(replace_dict)

    # gt_df = gt_df.drop(columns=["group", "note"])
    columns_to_drop = ["group", "note"]  # Dropping these columns if they exist
    gt_df = gt_df.drop(columns=[col for col in columns_to_drop if col in gt_df.columns])

    # gt_df['location_type'] = gt_df['location_type'].fillna('NA')
    mode_location_type = gt_df.groupby("site_name")["location_type"].agg(
        lambda x: x.mode().iloc[0] if not x.mode().empty else None
    )
    gt_df["location_type"] = gt_df.apply(
        lambda row: mode_location_type[row["site_name"]]
        if pd.isna(row["location_type"])
        else row["location_type"],
        axis=1,
    )
    gt_df["location_type"] = gt_df["location_type"].replace({"Mix": "M"})
    if excluded_sites:
        gt_df = gt_df[~gt_df["site_name"].isin(excluded_sites)]
    return gt_df
