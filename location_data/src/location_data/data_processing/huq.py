from pathlib import PosixPath
from typing import Dict, Optional, Tuple, Union

import geopandas as gpd
import pandas as pd


def process_huq_annual_estimates(
    huq_annual_estimates_path: Union[str, PosixPath], years: Optional[Tuple[int]] = None
) -> pd.DataFrame:
    """Reads annual huq data estimates, cleans data and optionally filters by target years

    Parameters
    ----------
    huq_annual_estimates_path : Union[str,PosixPath]
        path to annual huq estimates
    years : Optional[Tuple[int]]
        years to filter by

    Returns
    -------
    pd.DataFrame
        dataframe of estimated huq annual estimates
    """
    huq_df = pd.read_csv(huq_annual_estimates_path, parse_dates=["year"])
    huq_df.rename(columns={"name": "site_name"}, inplace=True)
    huq_df["year"] = huq_df["year"].dt.year
    if years:
        huq_df = huq_df[huq_df["year"].between(years[0], years[1])]
    huq_df.dropna(
        subset=[
            "estimated_actual_footfall_sum",
            "estimated_actual_footfall_interpolated_sum",
        ],
        inplace=True,
    )
    return huq_df


def process_huq_daily_estimates(
    huq_daily_estimates_path: Union[str, PosixPath],
    years: Optional[Tuple[int]] = None,
    dates_of_interest: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """Gets daily estimates data and optionally filters based on either years or selected day

    Parameters
    ----------
    huq_daily_estimates_path : Union[str,PosixPath]
        Path to huq daily estimates
    years : Optional[Tuple[int]]
        years to filter by
    dates_of_interest : Optional[Dict[str,str]]
        days of interest to filter by

    Returns
    -------
    pd.DataFrame
        Dataframe of daily estimates filtered by specific years/dates
    """
    huq_daily_df = pd.read_csv(huq_daily_estimates_path, parse_dates=["datestamp"])
    huq_daily_df = huq_daily_df.rename(columns={"polygon_id": "site_name"})
    huq_daily_df["site_name"] = huq_daily_df["site_name"].apply(
        lambda x: x.split("/")[1]
    )
    huq_daily_df["year"] = huq_daily_df["datestamp"].dt.year
    if years:
        huq_daily_df = huq_daily_df[huq_daily_df["year"].between(years[0], years[1])]
    if dates_of_interest:
        huq_daily_df = huq_daily_df[
            huq_daily_df.apply(
                lambda row: row["site_name"] in dates_of_interest
                and row["datestamp"]
                == pd.Timestamp(dates_of_interest[row["site_name"]]),
                axis=1,
            )
        ]
        huq_daily_df["year"] = huq_daily_df["datestamp"].dt.year
    return huq_daily_df

    # Getting Annual Visitors Days


def find_huq_annual_visitors(huq_daily_df: pd.DataFrame) -> pd.DataFrame:
    """Finds a dataframe of number of visitor days per year

    Parameters
    ----------
    huq_daily_df : pd.DataFrame
        dataframe of daily estimates of huq data

    Returns
    -------
    pd.DataFrame
        data frame of counts of visitor days by site name
    """
    huq_vd_annual_df = huq_daily_df.copy()
    huq_vd_annual_df = huq_vd_annual_df.sort_values(by=["site_name", "datestamp"])
    huq_vd_annual_df = (
        huq_vd_annual_df.groupby(["site_name", "year"])["datestamp"]
        .count()
        .reset_index(name="visitors_day")
    )
    return huq_vd_annual_df

    # Getting Unique Users


def find_huq_unique_users(huq_daily_df: pd.DataFrame) -> pd.DataFrame:
    """Gets number of unique users for each test site based
    on huq_daily_data_df

    Parameters
    ----------
    huq_daily_df : pd.DataFrame
        dataframe of daily estimates of huq data

    Returns
    -------
    pd.DataFrame
        Dataframe of number of unique users per site
    """
    huq_unique_users_annual_df = huq_daily_df[
        ["site_name", "year", "device_iid"]
    ].copy()
    huq_unique_users_annual_df = (
        huq_unique_users_annual_df.groupby(["site_name", "year"])["device_iid"]
        .apply(lambda x: x.nunique())
        .reset_index(name="unique_users")
    )
    return huq_unique_users_annual_df


def process_huq_bayesian(
    huq_bayesian_data_path: Union[str, PosixPath],
    years: Optional[Tuple[int]] = None,
    dates_of_interest: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """Reads and aggregates huq bayesian statistics by year, Optional filtering for either specific years or event dates

    Parameters
    ----------
    huq_bayesian_data_path : Union[str,PosixPath]
        path to huq bayesian data
    years : Optional[Tuple[int]]
        years to filter by
    dates_of_interest : Optional[Dict[str,str]]
        days of interest to filter by

    Returns
    -------
    pd.DataFrame
        dataframe of huq bayesian estimates
    """

    # Huq Bayesian Estimates
    huq_bys_df = pd.read_csv(huq_bayesian_data_path, parse_dates=["datestamp"])
    huq_bys_df = huq_bys_df.rename(
        columns={
            "polygon_id": "site_name",
            "visitation_approximation": "bayesian_visitation_approximation",
        }
    )
    huq_bys_df["site_name"] = huq_bys_df["site_name"].apply(lambda x: x.split("/")[1])
    huq_bys_df["year"] = huq_bys_df["datestamp"].dt.year
    if years:
        huq_bys_df = huq_bys_df[huq_bys_df["year"].between(years[0], years[1])]
    if dates_of_interest:
        huq_bys_df = huq_bys_df[
            huq_bys_df.apply(
                lambda row: row["site_name"] in dates_of_interest
                and row["datestamp"]
                == pd.Timestamp(dates_of_interest[row["site_name"]]),
                axis=1,
            )
        ]
    huq_bys_df = (
        huq_bys_df.groupby(["site_name", "year"])["bayesian_visitation_approximation"]
        .sum()
        .reset_index()
    )
    return huq_bys_df


def get_area(site_shapes_path: Union[str, PosixPath]) -> gpd.DataFrame:
    """Reads in shape files of events and gets area and population density

    Parameters
    ----------
    site_shapes_path : Union[str,PosixPath]
        path to .shp file

    Returns
    -------
    gpd.DataFrame
        geoDataframe of site names, densitys and areas
    """
    shape_df = gpd.read_file(site_shapes_path)
    shape_df = shape_df.rename(columns={"name": "site_name", "density": "pop_density"})
    shape_df = shape_df.to_crs("EPSG:27700")
    shape_df["site_area"] = (shape_df["geometry"].area) / 1000
    return shape_df


def process_one_day_event_data(
    one_day_events_path: Union[str, PosixPath],
    dates_of_interest: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """Processes Huq attendence data for one day events, optionally filters by selected dates

    Parameters
    ----------
    one_day_events_path : Union[str,PosixPath]
        path to one day events huq data
    dates_of_interest : Optional[Dict[str,str]], optional
        days of interest to filter by, by default None

    Returns
    -------
    pd.DataFrame
        dataframe of huq event attendence for selected dates
    """
    one_day_events_df = pd.read_csv(one_day_events_path, parse_dates=["datestamp"])
    one_day_events_df = one_day_events_df.rename(columns={"polygon_id": "site_name"})
    one_day_events_df["site_name"] = one_day_events_df["site_name"].transform(
        lambda x: x.split("/")[1]
    )
    one_day_events_df = one_day_events_df.rename(
        columns={
            "estimated_actual_footfall": "estimated_actual_footfall_sum",
            "estimated_actual_footfall_interpolated": "estimated_actual_footfall_interpolated_sum",
        }
    )

    one_day_events_df = one_day_events_df[
        one_day_events_df.apply(
            lambda row: row["site_name"] in dates_of_interest
            and row["datestamp"] == pd.Timestamp(dates_of_interest[row["site_name"]]),
            axis=1,
        )
    ]
    one_day_events_df["year"] = one_day_events_df["datestamp"].dt.year
    return one_day_events_df


def clean_huq_data(
    huq_estimates_path: Union[str, PosixPath],
    huq_daily_estimates_path: Union[str, PosixPath],
    huq_bayesian_data_path: Union[str, PosixPath],
    sites_shape_path: Union[str, PosixPath],
    data_freq: str = "Annual",
    years: Optional[Tuple[int]] = None,
    dates_of_interest: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """Processes all Huq data

    Parameters
    ----------
    huq_estimates_path : Union[str,PosixPath]
        path for huq estimate counts
    huq_daily_estimates_path : Union[str,PosixPath]
        path for daily huq estimate counts
    huq_bayesian_data_path : Union[str,PosixPath]
        path for bayesian huq estimate counts
    sites_shape_path : Union[str,PosixPath]
        path to shape files for event locations
    data_freq : str, optional
        whether the data is collected annually or monthly, by default "Annual"
    years : Optional[Tuple[int]], optional
        years to filter by, by default None
    dates_of_interest : Optional[Dict[str,str]], optional
        dates of interest to filter by, by default None

    Returns
    -------
    pd.DataFrame
        dataframe of clean huq data
    """
    if data_freq == "Annual":
        huq_df = process_huq_annual_estimates(huq_daily_estimates_path, years)
        huq_df["data_freq"] = "annual"
    else:
        huq_df = process_one_day_event_data(huq_estimates_path, dates_of_interest)
        huq_df["data_freq"] = "monthly"

    huq_bys_df = process_huq_bayesian(
        huq_bayesian_data_path, years=years, dates_of_interest=dates_of_interest
    )
    huq_daily_df = process_huq_daily_estimates(
        huq_daily_estimates_path, years=years, dates_of_interest=dates_of_interest
    )
    huq_vd_annual_df = find_huq_annual_visitors(huq_daily_df)
    huq_unique_users_annual_df = find_huq_unique_users(huq_daily_df)
    shape_df = get_area(sites_shape_path)
    huq_df = huq_df.merge(
        huq_unique_users_annual_df, on=["site_name", "year"], how="left"
    )  # Adding Unique Users Column
    huq_df = huq_df.merge(
        huq_vd_annual_df[["site_name", "year", "visitors_day"]],
        on=["site_name", "year"],
        how="left",
    )  # Adding the detected visitors days
    huq_df = huq_df.merge(
        huq_bys_df[["site_name", "year", "bayesian_visitation_approximation"]],
        on=["site_name", "year"],
        how="left",
    )  # Adding annual Bayesian estimates

    huq_df = huq_df.merge(
        shape_df[["site_name", "site_area", "pop_density"]],
        on=["site_name"],
        how="left",
    )  # Adding site area
    huq_df["pop_density"] = huq_df["pop_density"].fillna(0)

    if huq_df.duplicated(subset=["site_name", "year"]).any() is False:
        print("duplicated sites found dropping duplicates")
        huq_df = huq_df.drop_duplicates(subset=["site_name", "year"])
    return huq_df
