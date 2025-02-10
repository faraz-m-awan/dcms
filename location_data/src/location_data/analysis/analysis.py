from pathlib import PosixPath
from typing import List, Optional, Union

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from shapely.wkt import loads

plt.style.use("ggplot")
plt.rcParams["axes.facecolor"] = "white"
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.color"] = "gray"
plt.rcParams["grid.alpha"] = 0.5
plt.rcParams["grid.linestyle"] = ":"


def plot_demographic_heat_map(test_df: pd.DataFrame, site_name: str):
    """
    Description:
    ------------
    Generates a heatmap visualization to analyze the percentage of visitors
    to a given site based on the Index of Multiple Deprivation (IMD) deciles
    across different years.

    Parameters:
    ----------
    df : pd.DataFrame
        A dataset containing visitor percentage data categorized by
        'year' (rows) and 'LA_decile' (columns).
    site_name : str
        The name of the site for which the heatmap is generated.

    Returns:
    -------
    go.Figure
        Plotly Figure of catchment area for a given site for each year

    Example:
    --------
    >>> plotDemographicHeatMap(visitor_data, "Natural History Museum")

    This will generate a heatmap visualizing the percentage of visitors to
    the 'Natural History Museum' across different deprivation levels and years.

    """
    tdf = test_df[test_df["site_name"] == site_name]
    df = (
        tdf.groupby(["year", "site_name", "LA_decile"])
        .size()
        .reset_index(name="visits")
    )
    df["visits_perc"] = df.groupby(["year", "site_name"])["visits"].transform(
        lambda x: (x / x.sum()) * 100
    )
    plt.figure(figsize=(15, 7))
    pivot_df = df.pivot_table(index="year", columns="LA_decile", values="visits_perc")
    sns.heatmap(
        pivot_df,
        cmap="coolwarm",
        annot=True,
        fmt=".1f",
        cbar_kws={"label": "Percentage"},
    )
    plt.title(
        f"Percentage of Visitors to {site_name} by Index of Multiple Deprivation Decile"
    )
    plt.xlabel("LA Decile")
    plt.ylabel("Year")
    plt.tight_layout()

    return plt


def plot_catchment_map(
    uk_sf: gpd.GeoDataFrame, site_df: gpd.GeoDataFrame, site_name: str, years: List[int]
):
    """
    Description:
    ------------
    Generates a series of catchment area maps for a specified site across multiple years,
    visualizing visitor distribution within geographic regions using a color-coded heatmap.

    Parameters:
    ----------
    uk_sf : gpd.GeoDataFrame
        A GeoDataFrame containing the geographic boundaries of the UK regions.
    site_df : gpd.GeoDataFrame
        A dataset containing visitor percentage data for the specified site,
        including 'site_name', 'geo_code', 'year', and 'pct_total'.
    site_name : str
        The name of the site for which the catchment area maps are generated.
    years : list
        A list of years for which the visitor distribution maps will be created.

    Returns:
    -------
    go.Figure
        Plotly Figure of catchment area for a given site for each year

    Example:
    --------
    >>> plotCatchmentMap(uk_shapefile, visitor_data, "Natural History Museum", [2019, 2020, 2021, 2022, 2023])

    This will generate **catchment area heatmaps** showing the visitor distribution
    for the 'Natural History Museum' across the specified years.

    Notes:
    ------
    - The **log transformation (`log_pct_total`)** helps normalize visitor percentage values
      for better visualization of variations across different regions.
    - The function is designed for **spatial analysis** of visitor trends and geographic coverage.
    """
    tdf = site_df[site_df["site_name"] == site_name]
    tdf = (
        tdf.groupby(["site_name", "year", "geo_code"])["device_iid"]
        .count()
        .reset_index(name="n_visitors")
    )
    tdf["total_visitors"] = tdf.groupby(["site_name", "year"])["n_visitors"].transform(
        lambda x: sum(x)
    )
    tdf["pct_total"] = (tdf["n_visitors"] / tdf["total_visitors"]) * 100
    _, ax = plt.subplots(2, 3, figsize=(20, 15))
    plt.subplots_adjust(hspace=0.9)
    ax = ax.flatten()
    for ind, year in enumerate(years):
        tdf_uk = uk_sf.copy()
        tdf_site = site_df[site_df["year"] == year][
            ["site_name", "geo_code", "pct_total"]
        ]
        tdf_uk = tdf_uk.merge(tdf_site, on="geo_code", how="left")
        tdf_uk["log_pct_total"] = np.log1p(tdf_uk["pct_total"])

        tdf_uk.plot(
            column="log_pct_total",
            cmap="viridis",
            legend=True,
            legend_kwds={"label": "Visitor Percentage (%)"},
            edgecolor="black",
            ax=ax[ind],
        )
        ax[ind].set_title(f"{site_name} Catchment ({year})")

    ax[5].set_visible(False)
    plt.tight_layout()
    return plt


def spacial_demographic_data(
    local_area_shape_path: Union[str, PosixPath],
    demographic_data_path: Union[str, PosixPath],
) -> gpd.GeoDataFrame:
    """Creates geoDataframe of local area shapes combined with deprivation data

    Parameters
    ----------
    local_area_shape_path : Union[str,PosixPath]
        shp file with geographical boundries at Lower Layer Super Output Area (LSOA) level,
    demographic_data_path : Union[str,PosixPath]
        The demographics file contains socioeconomic indices at the LSOA level, including:
        SOA_decile: Socioeconomic rank within smaller, localized areas.
        LA_decile: Socioeconomic rank within broader local authority areas.

    Returns
    -------
    gpd.GeoDataFrame
        combined dataframe of geographical bounds and measures of socioeconimic
        indices
    """
    uk_shp_df = gpd.read_file(
        local_area_shape_path
    )  # Reading UK shape file at LSOA level
    uk_shp_df = uk_shp_df.to_crs("EPSG:4326")  # Chanding the projection to 4326
    demog_info = pd.read_csv(demographic_data_path)  # Reading demographics file
    uk_shp_df = uk_shp_df.merge(
        demog_info[["LSOA", "SOA_decile", "LA_decile"]],
        how="left",
        left_on="geo_code",
        right_on="LSOA",
    )  # Merging demographics with the UK shape file
    return uk_shp_df


def huq_plotting_data(
    huq_daily_estimates_path: Union[str, PosixPath],
    uk_shp_df: gpd.GeoDataFrame,
    years: Optional[List[int]] = None,
    test_sites: Optional[List[str]] = None,
) -> gpd.GeoDataFrame:
    """Prepare huq_daily_estimates data for plotting and merging with demographic shape data

    Parameters
    ----------
    huq_daily_estimates_path : Union[str,PosixPath]
        Path to huq daily estimates
    uk_shp_df : gpd.GeoDataFrame
        combined dataframe of geographical bounds and measures of socioeconomic
        indices
    years : Optiona;List[int]
        List of years to filter by
    test_sites : List[str]
        List of test sites to filter by

    Returns
    -------
    gpd.GeoDataFrame
    combined HUQ daily estimates dataframe with geographic boundaries and socioeconomic data

    """
    test_df = gpd.read_file(huq_daily_estimates_path)
    test_df["datestamp"] = pd.to_datetime(test_df["datestamp"])
    test_df["year"] = test_df["datestamp"].dt.year
    test_df["month"] = test_df["datestamp"].dt.month
    if years:
        test_df = test_df[test_df["datestamp"].dt.year.between(years[0], years[1])]
    test_df = test_df.rename(columns={"polygon_id": "site_name"})
    test_df["site_name"] = test_df["site_name"].apply(lambda x: x.split("/")[1])
    if test_sites:
        test_df = test_df[test_df["site_name"].isin(test_sites)]
    test_df["geometry"] = test_df["home_geog"].apply(lambda x: loads(x))
    test_df = test_df.drop(columns=["home_geog"])
    test_df = gpd.GeoDataFrame(test_df, geometry="geometry", crs="EPSG:4326")
    test_df = gpd.sjoin(
        test_df,
        uk_shp_df[["geo_code", "geometry", "LA_decile"]],
        how="left",
        predicate="intersects",
    )
    test_df = test_df.drop(columns=["index_right"])
    return test_df
