# Attendance Counts from Location Data
## Overview
This package contains code for processing and predicting attendance figures for events and sites using location data from HUQ which is based on mobile phone location data.
## Set up and install
1. Clone dcms_engagement repo from gitlab
```bash
git clone [link to package]
```
2. Install `activity_data` package:
```bash
cd dcms_engagement/
pip install activity_data/
pip install activity_data/requirements.txt
```
## Folder Structure
```
location_data
 ┣ src
 ┃ ┣ location_data
 ┃ ┃ ┣ data_processing
 ┃ ┃ ┃ ┣ __init__.py
 ┃ ┃ ┃ ┣ alva.py
 ┃ ┃ ┃ ┣ data_processing.py
 ┃ ┃ ┃ ┗ huq.py
 ┃ ┃ ┣ models
 ┃ ┃ ┃ ┣ __init__.py
 ┃ ┃ ┃ ┗ location_data_model.py
 ┃ ┃ ┣ plots
 ┃ ┃ ┃ ┣ __init__.py
 ┃ ┃ ┃ ┗ plots.py
 ┃ ┃ ┗ __init__.py
 ┣ README.md
 ┗ pyproject.toml
```
## Data Sources
1. **HUQ Data:** HUQ data provides counts in three main csvs:

    1. **estimated actual footfall**: provides scaled footfall estimation by month for each site
    2. **raw daily visitors**: provides device IDs and home regions of visitors, used for estimating number of unique visitors and information on where they traveled from
    3. **estimated bayesian counts**: provides approximations of footfall based on bayesian estimations
2. **ALVA Data**: this data contains the ground truth counts of the sites used for training models, it also contains other useful variables for modeling such as whether the site was outside or inside, or whether it is free to access.
3. **site shp files**: These files contain outlines of the sites of interest represented as polygons with geographic coordinates, used in estimating area of the site and also population density, these have been extracted from open street map.
## Usage
### Data Processing
#### HUQ data
Processing HUQ data combines the output from the three HUQ csvs, extracts unique visitors and number of visitor days and aggragates to year level if appropriate, filtering for specific dates or years.
```Python
from location_data.data_processing.huq import clean_huq_data

huq_df = clean_huq_data(
    huq_estimates_path ='path/to/huq_estimated_actual_footfall.csv',
    huq_daily_estimates_path = 'path/to/huq_raw_daily_visitors.csv',
    huq_bayesian_data_path = 'path/to/huq_bayesian_estimates.csv',
    sites_shape_path = 'path/to/sites_shape_file.shp',
    data_freq = "Annual"
    years = (2019,2023)
)
```
#### ALVA data (ground truth data)
Key columns in alva data include:
- site_name: Standardized names of touristic sites.
- gt_total_visits: The total number of visits to each site.
- entry: Denotes if the site charges entry (free, chargeable, or mixed).
- location_type: Indicates whether the site is indoor, outdoor, or a mix.
- pct_changed: The percentage change in visits compared to previous years.
- region, year, and note: Additional metadata.
column and site names are standardised and data is cleaned
```python
from location_data.data_processing.alva import clean_alva_data

alva_df = clean_alva_data(alva_data_path="path/to/alva_data.csv")
```
### Full data Processing
1. Before ALVA data and HUQ data is combined, as sites are given different names in different datasets, they are matched based on TFIDF and cosign similarity, as well as allowing for manual matching
```Python
from location_data.data_processing.data_processing import get_matched_sites

matched_df = get_matched_sites(alva_sites,huq_sites)
```
2. The data sets are then combined and selected variables can be one hot encoded and attributes for modelling are selected
```Python
from location_data.data_processing.data_processing import process_data

processed_data = process_data(
    huq_dfs = huq_df,
    match_df = matched_df,
    alva_df = alva_df
    one_hot_columns = [
        "entry",
        "location_type",
        "data_freq",
        "year",
    ],
)
```
### Modelling
The `LocationDataModel` has four base models using the same parameters in experimentation:
- linear_model
- SVM
- random_forest
- xgb_model
These models can be used for training and inference and saved out
```Python
from location_data.models.location_data_model import LocationDataModel

location_data_model = LocationDataModel(model = "linear_model")
location_data_model.fit(X_train,y_train)
y_pred = location_data_model.predict(X_test)
location_data.save("directory/to/save_model")
```

Previosly trained models can be loaded for inference
```Python
from location_data.models.location_data_model import LocationDataModel

location_data_model = LocationDataModel(
    model = "linear_model",
    load_path = "path/to/saved_model.pkl")
```
### Plotting
Plots including attendance by socioeconomic indicators, and catchment can be used to further analyse event/ site attendance
```Python
from location_data.plots.plots import plot_catchment_map, plot_demographic_heat_map,huq_plotting_data,spacial_demographic_data

# get data for local area geographical shapes and socioeconomic data
uk_shp_df = spacial_demographic_data(
    local_area_shape_path = "path/to/local_area_geographical_shapes.shp"
    demographic_data_path = "path/to/socioeconomic_data.csv"
)

# get HUQ data for plotting
huq_data = huq_plotting_data(
    huq_daily_estimates_path = 'path/to/huq_raw_daily_visitors.csv',
    uk_shp_df= uk_shp_df,
    years = (2019,2023),
    test_sites = ['British Museum','Giant\'s Causeway'])

# plot catchment map
plot_catchment_map(
    uk_sf = uk_shp_df,
    site_df = huq_data,
    site_name="British Museum",
    years = [2019,2020]
    )

# Plot attendance by socioeconomic indexes
plot_demographic_heat_map(
    test_df = huq_data,
    site_name = "British Museum")

```
