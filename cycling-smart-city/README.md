# Cycling Smart City KrakHack: Data Sourcing & Exploration

**Cycling Smart City AI KrakHack** repository is designed to give you a head start on the challenge.

This code provides a **sourcing module** (`sourcing_data.py`) and an **exploration notebook** (`cycling_data_exploration.ipynb`). 

## What's Inside?


**Data Ingestion**: Classes to load local `GeoJSON` and `Shapefiles` (ZTPK, MSIP, GUGIK), fetch live OpenStreetMap (OSM) data and Bicycle Counter Data

**Visual Glance**: Example visualizations to help understand the spatial distribution of existing infrastructure, noise pollution, green areas etc. 



---

## Data Sources

| Name | URL | Code Reference (from `sourcing_data`)| Description |
| --- | --- | --- | --- |
| **Bicycle Racks** | [ZTPK Hub](https://ztpk-gmk-2.hub.arcgis.com/) | `LocalGeoData.bike_racks_df` | Point data for bicycle stands in Kraków.|
| **Point Infrastructure** | [ZTPK Hub](https://ztpk-gmk-2.hub.arcgis.com/) | `LocalGeoData.bike_infrastructure_df` | Miscellaneous bicycle-related point assets (e.g., service stations).|
| **Cycling Paths** | [ZTPK Hub](https://ztpk-gmk-2.hub.arcgis.com/) | `LocalGeoData.cycling_paths_df` | Existing linear bicycle infrastructure network.|
| **Noise Maps** | [MSIP Kraków](https://msip.krakow.pl/) | `LocalGeoData.noise_map_df` | Consolidated layers of road, rail, and tram noise levels.|
| **Greenery (GUGIK)** | [OpenData Geoportal](https://opendata.geoportal.gov.pl) | `LocalGeoData.greenery_df` | Consolidated BDOT10k layers: Forests (**PTLZ**), Grasslands (**PTTR**), and Managed Areas (**PTUT**).|
| **Buildings** |-|`OpenStreetMap.buildings_df`| Real-time fetch of building footprints via `osmnx`.|
| **Streets** |-|`OpenStreetMap.streets_df`| Real-time fetch of the street graph via `osmnx`.|
| **Bicycle Counters** | [Liczniki Rowerowe](https://liczniki-rowerowe.pl/city/krakow) | `BicycleCounterData.counters_df` | Live data from city-wide bicycle traffic counters.|

---

## Getting Started

**Configure Paths**: Update the `[DATA]` section in your `config.ini` to point to the directories where you have extracted the provided datasets.

**Explore**: Use the provided notebook to visualize the layers. For example, you can see how noise maps overlap with existing paths to identify "unpleasant" routes that may need alternatives.
