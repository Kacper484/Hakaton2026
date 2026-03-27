import configparser
import logging
import urllib3
from pathlib import Path
from dataclasses import dataclass, field

import geopandas as gpd
import pandas as pd
import osmnx as ox
import requests


logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
ox.settings.requests_timeout = 600

def load_config() -> configparser.ConfigParser:
    """Loads the configuration from the config.ini file."""
    config = configparser.ConfigParser()
    config.read("config.ini")
    return config

@dataclass
class OpenStreetMapData:
    """Class for retrieving OpenStreetMap (OSM) data, including buildings and streets.

    Attributes:
        place (str): Name of the location to retrieve data for.
        buildings_df (gpd.GeoDataFrame): GeoDataFrame containing building geometries.
        streets_df (gpd.GeoDataFrame): GeoDataFrame containing street geometries.
    """

    place: str = "Kraków, Poland"
    buildings_df: gpd.GeoDataFrame = field(default_factory=gpd.GeoDataFrame)
    streets_df: gpd.GeoDataFrame = field(default_factory=gpd.GeoDataFrame)

    def __post_init__(self) -> None:
        """Initializes the OpenStreetMapData class by retrieving buildings and streets."""
        logging.info(f"Initializing OSM data retrieval for: {self.place}")
        try:
            self._fetch_buildings()
            self._fetch_streets()
        except Exception as e:
            logging.error(f"Error initializing OSM data: {e}")

    def _fetch_buildings(self) -> None:
        """Fetches building geometries from OSM and computes their centroids."""
        self.buildings_df = ox.features_from_place(
            self.place, tags={"building": True}
        )[['geometry']].dropna()
        logging.info(f"Successfully retrieved {len(self.buildings_df)} buildings.")

        self.buildings_df = self._calculate_centroid(self.buildings_df)

    def _fetch_streets(self) -> None:
        """Fetches street geometries from OSM as a GeoDataFrame."""
        graph = ox.graph_from_place(self.place, network_type="all")
        self.streets_df = ox.graph_to_gdfs(graph, nodes=False)
        logging.info(f"Successfully retrieved {len(self.streets_df)} street segments.")

    @staticmethod
    def _calculate_centroid(df: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Calculates the centroid of each building polygon."""
        df = df.copy()
        df["centroid"] = df.geometry.centroid
        df["lat"] = df["centroid"].y
        df["lng"] = df["centroid"].x
        return df.drop(columns=["centroid"])

@dataclass
class LocalGeoData:
    """
    Class to load and consolidate local geospatial data for Kraków.
    
    Attributes:
        bike_racks_df (gpd.GeoDataFrame): Data for bike racks.
        bike_infrastructure_df (gpd.GeoDataFrame): Point infrastructure data.
        noise_map_df (gpd.GeoDataFrame): Consolidated noise data from all subfolders.
        greenery_df (gpd.GeoDataFrame): Consolidated PTLZ, PTTR, and PTUT areas.
        cycling_paths_df (gpd.GeoDataFrame): Existing cycling path network.
    """
    config: configparser.ConfigParser = field(default_factory=load_config)
    bike_racks_df: gpd.GeoDataFrame = field(init=False)
    bike_infrastructure_df: gpd.GeoDataFrame = field(init=False)
    noise_map_df: gpd.GeoDataFrame = field(init=False)
    greenery_df: gpd.GeoDataFrame = field(init=False)
    cycling_paths_df: gpd.GeoDataFrame = field(init=False)

    def __post_init__(self):
        logging.info("Starting local data ingestion...")
        data_cfg = self.config['DATA']

        self.bike_racks_df = self._read_geo_file(data_cfg.get("BIKE_RACKS_PATH"))
        self.bike_infrastructure_df = self._read_geo_file(data_cfg.get("BIKE_INFRASTRUCTURE_PATH"))
        self.cycling_paths_df = self._read_geo_file(data_cfg.get("CYCLING_PATHS_PATH"))
        
        noise_root = Path(data_cfg.get("NOISE_DIR"))
        self.noise_map_df = self._load_recursive_geojson(noise_root)
        
        greenery_root = Path(data_cfg.get("GREENERY_DIR"))
        self.greenery_df = self._load_greenery(greenery_root)
        
        logging.info("Data successfully loaded into DataClass.")

    def _read_geo_file(self, path_str: str) -> gpd.GeoDataFrame:
        """Helper to safely read a single file."""
        if path_str and Path(path_str).exists():
            return gpd.read_file(path_str)
        logging.warning(f"File path not found: {path_str}")
        return gpd.GeoDataFrame()

    def _load_recursive_geojson(self, folder_path: Path) -> gpd.GeoDataFrame:
        """Finds all geojson files in nested folders and merges them."""
        files = list(folder_path.rglob("*.geojson"))
        if not files:
            return gpd.GeoDataFrame()
        
        logging.info(f"Merging {len(files)} noise map layers...")
        return gpd.GeoDataFrame(pd.concat([gpd.read_file(f) for f in files], ignore_index=True))

    def _load_greenery(self, base_path: Path) -> gpd.GeoDataFrame:
        """Merges PTLZ, PTTR, and PTUT shapefiles into one GeoDataFrame."""
        categories = ["PTLZ", "PTTR", "PTUT"]
        all_dfs = []
        
        for cat in categories:
            cat_folder = base_path / cat
            for shp in cat_folder.glob("*.shp"):
                df = gpd.read_file(shp)
                df['greenery_category'] = cat
                all_dfs.append(df)
        
        if all_dfs:
            logging.info(f"Merged {len(all_dfs)} greenery files into one GeoDataFrame.")
            return gpd.GeoDataFrame(pd.concat(all_dfs, ignore_index=True))
        return gpd.GeoDataFrame()


@dataclass
class BicycleCounterData:
    """Scrapes live bicycle traffic data from the Kraków counter website."""
    config: configparser.ConfigParser = field(default_factory=load_config)
    counters_df: pd.DataFrame = field(default_factory=pd.DataFrame)

    def __post_init__(self) -> None:
        """Initializes fetching using the URL from config.ini."""
        url = self.config.get("API", "COUNTER_URL", fallback=None)
        
        if not url or url == "None":
            logging.error("Counter URL is missing in config.ini [API] section.")
            return

        logging.info(f"Fetching live counter data from: {url} (SSL verification disabled)")
        try:
            headers = {"User-Agent": "Mozilla/5.0"}
            response = requests.get(url, headers=headers, timeout=30, verify=False)
            response.raise_for_status()
            
            tables = pd.read_html(response.text)
            if tables:
                self.counters_df = tables[0]
                self.counters_df.dropna(axis=1, how="all", inplace=True)
                
                logging.info(f"Successfully retrieved {len(self.counters_df)} counter records.")
        except Exception as e:
            logging.error(f"Failed to fetch counter data: {e}")
