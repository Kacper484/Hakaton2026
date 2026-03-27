from __future__ import annotations

import logging
import sys
from pathlib import Path

import dash
import dash_bootstrap_components as dbc
import geopandas as gpd
import h3
import numpy as np
import pandas as pd
import plotly.express as px
from dash import Input, Output, dcc, html
from shapely.geometry import Polygon

sys.path.insert(0, str(Path(__file__).parent.parent))

DATA_PATH = Path("app/model/h3_features.parquet")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _generate_test_h3_data() -> gpd.GeoDataFrame:
    """Generate dummy H3 data for testing/demo when parquet not available."""
    kraków_center = (50.0647, 19.9450)
    resolution = 7
    
    cell_ids = h3.grid_disk(h3.latlng_to_cell(kraków_center[0], kraków_center[1], resolution), k=3)
    
    polygons = []
    for cell_id in cell_ids:
        boundary_latlng = h3.cell_to_boundary(cell_id)
        polygons.append(Polygon([(lng, lat) for lat, lng in boundary_latlng]))
    
    scores = np.random.uniform(0.2, 0.95, len(cell_ids))
    
    gdf = gpd.GeoDataFrame(
        {
            "h3_id": list(cell_ids),
            "demand_score": scores,
            "buildings_count": np.random.randint(10, 200, len(cell_ids)),
            "bike_path_coverage_ratio": np.random.uniform(0.0, 1.0, len(cell_ids)),
            "noise_area_share": np.random.uniform(0.0, 0.8, len(cell_ids)),
            "greenery_area_share": np.random.uniform(0.0, 1.0, len(cell_ids)),
        },
        geometry=polygons,
        crs="EPSG:4326",
    )
    
    gdf["demand_priority"] = pd.cut(
        gdf["demand_score"],
        bins=[-1.0, 0.33, 0.66, 1.0],
        labels=["low", "medium", "high"],
    )
    
    logger.info(f"Generated test H3 data with {len(gdf)} cells")
    return gdf


def load_map_data() -> gpd.GeoDataFrame:
    if DATA_PATH.exists():
        try:
            gdf = gpd.read_parquet(DATA_PATH)
            if not gdf.empty:
                if gdf.crs is None:
                    gdf = gdf.set_crs("EPSG:4326")
                else:
                    gdf = gdf.to_crs("EPSG:4326")
                logger.info(f"Loaded production data: {len(gdf)} H3 cells")
                return gdf
        except Exception as e:
            logger.warning(f"Could not load parquet: {e}, using test data")
    
    logger.info("Using test/demo H3 data")
    return _generate_test_h3_data()


def build_map_figure(data: gpd.GeoDataFrame):
    if data.empty:
        return px.scatter_mapbox(lat=[], lon=[], zoom=11, title="No data available")

    frame = data.copy()
    frame = frame.set_index("h3_id")
    feature_collection = frame.__geo_interface__
    center = data.unary_union.centroid

    fig = px.choropleth_mapbox(
        frame,
        geojson=feature_collection,
        locations=frame.index,
        color="demand_score",
        color_continuous_scale="YlOrRd",
        mapbox_style="carto-positron",
        zoom=11,
        center={"lat": center.y, "lon": center.x},
        opacity=0.7,
        labels={"demand_score": "Demand Score"},
        hover_data=[
            "demand_priority",
            "buildings_count",
            "bike_path_coverage_ratio",
            "noise_area_share",
            "greenery_area_share",
        ],
    )
    fig.update_layout(
        margin={"r": 0, "t": 40, "l": 0, "b": 0},
        title_text="<b>H3 Cycling Infrastructure Demand Map</b> (Kraków)",
    )
    return fig


app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])

app.layout = dbc.Container(
    [
        html.H1("🚴 Cycling Infrastructure Demand Analysis", className="mb-4"),
        html.P(
            "Interactive H3 hexagon map showing interpretable demand scores for planning new cycling paths in Kraków. "
            "Data updates automatically when pipeline completes."
        ),
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.Label("Filter by Priority:", className="fw-bold"),
                        dcc.Dropdown(
                            id="priority-filter",
                            options=[
                                {"label": "All Areas", "value": "all"},
                                {"label": "High Demand", "value": "high"},
                                {"label": "Medium Demand", "value": "medium"},
                                {"label": "Low Demand", "value": "low"},
                            ],
                            value="all",
                            clearable=False,
                        ),
                    ],
                    width=3,
                ),
                dbc.Col(
                    html.Div(
                        id="data-status",
                        className="alert alert-info",
                        children="📊 Using demo data. Production data will load when pipeline completes...",
                    ),
                    width=9,
                ),
            ],
            className="mb-4",
        ),
        dcc.Graph(id="demand-map", style={"height": "75vh"}),
        dcc.Interval(id="interval-component", interval=5000, n_intervals=0),
    ],
    fluid=True,
    className="mt-4",
)


@app.callback(
    Output("demand-map", "figure"),
    [Input("priority-filter", "value"), Input("interval-component", "n_intervals")],
)
def update_map(priority: str, n: int):
    data = load_map_data()
    if priority != "all" and not data.empty:
        data = data[data["demand_priority"].astype(str) == priority]
    return build_map_figure(data)


if __name__ == "__main__":
    app.run(debug=True)
