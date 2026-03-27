"""Extended dashboard with model comparison and analysis."""

from __future__ import annotations

from functools import lru_cache
import json
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
from hdbscan import HDBSCAN
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from shapely.geometry import Polygon

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.models.unsupervised import EnsembleAnalyzer

DATA_PATH = Path("app/model/h3_features.parquet")
RESULTS_PATH = Path("app/model/model_comparison.json")
PHASE4_PATH = Path("app/model/phase4_optimization.json")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

FEATURE_COLS = [
    "bike_racks_count",
    "bike_infra_points_count",
    "buildings_count",
    "noise_area_share",
    "greenery_area_share",
    "cycling_path_length_m",
    "street_length_m",
    "bike_path_coverage_ratio",
]


def _generate_test_h3_data() -> gpd.GeoDataFrame:
    """Generate dummy H3 data for testing/demo."""
    kraków_center = (50.0647, 19.9450)
    resolution = 9
    
    try:
        cell_ids = h3.grid_disk(h3.latlng_to_cell(kraków_center[0], kraków_center[1], resolution), k=5)
    except Exception:
        cell_ids = [h3.latlng_to_cell(kraków_center[0] + 0.01 * i, kraków_center[1] + 0.01 * j, resolution) 
                    for i in range(-2, 3) for j in range(-2, 3)]
    
    polygons = []
    for cell_id in cell_ids:
        boundary_latlng = h3.cell_to_boundary(cell_id)
        polygons.append(Polygon([(lng, lat) for lat, lng in boundary_latlng]))
    
    scores = np.random.uniform(0.2, 0.95, len(cell_ids))
    
    gdf = gpd.GeoDataFrame(
        {
            "h3_id": list(cell_ids),
            "demand_score": scores,
            "bike_racks_count": np.random.randint(0, 50, len(cell_ids)),
            "bike_infra_points_count": np.random.randint(0, 30, len(cell_ids)),
            "buildings_count": np.random.randint(10, 200, len(cell_ids)),
            "bike_path_coverage_ratio": np.random.uniform(0.0, 1.0, len(cell_ids)),
            "noise_area_share": np.random.uniform(0.0, 0.8, len(cell_ids)),
            "greenery_area_share": np.random.uniform(0.0, 1.0, len(cell_ids)),
            "cycling_path_length_m": np.random.uniform(0, 5000, len(cell_ids)),
            "street_length_m": np.random.uniform(1000, 10000, len(cell_ids)),
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


@lru_cache(maxsize=1)
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


@lru_cache(maxsize=1)
def get_phase4_results() -> dict | None:
    if PHASE4_PATH.exists():
        try:
            with open(PHASE4_PATH, encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load phase4 optimization: {e}")
    return None


def get_model_results() -> dict | None:
    if RESULTS_PATH.exists():
        try:
            with open(RESULTS_PATH) as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load model results: {e}")
    return None


def _fit_best_model_labels(frame: pd.DataFrame, phase4: dict | None) -> np.ndarray:
    if phase4 is None:
        return np.full(len(frame), -999, dtype=int)

    best_model = phase4.get("best_model")
    if not best_model:
        return np.full(len(frame), -999, dtype=int)

    X = frame[FEATURE_COLS].fillna(0.0).values
    Xs = StandardScaler().fit_transform(X)

    if best_model.startswith("HDBSCAN_mcs"):
        mcs = int(best_model.replace("HDBSCAN_mcs", ""))
        model = HDBSCAN(min_cluster_size=mcs, core_dist_n_jobs=-1)
        return model.fit_predict(Xs)

    if best_model.startswith("KMeans_k"):
        k = int(best_model.replace("KMeans_k", ""))
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        return model.fit_predict(Xs)

    if best_model.startswith("IsolationForest_c"):
        contamination = float(best_model.replace("IsolationForest_c", ""))
        model = IsolationForest(contamination=contamination, random_state=42, n_jobs=-1)
        return model.fit_predict(Xs)

    return np.full(len(frame), -999, dtype=int)


def build_recommendations(data: gpd.GeoDataFrame) -> tuple[gpd.GeoDataFrame, str]:
    frame = data.copy()
    phase4 = get_phase4_results()
    best_model = phase4.get("best_model") if phase4 else "rule_based"

    if not set(FEATURE_COLS).issubset(frame.columns):
        frame["recommendation_tier"] = np.where(
            frame.get("demand_score", pd.Series(np.zeros(len(frame)))) >= 0.66,
            "high-priority",
            "watch",
        )
        return frame, "rule_based"

    labels = _fit_best_model_labels(frame, phase4)
    frame["model_label"] = labels

    demand = frame.get("demand_score", pd.Series(np.zeros(len(frame))))
    high_demand = demand >= 0.66

    if str(best_model).startswith("HDBSCAN"):
        noise = frame["model_label"] == -1
        cluster_avg = frame[~noise].groupby("model_label")["demand_score"].mean()
        top_clusters = set(cluster_avg[cluster_avg >= cluster_avg.quantile(0.75)].index)
        model_hotspot = frame["model_label"].isin(top_clusters)
        high_priority = (noise & high_demand) | model_hotspot
    elif str(best_model).startswith("KMeans"):
        cluster_avg = frame.groupby("model_label")["demand_score"].mean()
        top_clusters = set(cluster_avg[cluster_avg >= cluster_avg.quantile(0.75)].index)
        high_priority = frame["model_label"].isin(top_clusters) & high_demand
    elif str(best_model).startswith("IsolationForest"):
        anomalies = frame["model_label"] == -1
        high_priority = anomalies & high_demand
    else:
        high_priority = high_demand

    medium_priority = (~high_priority) & (demand >= 0.45)
    frame["recommendation_tier"] = np.select(
        [high_priority, medium_priority],
        ["high-priority", "watch"],
        default="low-priority",
    )

    return frame, str(best_model)


app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])


app.layout = dbc.Container(
    [
        dcc.Tabs(
            id="main-tabs",
            value="tab-map",
            children=[
                dcc.Tab(
                    label="🗺️ Demand Map",
                    value="tab-map",
                    children=[
                        html.H2("Cycling Infrastructure Demand Map"),
                        html.P("Interactive H3 hexagon map with interpretable demand scoring."),
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        html.Label("Map Layer:", className="fw-bold"),
                                        dcc.Dropdown(
                                            id="map-layer",
                                            options=[
                                                {"label": "Demand Score", "value": "demand"},
                                                {"label": "Model Recommendations", "value": "recommendations"},
                                            ],
                                            value="demand",
                                            clearable=False,
                                            className="mb-2",
                                        ),
                                        html.Label("Filter by Priority:", className="fw-bold"),
                                        dcc.Dropdown(
                                            id="priority-filter",
                                            options=[
                                                {"label": "All Areas", "value": "all"},
                                                {"label": "High Priority", "value": "high-priority"},
                                                {"label": "Watch", "value": "watch"},
                                                {"label": "Low Priority", "value": "low-priority"},
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
                                        children="📊 Using demo data. Production data loads when pipeline completes.",
                                    ),
                                    width=9,
                                ),
                            ],
                            className="mb-4",
                        ),
                        dcc.Graph(id="demand-map", style={"height": "70vh"}),
                        dcc.Interval(id="interval-component", interval=5000, n_intervals=0),
                    ],
                ),
                dcc.Tab(
                    label="🤖 Model Comparison",
                    value="tab-models",
                    children=[
                        html.H2("Unsupervised Model Comparison"),
                        html.P("Compare clustering stability and quality metrics across multiple algorithms."),
                        dbc.Row(
                            [
                                dbc.Col(
                                    html.Button(
                                        "Run Models",
                                        id="run-models-btn",
                                        className="btn btn-primary",
                                    ),
                                    width=3,
                                ),
                                dbc.Col(
                                    html.Div(id="model-status", className="alert alert-info d-none"),
                                    width=9,
                                ),
                            ],
                            className="mb-4",
                        ),
                        dcc.Graph(id="model-metrics-chart"),
                        html.Hr(),
                        html.H4("Detailed Results"),
                        html.Div(id="model-results-table"),
                        dcc.Interval(id="interval-models", interval=3000, n_intervals=0),
                    ],
                ),
                dcc.Tab(
                    label="📈 Analytics",
                    value="tab-analytics",
                    children=[
                        html.H2("Feature Statistics & Distribution"),
                        html.P("Analyze feature distributions across generated H3 hexagons."),
                        dcc.Dropdown(
                            id="feature-selector",
                            options=[
                                {"label": "Building Density", "value": "buildings_count"},
                                {"label": "Bike Path Coverage", "value": "bike_path_coverage_ratio"},
                                {"label": "Noise Exposure", "value": "noise_area_share"},
                                {"label": "Greenery Coverage", "value": "greenery_area_share"},
                            ],
                            value="buildings_count",
                        ),
                        dcc.Graph(id="feature-histogram"),
                    ],
                ),
            ],
        ),
    ],
    fluid=True,
    className="mt-4",
)


@app.callback(
    [Output("demand-map", "figure"), Output("data-status", "children")],
    [Input("priority-filter", "value"), Input("map-layer", "value"), Input("interval-component", "n_intervals")],
)
def update_map(priority: str, map_layer: str, n: int):
    data = load_map_data()
    data, best_model = build_recommendations(data)

    priority_column = "demand_priority" if map_layer == "demand" else "recommendation_tier"
    if priority != "all" and not data.empty and priority_column in data.columns:
        data = data[data[priority_column].astype(str) == priority]
    
    if data.empty:
        return px.scatter_mapbox(lat=[], lon=[], zoom=11, title="No data available"), "No data available"

    frame = data.copy()
    frame = frame.set_index("h3_id") if "h3_id" in frame.columns else frame
    feature_collection = frame.__geo_interface__
    center = data.unary_union.centroid

    if map_layer == "recommendations":
        fig = px.choropleth_mapbox(
            frame,
            geojson=feature_collection,
            locations=frame.index,
            color="recommendation_tier",
            color_discrete_map={
                "high-priority": "#d73027",
                "watch": "#fdae61",
                "low-priority": "#1a9850",
            },
            mapbox_style="carto-positron",
            zoom=11,
            center={"lat": center.y, "lon": center.x},
            opacity=0.75,
            hover_data=["demand_score", "recommendation_tier"],
        )
        status = f"Using best model from Phase 4: {best_model}"
    else:
        fig = px.choropleth_mapbox(
            frame,
            geojson=feature_collection,
            locations=frame.index,
            color="demand_score" if "demand_score" in frame.columns else None,
            color_continuous_scale="YlOrRd",
            mapbox_style="carto-positron",
            zoom=11,
            center={"lat": center.y, "lon": center.x},
            opacity=0.7,
        )
        status = f"Demand layer loaded; best model available: {best_model}"

    fig.update_layout(margin={"r": 0, "t": 40, "l": 0, "b": 0})
    return fig, status


@app.callback(
    Output("model-metrics-chart", "figure"),
    Input("interval-models", "n_intervals"),
)
def update_model_metrics(n: int):
    results = get_model_results()
    if not results:
        return px.bar(title="No model results yet. Click 'Run Models' first.")

    metrics_data = []
    for model_name, data in results.items():
        metrics = data.get("metrics", {})
        metrics_data.append({
            "model": model_name,
            **metrics,
        })

    df = pd.DataFrame(metrics_data)
    
    if "silhouette" in df.columns:
        fig = px.bar(df, x="model", y="silhouette", title="Silhouette Score (higher is better)")
        fig.update_yaxes(title_text="Silhouette")
        return fig

    return px.bar(title="No clustering metrics available")


@app.callback(
    Output("model-results-table", "children"),
    Input("interval-models", "n_intervals"),
)
def update_model_table(n: int):
    results = get_model_results()
    if not results:
        return html.Div("No model results yet.", className="alert alert-warning")

    rows = []
    for model_name, data in results.items():
        metrics = data.get("metrics", {})
        row = html.Tr([
            html.Td(model_name),
            html.Td(f"{metrics.get('n_clusters', 'N/A')}"),
            html.Td(f"{metrics.get('silhouette', 'N/A'):.3f}" if isinstance(metrics.get('silhouette'), (int, float)) else "N/A"),
            html.Td(f"{metrics.get('davies_bouldin', 'N/A'):.3f}" if isinstance(metrics.get('davies_bouldin'), (int, float)) else "N/A"),
        ])
        rows.append(row)

    table = html.Table([
        html.Thead(html.Tr([
            html.Th("Model"),
            html.Th("Clusters"),
            html.Th("Silhouette"),
            html.Th("Davies-Bouldin"),
        ])),
        html.Tbody(rows),
    ], className="table table-striped")

    return table


@app.callback(
    Output("feature-histogram", "figure"),
    Input("feature-selector", "value"),
)
def update_feature_hist(feature: str):
    data = load_map_data()
    if feature not in data.columns:
        return px.histogram(title=f"Feature '{feature}' not in data")

    fig = px.histogram(
        data,
        x=feature,
        nbins=30,
        title=f"Distribution: {feature}",
        labels={feature: feature},
    )
    return fig


if __name__ == "__main__":
    app.run(debug=True, port=8050)
