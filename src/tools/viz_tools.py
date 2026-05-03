"""
(Visualization tools)

folium () + contextily () + seaborn ,
matplotlib ,

  - plot_map -> OSM PNG (contextily)
  - create_heatmap -> folium HTML + PNG 
"""
import os
from typing import Optional, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import geopandas as gpd
import numpy as np

from .registry import tool_registry
from .vector_tools import get_loaded_datasets

# -- ------------------------------
# ColorBrewer / GIS 
LAYER_COLORS = [
    "#2166ac",
    "#d6604d",
    "#4daf4a",
    "#ff7f00",
    "#984ea3",
    "#17becf",
]

# folium HTMLmatplotlib 
MAX_FOLIUM_FEATURES = 10000
MAX_MATPLOTLIB_POLYGONS = 50000

# GIS 
def _apply_map_style():
    """ GIS matplotlib """
    plt.rcParams.update({
        "figure.facecolor": "#f8f9fa",
        "axes.facecolor": "#e9ecef",
        "axes.edgecolor": "#adb5bd",
        "axes.labelcolor": "#343a40",
        "axes.grid": True,
        "grid.color": "#dee2e6",
        "grid.linewidth": 0.5,
        "grid.alpha": 0.7,
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "figure.dpi": 150,
    })

def _try_add_basemap(ax, gdf):
    """ contextily failure"""
    try:
        import contextily as ctx
        # Web Mercator 
        gdf_3857 = gdf.to_crs(epsg=3857)
        ax.set_xlim(gdf_3857.total_bounds[0], gdf_3857.total_bounds[2])
        ax.set_ylim(gdf_3857.total_bounds[1], gdf_3857.total_bounds[3])
        ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron, zoom="auto")
        return True
    except Exception:
        # failurereplacement
        return False

def _make_folium_map(gdf, popup_col=None, color="#2166ac", tiles="CartoDB positron"):
    """ folium Leaflet """
    import folium
    from folium.plugins import HeatMap

    centroid = gdf.dissolve().centroid.iloc[0]
    m = folium.Map(
        location=[centroid.y, centroid.x],
        zoom_start=13,
        tiles=tiles,
    )
    return m

@tool_registry.register(
    name="plot_map",
    description=(
        "Create a professional map visualization of one or more datasets. "
        "Produces a high-quality PNG with contextily basemap and optional choropleth coloring. "
        "Also generates an interactive HTML map viewable in a browser."
    ),
    parameters={
        "dataset_names": {
            "type": "string",
            "description": "Comma-separated names of datasets to plot (e.g., 'buildings,roads')"
        },
        "output_path": {
            "type": "string",
            "description": "Output image file path (e.g., 'output/map.png')"
        },
        "title": {
            "type": "string",
            "description": "Map title"
        },
        "color_column": {
            "type": "string",
            "description": "Optional: column name for choropleth coloring (applies to first dataset). Use 'none' to skip.",
            "optional": True,
        },
    },
    returns="Confirmation message with the output path",
)
def plot_map(
    dataset_names: str,
    output_path: str,
    title: str,
    color_column: str = "none",
) -> str:
    """"""
    try:
        datasets = get_loaded_datasets()
        names = [n.strip() for n in dataset_names.split(",")]
        _apply_map_style()

        # -- 1. contextily (PNG) --
        fig, ax = plt.subplots(1, 1, figsize=(14, 11))
        legends = []
        has_basemap = False

        # Web Mercator
        gdfs_3857 = {}
        for name in names:
            gdf = datasets.get(name)
            if gdf is None:
                return f"Error: Dataset '{name}' not found."
            gdfs_3857[name] = gdf.to_crs(epsg=3857)

        for i, name in enumerate(names):
            gdf = gdfs_3857[name]
            color = LAYER_COLORS[i % len(LAYER_COLORS)]

            if i == 0 and color_column != "none" and color_column in gdf.columns:
                gdf.plot(
                    ax=ax, column=color_column, legend=True,
                    cmap="RdYlBu_r", edgecolor="#555", linewidth=0.3,
                    legend_kwds={"label": color_column, "shrink": 0.5,
                                 "orientation": "horizontal", "pad": 0.05},
                    alpha=0.75,
                )
                legends.append(f"{name} ({color_column})")
            else:
                geom_types = gdf.geometry.geom_type.unique()
                plot_gdf = gdf
                if len(gdf) > MAX_MATPLOTLIB_POLYGONS:
                    plot_gdf = gdf.sample(n=MAX_MATPLOTLIB_POLYGONS, random_state=42)
                if any(t in ["Point", "MultiPoint"] for t in geom_types):
                    plot_gdf.plot(ax=ax, color=color, markersize=20,
                             alpha=0.8, edgecolor="white", linewidth=0.5,
                             zorder=5)
                elif any(t in ["LineString", "MultiLineString"] for t in geom_types):
                    plot_gdf.plot(ax=ax, color=color, linewidth=1.2, alpha=0.85)
                else:
                    plot_gdf.plot(ax=ax, facecolor=color, edgecolor="#333",
                             linewidth=0.3, alpha=0.45)
                sampled = f" (sampled {MAX_MATPLOTLIB_POLYGONS})" if len(gdf) > MAX_MATPLOTLIB_POLYGONS else ""
                legends.append(f"{name}{sampled}")

        try:
            import contextily as ctx
            ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron, zoom="auto")
            has_basemap = True
        except Exception:
            ax.set_facecolor("#e8e8e8")

        ax.set_title(title, fontsize=16, fontweight="bold", pad=15,
                     color="#1a1a2e")
        ax.set_axis_off()

        if len(names) > 1 or color_column == "none":
            from matplotlib.lines import Line2D
            handles = []
            for i, n in enumerate(legends):
                c = LAYER_COLORS[i % len(LAYER_COLORS)]
                handles.append(Line2D([0], [0], marker='s', color='w',
                               markerfacecolor=c, markersize=12,
                               label=n, alpha=0.8))
            ax.legend(handles=handles, loc="upper right",
                      fontsize=10, framealpha=0.9,
                      edgecolor="#ccc", fancybox=True)

        _add_scalebar(ax, gdf)

        plt.tight_layout()
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
        fig.savefig(output_path, dpi=200, bbox_inches="tight",
                    facecolor=fig.get_facecolor(), edgecolor="none")
        plt.close(fig)

        # -- 2. folium (HTML) --
        # folium 
        total_features = sum(len(datasets.get(n, [])) for n in names)
        html_path = output_path.replace(".png", ".html")
        html_msg = ""

        if total_features <= MAX_FOLIUM_FEATURES:
            try:
                import folium
                first_gdf = datasets.get(names[0])
                centroid = first_gdf.to_crs(epsg=4326).dissolve().centroid.iloc[0]
                m = folium.Map(
                    location=[centroid.y, centroid.x],
                    zoom_start=13,
                    tiles="CartoDB positron",
                )

                for i, name in enumerate(names):
                    gdf = datasets.get(name)
                    color = LAYER_COLORS[i % len(LAYER_COLORS)]
                    geom_types = gdf.geometry.geom_type.unique()

                    if any(t in ["Point", "MultiPoint"] for t in geom_types):
                        for _, row in gdf.iterrows():
                            popup_text = str(row.get("name", "")) if row.get("name") else name
                            folium.CircleMarker(
                                location=[row.geometry.centroid.y, row.geometry.centroid.x],
                                radius=5, color=color, fill=True,
                                fill_color=color, fill_opacity=0.7,
                                popup=popup_text,
                            ).add_to(m)
                    else:
                        # Polygon/Line 
                        plot_gdf = gdf if len(gdf) <= MAX_FOLIUM_FEATURES else gdf.sample(n=MAX_FOLIUM_FEATURES, random_state=42)
                        style = lambda feature, c=color: {
                            "fillColor": c, "color": c,
                            "weight": 1, "fillOpacity": 0.3,
                        }
                        folium.GeoJson(
                            plot_gdf.__geo_interface__,
                            name=name,
                            style_function=style,
                        ).add_to(m)

                folium.LayerControl().add_to(m)
                m.save(html_path)
                html_msg = f" Interactive HTML map also saved to '{html_path}'."
            except Exception as e:
                html_msg = ""

        basemap_note = " (with OpenStreetMap basemap)" if has_basemap else ""
        return f"Map saved to '{output_path}'{basemap_note}. Layers: {legends}.{html_msg}"
    except Exception as e:
        return f"Error: {str(e)}"

@tool_registry.register(
    name="create_heatmap",
    description=(
        "Create a professional heatmap visualization based on point density or a numeric attribute. "
        "Produces both a high-quality PNG with basemap and an interactive HTML heatmap."
    ),
    parameters={
        "dataset_name": {
            "type": "string",
            "description": "Name of the point dataset"
        },
        "output_path": {
            "type": "string",
            "description": "Output image file path"
        },
        "title": {
            "type": "string",
            "description": "Map title"
        },
        "value_column": {
            "type": "string",
            "description": "Optional: numeric column for weighted heatmap. Use 'density' for point density.",
            "optional": True,
        },
    },
    returns="Confirmation message with the output path",
)
def create_heatmap(
    dataset_name: str,
    output_path: str,
    title: str,
    value_column: str = "density",
) -> str:
    """PNG + HTML"""
    try:
        datasets = get_loaded_datasets()
        gdf = datasets.get(dataset_name)
        if gdf is None:
            return f"Error: Dataset '{dataset_name}' not found."

        _apply_map_style()

        # -- 1. contextily + KDE (PNG) --
        fig, ax = plt.subplots(1, 1, figsize=(14, 11))

        gdf_3857 = gdf.to_crs(epsg=3857)
        x = gdf_3857.geometry.centroid.x.values
        y = gdf_3857.geometry.centroid.y.values

        if value_column != "density" and value_column in gdf.columns:
            c = gdf[value_column].values.astype(float)
            scatter = ax.scatter(x, y, c=c, cmap="magma_r", s=30,
                                 alpha=0.8, edgecolor="white", linewidth=0.3,
                                 zorder=5)
            plt.colorbar(scatter, ax=ax, label=value_column, shrink=0.5,
                         orientation="horizontal", pad=0.05)
        else:
            # hexbin + 
            hb = ax.hexbin(x, y, gridsize=40, cmap="YlOrRd",
                           mincnt=1, alpha=0.7, zorder=3)
            ax.scatter(x, y, c="#d62828", s=8, alpha=0.5,
                       edgecolor="none", zorder=4)
            plt.colorbar(hb, ax=ax, label="Feature Density", shrink=0.5,
                         orientation="horizontal", pad=0.05)

        try:
            import contextily as ctx
            ctx.add_basemap(ax, source=ctx.providers.CartoDB.DarkMatter, zoom="auto")
        except Exception:
            ax.set_facecolor("#1a1a2e")

        ax.set_title(title, fontsize=16, fontweight="bold", pad=15,
                     color="#e8e8e8" if True else "#1a1a2e")
        ax.set_axis_off()

        _add_scalebar(ax, gdf_3857, dark=True)

        plt.tight_layout()
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
        fig.savefig(output_path, dpi=200, bbox_inches="tight",
                    facecolor="#1a1a2e", edgecolor="none")
        plt.close(fig)

        # -- 2. folium (HTML) --
        html_path = output_path.replace(".png", ".html")
        try:
            import folium
            from folium.plugins import HeatMap

            centroid = gdf.dissolve().centroid.iloc[0]
            m = folium.Map(
                location=[centroid.y, centroid.x],
                zoom_start=13,
                tiles="CartoDB dark_matter",
            )

            heat_data = [[row.geometry.centroid.y, row.geometry.centroid.x]
                         for _, row in gdf.iterrows()]

            if value_column != "density" and value_column in gdf.columns:
                heat_data = [[row.geometry.centroid.y, row.geometry.centroid.x,
                              float(row[value_column])]
                             for _, row in gdf.iterrows()
                             if not np.isnan(float(row[value_column]))]

            HeatMap(
                heat_data,
                radius=18,
                blur=25,
                max_zoom=15,
                gradient={0.2: "#2a9d8f", 0.5: "#e9c46a",
                          0.7: "#f4a261", 1.0: "#e76f51"},
            ).add_to(m)

            m.save(html_path)
            html_msg = f" Interactive heatmap also saved to '{html_path}'."
        except Exception:
            html_msg = ""

        return f"Heatmap saved to '{output_path}'. Points: {len(gdf)}.{html_msg}"
    except Exception as e:
        return f"Error: {str(e)}"

def _add_scalebar(ax, gdf, dark=False):
    """"""
    try:
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        # 20%
        map_width = xlim[1] - xlim[0]
        bar_length = map_width * 0.2

        nice_lengths = [100, 200, 500, 1000, 2000, 5000, 10000]
        bar_m = min(nice_lengths, key=lambda x: abs(x - bar_length))

        bar_x = xlim[0] + map_width * 0.05
        bar_y = ylim[0] + (ylim[1] - ylim[0]) * 0.05

        color = "white" if dark else "#333"
        ax.plot([bar_x, bar_x + bar_m], [bar_y, bar_y],
                color=color, linewidth=3, zorder=10)

        label = f"{bar_m}m" if bar_m < 1000 else f"{bar_m//1000}km"
        ax.text(bar_x + bar_m / 2, bar_y + (ylim[1] - ylim[0]) * 0.015,
                label, ha="center", fontsize=10, fontweight="bold",
                color=color, zorder=10)
    except Exception:
        pass
