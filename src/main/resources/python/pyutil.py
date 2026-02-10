import numpy as np
from typing import Any
from shapely.geometry import (
    Polygon, MultiPolygon, LineString, MultiLineString, Point, GeometryCollection
)
import geopandas as gpd
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

def render_plan_static(
        plan: dict[str, Any],
        dpi=100, 
        figsize=(8, 8)
) -> np.ndarray:

    def get_geometries(geom_data: Any) -> list[Any]:
        """Safely extract individual geometries from single/multi/collections."""
        if geom_data is None:
            return []
        if isinstance(geom_data, (Polygon, LineString, Point)):
            return [] if geom_data.is_empty else [geom_data]
        if isinstance(geom_data, (MultiPolygon, MultiLineString, GeometryCollection)):
            return [g for g in geom_data.geoms if g is not None and not g.is_empty]
        return []

    colors = {
        "living": "#d9d9d9",     # light gray
        "bedroom": "#66c2a5",    # greenish
        "bathroom": "#fc8d62",   # orange
        "kitchen": "#8da0cb",    # blue
        "door": "#e78ac3",       # pink
        "window": "#a6d854",     # lime
        "wall": "#ffd92f",       # yellow
        "front_door": "#a63603", # dark reddish-brown
        "balcony": "#b3b3b3"     # dark gray
    }

    categories = ["living","bedroom","bathroom","kitchen","door","window","wall","front_door","balcony"]

    geoms, color_list = [], []
    for key in categories:
        geom = plan.get(key)
        if geom is None:
            continue
        parts = get_geometries(geom)
        if not parts:
            continue
        geoms.extend(parts)
        color_list.extend([colors.get(key, "#000000")] * len(parts))

    assert len(geoms) > 0, "No geometries to plot."

    # --- Use Matplotlib's OO API (process-safe) ---
    # 1. Create Figure and Canvas objects, NOT using plt
    fig = Figure(figsize=figsize, dpi=dpi)
    canvas = FigureCanvas(fig)
    ax = fig.add_subplot(111)
    # ---

    # Your plotting logic remains the same
    gseries = gpd.GeoSeries(geoms)
    gseries.plot(ax=ax, color=color_list, edgecolor="black", linewidth=0.5)
    ax.set_aspect("equal", adjustable="box")
    ax.set_axis_off()
    
    # 2. Use fig.tight_layout(), NOT plt.tight_layout()
    fig.tight_layout()

    # 3. Convert plot to NumPy array using the canvas
    canvas.draw()
    buf = canvas.buffer_rgba()
    img_np = np.asarray(buf)
    
    # 4. No need for plt.close(fig), fig is a local object
    #    that will be garbage collected.

    return img_np[:, :, :3]  # Discard alpha channel