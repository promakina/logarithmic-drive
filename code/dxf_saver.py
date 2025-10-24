# dxf_saver.py
import math
from typing import Dict, List, Tuple, Iterable, Optional

import ezdxf
from ezdxf.entities import Spline

Point2D = Tuple[float, float]


# ----------------------------
# Geometry helpers (closedness)
# ----------------------------
def _is_closed(points: Iterable[Point2D], tol: float = 1e-9) -> bool:
    pts = list(points)
    if len(pts) < 2:
        return False
    (x0, y0), (x1, y1) = pts[0], pts[-1]
    return (x0 - x1) ** 2 + (y0 - y1) ** 2 <= tol * tol


def _dedupe_closure(points: Iterable[Point2D], tol: float = 1e-9) -> List[Point2D]:
    """If first and last point are the same (within tol), drop the last one to
    avoid issues with periodic SPLINE creation."""
    pts = list(points)
    if len(pts) >= 2 and _is_closed(pts, tol):
        return pts[:-1]
    return pts


# ----------------------------
# Offset helper
# ----------------------------
def _offset_curve(points: List[Point2D], dist: float, closed: bool, tol: float = 1e-12) -> List[Point2D]:
    """
    Offset a sampled 2D curve by distance 'dist' along the local unit normal.
    Uses central differences for tangents and wraps indices if 'closed' is True.
    """
    n = len(points)
    if n < 2 or abs(dist) < tol:
        return points[:]

    # Precompute tangents with central differences
    tangents = []
    for i in range(n):
        if closed:
            i_prev = (i - 1) % n
            i_next = (i + 1) % n
        else:
            i_prev = max(0, i - 1)
            i_next = min(n - 1, i + 1)
        x_prev, y_prev = points[i_prev]
        x_next, y_next = points[i_next]
        tx = x_next - x_prev
        ty = y_next - y_prev
        L = math.hypot(tx, ty)
        if L > tol:
            tx /= L
            ty /= L
        else:
            # Degenerate: copy previous tangent if exists, else 0,1
            if tangents:
                tx, ty = tangents[-1]
            else:
                tx, ty = (0.0, 1.0)
        tangents.append((tx, ty))

    # Normals are +90° rotations of tangents: n = (-ty, tx)
    off_pts: List[Point2D] = []
    for (p, t) in zip(points, tangents):
        tx, ty = t
        nx, ny = -ty, tx
        off_pts.append((p[0] + dist * nx, p[1] + dist * ny))

    return off_pts


# ----------------------------
# DXF writer
# ----------------------------
def save_geometry_to_dxf(
    filename: str,
    curves: Dict[str, List[Point2D]] = None,
    circles: Dict[str, List[Tuple[float, float, float]]] = None,
    *,
    # NEW: pass ball radius (+ optional clearance) to emit ± offset hypocycloid curves
    ball_radius: Optional[float] = None,
    clearance: float = 0.0,
    hypocycloid_layer_name: str = "hypocycloid",
    offset_layer_plus: str = "hypocycloid_offset_plus",
    offset_layer_minus: str = "hypocycloid_offset_minus",
) -> None:
    """
    Saves multiple 2D curves and circles to a single DXF file, each on its own layer.

    Args:
        filename: Output DXF path (e.g., 'output.dxf').
        curves:   {layer_name: [(x, y), ...]} for each curve.
                  - 'hypocycloid' (or name in 'hypocycloid_layer_name') is exported
                    as a single SPLINE entity (continuous curve).
                  - other layers are exported as closed LWPOLYLINE by default.
        circles:  {layer_name: [(cx, cy, r), ...]} circles per layer.

        ball_radius: If provided, two additional hypocycloid offset curves are exported
                     as single SPLINEs at ±(ball_radius + clearance).
        clearance:   Added to ball_radius for the offset distance.
        hypocycloid_layer_name: which layer name in 'curves' is treated as the hypocycloid.
        offset_layer_plus / offset_layer_minus: names for the two offset layers (+ and −).
    """
    # Create a new DXF (default R12/R2000 settings for broad compatibility)
    doc = ezdxf.new()
    msp = doc.modelspace()

    # Helper: ensure a layer exists
    def ensure_layer(name: str) -> None:
        try:
            if name not in doc.layers:
                doc.layers.add(name)
        except Exception:
            try:
                doc.layers.new(name)
            except Exception:
                pass

    # 1) Curves
    if curves:
        for layer_name, pts in curves.items():
            if not pts:
                continue

            ensure_layer(layer_name)
            # Clean up duplicated closure for better SPLINE behavior
            pts2 = _dedupe_closure(pts, tol=1e-9)
            is_closed = _is_closed(pts, tol=1e-9) or _is_closed(pts2, tol=1e-9)

            if layer_name.lower() == hypocycloid_layer_name.lower():
                # ---- Primary hypocycloid: one continuous SPLINE (fit-point based) ----
                # (keeps your existing behavior)
                try:
                    spline: Spline = msp.add_spline(
                        fit_points=pts2,
                        degree=3,
                        periodic=is_closed,
                        dxfattribs={"layer": layer_name},
                    )
                except TypeError:
                    spline: Spline = msp.add_spline(
                        fit_points=pts2, dxfattribs={"layer": layer_name}
                    )
                    if is_closed and not _is_closed(pts2, tol=1e-9):
                        try:
                            spline.set_fit_points(pts2 + [pts2[0]])
                        except Exception:
                            pass

                # ---- Offsets (optional) ----
                if ball_radius is not None and ball_radius > 0.0:
                    d = float(ball_radius) + float(clearance)
                    if d > 0.0:
                        # Compute ± offsets
                        off_plus = _offset_curve(pts2, +d, closed=is_closed)
                        off_minus = _offset_curve(pts2, -d, closed=is_closed)

                        # Remove duplicate closure for each
                        off_plus = _dedupe_closure(off_plus, tol=1e-9)
                        off_minus = _dedupe_closure(off_minus, tol=1e-9)

                        # Ensure their layers exist
                        ensure_layer(offset_layer_plus)
                        ensure_layer(offset_layer_minus)

                        # Export each as one SPLINE (continuous)
                        try:
                            msp.add_spline(
                                fit_points=off_plus,
                                degree=3,
                                periodic=is_closed,
                                dxfattribs={"layer": offset_layer_plus},
                            )
                        except TypeError:
                            msp.add_spline(
                                fit_points=off_plus,
                                dxfattribs={"layer": offset_layer_plus},
                            )

                        try:
                            msp.add_spline(
                                fit_points=off_minus,
                                degree=3,
                                periodic=is_closed,
                                dxfattribs={"layer": offset_layer_minus},
                            )
                        except TypeError:
                            msp.add_spline(
                                fit_points=off_minus,
                                dxfattribs={"layer": offset_layer_minus},
                            )

            else:
                # Default: a single closed lightweight polyline
                # (Still one entity; most CAD will show straight segments)
                msp.add_lwpolyline(
                    pts2,
                    format="xy",
                    close=is_closed,  # close if the path is closed
                    dxfattribs={"layer": layer_name},
                )

    # 2) Circles
    if circles:
        for layer_name, circle_params in circles.items():
            if not circle_params:
                continue
            ensure_layer(layer_name)
            for (cx, cy, r) in circle_params:
                msp.add_circle(center=(cx, cy), radius=float(r), dxfattribs={"layer": layer_name})

    # 3) Save
    try:
        doc.saveas(filename)
        print(f"✅ Successfully saved geometry to '{filename}'")
    except IOError as e:
        print(f"❌ Could not save DXF file: '{filename}'. Reason: {e}")
