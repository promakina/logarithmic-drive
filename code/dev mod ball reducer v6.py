#####################################
# Modified Equations from the patent
######################################
import numpy as np
import matplotlib.pyplot as plt

# ----- USER INPUTS -----
# Edit these parameters to generate a clean (non-intersecting) curve.
# Default: 22, 41, 4

SHOW_EVERY_OTHER = True

clearance = 0.0       # set if you want extra running clearance between ball and offset curves

k = 26   # Desired number of cusps for the hypocycloid (must be an integer >= 3)
Rx = 25.0  # Ellipse short-axis (minor) radius.
Rp = 2.50   # Radius of the balls (Rp). A reasonable value is f or slightly less.

# ------------------ SCRIPT BELOW ------------------

# --- Input Validation ---
if not isinstance(k, int) or k < 3:
    raise ValueError("Input 'k' must be an integer of 3 or greater to ensure a valid, non-degenerate shape.")

# --- Derived Geometry ---
# Condition for clean curve: Rw / f = k
# Rw = Ry & f = (Ry - Rx)/2
# Rx and k are known
# Solve for Ry in terms of Rx and k → Ry = Rx * k / (k - 2)

Ry = Rx * k / (k - 2)          # long-axis (major) radius of ellipse (x-axis)
f  = 0.5 * (Ry - Rx)           # rolling circle radius
Rs = 0.5 * (Ry + Rx)           # retaining-hole circle radius
Rw = Ry                        # base circle radius
num_balls = k - 2              # number of balls (and holes)
Ru = Rp + f                    # hole radius

print(f"--- Derived Parameters ---")
print(f"Given k={k}, Rx={Rx:.2f}, the calculated Ry is: {Ry:.2f}")
print(f"Rolling radius f = {f:.2f}")
print(f"Base radius Rw = {Rw:.2f}")
print(f"Number of balls = {num_balls}")
print(f"Ball radius Rp = {Rp:.2f}")
print(f"Hole radius Ru = {Ru:.2f}")
print(f"Verification: Rw / f = {Rw/f:.2f} (should equal k)")
print(f"Reduction Ratio = {k/2:.2f}")
print("--------------------------")

# --- Curves ---
# Ellipse: x = Ry cos θ, y = Rx sin θ (major axis along +x)
theta = np.linspace(0.0, 2 * np.pi, 6001)
x_ell = Ry * np.cos(theta)
y_ell = Rx * np.sin(theta)

# Hypocycloid centerline (standard parametric form)
T = 2 * np.pi
R_hypo, r_hypo = Rw, f
k_ratio = (R_hypo - r_hypo) / r_hypo  # = k - 1

def hypo_xy(t):
    return ((R_hypo - r_hypo) * np.cos(t) + r_hypo * np.cos(k_ratio * t),
            (R_hypo - r_hypo) * np.sin(t) - r_hypo * np.sin(k_ratio * t))

# --- Robust intersections (ellipse ∩ hypocycloid) via root-finding ---
# We solve g(t) = x(t)^2/Ry^2 + y(t)^2/Rx^2 - 1 = 0 for t ∈ [0, 2π)
# This avoids polygonal discretization artifacts and fixes the missing
# intersections at 0°, 90°, 180°, and 270°.

def g_val(t):
    x, y = hypo_xy(t)
    return (x * x) / (Ry * Ry) + (y * y) / (Rx * Rx) - 1.0

roots = []
Nscan = 40000
ts = np.linspace(0.0, 2 * np.pi, Nscan + 1)
vals = np.array([g_val(t) for t in ts])
EPS_EQ = 1e-9

for i in range(Nscan):
    t0, t1 = ts[i], ts[i + 1]
    g0, g1 = vals[i], vals[i + 1]

    # if exactly (or very nearly) on the ellipse, take it
    if abs(g0) < 1e-7:
        roots.append(t0)
        continue

    # bracketed root
    if g0 * g1 < 0.0:
        a, b = t0, t1
        fa, fb = g0, g1
        # bisection refinement
        for _ in range(60):
            m = 0.5 * (a + b)
            fm = g_val(m)
            if abs(fm) < EPS_EQ:
                a = b = m
                break
            if fa * fm <= 0:
                b, fb = m, fm
            else:
                a, fa = m, fm
        roots.append(0.5 * (a + b))

# Deduplicate close roots (wrap-aware)
roots = np.mod(np.array(roots), 2 * np.pi)
if roots.size == 0:
    print("Warning: No intersections found.")
roots.sort()
merged = []
for t in roots:
    if not merged:
        merged.append(t)
    else:
        # wrap-aware spacing
        if abs(((t - merged[-1] + np.pi) % (2 * np.pi) - np.pi)) > 1e-5:
            merged.append(t)
# also check first/last wrap-around
if len(merged) >= 2 and abs(((merged[0] + 2 * np.pi) - merged[-1] + np.pi) % (2 * np.pi) - np.pi) <= 1e-5:
    merged[0] = 0.5 * ((merged[0] + (merged[-1] - 2 * np.pi)))
    merged.pop()

intersection_points = [hypo_xy(t) for t in merged]
print(f"Found {len(intersection_points)} intersection points after refinement.")

# Sort intersection points by their polar angle
def ang(x, y):
    return np.arctan2(y, x)

inter_angles = [ang(x, y) for (x, y) in intersection_points]
order = np.argsort(inter_angles)
intersection_points = [intersection_points[i] for i in order]
inter_angles = [inter_angles[i] for i in order]

# --- Pre-calculation of Positions ---
# Equally spaced hole centers on radius Rs
hole_angles = np.linspace(0.0, 2 * np.pi, num_balls, endpoint=False)
hole_centers = [(Rs * np.cos(a), Rs * np.sin(a)) for a in hole_angles]

# --- Selection policy: one ball per hole ---
# A ball center must be (i) on ellipse & hypocycloid (true for our intersections)
# and (ii) fully contained in its hole: dist(center, hole_center) ≤ Ru - Rp = f.
# We use a slightly relaxed tolerance to avoid floating-point misses at cardinal angles.

TOL = max(1e-6 * max(Ru, 1.0), 1e-8)
assigned = np.zeros(len(intersection_points), dtype=bool)
ball_assignments = []  # list of (i_hole, x, y)

for i, (cx_h, cy_h) in enumerate(hole_centers):
    A_i = hole_angles[i]
    best = None
    for j, ((x, y), theta_j) in enumerate(zip(intersection_points, inter_angles)):
        if assigned[j]:
            continue
        d = np.hypot(x - cx_h, y - cy_h)
        if d <= (Ru - Rp + TOL):  # ball entirely inside this hole
            dA = abs(((theta_j - A_i + np.pi) % (2 * np.pi)) - np.pi)
            key = (dA, d)  # prioritize angular alignment, then proximity to hole center
            if (best is None) or (key < best[0]):
                best = (key, j, x, y)
    if best is not None:
        _, j_sel, bx, by = best
        assigned[j_sel] = True
        ball_assignments.append((i, bx, by))
    else:
        print(f"Warning: No valid intersection inside hole {i} (angle {A_i:.3f} rad). Hole will be empty.")

print(f"Assigned {len(ball_assignments)} balls to {num_balls} holes.")

# --- Plotting ---
fig, ax = plt.subplots(figsize=(8, 8))
ax.plot(x_ell, y_ell, label="Elliptic groove (driving disc)")
# plot hypocycloid for visualization
tt_plot = np.linspace(0, 2 * np.pi, 4001)
xh, yh = hypo_xy(tt_plot)
ax.plot(xh, yh, color='moccasin', linestyle='-', label=f"Hypocycloid Centerline (k={k})")

# Show all holes or every other
if SHOW_EVERY_OTHER:
    divideby = 2
else:
    divideby = 1

# Draw holes (equally spaced, dashed)
for i, (cx_h, cy_h) in enumerate(hole_centers):
    if i % divideby == 0:  # Only plot every other hole
        hole = plt.Circle((cx_h, cy_h), Ru, facecolor='none', edgecolor='gray', linestyle='--')
        ax.add_patch(hole)

# Draw balls (one per hole at a qualified intersection)
for (i, bx, by) in ball_assignments:
    if i % divideby == 0:  # Only plot every other ball
        ball = plt.Circle((bx, by), Rp, facecolor='skyblue', edgecolor='black', zorder=10, alpha=0.5)
        ax.add_patch(ball)
        ax.plot(bx, by, 'ro', markersize=2, zorder=11)

############################################################
############################################################
# --- DXF EXPORT ---
# This section uses the function from the dxf_saver.py file
# to save the generated geometry.

from dxf_saver import save_geometry_to_dxf

print("\n--- Exporting to DXF ---")

# 1. Prepare the curve data for the ellipse and hypocycloid
# The zip() function is a clean way to pair the x and y coordinate arrays.
ellipse_points = list(zip(x_ell, y_ell))
hypocycloid_points = list(zip(xh, yh)) # Use the 'xh, yh' from the plotting section

curves_data = {
    'ellipse': ellipse_points,
    'hypocycloid': hypocycloid_points
}

# 2. Prepare the circle data for the retaining holes and balls
retaining_holes_data = [(center[0], center[1], Ru) for center in hole_centers]
# We only want to export the assigned ball positions
ball_data = [(bx, by, Rp) for (i, bx, by) in ball_assignments]

circles_data = {
    'retaining_holes': retaining_holes_data,
    'balls': ball_data
}

# 3. Define a filename and call the export function
output_filename = f"gears_k{k}_Rx{int(Rx)}.dxf"
#save_geometry_to_dxf(output_filename, curves=curves_data, circles=circles_data)

save_geometry_to_dxf(
    output_filename,
    curves=curves_data,
    circles=circles_data,
    ball_radius=Rp,
    clearance=clearance,
    hypocycloid_layer_name="hypocycloid",             # matches your dict key
    offset_layer_plus="hypocycloid_offset_plus",      # rename if you prefer
    offset_layer_minus="hypocycloid_offset_minus",
)

# --- End of DXF Export ---
############################################################
############################################################


# Legend entries for the balls and holes
ax.plot([], [], 'o', color='skyblue', markeredgecolor='black', markersize=10, label='Balls', alpha=0.5)
ax.plot([], [], 'o', mfc='none', mec='gray', mew=1, ls='--', markersize=15, label='Retaining Holes')

ax.axis('equal')
ax.grid(True, linestyle='--', linewidth=0.5)
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_title(f"Groove Shapes for k={k}, Rx={Rx}, {num_balls} Balls with {Rp*2} Diam")
ax.legend()
plt.tight_layout()
plt.show()
