import numpy as np
# import matplotlib
# import matplotlib.pyplot as plt
import plotly.graph_objects as go
from track import pyplot_track

import sys, os
BASEPATH = os.path.abspath(__file__).split("plots/", 1)[0]+"plots/"
ROOTPATH = os.path.abspath(__file__).split("plots/", 1)[0]+".."
sys.path += [BASEPATH]
sys.path += [ROOTPATH]

track_file = ROOTPATH + "/resources/racetrack/straight_line.yaml"
traj_file = ROOTPATH + "/resources/trajectory/crazyflie_traj.csv"
arclen_traj_file = ROOTPATH + "/resources/trajectory/crazyflie_arclen_traj.csv"

# 1. Load Optimized Trajectory (CSV)
data_ocp = np.genfromtxt(traj_file, dtype=float, delimiter=',', names=True)
p_x, p_y, p_z = data_ocp['p_x'], data_ocp['p_y'], data_ocp['p_z']
vs_norm = np.sqrt(data_ocp['v_x']**2 + data_ocp['v_y']**2 + data_ocp['v_z']**2)

# 2. Load Reference Path (Tab-separated s, x, y, z)
# 'skiprows=1' if your file has a header like "s x(s) y(s) z(s)"
try:
    data_ref = np.loadtxt(arclen_traj_file, skiprows=1, delimiter=None) # delimiter=None handles any whitespace (tabs/spaces)

    s_ref = data_ref[:, 0]
    x_ref = data_ref[:, 1]
    y_ref = data_ref[:, 2]
    z_ref = data_ref[:, 3]
except Exception as e:
    print(f"Could not load reference file: {e}")
    x_ref, y_ref, z_ref = None, None, None

# --- Main Plotting Execution ---
fig = go.Figure()

# A. Plot Optimized Trajectory (Gradient by Speed)
fig.add_trace(go.Scatter3d(
    x=p_x, y=p_y, z=p_z,
    mode='lines',
    line=dict(
        color=vs_norm, 
        colorscale='Viridis', 
        width=6, 
        colorbar=dict(title="Speed [m/s]", thickness=15)
    ),
    name='Optimized (OCP)'
))

# B. Plot Reference Path (Dashed gray line)
if x_ref is not None:
    fig.add_trace(go.Scatter3d(
        x=x_ref, y=y_ref, z=z_ref,
        mode='lines',
        line=dict(color='gray', width=3, dash='dash'),
        name='Reference Path'
    ))

# C. Plot Gates (Using your ported function)
pyplot_track(fig, track_file)

# --- Final Layout ---
fig.update_layout(
    scene=dict(
        aspectmode='data', 
        xaxis_title='X [m]', 
        yaxis_title='Y [m]', 
        zaxis_title='Z [m]',
        camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)) # Nice starting angle
    ),
    title="Comparison: Reference Path vs. Optimized Trajectory",
    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
)

fig.show()

