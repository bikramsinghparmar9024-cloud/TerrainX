# app.py
# Hard-constrained terrain-aware route planner
# Enemy & restricted zones are STRICTLY forbidden

import streamlit as st
import numpy as np
import cv2
import math
import heapq
import plotly.graph_objects as go
import plotly.express as px
from scipy.ndimage import gaussian_filter

# -------------------------------
# UI Configuration (Must be first)
# -------------------------------
st.set_page_config(
    layout="wide", 
    page_title="Tactical Route Planner",
    page_icon="🛡️",
    initial_sidebar_state="expanded"
)

# Custom CSS for a "Tactical" look
st.markdown("""
<style>
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
        border-left: 5px solid #1f77b4;
    }
    [data-testid="stSidebar"] {
        background-color: #f8f9fa;
    }
    h1, h2, h3 {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------
# Image loading
# -------------------------------

def load_image_from_bytes(file_bytes):
    arr = np.frombuffer(file_bytes.getvalue(), np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError("Invalid image")

    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.ndim == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    elif img.ndim == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        raise ValueError("Unsupported image format")

    return cv2.resize(img, (256, 256))


# -------------------------------
# Terrain & Risk Calculation
# -------------------------------

def compute_slope(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    slope = np.sqrt(gx**2 + gy**2)
    return cv2.normalize(slope, None, 0.0, 1.0, cv2.NORM_MINMAX)


def compute_combined_risk(slope):
    """
    Combines natural terrain slope risk. 
    Structure detection has been removed.
    """
    # Base slope risk
    risk = np.zeros_like(slope)
    risk[slope < 0.10] = 0.15
    risk[(slope >= 0.10) & (slope < 0.30)] = 0.50
    risk[slope >= 0.30] = 1.00
    
    return risk


# -------------------------------
# Forbidden zones
# -------------------------------

def build_forbidden_mask(shape, zones):
    h, w = shape
    mask = np.zeros((h, w), dtype=bool)
    Y, X = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")

    for x, y, r in zones:
        dist = np.sqrt((X - x)**2 + (Y - y)**2)
        mask |= dist <= r

    return mask


# -------------------------------
# A* Algorithm
# -------------------------------

def heuristic(a, b):
    dx = abs(a[0] - b[0])
    dy = abs(a[1] - b[1])
    return max(dx, dy) + (math.sqrt(2) - 1) * min(dx, dy)


def astar(risk, forbidden, start, goal, w_risk, w_dist):
    h, w = risk.shape
    neighbors = [(-1,0),(1,0),(0,-1),(0,1),
                 (-1,-1),(-1,1),(1,-1),(1,1)]

    if forbidden[start[1], start[0]] or forbidden[goal[1], goal[0]]:
        return []

    pq = [(0, start)]
    came_from = {}
    g = {start: 0}
    visited = set()

    while pq:
        _, current = heapq.heappop(pq)

        if current == goal:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            return path[::-1]

        if current in visited:
            continue
        visited.add(current)

        cx, cy = current
        
        for dx, dy in neighbors:
            nx, ny = cx + dx, cy + dy
            if not (0 <= nx < w and 0 <= ny < h):
                continue
            if forbidden[ny, nx]:
                continue

            next_risk = risk[ny, nx]
            step = math.sqrt(2) if dx and dy else 1
            
            base_cost = w_dist * step
            
            # Non-linear risk penalties
            if next_risk > 0.9:  # Extreme Steep
                risk_cost = w_risk * next_risk * step * 15.0
            elif next_risk > 0.7: # Red
                risk_cost = w_risk * next_risk * step * 10.0
            elif next_risk > 0.5: # Orange
                risk_cost = w_risk * next_risk * step * 3.0
            elif next_risk > 0.3: # Yellow
                risk_cost = w_risk * next_risk * step * 1.5
            else:
                risk_cost = w_risk * next_risk * step * 0.5
            
            # Turn penalty
            turn_cost = 0
            if current in came_from:
                prev = came_from[current]
                px, py = prev
                v1 = (cx - px, cy - py)
                v2 = (nx - cx, ny - cy)
                norm1 = math.sqrt(v1[0]**2 + v1[1]**2)
                norm2 = math.sqrt(v2[0]**2 + v2[1]**2)
                if norm1 > 0 and norm2 > 0:
                    dot = v1[0]*v2[0] + v1[1]*v2[1]
                    cos_angle = dot / (norm1 * norm2)
                    if cos_angle < 0.8:
                        turn_cost = (0.8 - cos_angle) * 0.5
            
            total_cost = g[current] + base_cost + risk_cost + turn_cost

            if total_cost < g.get((nx, ny), 1e9):
                g[(nx, ny)] = total_cost
                came_from[(nx, ny)] = current
                f = total_cost + heuristic((nx, ny), goal)
                heapq.heappush(pq, (f, (nx, ny)))

    return []


# -------------------------------
# Analysis
# -------------------------------

def analyze_route(path, risk, forbidden):
    if len(path) < 2:
        return 0, 0, "Invalid", 0, 0, 0, 0, 0, 0, 0
    
    total_distance = 0.0
    total_risk = 0.0
    max_risk = 0.0
    red_zone_count = 0
    red_zone_distance = 0.0
    sharp_turns = 0
    
    for i in range(1, len(path)):
        x1, y1 = path[i-1]
        x2, y2 = path[i]
        
        dx = x2 - x1
        dy = y2 - y1
        step_dist = math.sqrt(dx*dx + dy*dy)
        total_distance += step_dist
        
        risk_value = risk[y2, x2]
        max_risk = max(max_risk, risk_value)
        
        if risk_value > 0.7:
            weighted_risk = risk_value * step_dist * 3.0
            red_zone_count += 1
            red_zone_distance += step_dist
        elif risk_value > 0.5:
            weighted_risk = risk_value * step_dist * 1.5
        else:
            weighted_risk = risk_value * step_dist
        
        total_risk += weighted_risk
        
        if i > 1:
            x0, y0 = path[i-2]
            v1 = (x1 - x0, y1 - y0)
            v2 = (x2 - x1, y2 - y1)
            norm1 = math.sqrt(v1[0]**2 + v1[1]**2)
            norm2 = math.sqrt(v2[0]**2 + v2[1]**2)
            if norm1 > 0 and norm2 > 0:
                dot = v1[0]*v2[0] + v1[1]*v2[1]
                cos_angle = dot / (norm1 * norm2)
                if cos_angle < 0.3:
                    sharp_turns += 1
    
    avg_risk = total_risk / total_distance if total_distance > 0 else 0
    
    if red_zone_count > 3 or red_zone_distance > 30 or max_risk > 0.9:
        difficulty = "Extreme"
        difficulty_color = "red"
    elif red_zone_count > 1 or red_zone_distance > 15 or max_risk > 0.7:
        difficulty = "High"
        difficulty_color = "orange"
    elif avg_risk > 0.5 or max_risk > 0.5:
        difficulty = "Medium"
        difficulty_color = "yellow"
    else:
        difficulty = "Low"
        difficulty_color = "green"
    
    safety_score = 100.0
    safety_score -= red_zone_count * 15
    safety_score -= red_zone_distance * 0.5
    if max_risk > 0.8: safety_score -= 30
    elif max_risk > 0.6: safety_score -= 15
    safety_score -= sharp_turns * 5
    safety_score = max(0, safety_score)
    
    if safety_score >= 80: safety_rating = "Excellent"
    elif safety_score >= 60: safety_rating = "Good"
    elif safety_score >= 40: safety_rating = "Fair"
    else: safety_rating = "Dangerous"
    
    return (total_distance, total_risk, difficulty, difficulty_color,
            safety_score, safety_rating, red_zone_count, red_zone_distance,
            max_risk, sharp_turns)


# -------------------------------
# Visualization Helpers (UPDATED to Plotly for Zoom/Pan)
# -------------------------------

def create_interactive_tactical_map(image, path, enemies, restricted, sx, sy, ex, ey):
    """
    Creates a Zoomable/Pannable Plotly 2D Map.
    """
    # Create base image figure using px.imshow
    fig = px.imshow(image)
    
    # Add Path trace
    if path:
        px_coords, py_coords = zip(*path)
        fig.add_trace(go.Scatter(
            x=px_coords, y=py_coords, 
            mode='lines', 
            line=dict(color='#00FF00', width=4), 
            name='Optimal Route',
            hoverinfo='skip'
        ))
    
    # Add Start/End Points
    fig.add_trace(go.Scatter(
        x=[sx], y=[sy], 
        mode='markers', 
        marker=dict(size=14, color='#00FF00', symbol='circle', line=dict(color='white', width=2)), 
        name='Start'
    ))
    fig.add_trace(go.Scatter(
        x=[ex], y=[ey], 
        mode='markers', 
        marker=dict(size=14, color='#FF0000', symbol='x', line=dict(color='white', width=2)), 
        name='Extraction'
    ))

    # Add Zones as Shapes
    shapes = []
    # Enemy Zones (Red)
    for x, y, r in enemies:
        shapes.append(dict(type="circle", x0=x-r, y0=y-r, x1=x+r, y1=y+r, line_color="red", fillcolor="rgba(255, 0, 0, 0.3)"))
    # Restricted Zones (Purple)
    for x, y, r in restricted:
        shapes.append(dict(type="circle", x0=x-r, y0=y-r, x1=x+r, y1=y+r, line_color="purple", fillcolor="rgba(128, 0, 128, 0.3)"))

    fig.update_layout(
        shapes=shapes,
        title="Tactical Map (Interactive - Scroll to Zoom)",
        margin=dict(l=0, r=0, t=30, b=0),
        height=600,
        dragmode='pan', # Set default drag mode to pan
        xaxis=dict(range=[0, 256], visible=False),
        yaxis=dict(range=[256, 0], visible=False) # Invert Y to match image coords
    )
    return fig

def create_interactive_heatmap(risk, path):
    """
    Creates a Zoomable/Pannable Plotly Heatmap.
    """
    # Smooth the risk for better visuals
    risk_smooth = gaussian_filter(risk, sigma=2)
    
    fig = go.Figure(data=go.Heatmap(
        z=risk_smooth,
        colorscale='RdYlGn_r', # Red (High Risk) to Green (Low Risk)
        opacity=0.9,
        showscale=True
    ))
    
    # Overlay Path
    if path:
        px_coords, py_coords = zip(*path)
        fig.add_trace(go.Scatter(
            x=px_coords, y=py_coords, 
            mode='lines', 
            line=dict(color='black', width=3), 
            name='Route'
        ))

    fig.update_layout(
        title="Risk Heatmap Analysis",
        margin=dict(l=0, r=0, t=30, b=0),
        height=600,
        yaxis=dict(autorange='reversed')
    )
    return fig


# -------------------------------
# UI LAYOUT
# -------------------------------

st.title("🛡️ AI Tactical Route Planner")
st.markdown("---")

# --- SIDEBAR: Mission Control ---
with st.sidebar:
    st.header("🎛️ Mission Control")
    
    uploaded = st.file_uploader("📂 Upload Terrain Map", ["png", "jpg", "jpeg"])
    
    # Handle image loading - REMOVED SAMPLE MAP FEATURE
    if uploaded is None:
        st.warning("⚠️ Please upload a terrain map to begin planning.")
        st.stop()
    
    image = load_image_from_bytes(uploaded)
    
    with st.expander("📍 Route Coordinates", expanded=True):
        st.caption("Set Start and Extraction points")
        sx = st.slider("Start X", 0, 255, 10)
        sy = st.slider("Start Y", 0, 255, 10)
        st.markdown("---")
        ex = st.slider("End X", 0, 255, 245)
        ey = st.slider("End Y", 0, 255, 245)

    with st.expander("⚠️ Threat Intelligence", expanded=True):
        st.caption("Define Known Hostile Areas")
        enemy_n = st.slider("Active Enemy Zones", 0, 5, 2)
        enemies = []
        for i in range(enemy_n):
            st.markdown(f"**Enemy Zone {i+1}**")
            col_e1, col_e2 = st.columns(2)
            x_e = col_e1.number_input(f"X", 0, 255, 128+i*20, key=f"ex_{i}")
            y_e = col_e2.number_input(f"Y", 0, 255, 128+i*20, key=f"ey_{i}")
            r_e = st.slider(f"Radius (m)", 10, 100, 40, key=f"er_{i}")
            enemies.append((x_e, y_e, r_e))

        st.markdown("---")
        res_n = st.slider("Restricted Zones", 0, 5, 1)
        restricted = []
        for i in range(res_n):
            st.markdown(f"**Restricted Zone {i+1}**")
            col_r1, col_r2 = st.columns(2)
            x_r = col_r1.number_input(f"X", 0, 255, 64, key=f"rx_{i}")
            y_r = col_r2.number_input(f"Y", 0, 255, 64, key=f"ry_{i}")
            r_r = st.slider(f"Radius (m)", 10, 100, 40, key=f"rr_{i}")
            restricted.append((x_r, y_r, r_r))

    with st.expander("⚙️ Tactics & Systems", expanded=False):
        risk_w = st.slider("Safety Priority", 0.0, 1.0, 0.7, help="Higher value prefers safer but longer routes")
        st.caption("Computer Vision detection disabled.")

# -------------------------------
# PROCESSING (Logic Updated)
# -------------------------------
slope = compute_slope(image)

# Structure detection logic removed
struct_mask = np.zeros(image.shape[:2], dtype=np.uint8) # Empty mask to satisfy 3D plotter

risk = compute_combined_risk(slope)

forbidden = build_forbidden_mask(risk.shape, enemies + restricted)
path = astar(risk, forbidden, (sx,sy), (ex,ey), risk_w, 1-risk_w)

if not path:
    st.error("❌ MISSION ABORTED: No valid route exists. Destination is blocked or unreachable.")
    st.stop()

# Analyze Logic
(dist, total_risk, difficulty, diff_color, safety_score, safety_rating,
 red_zones, red_dist, max_risk, sharp_turns) = analyze_route(path, risk, forbidden)

px_coords, py_coords = zip(*path)


# -------------------------------
# RESULTS DISPLAY
# -------------------------------

# 1. Mission Briefing (Metrics)
st.subheader("📊 Mission Briefing")
m1, m2, m3, m4 = st.columns(4)

with m1:
    st.metric(label="Total Distance", value=f"{dist:.1f} m")
with m2:
    st.metric(label="Safety Score", value=f"{safety_score:.0f}/100", delta=safety_rating, delta_color="normal" if safety_score > 60 else "inverse")
with m3:
    st.metric(label="Risk Level", value=difficulty, delta="Alert" if difficulty in ["Extreme", "High"] else "Stable", delta_color="inverse")
with m4:
    st.metric(label="Structures Detected", value="Disabled")

st.markdown("---")

# 2. Tabs for Visualizations
tab1, tab2, tab3, tab4 = st.tabs(["🗺️ 2D Interactive Map", "🔥 Risk Heatmap", "⛰️ 3D Simulation", "👁️ Intel Comparison"])

# --- TAB 1: 2D Interactive Map (New Plotly Version) ---
with tab1:
    st.markdown("Use mouse wheel to **Zoom**, Click & Drag to **Pan**.")
    fig_map = create_interactive_tactical_map(image, path, enemies, restricted, sx, sy, ex, ey)
    st.plotly_chart(fig_map, use_container_width=True)
    
    st.info("ℹ️ **Intel Summary**")
    st.markdown(f"""
    * **Max Risk Encountered:** {max_risk:.2f}
    * **Red Zone Travel:** {red_dist:.1f} m
    * **Sharp Turns:** {sharp_turns}
    """)
    if difficulty in ["High", "Extreme"]:
        st.warning("⚠️ **Advisory:** High risk detected. Consider adjusting route waypoints or reducing enemy zone radius.")
    else:
        st.success("✅ **Advisory:** Route is optimal for current parameters.")


# --- TAB 2: Heatmap (New Plotly Version) ---
with tab2:
    st.markdown("Use mouse wheel to **Zoom**, Click & Drag to **Pan**.")
    fig_heat = create_interactive_heatmap(risk, path)
    st.plotly_chart(fig_heat, use_container_width=True)
    st.caption("Color Gradient: Green (Safe) → Yellow (Caution) → Red (Danger)")

# --- TAB 3: 3D Simulation (Original Plotly) ---
with tab3:
    X, Y = np.meshgrid(np.arange(256), np.arange(256))
    Z = slope * 30

    fig_3d = go.Figure()

    # Surface
    fig_3d.add_surface(
        x=X, y=Y, z=Z, surfacecolor=risk,
        colorscale=[
            [0.0, "darkgreen"], [0.25, "green"], [0.45, "yellow"],
            [0.65, "orange"], [0.85, "red"], [1.0, "darkred"]
        ],
        opacity=0.95, showscale=False
    )

    # Structures
    struct_y, struct_x = np.where(struct_mask > 0)
    if len(struct_x) > 0:
        skip = 2
        fig_3d.add_trace(go.Scatter3d(
            x=struct_x[::skip], y=struct_y[::skip], z=Z[struct_y[::skip], struct_x[::skip]] + 2,
            mode='markers',
            marker=dict(size=2, color='cyan', opacity=0.5),
            name='Structures'
        ))

    # Route
    route_z = [Z[y, x] + 2 for x, y in path]
    fig_3d.add_trace(go.Scatter3d(
        x=px_coords, y=py_coords, z=route_z,
        mode="lines+markers", line=dict(color="black", width=5),
        marker=dict(size=3, color="black"), name="Route"
    ))

    # Start/End
    fig_3d.add_trace(go.Scatter3d(x=[sx], y=[sy], z=[Z[sy, sx]+4], mode="markers", marker=dict(size=8, color="green"), name="Start"))
    fig_3d.add_trace(go.Scatter3d(x=[ex], y=[ey], z=[Z[ey, ex]+4], mode="markers", marker=dict(size=8, color="red"), name="End"))

    # Zones
    for x, y, r in enemies + restricted:
        fig_3d.add_trace(go.Scatter3d(
            x=[x], y=[y], z=[Z[y, x] + 3],
            mode="markers",
            marker=dict(size=r/2, color="rgba(139,0,0,0.3)", symbol="circle", line=dict(color="darkred", width=2)),
            showlegend=False
        ))

    fig_3d.update_layout(
        scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Elev", aspectmode="data"),
        margin=dict(l=0, r=0, t=10, b=0),
        height=600
    )
    st.plotly_chart(fig_3d, use_container_width=True)

# --- TAB 4: Intel Comparison (New Feature) ---
with tab4:
    st.subheader("👁️ Intelligence Data Comparison")
    st.markdown("Side-by-side analysis of **Original Terrain Map** vs. **Computed Risk Heatmap** vs. **3D Simulation**.")
    
    col_c1, col_c2 = st.columns(2)
    
    with col_c1:
        st.markdown("**1. Interactive Tactical Map**")
        # Reuse fig_map
        st.plotly_chart(fig_map, use_container_width=True, key="comp_map")
    
    with col_c2:
        st.markdown("**2. Risk Heatmap**")
        st.plotly_chart(fig_heat, use_container_width=True, key="comp_heat")
    
    st.markdown("---")
    st.markdown("**3. 3D Terrain Simulation**")
    st.plotly_chart(fig_3d, use_container_width=True, key="comp_3d")