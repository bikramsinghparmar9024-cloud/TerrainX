# app.py
# TerrainX: Tactical Pathfinding System
# Author: [Your Name/Dev Team]
# Notes: Implements A* with custom cost functions for terrain analysis.
#        Now includes Computer Vision for Feature Extraction (Roads, Rivers).
#        Strict avoidance of designated enemy zones.

import streamlit as st
import numpy as np
import cv2
import math
import heapq
import plotly.graph_objects as go
import plotly.express as px
from scipy.ndimage import gaussian_filter
from PIL import Image
import io

# ==========================================
# 1. App Configuration & styling
# ==========================================

# Basic page setup - keeping the layout wide for map visibility
st.set_page_config(
    layout="wide", 
    page_title="TerrainX",
    page_icon="🦅",
    initial_sidebar_state="expanded"
)

# I'm injecting custom CSS here to override Streamlit's default light theme.
# We want a 'Black Ops' / Dark Mode aesthetic. 
# Using specific hex codes for that neon-on-black look.
st.markdown("""
<style>
    /* Main Background - Deep charcoal/black */
    .stApp {
        background-color: #0b0c10;
        color: #c5c6c7;
    }
    
    /* Custom Font styling for the headers to give it that 'terminal' feel */
    h1 {
        font-family: 'Courier New', monospace;
        color: #66fcf1;
        text-transform: uppercase;
        letter-spacing: 4px;
        text-shadow: 0px 0px 10px rgba(102, 252, 241, 0.5);
    }
    h2, h3 {
        font-family: 'Segoe UI', sans-serif;
        color: #45a29e;
    }
    
    /* Sidebar needs to match the main theme */
    [data-testid="stSidebar"] {
        background-color: #1f2833;
        border-right: 1px solid #45a29e;
    }
    
    /* Styling the metrics to look like HUD elements */
    div[data-testid="metric-container"] {
        background-color: #1f2833;
        border: 1px solid #45a29e;
        padding: 15px;
        border-radius: 4px;
        box-shadow: 0 0 10px rgba(69, 162, 158, 0.2);
    }
    [data-testid="stMetricLabel"] {
        color: #66fcf1 !important;
        font-size: 0.9rem;
    }
    [data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-family: 'Courier New', monospace;
    }
    
    /* Tweaking slider colors to match the neon cyan theme */
    .stSlider > div > div > div > div {
        background-color: #66fcf1;
    }
    
    /* Tab styling is tricky in Streamlit, manual overrides needed */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #1f2833;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
        color: #c5c6c7;
    }
    .stTabs [aria-selected="true"] {
        background-color: #0b0c10;
        border-bottom: 2px solid #66fcf1;
        color: #66fcf1;
    }
    
    /* Briefing card containers */
    .briefing-card {
        background-color: #1f2833;
        padding: 20px;
        border-radius: 5px;
        border-left: 3px solid #66fcf1;
        margin-bottom: 20px;
    }
    
    /* Sample images styling */
    .sample-image-container {
        background-color: #1f2833;
        padding: 15px;
        border-radius: 5px;
        border: 1px solid #45a29e;
        margin-bottom: 20px;
    }
    .sample-image-title {
        color: #66fcf1;
        font-weight: bold;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# Helper Functions & Feature Extraction
# ==========================================

def load_image_from_bytes(file_bytes):
    """
    Helper to safely decode the uploaded image stream into an OpenCV array.
    We resize everything to 256x256 to keep the pathfinding fast.
    """
    # Convert file buffer to numpy array
    arr = np.frombuffer(file_bytes.getvalue(), np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    
    if img is None:
        raise ValueError("Image decode failed. File might be corrupted.")

    # Handle different channel counts (grayscale, RGBA, etc.)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.ndim == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    elif img.ndim == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        raise ValueError("Unsupported image format")

    return cv2.resize(img, (256, 256))

# ... [rest of the helper functions remain exactly the same] ...
def compute_slope(image):
    """
    Estimates terrain steepness using Sobel operators.
    Returns a normalized value (0.0 flat -> 1.0 steep).
    """
    # Convert to grayscale first
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    
    # Calculate gradients in X and Y
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    
    # Magnitude of gradient
    slope = np.sqrt(gx**2 + gy**2)
    
    # Normalize for easier processing later
    return cv2.normalize(slope, None, 0.0, 1.0, cv2.NORM_MINMAX)

def detect_terrain_features(image):
    """
    Performs Computer Vision segmentation to identify distinct terrain types.
    Returns a dictionary of boolean masks.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    
    # --- 1. Water / Rivers (Blue range) ---
    lower_blue = np.array([90, 50, 50])
    upper_blue = np.array([130, 255, 255])
    mask_water = cv2.inRange(hsv, lower_blue, upper_blue)
    
    # --- 2. Vegetation / Trees (Green range) ---
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])
    mask_trees = cv2.inRange(hsv, lower_green, upper_green)
    
    # --- 3. Roads (Low Saturation, specific Value range, linear structure) ---
    # This assumes asphalt/dirt roads are greyish/brownish
    lower_road = np.array([0, 0, 50])
    upper_road = np.array([180, 50, 220])
    mask_road_raw = cv2.inRange(hsv, lower_road, upper_road)
    # Remove water areas from road mask
    mask_road_raw = cv2.bitwise_and(mask_road_raw, cv2.bitwise_not(mask_water))
    mask_road_raw = cv2.bitwise_and(mask_road_raw, cv2.bitwise_not(mask_trees))
    
    # Morphological operations to clean up road noise (roads should be connected)
    kernel = np.ones((3,3), np.uint8)
    mask_road = cv2.morphologyEx(mask_road_raw, cv2.MORPH_OPEN, kernel)
    
    # Return bool arrays for easier math later
    return {
        "water": mask_water > 0,
        "trees": mask_trees > 0,
        "roads": mask_road > 0
    }

def compute_enhanced_risk(slope, features):
    """
    Calculates cost map based on slope AND detected terrain features.
    Logic:
    - Roads: Low Cost (Preferred Route)
    - Water: Extreme Cost (Non-traversable)
    - Trees: Medium Cost (Concealment but slow)
    """
    # Base risk from slope
    risk = np.zeros_like(slope)
    
    # 1. Base Slope Risk
    risk[slope < 0.10] = 0.20
    risk[(slope >= 0.10) & (slope < 0.30)] = 0.50
    risk[slope >= 0.30] = 0.90
    
    # 2. Apply Feature Modifiers
    
    # Vegetation adds some difficulty (bushwhacking)
    risk[features["trees"]] = np.maximum(risk[features["trees"]], 0.4)
    
    # Water is nearly impassable
    risk[features["water"]] = 1.0 
    
    # ROADS REDUCE RISK (The Prompt Requirement)
    # We purposefully lower the cost here to encourage the A* algorithm 
    # to "snap" to the road network.
    risk[features["roads"]] = 0.1 # Very low cost
    
    return risk

def build_forbidden_mask(shape, zones):
    """
    Creates a boolean mask where True = Do Not Enter.
    Combines enemy zones and restricted fly zones.
    """
    h, w = shape
    mask = np.zeros((h, w), dtype=bool)
    Y, X = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")

    for x, y, r in zones:
        # Simple Euclidean distance check for circular zones
        dist = np.sqrt((X - x)**2 + (Y - y)**2)
        mask |= dist <= r

    return mask

# ==========================================
# 3. Pathfinding (A* Logic)
# ==========================================

def heuristic(a, b):
    # Octile distance is better than Manhattan for 8-direction movement
    dx = abs(a[0] - b[0])
    dy = abs(a[1] - b[1])
    return max(dx, dy) + (math.sqrt(2) - 1) * min(dx, dy)

def astar(risk_map, forbidden_mask, start, goal, weight_risk, weight_dist):
    """
    Standard A* implementation but with a weighted cost function 
    that balances distance vs. risk exposure.
    """
    h, w = risk_map.shape
    
    # 8-connected grid (can move diagonally)
    neighbors = [(-1,0),(1,0),(0,-1),(0,1),
                 (-1,-1),(-1,1),(1,-1),(1,1)]

    # Sanity check: Start or End shouldn't be inside a wall
    if forbidden_mask[start[1], start[0]] or forbidden_mask[goal[1], goal[0]]:
        return []

    # Priority queue: (f_score, coordinates)
    pq = [(0, start)]
    came_from = {}
    g_scores = {start: 0}
    visited = set()

    while pq:
        _, current = heapq.heappop(pq)

        if current == goal:
            # Reconstruct path backwards
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            return path[::-1] # Reverse it

        if current in visited:
            continue
        visited.add(current)

        cx, cy = current
        
        for dx, dy in neighbors:
            nx, ny = cx + dx, cy + dy
            
            # Check bounds
            if not (0 <= nx < w and 0 <= ny < h):
                continue
            
            # Check restricted zones (hard constraint)
            if forbidden_mask[ny, nx]:
                continue

            next_risk_val = risk_map[ny, nx]
            
            # Distance cost: 1 for cardinal, sqrt(2) for diagonal
            step_cost = math.sqrt(2) if dx and dy else 1
            base_movement_cost = weight_dist * step_cost
            
            # --- Dynamic Risk Penalties ---
            # We want to aggressively punish high-risk steps
            if next_risk_val >= 1.0: # Impassable (Water)
                 risk_penalty = 1000.0 # Effectively infinite
            elif next_risk_val > 0.9: 
                risk_penalty = weight_risk * next_risk_val * step_cost * 15.0 # Very high penalty
            elif next_risk_val > 0.7: 
                risk_penalty = weight_risk * next_risk_val * step_cost * 10.0
            elif next_risk_val > 0.5: 
                risk_penalty = weight_risk * next_risk_val * step_cost * 3.0
            else:
                # Minimal penalty for safe terrain / Roads
                risk_penalty = weight_risk * next_risk_val * step_cost * 0.5
            
            # --- Turn Penalty ---
            # Prevent jagged, zig-zag paths by penalizing sharp turns
            turn_penalty = 0
            if current in came_from:
                prev = came_from[current]
                px, py = prev
                
                # Vectors
                v1 = (cx - px, cy - py)
                v2 = (nx - cx, ny - cy)
                
                # Calculate angle cosine
                norm1 = math.sqrt(v1[0]**2 + v1[1]**2)
                norm2 = math.sqrt(v2[0]**2 + v2[1]**2)
                
                if norm1 > 0 and norm2 > 0:
                    dot = v1[0]*v2[0] + v1[1]*v2[1]
                    cos_angle = dot / (norm1 * norm2)
                    
                    # If angle is sharp (cos_angle < 0.8), add cost
                    if cos_angle < 0.8:
                        turn_penalty = (0.8 - cos_angle) * 0.5
            
            total_new_cost = g_scores[current] + base_movement_cost + risk_penalty + turn_penalty

            # Update if we found a better path
            if total_new_cost < g_scores.get((nx, ny), 1e9):
                g_scores[(nx, ny)] = total_new_cost
                came_from[(nx, ny)] = current
                
                # F = G + H
                f_score = total_new_cost + heuristic((nx, ny), goal)
                heapq.heappush(pq, (f_score, (nx, ny)))

    # If we get here, no path was found
    return []

# ==========================================
# 4. Result Analysis & Visualization
# ==========================================

def analyze_route(path, risk_map, forbidden_mask):
    """
    Post-processing to generate mission stats.
    Calculates safety scores based on how much time we spend in red zones.
    """
    if len(path) < 2:
        return 0, 0, "Invalid", 0, 0, 0, 0, 0, 0, 0
    
    total_dist = 0.0
    accumulated_risk = 0.0
    max_risk_encountered = 0.0
    
    red_zone_count = 0
    red_zone_meters = 0.0
    sharp_turn_count = 0
    
    for i in range(1, len(path)):
        x1, y1 = path[i-1]
        x2, y2 = path[i]
        
        # Euclidean step distance
        step_d = math.sqrt((x2-x1)**2 + (y2-y1)**2)
        total_dist += step_d
        
        r_val = risk_map[y2, x2]
        max_risk_encountered = max(max_risk_encountered, r_val)
        
        # Weight the risk contribution
        if r_val > 0.7:
            accumulated_risk += r_val * step_d * 3.0
            red_zone_count += 1
            red_zone_meters += step_d
        elif r_val > 0.5:
            accumulated_risk += r_val * step_d * 1.5
        else:
            accumulated_risk += r_val * step_d
        
        # Turn detection
        if i > 1:
            x0, y0 = path[i-2]
            v1 = (x1 - x0, y1 - y0)
            v2 = (x2 - x1, y2 - y1)
            # Dot product for angle again...
            # Note: Could refactor this into a helper since we use it in A* too
            n1 = math.sqrt(v1[0]**2 + v1[1]**2)
            n2 = math.sqrt(v2[0]**2 + v2[1]**2)
            if n1 > 0 and n2 > 0:
                val = (v1[0]*v2[0] + v1[1]*v2[1]) / (n1*n2)
                if val < 0.3: # Threshold for "sharp"
                    sharp_turn_count += 1
    
    # Determine Mission Difficulty
    if red_zone_count > 3 or red_zone_meters > 30 or max_risk_encountered > 0.9:
        diff_label = "Extreme"
        diff_color = "#FF0000"
    elif red_zone_count > 1 or red_zone_meters > 15 or max_risk_encountered > 0.7:
        diff_label = "High"
        diff_color = "#FFA500"
    elif max_risk_encountered > 0.5:
        diff_label = "Medium"
        diff_color = "#FFFF00"
    else:
        diff_label = "Low"
        diff_color = "#00FF00"
    
    # Calculate a readable 0-100 score
    score = 100.0
    score -= red_zone_count * 15
    score -= red_zone_meters * 0.5
    if max_risk_encountered > 0.8: score -= 30
    elif max_risk_encountered > 0.6: score -= 15
    score -= sharp_turn_count * 5
    score = max(0, score)
    
    # Rating text
    if score >= 80: rating = "Excellent"
    elif score >= 60: rating = "Good"
    elif score >= 40: rating = "Fair"
    else: rating = "Dangerous"
    
    return (total_dist, accumulated_risk, diff_label, diff_color,
            score, rating, red_zone_count, red_zone_meters,
            max_risk_encountered, sharp_turn_count)

def create_interactive_tactical_map(img, path, enemies, restricted, sx, sy, ex, ey):
    """ Builds the main 2D Plotly map with zoom/pan enabled. """
    fig = px.imshow(img)
    
    # Draw path if we have one
    if path:
        xs, ys = zip(*path)
        fig.add_trace(go.Scatter(
            x=xs, y=ys, 
            mode='lines', 
            line=dict(color='#66fcf1', width=3, dash='solid'), 
            name='Optimal Route',
            hoverinfo='skip'
        ))
    
    # Markers for Start/End
    fig.add_trace(go.Scatter(
        x=[sx], y=[sy], 
        mode='markers', 
        marker=dict(size=14, color='#66fcf1', symbol='circle', line=dict(color='white', width=1)), 
        name='Start'
    ))
    fig.add_trace(go.Scatter(
        x=[ex], y=[ey], 
        mode='markers', 
        marker=dict(size=14, color='#ff0000', symbol='x', line=dict(color='white', width=1)), 
        name='Extraction'
    ))

    # Draw Zones (Enemies = Red, Restricted = Purple)
    shapes = []
    for x, y, r in enemies:
        shapes.append(dict(type="circle", x0=x-r, y0=y-r, x1=x+r, y1=y+r, 
                           line_color="#ff4444", fillcolor="rgba(255, 0, 0, 0.2)"))
        
    for x, y, r in restricted:
        shapes.append(dict(type="circle", x0=x-r, y0=y-r, x1=x+r, y1=y+r, 
                           line_color="#bd93f9", fillcolor="rgba(189, 147, 249, 0.2)"))

    # Plotly dark template + removal of axes for cleaner look
    fig.update_layout(
        template="plotly_dark",
        shapes=shapes,
        title="Tactical Overview",
        margin=dict(l=0, r=0, t=40, b=0),
        height=600,
        dragmode='pan',
        xaxis=dict(range=[0, 256], visible=False),
        yaxis=dict(range=[256, 0], visible=False),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    return fig

# ==========================================
# Heatmap Viz
# ==========================================

def create_interactive_heatmap(risk_map, path):
    """ Visualizes the cost map (Thermal Scan). """
    # Gaussian blur makes the heatmap look more organic/thermal
    risk_smooth = gaussian_filter(risk_map, sigma=2)

    # Define the custom colorscale to match the 3D view exactly
    # This transitions from deep teal (safe) -> yellow -> red/purple (dangerous)
    custom_terrain_colorscale = [
        [0.0, "#004d40"],   # Deep Teal
        [0.25, "#00695c"], # TealGreen
        [0.45, "#fbc02d"], # Yellow
        [0.65, "#f57f17"], # Orange
        [0.85, "#b71c1c"], # Red
        [1.0, "#880e4f"]   # Deep PurpleRed
    ]
    
    fig = go.Figure(data=go.Heatmap(
        z=risk_smooth,
        colorscale=custom_terrain_colorscale, 
        zmin=0.0, 
        zmax=1.0,
        opacity=1.0,
        showscale=True,
        colorbar=dict(title="Risk Index")
    ))
    
    if path:
        xs, ys = zip(*path)
        fig.add_trace(go.Scatter(
            x=xs, y=ys, 
            mode='lines', 
            line=dict(color='#66fcf1', width=3), 
            name='Route'
        ))

    fig.update_layout(
        template="plotly_dark",
        title="Risk Density Scan",
        margin=dict(l=0, r=0, t=40, b=0),
        height=600,
        yaxis=dict(autorange='reversed', visible=False), 
        xaxis=dict(visible=False),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    return fig

# ==========================================
# 5. Main Application Logic
# ==========================================

def main():
    st.title(" TerrainX")

    # --- Introduction Section ---
    st.markdown("""
    <div style='background: linear-gradient(135deg, #1f2833 0%, #0b0c10 100%); 

<p>TerrainX is a smart routing system that helps find the safest path for a mission. It does not just choose the shortest route, but also avoids steep land and dangerous areas. By focusing on safety and easy movement, TerrainX creates routes that are safer and more suitable for real missions.
 </p>

 <h3> What TerrainX Does </h3>
<p>This system studies how steep or flat the land is using satellite images and turns that information into a clear risk map. It strictly avoids dangerous or restricted areas to keep missions safe. During the mission, it continuously shows live details such as how exposed the route is, how difficult it is, and an overall safety score, making everything easy to understand and monitor.</p>

 TerrainX is useful for organizations that must plan routes in dangerous terrain where shortest paths are not safe paths.

  <h3> Applications </h3>
                
1. Military and security tactical route planning
                
2. Space agencies – launchpad and spaceport site selection

3. Road construction and highway alignment planning

4. Real estate land suitability and investment analysis

5. Disaster response and emergency evacuation routing

6. Autonomous drones and robotic navigation planning

7. Mining and large infrastructure logistics optimization
        
    """, unsafe_allow_html=True)



    # --- Briefing Section ---
    st.markdown("---")
    st.subheader(" Operational Briefing")
    st.markdown("System initialized. Tactical pathfinding active.")

    # Using columns for the briefing cards
    b1, b2, b3 = st.columns(3)
    
    with b1:
        st.markdown("""
        
            <h4> 1. Recon</h4>
            <p>Upload satellite imagery. We detect Roads, Rivers, and Vegetation.</p>
        </div>
        """, unsafe_allow_html=True)

    with b2:
        st.markdown("""
       
            <h4> 2. Designate</h4>
            <p>Set extraction points and mark hostility zones. These are treated as hard constraints.</p>
        </div>
        """, unsafe_allow_html=True)

    with b3:
        st.markdown("""
        
            <h4> 3. Execute</h4>
            <p>Routes prioritize roads. Visualize via 3D models.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    

    # --- Sample Images Section ---
    st.subheader("📸 Sample Images")
    st.markdown("Download and test these terrain samples to explore TerrainX capabilities.")
    
    sample_col1, sample_col2, sample_col3 = st.columns(3)
    
    with sample_col1:
        
        # Display image from ImgBB - using direct link

        st.markdown("[🔗 Download Image 1](https://ibb.co/TqDhG9pc)")
    
    with sample_col2:
        
        # Display image from ImgBB - using direct link
       
        st.markdown("[🔗 Download Image 2](https://ibb.co/0jxmPJbk)")
    
    with sample_col3:
        
        # Display image from ImgBB - using direct link
    
        
        st.markdown("[🔗 Download Image 3](https://ibb.co/JR9j0dSM)")
    
    st.info("💡 **Tip:** Right-click on any download link and save the images, then upload via the sidebar to test TerrainX pathfinding.")
    
    st.markdown("---")


    # --- Sidebar Controls ---
    with st.sidebar:
        st.header(" COMMAND MODULE")
        
        uploaded_file = st.file_uploader("📂 Upload Terrain", ["png", "jpg", "jpeg"])
        
        # Stop execution here if no file is uploaded yet
        if uploaded_file is None:
            st.warning("⚠️ Waiting for satellite feed...")
            st.stop()
        
        # Process image
        try:
            image = load_image_from_bytes(uploaded_file)
        except Exception as e:
            st.error(f"Error loading image: {e}")
            st.stop()
        
        st.markdown("### 📍 Waypoints")
        with st.expander("Coordinate Input", expanded=True):
            col_s1, col_s2 = st.columns(2)
            start_x = col_s1.number_input("Start X", 0, 255, 10)
            start_y = col_s2.number_input("Start Y", 0, 255, 10)
            st.markdown("---")
            col_e1, col_e2 = st.columns(2)
            end_x = col_e1.number_input("End X", 0, 255, 245)
            end_y = col_e2.number_input("End Y", 0, 255, 245)

        st.markdown("### ⚠️ Threat Matrix")
        with st.expander("Hostile Zones", expanded=True):
            # Enemy Inputs
            num_enemies = st.slider("Hostile Units", 0, 5, 2)
            enemy_list = []
            for i in range(num_enemies):
                st.markdown(f"**Unit {i+1}**")
                c1, c2 = st.columns(2)
                e_x = c1.number_input(f"X", 0, 255, 128+i*20, key=f"ex_{i}")
                e_y = c2.number_input(f"Y", 0, 255, 128+i*20, key=f"ey_{i}")
                e_r = st.slider(f"Radius", 10, 100, 40, key=f"er_{i}")
                enemy_list.append((e_x, e_y, e_r))

            st.markdown("---")
            # Restricted Zone Inputs
            num_restricted = st.slider("No-Fly Zones", 0, 5, 1)
            restricted_list = []
            for i in range(num_restricted):
                st.markdown(f"**Zone {i+1}**")
                c1, c2 = st.columns(2)
                r_x = c1.number_input(f"X", 0, 255, 64, key=f"rx_{i}")
                r_y = c2.number_input(f"Y", 0, 255, 64, key=f"ry_{i}")
                r_r = st.slider(f"Radius", 10, 100, 40, key=f"rr_{i}")
                restricted_list.append((r_x, r_y, r_r))

        st.markdown("### ⚙️ Algorithm Params")
        # Gives the user control over the greedy vs safe behavior of A*
        risk_weight = st.slider("Safety Heuristic Weight", 0.0, 1.0, 0.7)

    # --- Processing ---
    
    # 1. Compute Terrain Data & Features
    slope_map = compute_slope(image)
    features = detect_terrain_features(image) # Detects Roads, Water, Trees
    
    # 2. Compute Enhanced Risk Map (Prioritize Roads, Avoid Water)
    risk_map = compute_enhanced_risk(slope_map, features)
    
    # 2. Build Masks
    forbidden_mask = build_forbidden_mask(risk_map.shape, enemy_list + restricted_list)
    
    # 3. Run Pathfinding
    # Note: weight_dist is inverse of risk_weight to balance the two
    path = astar(risk_map, forbidden_mask, (start_x, start_y), (end_x, end_y), 
                 risk_weight, 1.0 - risk_weight)

    if not path:
        st.error("❌ CRITICAL FAILURE: Path blocked. Re-evaluate waypoints or adjust constraints.")
        st.stop()

    # 4. Analyze Results
    stats = analyze_route(path, risk_map, forbidden_mask)
    (dist, tot_risk, diff_lbl, diff_col, safe_score, safe_rating,
     red_cnt, red_meters, max_r, sharps) = stats
    
    path_x, path_y = zip(*path)

    # --- Display Dashboard ---
    
    st.subheader("📊 Live Telemetry")
    m1, m2, m3, m4 = st.columns(4)

    with m1:
        st.metric(label="Traversal Distance", value=f"{dist:.1f} m")
    with m2:
        st.metric(label="Integrity Score", value=f"{safe_score:.0f}%", delta=safe_rating)
    with m3:
        st.metric(label="Threat Level", value=diff_lbl, delta_color="inverse")
    with m4:
        st.metric(label="Satellite Link", value="Active")

    st.markdown("---")

    # Tabs for different views
    t1, t2, t3, t4 = st.tabs(["🗺️ TACTICAL MAP", "🔥 THERMAL SCAN", "⛰️ 3D MODEL", "👁️ INTEL REPORT"])

    # View 1: Main Map
    with t1:
        fig_map = create_interactive_tactical_map(image, path, enemy_list, restricted_list, 
                                                  start_x, start_y, end_x, end_y)
        st.plotly_chart(fig_map, use_container_width=True)
        
        # Dynamic briefing note based on analysis
        st.markdown(f"""
        <div style='background-color: #1f2833; padding: 10px; border-radius: 5px; border-left: 5px solid {diff_col};'>
            <strong>Briefing Note:</strong> Detected max risk index of <strong>{max_r:.2f}</strong>. 
            Unit spends <strong>{red_meters:.1f}m</strong> in high-probability contact zones.
        </div>
        """, unsafe_allow_html=True)

    # View 2: Heatmap
    with t2:
        fig_heat = create_interactive_heatmap(risk_map, path)
        st.plotly_chart(fig_heat, use_container_width=True)

    # View 3: 3D Model
    with t3:
        # Building the 3D mesh
        X, Y = np.meshgrid(np.arange(256), np.arange(256))
        Z = slope_map * 30 # Exaggerate height for visibility

        fig_3d = go.Figure()

        # Terrain Surface
        fig_3d.add_surface(
            x=X, y=Y, z=Z, surfacecolor=risk_map,
            colorscale=[
                [0.0, "#004d40"], [0.25, "#19c2ae"], [0.45, "#fbc02d"],
                [0.65, "#f57f17"], [0.85, "#b71c1c"], [1.0, "#880e4f"]
            ],
            opacity=0.9, showscale=False
        )
        
        # Route Line
        route_z = [Z[y, x] + 2 for x, y in path]
        fig_3d.add_trace(go.Scatter3d(
            x=path_x, y=path_y, z=route_z,
            mode="lines", line=dict(color="#66fcf1", width=6),
            name="Route"
        ))

        # Start/End Markers
        fig_3d.add_trace(go.Scatter3d(
            x=[start_x], y=[start_y], z=[Z[start_y, start_x]+4], 
            mode="markers", marker=dict(size=8, color="#66fcf1"), name="Start"
        ))
        fig_3d.add_trace(go.Scatter3d(
            x=[end_x], y=[end_y], z=[Z[end_y, end_x]+4], 
            mode="markers", marker=dict(size=8, color="#ff0000"), name="End"
        ))

        # --- UPDATED: Wireframe cylinders with separate colors ---
        
        # 1. Enemy Zones (Red Rings)
        for x, y, r in enemy_list:
            theta = np.linspace(0, 2*np.pi, 20)
            xc = x + r * np.cos(theta)
            yc = y + r * np.sin(theta)
            # Safe z-access: clamp coords to image bounds just in case
            safe_x, safe_y = min(max(int(x), 0), 255), min(max(int(y), 0), 255)
            zc = Z[safe_y, safe_x] + 5
            
            fig_3d.add_trace(go.Scatter3d(
                x=xc, y=yc, z=[zc]*20,
                mode="lines",
                line=dict(color="#080808", width=4), # RED for Enemy
                name="Hostile Zone",
                showlegend=False
            ))

        # 2. Restricted Zones (Purple Rings)
        for x, y, r in restricted_list:
            theta = np.linspace(0, 2*np.pi, 20)
            xc = x + r * np.cos(theta)
            yc = y + r * np.sin(theta)
            safe_x, safe_y = min(max(int(x), 0), 255), min(max(int(y), 0), 255)
            zc = Z[safe_y, safe_x] + 5

            fig_3d.add_trace(go.Scatter3d(
                x=xc, y=yc, z=[zc]*20,
                mode="lines",
                line=dict(color="#0b045d", width=5), # PURPLE for Restricted
                name="No-Fly Zone",
                showlegend=False
            ))

        fig_3d.update_layout(
            template="plotly_dark",
            scene=dict(
                xaxis_title="", yaxis_title="", zaxis_title="",
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                zaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                aspectmode="data",
                bgcolor="#0b0c10"
            ),
            margin=dict(l=0, r=0, t=10, b=0),
            height=600,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_3d, use_container_width=True)

    # View 4: Comparison
    with t4:
        st.subheader("👁️ Comparative Analysis")
        
        c1, c2 = st.columns(2)
        
        with c1:
            st.markdown("**1. Optical Feed**")
            # We need unique keys for the same chart in different tabs
            st.plotly_chart(fig_map, use_container_width=True, key="comp_map")
        
        with c2:
            st.markdown("**2. Thermal Risk Feed**")
            st.plotly_chart(fig_heat, use_container_width=True, key="comp_heat")

if __name__ == "__main__":
    main()
