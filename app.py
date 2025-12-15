import csv
import math
import heapq
from collections import deque
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import LabelEncoder
import streamlit as st
import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score, confusion_matrix


graph = {}
coords = {}

# Load nodes
try:
    with open("giki_nodes.csv", newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            coords[row["name"]] = (
                float(row["latitude"]),
                float(row["longitude"])
            )
except FileNotFoundError:
    st.error("Error: giki_nodes.csv not found. Please ensure all CSV files are in the same directory.")
    st.stop()

# Load edges and store road data for Plotly lines
ROAD_DATA = []
try:
    with open("giki_edges.csv", newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            u, v = row["from_node"], row["to_node"]
            w = float(row["distance_m"])

            graph.setdefault(u, {})[v] = w
            graph.setdefault(v, {})[u] = w
            ROAD_DATA.append((u, v, w)) # Store for Plotly visualization
            
            # Since the graph is undirected, also store the reverse edge for ROAD_DATA
            ROAD_DATA.append((v, u, w)) 
except FileNotFoundError:
    st.error("Error: giki_edges.csv not found. Please ensure all CSV files are in the same directory.")
    st.stop()


available_nodes = sorted(list(coords.keys()))

def path_distance(path):
    dist = 0
    for i in range(len(path) - 1):
        if path[i+1] in graph.get(path[i], {}):
            dist += graph[path[i]][path[i+1]]
        else:
            return float('inf') 
    return dist

def heuristic(a, b):
    lat1, lon1 = coords[a]
    lat2, lon2 = coords[b]
    R = 6371000
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    h = math.sin(dlat/2)**2 + \
        math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * \
        math.sin(dlon/2)**2
    return 2 * R * math.atan2(math.sqrt(h), math.sqrt(1-h))

def bfs(start, goal):
    queue = deque([[start]])
    visited = {start}
    while queue:
        path = queue.popleft()
        node = path[-1]
        if node == goal: return path, path_distance(path)
        for neighbor in graph.get(node, {}):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(path + [neighbor])
    return None, float('inf')

def dfs(start, goal):
    stack = [[start]]
    visited = {start}
    while stack:
        path = stack.pop()
        node = path[-1]
        if node == goal: return path, path_distance(path)
        for neighbor in sorted(graph.get(node, {}).keys(), reverse=True):
            if neighbor not in visited:
                visited.add(neighbor)
                stack.append(path + [neighbor])
    return None, float('inf')

def gbfs(start, goal):
    pq = [(heuristic(start, goal), start, [start])]
    visited = set()
    while pq:
        _, node, path = heapq.heappop(pq)
        if node == goal: return path, path_distance(path)
        if node in visited: continue
        visited.add(node)
        for neighbor in graph.get(node, {}):
            if neighbor not in visited:
                heapq.heappush(pq, (heuristic(neighbor, goal), neighbor, path + [neighbor]))
    return None, float('inf')

def astar(start, goal):
    pq = [(0 + heuristic(start, goal), 0, start, [start])]
    g_scores = {start: 0}
    while pq:
        f, g, node, path = heapq.heappop(pq)
        if node == goal: return path, g
        if g > g_scores.get(node, float('inf')): continue
        for neighbor, weight in graph.get(node, {}).items():
            g_new = g + weight
            if g_new < g_scores.get(neighbor, float('inf')):
                g_scores[neighbor] = g_new
                f_new = g_new + heuristic(neighbor, goal)
                heapq.heappush(pq, (f_new, g_new, neighbor, path + [neighbor]))
    return None, float("inf")

def dijkstra(start, goal):
    pq = [(0, start, [start])]
    distances = {start: 0}
    while pq:
        cost, node, path = heapq.heappop(pq)
        if node == goal: return path, cost
        if cost > distances.get(node, float('inf')): continue
        for neighbor, weight in graph.get(node, {}).items():
            new_cost = cost + weight
            if new_cost < distances.get(neighbor, float('inf')):
                distances[neighbor] = new_cost
                heapq.heappush(pq, (new_cost, neighbor, path + [neighbor]))
    return None, float("inf")

def count_turns(path):
    if path is None or len(path) < 3: return 0
    return len(path) - 2

# ML Functions
# Training setup
algo_map = {"bfs": 0, "dfs": 1, "dijkstra": 2, "gbfs": 3, "astar": 4}
ALGORITHM_CHOICES = {
    "A* Search (Optimal, Informed)": "astar",
    "Dijkstra's (Optimal, Uninformed)": "dijkstra",
    "Greedy Best-First Search (GBFS)": "gbfs",
    "Breadth-First Search (BFS)": "bfs",
    "Depth-First Search (DFS)": "dfs",
}
ALGORITHM_MAP = {v: k for k, v in ALGORITHM_CHOICES.items()}

try:
    data = pd.read_csv("giki_training_data.csv")
    data["algorithm_code"] = data["algorithm"].map(algo_map)
    le = LabelEncoder()
    data["route_type_code"] = le.fit_transform(data["route_type"])
    X = data[["distance_m", "num_turns", "algorithm_code"]]
    
    lr_model = LinearRegression()
    lr_model.fit(X, data["eta_minutes"])

    cls_model = LogisticRegression(max_iter=500)
    cls_model.fit(X, data["route_type_code"])
    
    ALL_CLASSES = le.inverse_transform(data["route_type_code"].unique())
    
except FileNotFoundError:
    st.warning("Warning: giki_training_data.csv not found. ML predictions will be disabled.")
    lr_model, cls_model = None, None
except Exception as e:
    st.error(f"An error occurred during ML model training: {e}")
    lr_model, cls_model = None, None

def get_ml_predictions(distance, turns, algorithm_name):
    #Predicts ETA and Route Reliability Class/Probability.
    if lr_model is None or cls_model is None:
        return "--", "--", "Unknown"
    
    alg_code = algo_map.get(algorithm_name.lower())
    if alg_code is None:
        return "--", "--", "Unknown"

    df_input = pd.DataFrame(
        [[distance, turns, alg_code]],
        columns=["distance_m", "num_turns", "algorithm_code"]
    )
    
    pred_eta = lr_model.predict(df_input)[0]

    pred_class_code = cls_model.predict(df_input)[0]
    pred_class_label = le.inverse_transform([pred_class_code])[0]
    
    pred_proba = cls_model.predict_proba(df_input)[0]
    class_indices = cls_model.classes_
    pred_index = list(class_indices).index(pred_class_code)
    pred_reliability = pred_proba[pred_index] * 100
    
    return f"{pred_eta:.1f} min", f"{pred_reliability:.0f}% ({pred_class_label})", pred_class_label

# ---------------- MODEL PERFORMANCE METRICS ----------------
model_metrics = {}

if lr_model is not None and cls_model is not None:
    # Predictions
    y_eta_pred = lr_model.predict(X)
    y_cls_pred = cls_model.predict(X)

    # Regression metrics
    model_metrics["MAE"] = mean_absolute_error(data["eta_minutes"], y_eta_pred)
    model_metrics["R2"] = r2_score(data["eta_minutes"], y_eta_pred)

    # Classification metrics
    model_metrics["Accuracy"] = accuracy_score(data["route_type_code"], y_cls_pred)



def generate_landmark_directions(path):
    if not path or len(path) < 2:
        return "No path found."
    
    directions = []
    directions.append(f"1. Start at **{path[0]}**.")
    
    for i in range(len(path) - 1):
        current = path[i]
        next_node = path[i+1]
        
        if i == len(path) - 2:
            directions.append(f"{i+2}. You have arrived at your destination: **{next_node}**.")
        else:
            directions.append(f"{i+2}. From **{current}**, proceed towards **{next_node}**.")
    
    return "\n".join(directions)


#Map Visualization Function
def create_campus_map_simple(path=None):
    fig = go.Figure()
    #Draw ALL roads
    for u, v, _ in ROAD_DATA:
        if u in coords and v in coords:
            x1, y1 = coords[u]
            x2, y2 = coords[v]

            fig.add_trace(go.Scatter(
                x=[x1, x2],
                y=[y1, y2],
                mode="lines",
                line=dict(color="lightgray", width=2),
                hoverinfo="none",
                showlegend=False
            ))

    #Draw path (highlighted)
    if path and len(path) > 1:
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            x1, y1 = coords[u]
            x2, y2 = coords[v]

            fig.add_trace(go.Scatter(
                x=[x1, x2],
                y=[y1, y2],
                mode="lines",
                line=dict(color="green", width=4),
                hoverinfo="none",
                showlegend=False
            ))

    #Draw nodes
    node_x = []
    node_y = []
    labels = []
    colors = []

    for node, (x, y) in coords.items():
        node_x.append(x)
        node_y.append(y)
        labels.append(node)
        colors.append("red" if path and node in path else "royalblue")

    fig.add_trace(go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        text=labels,
        textposition="top center",
        marker=dict(size=12, color=colors),
        showlegend=False
    ))

    #Layout
    fig.update_layout(
        height=550,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(showgrid=True, zeroline=False),
        yaxis=dict(showgrid=True, zeroline=False),
        plot_bgcolor="#f8f9fa"
    )

    return fig



# STREAMLIT UI

st.set_page_config(
    page_title="GIKAtlas - Intelligent Campus Navigation",
    page_icon="🗺️",
    layout="wide"
)
st.markdown("""
<style>
/* Page background */
.stApp {
    background-color: #f4f6f8;
}

/* Buttons */
div.stButton > button {
    background-color: #1f77b4;
    color: white;
    border-radius: 8px;
    height: 3em;
    font-size: 16px;
}

div.stButton > button:hover {
    background-color: #155a8a;
    color: white;
}

/* Metrics */
[data-testid="stMetricValue"] {
    color: #2ca02c;
    font-weight: bold;
}

/* Section headers */
h1, h2, h3 {
    color: #1f2c56;
}
</style>
""", unsafe_allow_html=True)

DEFAULT_STATE = {
    "path": None,
    "distance": 0.0,
    "turns": 0,
    "algorithm": "",
    "eta": "--",
    "reliability": "--",
    "route_type": "--",
}

for k, v in DEFAULT_STATE.items():
    if k not in st.session_state:
        st.session_state[k] = v
st.markdown(
    """
    <h1 style='text-align: center; margin-bottom: 0;'>GIKAtlas</h1>
    <p style='text-align: center; color: gray; margin-top: 0;'>
        Intelligent Campus Navigation & Decision Support
    </p>
    """,
    unsafe_allow_html=True
)

st.markdown("---")

st.subheader("🧭 Plan Your Route")

c1, c2, c3 = st.columns(3)

with c1:
    start_location = st.selectbox(
        "Start Location",
        available_nodes,
        index=available_nodes.index("Main Gate") if "Main Gate" in available_nodes else 0
    )

with c2:
    destination = st.selectbox(
        "Destination",
        available_nodes,
        index=available_nodes.index("Academic Block") if "Academic Block" in available_nodes else 1
    )

with c3:
    algorithm_key = st.selectbox(
        "Algorithm",
        list(ALGORITHM_CHOICES.keys())
    )

b1, b2 = st.columns([1.2, 1])
with b1:
    find_btn = st.button("🔍 Find Route", use_container_width=True)
with b2:
    reset_btn = st.button(" Reset Map", use_container_width=True)

if reset_btn:
    for k, v in DEFAULT_STATE.items():
        st.session_state[k] = v
    st.rerun()

if find_btn:
    algo_func_name = ALGORITHM_CHOICES[algorithm_key]
    algorithm_func = globals().get(algo_func_name)

    if algorithm_func:
        with st.spinner(f"Calculating route using {algorithm_key}..."):
            path, distance = algorithm_func(start_location, destination)

        if path and distance != float("inf"):
            turns = count_turns(path)

            eta, reliability, route_type = get_ml_predictions(
                distance, turns, algo_func_name
            )

            st.session_state.update({
                "path": path,
                "distance": distance,
                "turns": turns,
                "algorithm": algorithm_key,
                "eta": eta,
                "reliability": reliability,
                "route_type": route_type,
            })
        else:
            st.warning("No valid route found between selected locations.")

st.markdown("---")
left, right = st.columns([1.6, 1])
with left:
    st.subheader("🗺️ Campus Map Visualization")
    fig = create_campus_map_simple(st.session_state["path"])
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.subheader("🧠 Intelligent Guidance")

    m1, m2 = st.columns(2)
    m1.metric("Predicted ETA", st.session_state["eta"])
    m2.metric("Route Reliability", st.session_state["reliability"])

    st.markdown("")

    st.write(f"**Total Distance:** {st.session_state['distance']:.1f} meters")
    st.write(f"**Pathfinding Algorithm:** {st.session_state['algorithm']}")
    st.write(f"**Turns / Segments:** {st.session_state['turns']}")
    st.write(f"**Route Type:** {st.session_state['route_type']}")

    st.markdown("---")

    st.subheader("📍 Landmark Directions")
    if st.session_state["path"]:
        st.markdown(generate_landmark_directions(st.session_state["path"]))
    else:
        st.info("Select start and destination, then click **Find Route**.")

st.markdown("---")
st.subheader("📊 Model Performance")

if model_metrics:
    c1, c2, c3 = st.columns(3)
    c1.metric("ETA MAE", f"{model_metrics['MAE']:.2f} min")
    c2.metric("ETA R²", f"{model_metrics['R2']:.2f}")
    c3.metric("Route Accuracy", f"{model_metrics['Accuracy']*100:.1f}%")
else:
    st.info("Model performance unavailable.")




#Tarining data generate
# def generate_training_data(samples=200):
#     data = []
#     nodes = list(graph.keys())
#     algorithms = {
#         "bfs": bfs,
#         "dfs": dfs,
#         "gbfs": gbfs,
#         "dijkstra": dijkstra,
#         "astar": astar
#     }

#     for _ in range(samples):
#         start, goal = random.sample(nodes, 2)
#         algo_name = random.choice(list(algorithms.keys()))
#         algo_func = algorithms[algo_name]

#         if algo_name in ["dijkstra", "astar"]:
#             path, dist = algo_func(start, goal)
#         else:
#             path = algo_func(start, goal)
#             if path is None:
#                 continue
#             dist = path_distance(path)

#         if path is None:
#             continue

#         num_nodes = len(path)
#         turns = count_turns(path)

#         #ETA calculation (synthetic)
#         speed = random.uniform(1.2, 1.5)   # m/s
#         eta_minutes = (dist / speed) / 60
#         eta_minutes += turns * 0.15        # turn penalty
#         eta_minutes += random.uniform(-0.5, 0.5)  # noise

#         #Route classification
#         if dist < 400:
#             #short
#             route_type = 0   
#         elif dist < 800:
#             #medium
#             route_type = 1   
#         else:
#             #long
#             route_type = 2   

#         data.append([
#             round(dist, 2),
#             num_nodes,
#             turns,
#             algo_name,
#             round(eta_minutes, 2),
#             route_type
#         ])

#     return data

# #saving the training data
# training_data = generate_training_data(samples=300)
# with open("giki_training_data.csv", "w", newline="", encoding="utf-8") as f:
#     writer = csv.writer(f)
#     writer.writerow([
#         "distance_m",
#         "num_nodes",
#         "num_turns",
#         "algorithm",
#         "eta_minutes",
#         "route_type"
#     ])
#     writer.writerows(training_data)

# print("Training dataset generated: giki_training_data.csv")
