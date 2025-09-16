import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx

###
# Packages required to run this pyhthon file
#  1). plotly package : Install it by using command : pip install plotly
#  2). networkx package : Install it by using command : pip install networkx
#  3). streamlit package : Install it by using command : pip install streamlit
# Before running with command streamlit run ./src/ui/streamlit_app.py do the following
# Make sure current working directory is in SIH_STRIVAS.
#
# ###
API_URL = "http://localhost:8080/api/plan"

st.set_page_config(page_title="KMRL AI Induction Planner", layout="wide")
st.title("🚆 KMRL AI-Driven Train Induction Planner")


st.sidebar.header("⚖️ Objective Weights")
w1 = st.sidebar.slider("Safety (Failure Risk)", 0.0, 1.0, 0.6)
w2 = st.sidebar.slider("Shunting Cost", 0.0, 1.0, 0.2)
w3 = st.sidebar.slider("Branding", 0.0, 1.0, 0.1)
w4 = st.sidebar.slider("Mileage Balance", 0.0, 1.0, 0.1)

capacity = st.sidebar.number_input("Required Capacity", min_value=5, max_value=20, value=10)

# Button to call API
if st.sidebar.button("Generate Induction Plan"):
    payload = {"required_capacity": capacity, "weights": {"w1": w1, "w2": w2, "w3": w3, "w4": w4}}
    plan_response = requests.post(API_URL, json=payload).json()
else:
    # Dummy fallback data for first load
    plan_response = {
        "selected": ["T12", "T15", "T19"],
        "reasons": {
            "T12": "Low risk, valid certificate, minimal shunting",
            "T15": "Medium risk but branding requirement",
            "T19": "Balanced mileage"
        },
        "kpis": {"failure_risk_avg": 0.05, "shunting_time": 120, "branding_coverage": 0.9, "capacity_achieved": 12}
    }

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Fleet Status", "Optimization Plan", "Maintenance Drill-Down", "Simulation", "KPIs"]
)

# Tab 1: Fleet Status
with tab1:
    st.subheader("🚉 Fleet Dashboard")
    fleet_data = [
        {"train_id": "T12", "mileage": 124500, "certificate_days_left": 120, "failure_prob": 0.03, "cleaning_status": "clean"},
        {"train_id": "T15", "mileage": 132000, "certificate_days_left": 15, "failure_prob": 0.12, "cleaning_status": "due"},
        {"train_id": "T19", "mileage": 110400, "certificate_days_left": 250, "failure_prob": 0.07, "cleaning_status": "clean"},
    ]
    trains_df = pd.DataFrame(fleet_data)

    def risk_color(prob):
        if prob < 0.05: return "🟢 Low"
        elif prob < 0.1: return "🟡 Medium"
        else: return "🔴 High"

    trains_df["Risk Level"] = trains_df["failure_prob"].apply(risk_color)
    st.dataframe(trains_df, use_container_width=True)

# Tab 2: Optimization Plan
with tab2:
    st.subheader("🧠 Smart Induction Plan")
    plan_df = pd.DataFrame([
        {"Train": t, "Reason": plan_response["reasons"][t]}
        for t in plan_response["selected"]
    ])
    st.table(plan_df)

# Tab 3: Maintenance Drill-Down
with tab3:
    st.subheader("🔧 Predictive Maintenance")
    #Example fleet data (replace with your DB or API call)
    fleet_data = [
        {"train_id": "T12", "mileage": 124500, "certificate_days_left": 120, "failure_prob": 0.03, "cleaning_status": "clean"},
        {"train_id": "T15", "mileage": 132000, "certificate_days_left": 15, "failure_prob": 0.12, "cleaning_status": "due"},
        {"train_id": "T19", "mileage": 110400, "certificate_days_left": 250, "failure_prob": 0.07, "cleaning_status": "clean"},
    ]
    trains_df = pd.DataFrame(fleet_data)

    # Dropdown selector for train
    selected_train = st.selectbox(
        "Select a Train for Maintenance Insights",
        trains_df["train_id"].tolist()
    )

    # Example backend response (replace with API call)
    # In production, do:
    # train_details = requests.get(f"http://localhost:8080/api/trains/{selected_train}").json()
    train_predictions = {
        "T12": {"failure_prob": 0.03, "shap_top_features": {"days_since_last_service": 12, "rolling_mileage_30d": 5000, "fault_count_30d": 0}},
        "T15": {"failure_prob": 0.12, "shap_top_features": {"days_since_last_service": 30, "rolling_mileage_30d": 6500, "fault_count_30d": 2}},
        "T19": {"failure_prob": 0.07, "shap_top_features": {"days_since_last_service": 18, "rolling_mileage_30d": 4200, "fault_count_30d": 1}},
    }
    train_details = train_predictions[selected_train]

    # Show failure probability
    st.metric(
        f"Failure Probability (next 7 days) for {selected_train}",
        f"{train_details['failure_prob']*100:.1f}%"
    )

    # SHAP-style explanation
    shap_df = pd.DataFrame(
        list(train_details["shap_top_features"].items()),
        columns=["Feature","Value"]
    )
    st.bar_chart(shap_df.set_index("Feature"))
    st.bar_chart(shap_df.set_index("Feature"))

# Tab 4: Depot Simulation
with tab4:
    st.subheader("🏭 Depot Simulator")
    G = nx.Graph()
    G.add_edges_from([
        ("Depot","Bay1"),("Depot","Bay2"),("Bay1","Track1"),("Bay2","Track2")
    ])
    pos = nx.spring_layout(G)
    edge_x, edge_y = [], []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    node_x, node_y, node_text = [], [], []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode='lines'))
    fig.add_trace(go.Scatter(x=node_x, y=node_y, mode='markers+text', text=node_text, textposition="top center"))
    st.plotly_chart(fig, use_container_width=True)

# Tab 5: KPI Dashboard
with tab5:
    st.subheader("📊 Performance Metrics")
    kpi_df = pd.DataFrame(list(plan_response["kpis"].items()), columns=["Metric", "Value"])
    fig_kpi = px.bar(kpi_df, x="Metric", y="Value", text="Value", title="KPI Dashboard")
    st.plotly_chart(fig_kpi, use_container_width=True)
