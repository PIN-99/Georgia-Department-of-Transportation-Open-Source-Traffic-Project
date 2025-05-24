import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# Load data
@st.cache_data
def load_data():
    return pd.read_csv("final_predict_pos.csv")

df_all = load_data()

# kNN prediction function
def predict_traffic_by_knn(lat, lon, weekday, time_slot, df, k=5):
    sub = df[(df['weekday'] == weekday) & (df['time_slot'] == time_slot)]
    if sub.empty:
        return None
    coords = sub[['latitude', 'longitude']].to_numpy()
    target = np.array([lat, lon])
    dists = np.linalg.norm(coords - target, axis=1)
    idx = np.argsort(dists)[:min(k, len(dists))]
    return sub['traffic'].to_numpy()[idx].mean()

# Time format helper
def slot_to_time_str(ts):
    hour = (ts * 15) // 60
    minute = (ts * 15) % 60
    return f"{hour:02d}:{minute:02d}"

weekday_dict = {
    0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"
}

# Page setup
st.title("Weekly Traffic Prediction Map")
st.markdown("Select a location and weekday. Slide through time to view predicted traffic and trends.")

# Sidebar inputs
with st.sidebar:
    st.header("Input Parameters")
    lat = st.number_input("Latitude", value=33.7490, format="%.5f")
    lon = st.number_input("Longitude", value=-84.3880, format="%.5f")
    weekday = st.selectbox("Weekday", list(weekday_dict.keys()), format_func=lambda x: weekday_dict[x])
    time_slot = st.slider("Time Slot (Each step = 15 minutes)", 0, 95, 32)

# Prediction for map
pred = predict_traffic_by_knn(lat, lon, weekday, time_slot, df_all, k=5)
real_df = df_all[(df_all["weekday"] == weekday) & (df_all["time_slot"] == time_slot)].copy()

if pred is None or real_df.empty:
    st.error("No real data available at this time slot.")
else:
    st.success(f"Time: {weekday_dict[weekday]} {slot_to_time_str(time_slot)} | Predicted traffic: {pred:.2f}")

    # Build map with border-highlighted prediction point
    fig = go.Figure()

    # Real points
    fig.add_trace(go.Scattermapbox(
        lat=real_df["latitude"],
        lon=real_df["longitude"],
        mode="markers",
        marker=go.scattermapbox.Marker(
            size=3,
            color=real_df["traffic"],
            colorscale="Viridis",
            showscale=True,
            colorbar=dict(title="Traffic")
        ),
        text=real_df["Site_ID"],
        hoverinfo="text",
        name="Real Points"
    ))

    # Predicted point with border (white outline)
    fig.add_trace(go.Scattermapbox(
        lat=[lat],
        lon=[lon],
        mode="markers",
        marker=go.scattermapbox.Marker(
            size=30,
            color=[pred],
            colorscale="Viridis",
            cmin=real_df["traffic"].min(),
            cmax=real_df["traffic"].max(),
            #line=dict(width=2, color="white")
        ),
        text=[f"Predicted Point<br>Traffic: {pred:.2f}"],
        hoverinfo="text",
        name="Predicted Point"
    ))

    fig.update_layout(
        mapbox_style="carto-positron",
        mapbox_zoom=6,
        mapbox_center={"lat": lat, "lon": lon},
        height=700,
        margin={"r":0, "t":40, "l":0, "b":0},
        title=f"Traffic Map - {weekday_dict[weekday]} {slot_to_time_str(time_slot)}"
    )

    st.plotly_chart(fig, use_container_width=True)

# Line chart for entire day
times = list(range(96))
predicted_traffic_day = [
    predict_traffic_by_knn(lat, lon, weekday, t, df_all, k=5) for t in times
]

# Remove None (just in case)
valid_times = [t for t, p in zip(times, predicted_traffic_day) if p is not None]
valid_preds = [p for p in predicted_traffic_day if p is not None]
time_labels = [slot_to_time_str(t) for t in valid_times]

st.subheader(f"Predicted Traffic Trend - {weekday_dict[weekday]}")
fig_line = px.line(
    x=time_labels,
    y=valid_preds,
    labels={"x": "Time", "y": "Predicted Traffic"},
    title="Traffic Prediction Over the Day"
)
fig_line.update_traces(line=dict(color="darkblue", width=2), mode="lines+markers")
fig_line.update_layout(xaxis_tickangle=-45)
st.plotly_chart(fig_line, use_container_width=True)
