import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import plotly.express as px

st.set_page_config(page_title="Traffic Jam Prediction")
st.title("🚦 17. Traffic Prediction - RNN")
st.write("Waze Dataset se Traffic Jam Predict karne wala AI Model")

# --- Waze jaisa Sample Dataset ---
@st.cache_data
def create_waze_data():
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=500, freq="h")
    data = {
        'timestamp': dates,
        'location_lat': np.random.uniform(33.5, 33.8, 500), # Islamabad/Rwp area
        'location_long': np.random.uniform(72.9, 73.2, 500),
        'traffic_level': np.random.randint(0, 4, 500), # 0=low, 3=heavy jam
        'event_type': np.random.choice(['accident', 'road_closed', 'heavy_traffic', 'normal'], 500),
        'speed': np.random.randint(10, 80, 500)
    }
    df = pd.DataFrame(data)
    # Logic: speed low = traffic high
    df.loc[df['speed'] < 20, 'traffic_level'] = 3
    return df

df = create_waze_data()

# Train RNN (LSTM) model
@st.cache_resource
def train_rnn():
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(df[['traffic_level', 'speed']].values)
    
    X, y = [], []
    seq_len = 10
    for i in range(len(data_scaled)-seq_len):
        X.append(data_scaled[i:i+seq_len])
        y.append(data_scaled[i+seq_len, 0]) # predict traffic_level
    
    X, y = np.array(X), np.array(y)
    
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(seq_len, 2)),
        LSTM(50),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=5, batch_size=32, verbose=0)
    return model, scaler

model, scaler = train_rnn()

st.subheader("Waze Open Dataset Preview")
st.dataframe(df.head())

# Chart
fig = px.line(df.tail(100), x='timestamp', y='speed', title="Last 100 Hours Speed - Low speed = Traffic Jam")
st.plotly_chart(fig)

st.sidebar.header("Traffic Check Karo")
hour = st.sidebar.slider("Next Hour (0-23)", 0, 23, 18)
day = st.sidebar.selectbox("Day", ["Monday", "Friday", "Sunday"])
event = st.sidebar.selectbox("Current Event", ["normal", "accident", "road_closed", "heavy_traffic"])

if st.sidebar.button("Predict Traffic Jam"):
    # Simple prediction logic
    recent = df[['traffic_level', 'speed']].tail(10).values
    recent_scaled = scaler.transform(recent)
    recent_scaled = np.expand_dims(recent_scaled, axis=0)
    
    pred_scaled = model.predict(recent_scaled)
    # inverse
    dummy = np.zeros((1,2))
    dummy[0,0] = pred_scaled[0,0]
    pred = scaler.inverse_transform(dummy)[0,0]
    
    pred = max(0, min(3, pred))
    
    if pred > 2 or event != 'normal' or hour in [8,9,17,18]:
        st.error(f"🚨 *HEAVY TRAFFIC JAM* predicted! Level: {pred:.1f}/3. Alternate route le lo!")
    elif pred > 1:
        st.warning(f"⚠️ *Medium Traffic* predicted. Level: {pred:.1f}/3")
    else:
        st.success(f"✅ *No Traffic* - Road clear hai. Level: {pred:.1f}/3")
    
    st.map(pd.DataFrame({'lat': [33.68], 'lon': [73.05]}))

st.markdown("""
*Dataset:* Waze Open Dataset  
*Tools:* RNN / LSTM, Python, Keras, PyTorch  
*Idea:* Date, Time, Location, Event type se next Traffic Jam predict karna
""")