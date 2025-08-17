import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from openai import OpenAI
import os
import matplotlib.pyplot as plt
import re

def extract_sop_steps(raw_text):
    """
    Extracts the numbered SOP steps from the model output,
    ignoring any reasoning or <think>...</think> blocks.
    """
    # Remove all <think>...</think> blocks (including newlines inside)
    cleaned = re.sub(r'<think>.*?</think>', '', raw_text, flags=re.DOTALL).strip()

    # Sometimes model outputs steps separated by newlines or numbered list
    # Extract lines starting with a number + dot or just text lines after cleaning
    lines = cleaned.splitlines()

    # Filter lines that look like steps (e.g., start with number or nonempty lines)
    steps = [line.strip() for line in lines if line.strip() and
             (re.match(r'^\d+\.', line.strip()) or len(line.strip()) > 5)]

    # If no numbered lines found, just return all lines joined
    if not steps:
        return cleaned

    # Join steps nicely
    return "\n".join(steps)

st.set_page_config(page_title="Industrial Predictive Maintenance and SOP Generator", layout="wide")
st.title("🔧 Industrial Predictive Maintenance and SOP Generator")

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY", "lm-studio"),
    base_url=os.getenv("LMSTUDIO_BASE_URL", "http://localhost:1234/v1")
)

def generate_sensor_data():
    np.random.seed(42)
    timestamps = pd.date_range(start="2025-08-09 09:00", periods=100, freq="T")
    vibration = np.random.normal(1.0, 0.1, size=100)
    temperature = np.random.normal(50, 0.5, size=100)
    vibration[60:65] += np.linspace(3, 6, 5)
    temperature[60:65] += np.linspace(10, 15, 5)
    df = pd.DataFrame({"timestamp": timestamps, "vibration": vibration, "temperature": temperature})
    return df

def detect_anomalies(data):
    clf = IsolationForest(contamination=0.05, random_state=42)
    preds = clf.fit_predict(data[['vibration', 'temperature']])
    data['anomaly'] = preds == -1
    anomalies = data[data['anomaly']]
    return anomalies
def plot_anomalies(data):
    fig, axs = plt.subplots(2, 1, figsize=(10,6), sharex=True)
    
    # Vibration plot
    axs[0].plot(data['timestamp'], data['vibration'], label='Vibration (mm/s)')
    axs[0].scatter(data[data['anomaly']]['timestamp'], data[data['anomaly']]['vibration'], 
                   color='red', label='Anomaly')
    axs[0].set_ylabel('Vibration (mm/s)')
    axs[0].legend()
    axs[0].grid(True)
    
    # Temperature plot
    axs[1].plot(data['timestamp'], data['temperature'], label='Temperature (°C)', color='orange')
    axs[1].scatter(data[data['anomaly']]['timestamp'], data[data['anomaly']]['temperature'], 
                   color='red', label='Anomaly')
    axs[1].set_ylabel('Temperature (°C)')
    axs[1].set_xlabel('Timestamp')
    axs[1].legend()
    axs[1].grid(True)
    
    plt.tight_layout()
    return fig
def generate_diagnosis(anomalies):
    if anomalies.empty:
        return "No anomaly detected."
    summary = f"Motor vibration spiked from {anomalies['vibration'].min():.2f} to {anomalies['vibration'].max():.2f} mm/s, " \
              f"temperature rose from {anomalies['temperature'].min():.1f}°C to {anomalies['temperature'].max():.1f}°C."
    prompt = f"Diagnose the root cause for the following incident:\n{summary}"
    response = client.chat.completions.create(
        model="local-model",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.8,
    )
    return response.choices[0].message.content

def generate_sop(diagnosis):
    if diagnosis == "No anomaly detected.":
        return diagnosis
    prompt = f"Based on this diagnosis:\n{diagnosis}\nWrite a clear 6-step SOP for safe inspection and temporary mitigation.\nSOP:\n1."
    response = client.chat.completions.create(
        model="local-model",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.8
    )
    return response.choices[0].message.content

st.sidebar.header("Options")
use_upload = st.sidebar.checkbox("Upload your own sensor CSV?", value=False)

if use_upload:
    uploaded_file = st.file_uploader("Upload CSV (timestamp,vibration,temperature)", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file, parse_dates=['timestamp'])
    else:
        st.warning("Upload a CSV file to proceed.")
        st.stop()
else:
    df = generate_sensor_data()

st.write("### Sensor Data (first 10 rows)")
st.dataframe(df.head(10))

if st.button("Run Agentic AI Pipeline"):

    # Step 1: Detect anomalies
    anomalies = detect_anomalies(df)
    st.write("### Detected Anomalies")
    if anomalies.empty:
        st.info("No anomalies detected.")
    else:
        st.dataframe(anomalies)
        fig = plot_anomalies(df)
        st.pyplot(fig)

    # Step 2: Generate diagnosis
    diagnosis = generate_diagnosis(anomalies)
    diagnosis = extract_sop_steps(diagnosis)
    st.write("### Diagnosis")
    st.markdown(diagnosis)

    # Step 3: Generate SOP
    sop = generate_sop(diagnosis)
    sop = extract_sop_steps(sop)
    st.write("### Generated SOP")
    st.markdown(sop)
