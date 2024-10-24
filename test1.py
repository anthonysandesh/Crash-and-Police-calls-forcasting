import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from scipy import stats

# Load the data
crash_data = pd.read_csv('crash_data.csv')
police_data = pd.read_csv('police_data.csv')

# Extract date from CrashName and convert to datetime
crash_data['DATE'] = pd.to_datetime(crash_data['CrashName'].str[:8], format='%Y%m%d')
police_data['START_DATE'] = pd.to_datetime(police_data['START_DATE'])


# EDA functions
def plot_crash_severity(data):
    severity_counts = data['VehicleDamage'].value_counts()
    plt.figure(figsize=(10, 6))
    severity_counts.plot(kind='bar')
    plt.title('Crash Severity Distribution')
    plt.xlabel('Severity')
    plt.ylabel('Count')
    st.pyplot(plt)

def plot_call_types(data):
    call_type_counts = data['CALL_TYPE'].value_counts().head(10)
    plt.figure(figsize=(12, 6))
    call_type_counts.plot(kind='bar')
    plt.title('Top 10 Police Call Types')
    plt.xlabel('Call Type')
    plt.ylabel('Count')
    plt.xticks(rotation=45, ha='right')
    st.pyplot(plt)

def plot_time_series(crash_data, police_data):
    crash_counts = crash_data.groupby('DATE').size().resample('D').sum()
    police_counts = police_data.groupby('START_DATE').size().resample('D').sum()

    plt.figure(figsize=(12, 6))
    plt.plot(crash_counts.index, crash_counts.values, label='Crashes')
    plt.plot(police_counts.index, police_counts.values, label='Police Calls')
    plt.title('Daily Crashes and Police Calls')
    plt.xlabel('Date')
    plt.ylabel('Count')
    plt.legend()
    st.pyplot(plt)

def calculate_correlation(crash_data, police_data):
    crash_counts = crash_data.groupby('DATE').size().resample('D').sum()
    police_counts = police_data.groupby('START_DATE').size().resample('D').sum()

    merged_data = pd.merge(crash_counts, police_counts, left_index=True, right_index=True, how='inner')
    correlation = merged_data['DATE'].corr(merged_data['START_DATE'])
    st.write(f"Correlation between daily crashes and police calls: {correlation:.2f}")

# Streamlit app
st.title('Traffic Crash and Police Call Analysis')

st.header('Exploratory Data Analysis')

st.subheader('Crash Severity Distribution')
plot_crash_severity(crash_data)

st.subheader('Top 10 Police Call Types')
plot_call_types(police_data)

st.subheader('Daily Crashes and Police Calls')
plot_time_series(crash_data, police_data)

st.header('Correlation Analysis')
calculate_correlation(crash_data, police_data)