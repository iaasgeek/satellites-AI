import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np
import boto3
import json

# Load data function
@st.cache_data
def load_data():
    file_path = 'UCS-Satellite-Database-1-1-2023.csv'
    
    encodings = ['utf-8', 'iso-8859-1', 'windows-1252', 'latin1']
    for encoding in encodings:
        try:
            df = pd.read_csv(file_path, encoding=encoding)
            df['Date of Launch'] = pd.to_datetime(df['Date of Launch'], format='%m/%d/%Y', errors='coerce')
            return df
        except UnicodeDecodeError:
            continue
    
    try:
        with open(file_path, 'rb') as file:
            df = pd.read_csv(file, encoding='latin1')
        df['Date of Launch'] = pd.to_datetime(df['Date of Launch'], format='%m/%d/%Y', errors='coerce')
        return df
    except Exception as e:
        st.error(f"Failed to load the CSV file. Error: {str(e)}")
        return pd.DataFrame()

# Bedrock response function
def get_bedrock_response(prompt, model_id="anthropic.claude-v2"):
    bedrock = boto3.client(service_name='bedrock-runtime', region_name='us-east-1')
    
    body = json.dumps({
        "prompt": prompt,
        "max_tokens_to_sample": 500,
        "temperature": 0.5,
        "top_p": 1,
        "stop_sequences": ["\n\nHuman:"]
    })
    
    response = bedrock.invoke_model(
        body=body,
        modelId=model_id,
        accept='application/json',
        contentType='application/json'
    )
    
    response_body = json.loads(response.get('body').read())
    return response_body.get('completion')

# Load the data
df = load_data()

# Sidebar for navigation
st.sidebar.title('Navigation')
page = st.sidebar.radio('Go to', ['Overview', 'Satellite Search'])

if page == 'Overview':
    st.title('Satellite Quick Facts')

    # Last update date
    last_update = df['Date of Launch'].max()
    if pd.isna(last_update):
        last_update_str = "Date information not available"
    else:
        last_update_str = last_update.strftime('%B %d, %Y')
    st.write(f"Includes launches through {last_update_str}")

    # Key metrics
    col1, col2 = st.columns(2)
    col1.metric("Total Operating Satellites", len(df))
    col2.metric("Countries", df['Country of Operator/Owner'].nunique())

    # Top 10 Countries
    st.subheader('Top 10 Countries by Number of Satellites')
    top_countries = df['Country of Operator/Owner'].value_counts().nlargest(10)
    fig = px.bar(top_countries, title='Top 10 Countries by Number of Satellites')
    st.plotly_chart(fig)

    # Country filter
    countries = ['All'] + sorted(df['Country of Operator/Owner'].unique().tolist())
    selected_country = st.selectbox('Select a country', countries)

    if selected_country != 'All':
        filtered_df = df[df['Country of Operator/Owner'] == selected_country]
    else:
        filtered_df = df

    # Display country-specific information
    st.subheader(f"Satellite Information for {selected_country}")
    st.write(f"Total Satellites: {len(filtered_df)}")

    # Purpose breakdown
    purpose_counts = filtered_df['Purpose'].value_counts()
    fig = px.pie(values=purpose_counts.values, names=purpose_counts.index, title='Satellites by Purpose')
    st.plotly_chart(fig)

    # Type of Orbit
    st.subheader('Satellites by Orbit Type')
    orbit_counts = filtered_df['Class of Orbit'].value_counts()
    fig = px.pie(values=orbit_counts.values, names=orbit_counts.index, title='Satellites by Orbit Type')
    st.plotly_chart(fig)

    # Chat section
    st.subheader('Ask Questions About Satellites')
    st.write("""
    Example questions:
    - What are the main types of Earth observation satellites?
    - How do communication satellites work?
    - What is the average lifespan of a satellite?

    Feel free to ask any question related to satellites and space technology!
    """)

    question = st.text_input('Enter your question about satellites:')
    if question:
        with st.spinner('Generating response...'):
            prompt = f"Human: Answer this question about satellites: {question}\nOnly provide information related to satellites, space technology, and Earth observation. Do not discuss politics, military details, or any sensitive topics.\n\nAssistant:"
            response = get_bedrock_response(prompt)
            st.write(response)

elif page == 'Satellite Search':
    st.title('Satellite Search')

    satellite = st.selectbox('Select a satellite', df['Current Official Name of Satellite'].unique())
    satellite_data = df[df['Current Official Name of Satellite'] == satellite].iloc[0]

    st.header(f"{satellite_data['Current Official Name of Satellite']} / {satellite_data['COSPAR Number']}")
    
    launch_date = satellite_data['Date of Launch']
    if pd.isna(launch_date):
        days_in_orbit = "Unknown"
        launch_date_str = "Unknown"
    else:
        days_in_orbit = (datetime.now() - launch_date).days
        launch_date_str = launch_date.strftime('%Y-%m-%d')
        current_year = datetime.now().year
        st.subheader(f"{current_year} DAY {days_in_orbit}")

    st.write(f"Launch Date: {launch_date_str}")

    st.subheader(f"{satellite_data['Current Official Name of Satellite']} - {satellite_data['Class of Orbit']}")

    col1, col2 = st.columns(2)
    col1.write(f"Orbit: {satellite_data['Type of Orbit']}")
    col1.write(f"Spacecraft Age: {days_in_orbit} days")
    
    # Simulated data for demonstration
    speed = np.random.uniform(7, 8)  # km/s
    altitude = np.random.uniform(400, 1000)  # km
    lat = np.random.uniform(-90, 90)
    lon = np.random.uniform(-180, 180)

    col2.write(f"Vel (km/s): {speed:.2f}  Alt(km): {altitude:.2f}")
    col2.write(f"Lat: {lat:.2f}  Lon: {lon:.2f}")

    # Speed and Altitude gauges
    fig = go.Figure()
    fig.add_trace(go.Indicator(
        mode = "gauge+number",
        value = speed,
        title = {'text': "Speed (km/s)"},
        domain = {'x': [0, 0.5], 'y': [0, 1]}))
    fig.add_trace(go.Indicator(
        mode = "gauge+number",
        value = altitude,
        title = {'text': "Altitude (km)"},
        domain = {'x': [0.5, 1], 'y': [0, 1]}))
    st.plotly_chart(fig)

    # 3D Orbit visualization with detailed Earth
    inclination = float(satellite_data['Inclination (degrees)']) if pd.notna(satellite_data['Inclination (degrees)']) else 0
    perigee = float(satellite_data['Perigee (km)']) if pd.notna(satellite_data['Perigee (km)']) else 6371
    apogee = float(satellite_data['Apogee (km)']) if pd.notna(satellite_data['Apogee (km)']) else 6371

    # Calculate orbit
    t = np.linspace(0, 2*np.pi, 100)
    r = (perigee + apogee) / 2 + 6371  # Average orbit radius
    x = r * np.cos(t)
    y = r * np.sin(t) * np.cos(np.radians(inclination))
    z = r * np.sin(t) * np.sin(np.radians(inclination))

    # Convert to lat/lon
    orbit_lon = np.degrees(np.arctan2(y, x))
    orbit_lat = np.degrees(np.arcsin(z/r))

    fig = go.Figure()

    # Add orbit
    fig.add_trace(go.Scattergeo(
        lon = orbit_lon,
        lat = orbit_lat,
        mode = 'lines',
        line = dict(width = 2, color = 'yellow'),
        name = 'Orbit'
    ))

    # Add satellite position
    fig.add_trace(go.Scattergeo(
        lon = [lon],
        lat = [lat],
        mode = 'markers',
        marker = dict(size = 15, color = 'red', symbol = 'star'),
        name = 'Satellite'
    ))

    fig.update_geos(
        projection_type = "orthographic",
        landcolor = "lightgreen",
        oceancolor = "lightblue",
        showocean = True,
        showland = True,
        showcoastlines = True,
        coastlinecolor = "black",
        showlakes = True,
        lakecolor = "blue",
        showcountries = True,
        countrycolor = "black",
        center = dict(lon = lon, lat = lat),  # Center on satellite position
        projection_rotation = dict(lon = lon, lat = lat, roll = 0)
    )

    fig.update_layout(height=600, margin={"r":0,"t":0,"l":0,"b":0})
    st.plotly_chart(fig)

    # Satellite details
    col1, col2, col3, col4 = st.columns(4)
    col1.write("**Name of Satellite:**\n" + str(satellite_data['Current Official Name of Satellite']))
    col1.write("**Country/Org of UN Registry:**\n" + str(satellite_data['Country/Org of UN Registry']))
    col1.write("**Country of Operator/Owner:**\n" + str(satellite_data['Country of Operator/Owner']))
    col1.write("**Operator/Owner:**\n" + str(satellite_data['Operator/Owner']))

    col2.write("**Users:**\n" + str(satellite_data['Users']))
    col2.write("**Purpose:**\n" + str(satellite_data['Purpose']))
    col2.write("**Class of Orbit:**\n" + str(satellite_data['Class of Orbit']))
    col2.write("**Type of Orbit:**\n" + str(satellite_data['Type of Orbit']))
    col2.write("**Date of Launch:**\n" + launch_date_str)

    col3.write("**Expected Lifetime (yrs):**\n" + str(satellite_data['Expected Lifetime (yrs.)']))
    col3.write("**Contractor:**\n" + str(satellite_data['Contractor']))
    col3.write("**Country of Contractor:**\n" + str(satellite_data['Country of Contractor']))
    col3.write("**Launch Site:**\n" + str(satellite_data['Launch Site']))
    col3.write("**Launch Vehicle:**\n" + str(satellite_data['Launch Vehicle']))

    col4.write("**Comments:**\n" + str(satellite_data['Comments']))
    col4.write(f"**Spacecraft Age:** {days_in_orbit} days")
    
    # Simulated altitude and speed graphs
    st.write("")  # Add some space
    
    dates = pd.date_range(end=datetime.now(), periods=30)
    altitudes = np.random.uniform(400, 1000, 30)
    speeds = np.random.uniform(7, 8, 30)

    fig1 = px.line(x=dates, y=altitudes, title='Altitude (km) - Last 30 Days')
    st.plotly_chart(fig1, use_container_width=True)

    fig2 = px.line(x=dates, y=speeds, title='Speed (km/s) - Last 30 Days')
    st.plotly_chart(fig2, use_container_width=True)

    # Chat about the selected satellite
    st.subheader('Ask Questions About This Satellite')
    satellite_question = st.text_input('Enter your question about this satellite:')
    if satellite_question:
        with st.spinner('Generating response...'):
            prompt = f"Human: Answer this question about the satellite {satellite}: {satellite_question}\nUse the following information about the satellite:\n{satellite_data.to_string()}\nOnly provide information related to this specific satellite, its purpose, and its technology. Do not discuss politics, military details, or any sensitive topics.\n\nAssistant:"
            response = get_bedrock_response(prompt)
            st.write(response)

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("Created with ❤️ by iaasgeek")
