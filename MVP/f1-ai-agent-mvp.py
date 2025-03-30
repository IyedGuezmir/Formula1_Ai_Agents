import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set page configuration
st.set_page_config(
    page_title="F1 AI Agent MVP",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Red Bull-inspired styling
st.markdown("""
<style>
/* Background and text styling */
.stApp {
    background-color: #0D1B2A; /* Dark blue, inspired by Red Bull Racing */
    color: #FFFFFF; /* White text */
}

/* Button styling */
.stButton>button {
    background-color: #FFCC00; /* Yellow, inspired by Red Bull logo */
    color: #0D1B2A; /* Dark blue text */
    border: none;
    font-weight: bold;
}
.stButton>button:hover {
    background-color: #FF0000; /* Red for hover effect */
    color: white;
}

/* Headers styling */
h1, h2, h3 {
    color: #FFCC00!important; /* Yellow headers */
}

/* Dataframe text color */
.dataframe {
    color: white;
}
</style>
""", unsafe_allow_html=True)


# Title and Header
st.title("🏎️ F1 AI Agent: Race Strategy Simulator")

# Sidebar for navigation
st.sidebar.image("https://upload.wikimedia.org/wikipedia/en/thumb/d/d4/Red_Bull_Racing_logo.svg/1920px-Red_Bull_Racing_logo.svg.png", width=200)
st.sidebar.header("F1 AI Agent")
st.sidebar.write("Explore AI-Driven Race Strategies")

# Simulated Race Data Generation
def generate_race_data():
    tracks = ['Monaco', 'Silverstone', 'Monza', 'Spa', 'Suzuka']
    weather_conditions = ['Dry', 'Wet', 'Mixed']
    
    data = {
        'Track': np.random.choice(tracks, 100),
        'Weather': np.random.choice(weather_conditions, 100),
        'Tire_Wear': np.random.uniform(0, 100, 100),
        'Fuel_Consumption': np.random.uniform(80, 110, 100),
        'Lap_Time': np.random.uniform(70, 120, 100)
    }
    
    return pd.DataFrame(data)

# Simulate Race Strategy Agent
class RaceStrategyAgent:
    def predict_strategy(self, track, weather, tire_wear):
        """
        Simple strategy prediction based on input parameters
        """
        risk_factor = np.random.uniform(0.3, 0.7)
        
        if weather == 'Wet':
            return f"Recommend Intermediate Tires (Risk Factor: {risk_factor:.2f})"
        elif tire_wear > 70:
            return f"Pit Stop Recommended (Tire Wear: {tire_wear:.2f})"
        elif track in ['Monaco', 'Singapore']:
            return f"Conservative Strategy (Tight Track: {track})"
        else:
            return f"Aggressive Push Strategy (Risk Factor: {risk_factor:.2f})"

# Main App Logic
def main():
    # Race Data Section
    st.header("Race Data Analysis")
    race_data = generate_race_data()
    
    # Visualization Options
    viz_option = st.selectbox(
        "Choose Visualization", 
        ["Lap Times", "Tire Wear Distribution", "Fuel Consumption"]
    )
    
    # Plotting
    plt.figure(figsize=(10, 6))
    if viz_option == "Lap Times":
        sns.histplot(race_data['Lap_Time'], color='red')
        plt.title("Lap Time Distribution")
    elif viz_option == "Tire Wear Distribution":
        sns.boxplot(x=race_data['Tire_Wear'], color='red')
        plt.title("Tire Wear Analysis")
    else:
        sns.scatterplot(data=race_data, x='Track', y='Fuel_Consumption', hue='Weather')
        plt.title("Fuel Consumption Across Tracks")
    
    st.pyplot(plt)
    
    # Strategy Prediction Section
    st.header("Race Strategy Predictor")
    
    # Inputs for Strategy Prediction
    col1, col2, col3 = st.columns(3)
    
    with col1:
        track = st.selectbox("Select Track", ['Monaco', 'Silverstone', 'Monza', 'Spa', 'Suzuka'])
    
    with col2:
        weather = st.selectbox("Weather Condition", ['Dry', 'Wet', 'Mixed'])
    
    with col3:
        tire_wear = st.slider("Tire Wear", 0, 100, 50)
    
    # Create Strategy Agent
    strategy_agent = RaceStrategyAgent()
    
    # Predict and Display Strategy 
    if st.button("Predict Race Strategy"):
        strategy = strategy_agent.predict_strategy(track, weather, tire_wear)
        st.success(f"Recommended Strategy: {strategy}")
    
    # Additional Information
    st.markdown("---")
    st.markdown("""
    ### 🏆 F1 AI Agent Project
    **Working on  Advanced AI Strategies for Formula 1 Racing**
    
    - Hands-on AI Agent Development
    - Real-world Racing Strategy Implementation
    """)

# Run the app
if __name__ == "__main__":
    main()