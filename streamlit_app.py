import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import warnings
import zipfile

warnings.filterwarnings('ignore')

# Set up the Streamlit page
st.set_page_config(
    page_title="Internet Speed and Plan Suggestion App",
    page_icon=":signal_strength:",
    layout="centered",
    initial_sidebar_state="auto"
)

# Load the datasets
df = pd.read_csv(r"C:\Users\arjun\Downloads\New_test.csv")

# Load the plans data from the zip file
with zipfile.ZipFile(r"C:\Users\arjun\Downloads\merged_county_data.zip", 'r') as z:
    with z.open('merged_county_data.csv') as f:
        plans_df = pd.read_csv(f)

# Clean the Income column to remove '$' and ',' and convert to float
df['Income'] = df['Income'].replace('[\\$,]', '', regex=True).astype(float)

# Select relevant features and the target variable
features = ['Age', 'Income', 'Number_of_Devices']
target = 'Current_Internet_Speed'

# Separate the features and target variable
X = df[features]
y = df[target]

# Split the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Random Forest model
model = RandomForestRegressor(n_estimators=50, random_state=42)
model.fit(X_train, y_train)

# Streamlit App UI
st.title("📶 Internet Speed and Plan Suggestion App")
st.markdown(
    """
    <style>
    .stButton>button {
        background-color: #4CAF50; /* Green */
        border: none;
        color: white;
        padding: 15px 32px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        cursor: pointer;
        border-radius: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Extract the unique list of counties for the dropdown
if 'county_name' in plans_df.columns:
    counties = plans_df['county_name'].unique()
else:
    st.error("The required 'county_name' column is missing from the data.")
    counties = []

# Create a dropdown menu for counties
county_name = st.selectbox("Select County", counties)

# User input fields for age, income, and number of devices
age = st.number_input("Age", min_value=0, max_value=120, value=25)
income = st.number_input("Income", min_value=0.0, value=50000.0, step=1000.0)
devices = st.number_input("Number of Devices", min_value=1, value=2, step=1)

# Prediction and Plan Suggestion
if st.button("Find Plan"):
    try:
        # Convert input values to appropriate types
        age = int(age)
        income = float(income)
        devices = int(devices)

        # Predict the internet speed
        input_data = pd.DataFrame([[age, income, devices]], columns=features)
        prediction = model.predict(input_data)[0]

        # Calculate confidence score
        confidence_score = model.score(X_test, y_test) * 100

        # Find top 3 closest matches for speed from the dataset
        top_n = 3
        if prediction > 0:
            unique_matches = plans_df[plans_df['county_name'] == county_name]

            if not unique_matches.empty:
                unique_matches = unique_matches.sort_values(by='max_advertised_download_speed')
                sorted_unique_matches = unique_matches.iloc[(unique_matches['max_advertised_download_speed'] - prediction).abs().argsort()]

                # Select the top 3 unique closest speeds
                top_unique_matches = sorted_unique_matches.head(top_n)
                if len(top_unique_matches) < top_n:
                    remaining_matches = unique_matches[~unique_matches.isin(top_unique_matches)]
                    top_unique_matches = pd.concat([top_unique_matches, remaining_matches.head(top_n - len(top_unique_matches))])
                top_unique_matches = top_unique_matches.sort_values(by='max_advertised_download_speed', ascending=True)
            else:
                top_unique_matches = pd.DataFrame()
        else:
            top_unique_matches = pd.DataFrame()  # Empty DataFrame if no matches

        # Display the results
        st.subheader(f"Suggested Speed: {prediction:.2f} Mbps")
        st.write(f"Confidence Score: {confidence_score:.2f}%")

        if not top_unique_matches.empty:
            st.subheader("Top 3 Unique Matches from Dataset:")
            for _, row in top_unique_matches.iterrows():
                st.write(f"**Brand Name**: {row['brand_name']}")
                st.write(f"**Download Speed**: {row['max_advertised_download_speed']:.2f} Mbps")
                st.write(f"**Upload Speed**: {row['max_advertised_upload_speed']:.2f} Mbps")
                st.write("---")
        else:
            st.write("No unique exact matches found.")
    except ValueError:
        st.error("Please enter valid numeric values.")
