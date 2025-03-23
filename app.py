import streamlit as st
import numpy as np
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# ✅ This must be the first Streamlit command in your script
st.set_page_config(
    page_title="Personal Fitness Tracker",  # Change this to your app name
    page_icon="appIcon.png",  # Change this to an emoji or image file
    layout="wide"  # Options: "centered" or "wide"
)

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')




# Custom CSS for Aesthetic UI
st.markdown(
    """
    <style>
        /* Background */
        body {
            background: linear-gradient(to right, #0f0c29, #302b63, #24243e);
            color: white;
        }

        /* Sidebar */
        [data-testid="stSidebar"] {
            background: #1e1e2f;
            color: white;
        }

        /* Titles */
        h1, h2, h3 {
            color: #f5f5f5;
            text-align: center;
            font-family: 'Poppins', sans-serif;
        }

        /* Buttons */
        .stButton>button {
            border-radius: 8px;
            background-color: #ff9800;
            color: white;
            font-size: 16px;
            padding: 8px 16px;
            border: none;
        }
        .stButton>button:hover {
            background-color: #ff5722;
        }

        /* DataFrame */
        .dataframe {
            border-radius: 10px;
            border: 2px solid white;
            background: rgba(255, 255, 255, 0.1);
        }

        /* Progress Bar */
        div[data-testid="stProgress"] > div {
            background-color: #ff9800 !important;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Title
st.title("💪 Personal Fitness Tracker")
st.markdown(
    "### 🌟 **Track Your Workout Like a Pro!**\
    Enter your **Age, Gender, BMI, Duration,** and more to see your predicted calorie burn."
)

# Sidebar for User Input
st.sidebar.header("🎯 User Input Parameters")

def user_input_features():
    col1, col2 = st.sidebar.columns(2)
    age = col1.slider("Age", 10, 100, 30)
    bmi = col2.slider("BMI", 15.0, 40.0, 20.0)
    duration = col1.slider("Duration (min)", 0, 35, 15)
    heart_rate = col2.slider("Heart Rate", 60, 130, 80)
    body_temp = col1.slider("Body Temperature (°C)", 36.0, 42.0, 38.0)
    gender = st.sidebar.radio("Gender", ["Male", "Female"], horizontal=True)
    
    gender_value = 1 if gender == "Male" else 0  # Proper encoding

    return pd.DataFrame({
        "Age": [age],
        "BMI": [bmi],
        "Duration": [duration],
        "Heart_Rate": [heart_rate],
        "Body_Temp": [body_temp],
        "Gender_male": [gender_value]  # Ensure correct encoding
    })

df = user_input_features()

# Display User Input
st.markdown("---")
st.subheader("📊 Your Parameters")
st.dataframe(df, width=700)

# Load Data Function
@st.cache_data
def load_data():
    calories = pd.read_csv("calories.csv")
    exercise = pd.read_csv("exercise.csv")
    data = exercise.merge(calories, on="User_ID").drop(columns=["User_ID"])

    # Ensure Gender is encoded correctly
    data["Gender_male"] = data["Gender"].map({"Male": 1, "Female": 0})
    data.drop(columns=["Gender"], inplace=True)  # Drop original gender column

    # Add BMI Column
    data["BMI"] = round(data["Weight"] / ((data["Height"] / 100) ** 2), 2)
    return data

exercise_df = load_data()

# Train-Test Split
exercise_train, exercise_test = train_test_split(exercise_df, test_size=0.2, random_state=1)

# Drop unnecessary columns
for dataset in [exercise_train, exercise_test]:
    dataset.drop(columns=["Weight", "Height"], inplace=True)

X_train = exercise_train.drop("Calories", axis=1)
y_train = exercise_train["Calories"]
X_test = exercise_test.drop("Calories", axis=1)
y_test = exercise_test["Calories"]

# Train Model
random_reg = RandomForestRegressor(n_estimators=1000, max_depth=6, random_state=1)
random_reg.fit(X_train, y_train)

# Align Input Data with Model
df = df.reindex(columns=X_train.columns, fill_value=0)

# Prediction
prediction = random_reg.predict(df)

# Display Prediction with Progress Bar
st.markdown("---")
st.subheader("🔥 **Estimated Calories Burned**")

progress_bar = st.progress(0)
for percent in range(100):
    time.sleep(0.01)
    progress_bar.progress(percent + 1)
progress_bar.empty()

st.success(f"💪 **{round(prediction[0], 2)} kilocalories**")

# Find Similar Results
st.markdown("---")
st.subheader("📋 Similar Workout Records")
calorie_range = [prediction[0] - 10, prediction[0] + 10]
similar_data = exercise_df[(exercise_df["Calories"] >= calorie_range[0]) & (exercise_df["Calories"] <= calorie_range[1])]

st.dataframe(similar_data.sample(5) if not similar_data.empty else "No similar records found.")

# General Information
st.markdown("---")
st.subheader("📈 Your Workout Compared to Others")
st.markdown(
    f"🔹 You are **older** than {round((exercise_df['Age'] < df['Age'][0]).mean() * 100, 2)}% of users."
)
st.markdown(
    f"🔹 Your **exercise duration** is longer than {round((exercise_df['Duration'] < df['Duration'][0]).mean() * 100, 2)}% of users."
)
st.markdown(
    f"🔹 Your **heart rate** is higher than {round((exercise_df['Heart_Rate'] < df['Heart_Rate'][0]).mean() * 100, 2)}% of users."
)
st.markdown(
    f"🔹 Your **body temperature** is higher than {round((exercise_df['Body_Temp'] < df['Body_Temp'][0]).mean() * 100, 2)}% of users."
)

# Footer
st.markdown("---")
st.markdown(
    "💡 **Tip:** Stay hydrated and maintain a balanced diet to optimize your workout efficiency!"
)
