import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
df = pd.read_csv('dataset.csv')

# Convert Date column to datetime 
df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y %H:%M')

# Extract date features
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day
df['Day_of_Week'] = df['Date'].dt.dayofweek

# Function to train and predict
def train_and_predict(state_name, future_date):
    X = df[['Month', 'Day', 'Day_of_Week']]
    y = df[state_name]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)

    future_prediction = model.predict(future_date)
    return future_prediction

# Streamlit app
def main():
    st.title('Electricity Consumption Prediction App')

    # Select box for states
    state_name = st.selectbox('Select State', df.columns[1:])

    # Date input box
    st.subheader('Select Future Date')
    month = st.number_input('Month', min_value=1, max_value=12, value=6)
    day = st.number_input('Day', min_value=1, max_value=31, value=15)
    day_of_week = st.number_input('Day of Week', min_value=0, max_value=6, value=5)

    future_date = pd.DataFrame({'Month': [month], 'Day': [day], 'Day_of_Week': [day_of_week]})

    if st.button('Predict'):
        future_prediction = train_and_predict(state_name, future_date)

        # Display predicted value
        st.subheader('Predicted Electricity Consumption')
        st.write(f'Predicted Electricity Consumption for {state_name}: {future_prediction[0]:.2f}')

        # Plot actual vs. predicted 
        X = df[['Month', 'Day', 'Day_of_Week']]
        y = df[state_name]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = RandomForestRegressor(random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(range(len(y_test)), y_test.values, label='Actual', marker='o')
        ax.plot(range(len(y_test)), y_pred, label='Predicted', marker='x')
        ax.set_xlabel('Index')
        ax.set_ylabel('Electricity Consumption')
        ax.set_title(f'Actual vs. Predicted Electricity Consumption for {state_name}')
        ax.legend()
        ax.grid(True)

        st.pyplot(fig)

if __name__ == '__main__':
    main()
