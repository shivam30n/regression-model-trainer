import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import time

# Function to display plots
def plot_predictions(y_true, y_pred):
    plt.figure(figsize=(6, 4))
    sns.regplot(x=y_true, y=y_pred, scatter_kws={'s': 10}, line_kws={'color': 'red'})
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Actual vs Predicted Values')
    plt.legend()
    st.pyplot(plt)

# Function to display metrics
def display_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    st.markdown(f"**📊 Model Evaluation**")
    st.write(f"**Mean Absolute Error (MAE):** {mae:.4f}")
    st.write(f"**Mean Squared Error (MSE):** {mse:.4f}")
    st.write(f"**Root Mean Squared Error (RMSE):** {rmse:.4f}")
    st.write(f"**R-squared (R2):** {r2:.4f}")

# Streamlit UI
st.title("🚀 Regression Model Trainer")

# Initialize step in session_state if not already
if 'step' not in st.session_state:
    st.session_state.step = 1  # Set initial step

def next_step():
    st.session_state.step += 1
st.markdown(
    """
    ## Welcome to the Regression Model Trainer! 🎉

    This web app allows you to train various regression models on your dataset, select features, and evaluate model performance. 
    Whether you're a beginner or an experienced analyst, this tool simplifies the process of creating accurate predictive models.

    **Features:**
    - Upload and preprocess your dataset
    - Select features and target variables for regression
    - Choose from multiple regression models
    - Evaluate model performance with metrics
    - Save your trained model for future use

    ⚠️ **Important:** Please upload a clean dataset with only numerical data. Convert any categorical data to numerical format (e.g., one-hot encoding) before uploading to ensure the model runs smoothly.
    """
)
# Step 1: Upload Dataset
if st.session_state.step >= 1:
    st.header("📤 Step 1: Upload Your Dataset")
    uploaded_file = st.file_uploader("Upload your CSV file", type="csv")

    if uploaded_file is not None:
        # Load dataset
        df = pd.read_csv(uploaded_file)
        st.session_state.df = df  # Store the dataset in session state
        st.write("Data Preview", df.head())

    if uploaded_file is not None:
        st.button("Next Step ➡️", key="next_step_1", on_click=lambda: st.session_state.update({"step": 2}))


# Step 2: Select Features and Target Variable
if st.session_state.step >= 2:
    st.header("⚙️ Step 2: Select Features and Target Variable")
    
    # Use the dataset already uploaded in Step 1
    if 'df' in st.session_state:
        df = st.session_state.df
        features = st.multiselect("Select Features 📊", df.columns.tolist())
        target = st.selectbox("Select Target Variable 🎯", df.columns.tolist())

        if features and target:
            st.session_state.X = df[features]
            st.session_state.y = df[target]
            st.write("Selected Features:", features)
            st.write("Selected Target:", target)

    if features and target:
        st.button("Next Step ➡️", key="next_step_2", on_click=lambda: st.session_state.update({"step": 3}))


# Step 3: Suggested Model Based on Features and Target
if st.session_state.step >= 3:
    st.header("🧠 Step 3: Suggested Model Based on Your Data")

    if len(st.session_state.X.columns) <= 3: 
        suggested_model = "Linear Regression"
    elif len(st.session_state.X.columns) > 3 and len(st.session_state.X.columns) <= 6:
        suggested_model = "Decision Tree Regression"
    else:
        suggested_model = "Random Forest Regression"

    st.write(f"Suggested Model: **{suggested_model}** based on your data.")

    model_options = ["Linear Regression", "Multiple Linear Regression", "Polynomial Regression", 
                     "Random Forest Regression", "Decision Tree Regression", "Support Vector Regression"]
    model_choice = st.selectbox("Choose a Regression Model", model_options)

    st.session_state.model_choice = model_choice
    st.button("Next Step ➡️", key="next_step_3", on_click=lambda: st.session_state.update({"step": 4}))


# Step 4: Split Dataset into Train and Test Sets
if st.session_state.step >= 4:
    st.header("📈Step 4: Split Dataset")
    split_data = st.radio("Do you want to split the data into train and test sets?", ["Yes", "No"])

    if split_data == "Yes":
        test_size = st.slider("Select test size", 0.1, 0.5, 0.2)
        X_train, X_test, y_train, y_test = train_test_split(st.session_state.X, st.session_state.y, test_size=test_size, random_state=42)
        st.session_state.X_train = X_train
        st.session_state.y_train = y_train
        st.session_state.X_test = X_test
        st.session_state.y_test = y_test

        st.write(f"Training set size: {len(X_train)}, Test set size: {len(X_test)}")
    else:
        X_train, y_train = st.session_state.X, st.session_state.y
        st.session_state.X_train = X_train
        st.session_state.y_train = y_train
        st.write(f"Using entire dataset for training: {len(X_train)} rows")

    st.button("Next Step ➡️", key="next_step_4", on_click=lambda: st.session_state.update({"step": 5}))


# Step 5: Feature Scaling Option
if st.session_state.step >= 5:
    st.header("⚖️Step 5: Feature Scaling Option")
    feature_scaling = st.radio("Apply Feature Scaling?", ["Yes", "No"])

    if feature_scaling == "Yes":
        st.session_state.scaler = StandardScaler()
        st.session_state.X_train = st.session_state.scaler.fit_transform(st.session_state.X_train)
        if 'X_test' in st.session_state:
            st.session_state.X_test = st.session_state.scaler.transform(st.session_state.X_test)

    st.button("Next Step ➡️", key="next_step_5", on_click=lambda: st.session_state.update({"step": 6}))


# Step 6: Find Best Parameters and Adjust Parameters (Combined)
if st.session_state.step >= 6:
    st.header("🚀Step 6: Find Best Parameters and Adjust Parameters")

    # GridSearchCV for best parameters
    if st.button("Find Best Parameters"):
        with st.spinner("⏳ Searching for the best parameters... This may take a while"):
            time.sleep(1)  # Short delay to show the spinner

            param_dist = {}
            rand_search = None

            if st.session_state.model_choice == "Support Vector Regression":
                param_dist = {'C': [0.1, 1, 10], 'epsilon': [0.1, 0.2, 0.5]}
                rand_search = RandomizedSearchCV(SVR(), param_dist, n_iter=5, cv=KFold(n_splits=3), scoring='r2', n_jobs=-1, random_state=42)

            elif st.session_state.model_choice == "Decision Tree Regression":
                param_dist = {'max_depth': [3, 5, 10], 'min_samples_split': [2, 5, 10]}
                rand_search = RandomizedSearchCV(DecisionTreeRegressor(), param_dist, n_iter=5, cv=KFold(n_splits=3), scoring='r2', n_jobs=-1, random_state=42)

            elif st.session_state.model_choice == "Random Forest Regression":
                param_dist = {'n_estimators': np.random.randint(50, 200, 5), 'max_depth': [3, 5, 10]}
                rand_search = RandomizedSearchCV(RandomForestRegressor(), param_dist, n_iter=5, cv=KFold(n_splits=3), scoring='r2', n_jobs=-1, random_state=42)

            if rand_search:
                try:
                    rand_search.fit(st.session_state.X_train, st.session_state.y_train)
                    best_params = rand_search.best_params_
                    st.write("✅ **Best parameters found:**", best_params)

                    # Set model to the best one
                    model = rand_search.best_estimator_
                    st.session_state.model = model
                    st.success("Model updated with best parameters!")
                except Exception as e:
                    st.error(f"An error occurred during the parameter search: {e}")

    # Adjusting parameters manually
    if st.session_state.model_choice == "Support Vector Regression":
        C = st.slider("C parameter", 0.1, 10.0, 1.0)
        epsilon = st.slider("Epsilon parameter", 0.1, 1.0, 0.1)
        st.session_state.model = SVR(C=C, epsilon=epsilon)

    elif st.session_state.model_choice == "Decision Tree Regression":
        max_depth = st.slider("Max Depth", 3, 20, 5)
        min_samples_split = st.slider("Min Samples Split", 2, 10, 2)
        st.session_state.model = DecisionTreeRegressor(max_depth=max_depth, min_samples_split=min_samples_split)

    elif st.session_state.model_choice == "Random Forest Regression":
        n_estimators = st.slider("Number of Estimators", 10, 200, 100)
        max_depth = st.slider("Max Depth", 3, 20, 5)
        st.session_state.model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth)

    st.button("Next Step ➡️", key="next_step_6", on_click=lambda: st.session_state.update({"step": 7}))


# Step 7: Train Model
if st.session_state.step >= 7:
    st.header("🏋️‍♂️Step 7: Train Your Model")

    if st.button("Train Model"):
        st.write("Training the model...")

        # Train the model
        model = st.session_state.model
        model.fit(st.session_state.X_train, st.session_state.y_train)

        st.write("Training completed successfully!")

        # Make predictions
        y_pred = model.predict(st.session_state.X_test)
        st.session_state.y_pred = y_pred

        # Display metrics
        display_metrics(st.session_state.y_test, y_pred)
        plot_predictions(st.session_state.y_test, y_pred)

        # Option to download pickle file
        pickle_button = st.download_button(
            label="Download Model as Pickle",
            data=pickle.dumps(model),
            file_name="trained_model.pkl",
            mime="application/octet-stream"
        )