# Regression Model Trainer 🚀📊

Welcome to the **Regression Model Trainer** app! This app helps you easily train regression models on your dataset without any hassle. With a simple, user-friendly interface, you can upload your data, select the appropriate features, choose a regression model, tune hyperparameters, and visualize predictions—all in one place!

## ✨ Features

- **Upload Dataset**: Easily upload your CSV files for regression model training.
- **Feature Selection**: Select features and the target variable for training.
- **Suggested Models**: Based on your dataset's features, the app suggests the most suitable regression model.
- **Data Splitting**: Choose to split your dataset into training and testing sets for model evaluation.
- **Feature Scaling**: Apply feature scaling to standardize the data before model training.
- **Hyperparameter Tuning**: Automatically tune hyperparameters using RandomizedSearchCV to optimize your models.
- **Visualize Results**: View actual vs predicted values and key performance metrics like MAE, MSE, RMSE, and R2.
- **Download Trained Model**: Save your trained model as a pickle file for future use.
- **Predict on New Data**: Once the model is trained, make predictions on new, unseen data.

## 📥 Getting Started

To get started, simply clone this repository and run the app using Streamlit. Here's how you can do it:

1. **Clone the repository**:

   ```bash
   git clone https://github.com/yourusername/regression-model-trainer.git
   cd regression-model-trainer

2. **Install the required dependencies:**

pip install -r requirements.txt

⚠️ Important Note:
Make sure that the dataset you upload contains clean, non-categorical data. If your dataset includes categorical data, please convert it to numerical format before uploading. This will ensure proper model training and avoid potential errors.

🎨 Customization
You can customize the appearance of this app, including colors and themes, by modifying the config.toml file located in the .streamlit directory.

📈 Model Choices
The app supports various regression models including:

Linear Regression
Multiple Linear Regression
Polynomial Regression
Decision Tree Regression
Random Forest Regression
Support Vector Regression
The app will suggest the best model based on the number of features in your dataset, but you can choose any model that best suits your needs.

📊 Visualizations & Metrics
Once the model is trained, the app provides visualizations to compare the predicted vs actual values. The following metrics are displayed to help you assess model performance:

Mean Absolute Error (MAE)
Mean Squared Error (MSE)
Root Mean Squared Error (RMSE)
R-squared (R2)
💾 Save Your Work
After training the model, you can download it as a .pkl file and use it for future predictions without needing to retrain.

🛠️ Technologies Used
Streamlit: For building the interactive app.
Pandas: For data manipulation.
Scikit-learn: For machine learning models and hyperparameter tuning.
Matplotlib & Seaborn: For visualizing results.
Pickle: For saving and loading models.

Scikit-learn
🤝 Contributing
If you'd like to contribute to this project, feel free to fork the repository and submit a pull request with your changes. Any suggestions or improvements are always welcome!

📄 License
This project is licensed under the MIT License - see the LICENSE file for details.
