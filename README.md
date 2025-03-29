# customer-churn-prediction

## **Overview**:
This is a machine learning project that predicts whether a bank's customers will churn.  
It is based on the [Churn Modeling Dataset from Kaggle](https://www.kaggle.com/datasets/shrutimechlearn/churn-modelling).  
The model is trained using XGBoost with preprocessed customer data, utilizing the following features from the dataset:
- **Age**
- **Balance**
- **Gender**
- **Geography**
  
The preprocessing steps include handling missing values and applying OneHotEncoding on the Gender and Geography features.

## **Prerequisites**:
In order to run this project, you need the following dependencies installed:
- Python 3.x
- pandas
- scikit-learn
- xgboost
- numpy

### **Running Steps**:
To run this project, follow these steps:

1. **Run the `customer_churn_data_preparer.py` script**  
   This script preprocesses the customer data and generates the One-Hot encoder. It applies the encoder to the "Gender" and "Geography" features in the dataset.

2. **Run the `customer_churn_data_training.py` script**  
   This script trains an XGBoost model using the preprocessed data to predict customer churn. It creates a model that will later be used for making predictions.

3. **Run the `customer_churn_data_predictor.py` script**  
   This script makes predictions on whether a customer will churn based on the trained model. You can modify the input customer data to test different scenarios and see various predictions.

### **Future Improvments**:
- **Improve error handling** to increase the resilience of the application.
- **Explore feature engineering** to enhance the performance of the model.
- **Add support for predicting multiple customers** at once.
- **Explore adding a web UI** to improve the interactivity of the project.
  
