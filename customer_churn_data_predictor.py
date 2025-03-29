import pandas as pd
import xgboost as xgb
import pickle

# Define the customer churn data to predict
customerChurnToPredict = pd.DataFrame({
     'Surname': ['Graves'],
     'CreditScore': [667],
     'Geography': ['Germany'],
     'Gender': ['Female'],
     'Age': [59],
     'Tenure': [4],
     "Balance": [103416.72],
     "NumOfProducts": [3],
     "HasCrCard": [1],
     "IsActiveMember": [1],
     "EstimatedSalary": [56678.20]
})


# Load the One-Hot encoder
with  open('./model/onehotEncoder.pkl', "rb") as one_hot_encoder_file:
      oneHotEncoder = pickle.load(one_hot_encoder_file)

# Convert encoded values into a DataFrame and merge them with the original dataset
oneHotEncodedValues = oneHotEncoder.transform(customerChurnToPredict[['Gender', 'Geography']])
oneHotEncodedDF = pd.DataFrame(oneHotEncodedValues, columns=oneHotEncoder.get_feature_names_out(['Gender', 'Geography']))
customerChurnToPredict = pd.concat([customerChurnToPredict, oneHotEncodedDF], axis=1)

# Drop columns that are unnecessary for the prediction
customerChurnToPredict = customerChurnToPredict.drop(labels=[ 'HasCrCard', 'CreditScore',  'NumOfProducts', 'IsActiveMember', 'Tenure', 'EstimatedSalary', 'Surname', 'Gender' , 'Geography'], axis=1)

# Load the model
model  = xgb.Booster(model_file="./model/customer_churn_model")

# Convert the DataFrame into DMatrix
d_pred = xgb.DMatrix(customerChurnToPredict)


# Predict if customer will churn or not
prediction= model.predict(d_pred)

print(prediction)

# TODO: Enhance the code to work with multiple customers.