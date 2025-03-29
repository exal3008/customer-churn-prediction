import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
import numpy as np
import pickle
import os

# Load the dataset 
churnDataSet = pd.read_csv('./dataset/Churn_Modelling.csv')

# Handle missing values in the Age Column
ageImputer = SimpleImputer(missing_values=np.nan, strategy='mean')
churnDataSet["Age"] = ageImputer.fit_transform(churnDataSet[["Age"]])

# Handle missing values in the Geography Column
geographyImputer = SimpleImputer(missing_values=np.nan, strategy="constant", fill_value="missing")
churnDataSet[["Geography"]] = geographyImputer.fit_transform(churnDataSet[["Geography"]])

# Apply One-Hot Encoding to the Gender and Geography categorical columns
oneHotEncoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False) 
oneHotEncodedValues = oneHotEncoder.fit_transform(churnDataSet[['Gender', 'Geography']])

# Ensure the 'model' directory exists
if not os.path.exists('./model'):
    os.mkdir('./model')

# Save the OneHotEncoder for future use
with open('./model/onehotEncoder.pkl', 'wb') as oneHotEncoderFile:
    pickle.dump(oneHotEncoder, oneHotEncoderFile)


# Convert encoded values into a DataFrame and merge them with the original dataset
oneHotEncodedDF = pd.DataFrame(oneHotEncodedValues, columns=oneHotEncoder.get_feature_names_out(['Gender', 'Geography']))
churnDataSet = pd.concat([churnDataSet, oneHotEncodedDF], axis=1)

# Drop columns that are unnecessary for training
churnDataSet = churnDataSet.drop(labels=[ 'HasCrCard', 'CustomerId', 'CreditScore', 'RowNumber', 'NumOfProducts', 'IsActiveMember', 'Tenure', 'EstimatedSalary', 'Surname', 'Gender' , 'Geography'], axis=1)

# Save the processed dataset for training
print(churnDataSet.head())
churnDataSet.to_csv("./dataset/Churn_Modelling_Prepared.csv",index=False)

# TODO: Add error handling for file paths and missing columns in future updates.