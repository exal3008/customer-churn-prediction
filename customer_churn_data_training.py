import pandas as pd 
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Load training data
training_data = pd.read_csv("./dataset/Churn_Modelling_Prepared.csv")

# Split training data in features and target
X = training_data.drop(columns=['Exited'])
y = training_data['Exited']

# Split the data into training and validation sets(80% train, 20% validation)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Create the dmatrixes from the previous splits
d_train = xgb.DMatrix(X_train, y_train)
d_val = xgb.DMatrix(X_val, y_val)

# Compute scale_post_weight since the churned customers are in a minority
non_churned_training =  y_train.value_counts()[0]
churned_training = y_train.value_counts()[1]
scale_post_weight = non_churned_training / churned_training

# Define the hyperparameters
hyperparameters = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'max_depth': 3,
    'eta': 0.1,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'scale_pos_weight': scale_post_weight
}

# Set the number of rounds the model will be trained
num_round = 100

# Define a watchlist to monitor model performance during training.
# This helps track progress on both the training and validation sets.
watchlist = [(d_train, 'train'), (d_val, 'eval')]

# Specify the watchlist to track the peformance on the validation set
# Early stopping is used to halt training if the validation loss doesn't improve
# for 10 consecutive rounds, preventing overfitting and saving computation time.
model = xgb.train(hyperparameters, d_train, num_boost_round=num_round, evals=watchlist, early_stopping_rounds=10)


# Predict the lables for the validation set
y_pred = model.predict(d_val)

# Convert predictions to binary classification(0 or 1)
y_pred_binary = [1 if x >  0.5 else 0 for x in y_pred]

# Calculate evaluation metrics
accuracy = accuracy_score(y_val, y_pred_binary)
precision = precision_score(y_val, y_pred_binary)
recall = recall_score(y_val, y_pred_binary)
f1 = f1_score(y_val, y_pred_binary)
roc_auc = roc_auc_score(y_val, y_pred)

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1-Score: {f1}")
print(f"AUC-ROC: {roc_auc}")
print('Finished training')

# Saving the model
model.save_model( './model/customer_churn_model')
