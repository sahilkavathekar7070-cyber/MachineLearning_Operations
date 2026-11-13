# Import Neccessory Libraries 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler,MinMaxScaler,RobustScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor

# Step1: Data Ingestion
def data_ingestion():
    filepath = r"C:\MachineLearning_Operations\data\Concrete_Data.csv"
    df = pd.read_csv(filepath)
    return df
# Step2: Data Preprocessing and Cleaning
def data_preprocessing(df):
  '''
  Step1: Feature Engineering and Selection

  Step2: Split the dataset into independent and dependent columns i.e X and y

  Step3: Use train and test split i.e. Split the dataset into Seen and Unseen
  (Seen Data: train data and Unseen Data: test data)

  Step4: Use Scaling Technique/ Normalization Technique
  (StandardScaler/MinMaxScaler/RobustScaler)

  Step5: Return Seen and Unseen Data
  '''
  # Step1: Feature Engineering : Create a new feature based on existing feature

  df['Water_Binder_Ratio'] = df['Water  (component 4)(kg in a m^3 mixture)'] / (df['Cement (component 1)(kg in a m^3 mixture)'] + df['Blast Furnace Slag (component 2)(kg in a m^3 mixture)'] + df['Fly Ash (component 3)(kg in a m^3 mixture)'])

  # Step2: Split the dataset into X and y
  target_col = 'Concrete compressive strength(MPa, megapascals) '
  X = df.drop(columns = [target_col])  # Independent Columns
  y = df[target_col]                   # Target Column

  # Step3: Split the dataset into Seen and Unseen (train and test data)
  from sklearn.preprocessing import MinMaxScaler,RobustScaler,StandardScaler
  from sklearn.model_selection import train_test_split

  X_train,X_test,y_train,y_test = train_test_split(X,y,
                                                  test_size = 0.3,
                                                  random_state = 1)

  # Step4: Use Scaling/Normalization Technique
  '''
  * Scaling for training data: fit_transform()

  * Scaling for test data: transform()
  '''
  sc = MinMaxScaler()
  X_train = sc.fit_transform(X_train)   # Seen Data
  X_test = sc.transform(X_test)         # Unseen Data

  # Step5: return seen and unseen data
  return X_train,X_test,y_train,y_test

# Step3: Model Building
# Step 3 : Model Training and Evaluation
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, SGDRegressor, HuberRegressor, RANSACRegressor, TheilSenRegressor
from sklearn.metrics import r2_score

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,ExtraTreesRegressor,GradientBoostingRegressor # Ensemble(Bagging techniques)

def model_build(X_train,X_test,y_train,y_test) :
  model= RandomForestRegressor()
  model.fit(X_train,y_train) # seen data
  y_pred = model.predict(X_test) # unseen data
  score =r2_score(y_test,y_pred)
  print(f'R2 Score : {score}')
  return score


# Define Orchestration Layer Design : Entry Point 

def main(): # Function Definition
    # Step1: Data Ingestion
    df = data_ingestion() 
    # Step2: Data Preprocessing and Cleaning
    X_train,X_test,y_train,y_test = data_preprocessing(df)
    # Step3: Model Building
    score = model_build(X_train,X_test,y_train,y_test)  
    print("R2 Score is : ",score)
    
main()      # Function calling
