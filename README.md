# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the necessary libraries
2. Load and analyse the dataset
3. Convert the dataset into pandas dataframe for easier access
4. Go with preprocessing if required
5. Assign the input features and target variable
6. Standardize the input features using StandardScaler
7. Train the model using SGDRegressor and MultiOutputRegressor
8. Now test the model with new values
9. And measure the accuracy using MSE

## Program:
```
/*
Program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor.
Developed by: MOPURI SARADEEPIKA
RegisterNumber:  212224040201
*/
    import numpy as np
    import pandas as pd
    from sklearn.datasets import fetch_california_housing
    from sklearn.linear_model import SGDRegressor
    from sklearn.multioutput import MultiOutputRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error
    from sklearn.preprocessing import StandardScaler
    
    data=fetch_california_housing()
    print(data)
    
    df=pd.DataFrame(data.data,columns=data.feature_names)
    df['target']=data.target
    print(df.head())
    print(df.tail())
    print(df.info())
    
    x=df.drop(columns=['AveOccup','target'])
    y=df['target']
    
    print(x.shape)
    print(y.shape)
    print(x.info())
    print(y.info())
    
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)
    print(x_train.shape)
    print(y_train.shape)
    print(x_test.shape)
    print(y_test.shape)
    
    scaler_x=StandardScaler()
    x_train=scaler_x.fit_transform(x_train)
    x_test=scaler_x.transform(x_test)
    scaler_y=StandardScaler()
    
    y_train = np.array(y_train).reshape(-1, 1)
    y_test = np.array(y_test).reshape(-1, 1)
    y_train=scaler_y.fit_transform(y_train)
    y_test=scaler_y.transform(y_test)
    
    sgd=SGDRegressor(max_iter=1000,tol=1e-3)
    multi_output_sgd=MultiOutputRegressor(sgd)
    multi_output_sgd.fit(x_train,y_train)
    y_pred=multi_output_sgd.predict(x_test)
    y_pred=scaler_y.inverse_transform(y_pred)
    y_test=scaler_y.inverse_transform(y_test)
    
    mse=mean_squared_error(y_test,y_pred)
    print("Mean Squared Error:",mse)
    
    print("\nPredictions:\n",y_pred[:5])
```

## Output:
<img width="1257" height="248" alt="ML 17" src="https://github.com/user-attachments/assets/37481a62-b0f2-46ae-8c7f-fc0185fbb1ba" />

<img width="727" height="485" alt="ML 18" src="https://github.com/user-attachments/assets/eb98130a-3929-42b9-b139-f158b99dfb95" />

<img width="454" height="276" alt="ML 19" src="https://github.com/user-attachments/assets/4d9fa244-a03e-4d35-8ebe-fefde911a8ee" />

<img width="594" height="458" alt="ML 20" src="https://github.com/user-attachments/assets/ba75d48c-50f0-41d0-9446-bd5c044596af" />

<img width="1119" height="88" alt="Screenshot 2025-09-01 163045" src="https://github.com/user-attachments/assets/8687b2cf-563d-4717-8908-c622a4f59798" />


<img width="438" height="150" alt="ML 22" src="https://github.com/user-attachments/assets/37e79891-030c-4de7-a7b9-dd64bc1ed30a" />








## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
