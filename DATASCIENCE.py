#!/usr/bin/env python
# coding: utf-8

# # TASK 1 ANALYZE THE IRIS DATASET USING MACHINE LEARNING ALGORITHMS

# In[ ]:


import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt


# In[5]:


data= pd.read_csv(r"C:\Users\ANIKET\OneDrive\Desktop\DATASCIENCE_INTRNSHIP\archive (14)\Iris.csv")
data


# In[20]:


data.shape


# In[22]:


data.head(20)


# In[23]:


data.tail(20)


# In[28]:


data.isnull().sum()


# In[29]:


data.describe()


# In[12]:


from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score,classification_report


from sklearn.datasets import load_iris
iris= load_iris()
X= iris.data
Y= iris.target
X_train,X_test,Y_train,Y_test= train_test_split(X,Y,test_size=0.3,random_state=42)

tree_clf= DecisionTreeClassifier(random_state=42)

tree_clf.fit(X_train,Y_train)

y_pred= tree_clf.predict(X_train)
y_pred_test= tree_clf.predict(X_test)


train_accuracy= accuracy_score(Y_train,y_pred)
test_accuracy= accuracy_score(Y_test,y_pred_test)
class_report= classification_report(Y_train,y_pred)
class_report_test=classification_report(Y_test,y_pred_test) 

print(train_accuracy)
print(test_accuracy)
print(class_report)
print(class_report_test)


import matplotlib.pyplot as plt 
plt.figure(figsize=(50,30))
plot_tree(tree_clf,filled= True, feature_names= iris.feature_names,class_names= iris.target_names)
plt.title('MODEL TO SHOW IRIS SPECIES RELATIONSHIP USING DECISION TREE CLASSIFIER ',fontsize=50, fontweight='bold', color='black', fontname='Georgia')
plt.show()


# In[24]:


from sklearn.datasets import load_iris
iris = load_iris()

X = iris.data
y = iris.target
fig, axes = plt.subplots(2, 2, figsize=(10, 10))


axes[0, 0].scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Set1)  
axes[0, 1].scatter(X[:, 2], X[:, 3], c=y, cmap=plt.cm.tab10)  
axes[1, 0].scatter(X[:, 0], X[:, 2], c=y, cmap=plt.cm.Purples) 
axes[1, 1].scatter(X[:, 1], X[:, 3], c=y, cmap=plt.cm.Greens)   


axes[0, 0].set_xlabel('Sepal Length (cm)')
axes[0, 0].set_ylabel('Sepal Width (cm)')
axes[0, 0].set_title('Sepal Measurements', fontsize=16)

axes[0, 1].set_xlabel('Petal Length (cm)')
axes[0, 1].set_ylabel('Petal Width (cm)')
axes[0, 1].set_title('Petal Measurements', fontsize=16)

axes[1, 0].set_xlabel('Sepal Length (cm)')
axes[1, 0].set_ylabel('Petal Length (cm)')
axes[1, 0].set_title('Sepal vs. Petal Length', fontsize=16)

axes[1, 1].set_xlabel('Sepal Width (cm)')
axes[1, 1].set_ylabel('Petal Width (cm)')
axes[1, 1].set_title('Sepal vs. Petal Width', fontsize=16)

fig.suptitle('Iris Feature Relationships', fontsize=30, fontweight='bold', color='black', fontname='Georgia')
plt.tight_layout()
plt.show()


# In[23]:


import pandas as pd 
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

iris = load_iris()
X= iris.data
y= iris.target

X_train,X_test,Y_train,Y_test= train_test_split(X,y,test_size=0.2,random_state=42)


random_forest_model= RandomForestClassifier( n_estimators = 50 , random_state= 42)

random_forest_model.fit(X_train, Y_train)
y_pred= random_forest_model.predict(X_test)

train_accuracy= accuracy_score(Y_test,y_pred)
print (f'Random Forest Model Accuracy:{train_accuracy}')


importances = random_forest_model.feature_importances_


plt.figure(figsize=(8, 6))
plt.bar(iris.feature_names, importances,color='red')
plt.xlabel('Feature Name')
plt.ylabel('Importance')
plt.title('Feature Importance for Random Forest Classifier',fontsize=30, fontweight='bold', color='black', fontname='Georgia')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()


# In[27]:


import pandas as pd 
from sklearn.ensemble import BaggingClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

iris = load_iris()
X= iris.data
y= iris.target

X_train,X_test,Y_train,Y_test= train_test_split(X,y,test_size=0.2,random_state=42)
base_model = DecisionTreeClassifier()

bagging_model= BaggingClassifier(base_model, n_estimators = 50 , random_state= 42)

bagging_model.fit(X_train, Y_train)
y_pred= bagging_model.predict(X_test)

train_accuracy= accuracy_score(Y_test,y_pred)
print (f'Bagging Model Accuracy:{train_accuracy}')


importances = []
for estimator in bagging_model.estimators_:
    importances.append(estimator.feature_importances_)

average_importances = np.mean(importances, axis=0)


plt.figure(figsize=(8, 6))
plt.bar(iris.feature_names, average_importances, color='purple')
plt.xlabel('Feature Name')
plt.ylabel('Importance')
plt.title('Feature Importance for Bagging Classification Technique', fontsize=30,
          fontweight='bold', color='black', fontname='Georgia')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()


# # TASK 2 - CAR PREDICTION USING MACHINE LEARNING 

# In[59]:


import pandas as pd 
import matplotlib.pyplot as plt 


# In[60]:


data= pd.read_csv(r"C:\Users\ANIKET\OneDrive\Desktop\DATASCIENCE_INTRNSHIP\archive (16)\car data.csv")
data


# In[46]:


data.describe()


# In[53]:


data.isnull().sum()


# In[63]:


data1=data.drop(columns=['Transmission','Selling_type','Owner'])
print(data1)


# In[64]:


data1


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[9]:


import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score


# In[43]:


data= pd.read_csv(r"C:\Users\ANIKET\OneDrive\Desktop\DATASCIENCE_INTRNSHIP\archive (16)\car data.csv")
x= data[['Driven_kms']]
y=data['Selling_Price']
x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.2,random_state=42)

model=LinearRegression()
model.fit(x_train,y_train)

y_prediction=model.predict(x_test)
mae=mean_absolute_error(y_test,y_prediction)
mse=mean_squared_error(y_test,y_prediction)
r2=r2_score(y_test,y_prediction)

print(f'MAE:{mae},MSE:{mse},R-SQUARE:{r2}')

plt.scatter(x,y,color='blue')
plt.plot(x,model.predict(x),color='green')
plt.xlabel('Driven Kms',fontsize=10)
plt.ylabel('Rate of Selling Price',fontsize=10)
plt.title('PREDICTION OF CAR SELLING PRICES USING LINEAR REGRESSION MODEL',fontsize=14, fontweight='bold', color='BLACK', fontname='Georgia')


# In[36]:


data= pd.read_csv(r"C:\Users\ANIKET\OneDrive\Desktop\DATASCIENCE_INTRNSHIP\archive (16)\car data.csv")
print(data.columns)
car_details = data.groupby('Car_Name')['Present_Price'].sum() 
price_rate = car_details.nlargest(10)

plt.figure(figsize=(13,10))
price_rate.plot(kind='bar', color='green')
plt.title('PREDICTION OF PRICES OF CARS FROM 2005-2017', fontsize=30, fontweight='bold', color='BLACK', fontname='Georgia')
plt.xlabel('Names of Cars',fontsize=17)
plt.ylabel('Rate of Present Price',fontsize=17)
plt.xticks(rotation=60, ha='right')  


for index, value in enumerate(price_rate):
    plt.text(index, value, str(value), ha='center', va='bottom')

plt.tight_layout()
plt.show()


# In[95]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

data= pd.read_csv(r"C:\Users\ANIKET\OneDrive\Desktop\DATASCIENCE_INTRNSHIP\archive (16)\car data.csv")

X = data[['Year', 'Present_Price', 'Driven_kms']]
y = data['Selling_Price']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

mlr = LinearRegression()
mlr.fit(X_train, y_train)

y_pred_mlr = mlr.predict(X_test)

mse_mlr = mean_squared_error(y_test, y_pred_mlr)
r2_mlr = r2_score(y_test, y_pred_mlr)

print("Multiple Linear Regression:")
print(f"Mean squared error: {mse_mlr}")
print(f"R-squared score: {r2_mlr}")


poly = PolynomialFeatures(degree=2)
X_poly_train = poly.fit_transform(X_train)
X_poly_test = poly.transform(X_test)

pr = LinearRegression()
pr.fit(X_poly_train, y_train)

y_pred_pr = pr.predict(X_poly_test)

mse_pr = mean_squared_error(y_test, y_pred_pr)
r2_pr = r2_score(y_test, y_pred_pr)

print("\nPolynomial Regression:")
print(f"Mean squared error: {mse_pr}")
print(f"R-squared score: {r2_pr}")

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_mlr, color='red', label='Multiple Linear Regression')
plt.scatter(y_test, y_pred_pr, color='darkgreen', label='Polynomial Regression')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Actual vs Predicted Prices')
plt.legend()
plt.show()


# In[96]:


pip install tensorflow


# In[99]:


pip install keras


# In[113]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Load the data
data = pd.read_csv(r"C:\Users\ANIKET\OneDrive\Desktop\DATASCIENCE_INTRNSHIP\archive (16)\car data.csv")

# Separate features and target
X = data.drop('Selling_Price', axis=1)
y = data['Selling_Price']

# Identify numeric and categorical columns
numeric_features = ['Year', 'Present_Price', 'Driven_kms']
categorical_features = ['Fuel_Type', 'Selling_type', 'Transmission', 'Owner']


# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit preprocessor on training data and transform both training and test data
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# Get the number of features after preprocessing
n_features = X_train_processed.shape[1]

# Create the DNN model
model = Sequential([
    Dense(64, activation='relu', input_shape=(n_features,)),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(1)
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

# Define early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model
history = model.fit(
    X_train_processed, y_train,
    validation_split=0.2,
    epochs=100,
    batch_size=32,
    callbacks=[early_stopping],
    verbose=1
)

# Evaluate the model
train_mse = model.evaluate(X_train_processed, y_train, verbose=0)
test_mse = model.evaluate(X_test_processed, y_test, verbose=0)

print(f"Train MSE: {train_mse}")
print(f"Test MSE: {test_mse}")

# Make predictions
y_pred = model.predict(X_test_processed)

# Calculate additional metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Root Mean Squared Error: {rmse}")
print(f"Mean Absolute Error: {mae}")
print(f"R-squared Score: {r2}")

# Plot actual vs predicted prices
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred,color='darkgreen')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=3)
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Actual vs Predicted Prices')
plt.show()

# Plot training history
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss',color='purple')
plt.plot(history.history['val_loss'], label='Validation Loss',color='brown')
plt.xlabel('Epoch')
plt.ylabel('Mean Squared Error')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

# Plot residuals
residuals = y_test - y_pred.flatten()
plt.figure(figsize=(10, 6))
plt.scatter(y_pred, residuals,color='darkblue')
plt.xlabel('Predicted Price')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.axhline(y=0, color='b', linestyle='--',lw=3)
plt.show()

# Plot feature importance
feature_importance = np.abs(model.layers[0].get_weights()[0]).mean(axis=1)
feature_names = preprocessor.get_feature_names_out()
plt.figure(figsize=(12, 6))
plt.bar(feature_names, feature_importance,color= 'green')
plt.xticks(rotation=90)
plt.xlabel('Features')
plt.ylabel('Importance')
plt.title('Feature Importance')
plt.tight_layout()
plt.show()


# In[ ]:




