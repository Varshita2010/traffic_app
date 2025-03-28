#!/usr/bin/env python
# coding: utf-8

# In[123]:


# import necessary libraries
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 
import statsmodels.formula.api as snf
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from statsmodels.graphics.regressionplots import influence_plot
warnings.filterwarnings('ignore')


# In[5]:


df = pd.read_csv('traffic.csv')
df.head()


# In[7]:


df['DateTime']=pd.to_datetime(df['DateTime'])
df["year"]=df['DateTime'].dt.year
df["Month"]=df['DateTime'].dt.month
df["Hour"]=df['DateTime'].dt.hour
df["Day"]=df['DateTime'].dt.strftime("%A")
df


# In[9]:


df.drop("DateTime",axis=1,inplace=True)


# In[11]:


df.drop("ID",axis=1,inplace=True)


# In[13]:


df.drop("Day",axis=1,inplace=True)


# In[15]:


df.head()


# In[17]:


df.isnull().sum()


# In[19]:


df.info()


# In[21]:


df.describe()


# In[23]:


sns.barplot(x='Junction',y='Vehicles',data=df)
plt.title('Average Traffic Volume')
plt.xlabel('Junctions')
plt.ylabel('Vehicle count')
plt.show()


# In[25]:


sns.barplot(x='Hour',y='Vehicles',data=df)
plt.title('Hourly Traffic')
plt.xlabel('Hours')
plt.ylabel('Vehicle count')
plt.show()


# In[27]:


sns.lineplot(x=df['Month'],y="Vehicles",data=df,hue='Junction')


# In[29]:


sns.scatterplot(x=df['Hour'],y="Vehicles",data=df,hue='Junction')


# In[31]:


sns.lineplot(x=df['year'],y="Vehicles",data=df,hue='Junction')


# In[37]:


x=df.drop("Vehicles", axis=1)
y=df["Vehicles"]


# In[39]:


x


# In[41]:


y


# In[43]:


y.isnull().sum()


# In[45]:


x.isnull().sum()


# In[49]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)


# In[51]:


rf_model = RandomForestRegressor(n_estimators=100,random_state=42)


# In[53]:


rf_model.fit(x_train, y_train)


# In[55]:


y_pred = rf_model.predict(x_test)


# In[57]:


y_pred


# In[71]:


mae = mean_absolute_error(y_test, y_pred)
print(f'Mean Absolute Error: {mae}')


# In[73]:


from sklearn.ensemble import RandomForestRegressor
a=RandomForestRegressor()
a.fit(x_train,y_train)
r1=a.predict(x_train)
r1=pd.DataFrame(r1)
print(a.score(x_train,y_train)*100)


# In[75]:


from sklearn.ensemble import AdaBoostRegressor
a=AdaBoostRegressor()
a.fit(x_train,y_train)
r2=a.predict(x_train)
r2=pd.DataFrame(r1)
print(a.score(x_train,y_train)*100)


# In[81]:


from sklearn.naive_bayes import GaussianNB
a=GaussianNB()
a.fit(x_train,y_train)
r3=a.predict(x_train)
r3=pd.DataFrame(r1)
print(a.score(x_train,y_train)*100)


# In[83]:


df=pd.DataFrame({'Actual':y_test,'Predicted':y_pred})
df.head()


# In[95]:


import cv2
import numpy as np


# In[97]:


# Load image 
img = cv2.imread('image2.png')


# In[99]:


# convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# In[101]:


# Apply GaussianBlur to Reduce noise and improve contour detection
blurred = cv2.GaussianBlur(gray, (5,5),0)


# In[103]:


# Apply Canny edges detection 
edges = cv2.Canny(blurred, 50, 150)


# In[105]:


# Find contours in the edges image 
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


# In[107]:


# Filers contours based on area to identify potential vehicle regions
min_contour_area = 500
vehicle_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]


# In[109]:


# Draw contours on the original image 
cv2.drawContours(img, vehicle_contours, -1, (0, 225,0), 2)


# In[111]:


# Count vehicles 1
vehicle_count = len(vehicle_contours)


# In[113]:


# Disply the result 
cv2.imshow("Image2", img)
cv2.waitKey(1)
cv2.destroyAllWindows()


# In[115]:


# Print the vehicles count
print("Number of vehicles:", vehicle_count)


# In[117]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor


# In[137]:


# Train final model 
modl = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(x_train,y_train)


# In[139]:


y_pred_rf = model.predict(x_test)


# In[141]:


y_pred_rf1 = model.predict(x_train)


# In[145]:


mse_rf1 = mean_absolute_error(y_train, y_pred_rf1)
r2_rf1 = r2_score(y_train, y_pred_rf1)


# In[153]:


print(f"Random Forest - Mean Squared Error: {mae:.2f}")
print(f"Random FoRrest - R² Score: {r2_rf1:.2f}")


# In[91]:


import warnings
warnings.filterwarnings('ignore')
plt.figure(figsize=(5,7))
ax=sns.distplot(y_test,hist=False,color="r",label="Actual Values")
sns.distplot(y_pred,hist=False,color="b",label="Predicted Values",ax=ax)
plt.title("Actual vs Predicted")
plt.show()
plt.close()


# In[163]:


import joblib


# In[127]:


scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)
x_scaled
print("Model Scaled succes")


# In[165]:


model.fit(x_scaled,y)
joblib.dump(model, 'model.pkl')
joblib.dump(scaler, 'scaler.pkl')


# In[ ]:




