#!/usr/bin/env python
# coding: utf-8

# ### Import Libraries

# In[102]:


import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# ### Import Datasets

# **Accidents Dataset**

# In[2]:


df_a = pd.read_csv("Road Safety Data - Accidents 2019.csv")
df_a.head(2)


# **Vehicles Dataset**

# In[3]:


df_v = pd.read_csv("Road Safety Data- Vehicles 2019.csv")
df_v.head(2)


# **Combine Datasets**

# In[4]:


df = pd.merge(df_a, df_v, how = "inner", on = "Accident_Index")
df.head(2)


# ### Questions

# **(a) Are there significant hours of the day, and days of the week, on which accidents occur?**

# Accidents occur by Hour

# In[5]:


hours = df_a["Time"].str.split(":").str.get(0)
hours = pd.DataFrame(hours)
hours = pd.DataFrame(hours.value_counts().sort_index()).reset_index()
hours.columns = ["Hours", "Accidents"]
hours


# In[116]:


sns.set_style("whitegrid")
plt.figure(figsize = (18, 7))
sns.barplot(x = "Hours", y = "Accidents", data = hours)
plt.title("Road Accidents by Hour", size = 20)
plt.xlabel("Hour", size = 20)
plt.ylabel("Accidents", size = 20)
plt.xticks(size = 15)
plt.yticks(size = 15)
plt.savefig("raby.png")
plt.show()


# Accidents Occur by Day of Week

# In[7]:


day_of_week = df_a["Day_of_Week"].value_counts().sort_index().to_frame().reset_index()
day_of_week.columns = ["Day of Week", "Accidents"]
day_of_week


# In[117]:


plt.figure(figsize = (18, 7))
sns.barplot(x = "Day of Week", y = "Accidents", data = day_of_week, palette = "hsv")
plt.title("Road Accidents by Day of Week", size = 20)
plt.xlabel("Day of Week", size = 20)
plt.ylabel("Accidents", size = 20)
plt.xticks(size = 15)
plt.yticks(size = 15)
plt.savefig("radw.png")
plt.show()


# **(b) For motorbikes, are there significant hours of the day, and days of the week, on which accidents occur?**

# In[9]:


df_a["Hour"] = df_a["Time"].str.split(":").str.get(0)


# In[10]:


df["Hour"] = df["Time"].str.split(":").str.get(0)


# In[11]:


motorbikes = df[(df["Vehicle_Type"] == 2) |
                (df["Vehicle_Type"] == 3) |
                (df["Vehicle_Type"] == 4) |
                (df["Vehicle_Type"] == 5) |
                (df["Vehicle_Type"] == 23)|
                (df["Vehicle_Type"] == 97)]
motorbikes.head(2)


# Motorbikes accidents occur by hour of day

# In[12]:


bikes_hours = motorbikes.groupby(["Hour", "Vehicle_Type"])["Vehicle_Type"].agg(["count"]).reset_index()
bikes_hours


# In[131]:


plt.figure(figsize = (18, 7))
sns.barplot(x = "Hour", y = "count", hue = "Vehicle_Type", data = bikes_hours)
plt.title("Motorbikes involved in Accidents by Hour", size = 20)
plt.xlabel("Hour", size = 20)
plt.ylabel("Motorbikes involved in Accidents", size = 20)
plt.xticks(size = 15)
plt.yticks(size = 15)
plt.savefig("mah.png")
plt.show()


# Motorbikes accidents occur by day of week

# In[14]:


bikes_day_of_week = motorbikes.groupby(["Day_of_Week", "Vehicle_Type"])["Vehicle_Type"].agg(["count"]).reset_index()
bikes_day_of_week


# In[143]:


bikes_day_of_week[bikes_day_of_week["Vehicle_Type"] == 97].sort_values(by = "count")


# In[144]:


plt.figure(figsize = (18, 7))
sns.barplot(x = "Day_of_Week", y = "count", hue = "Vehicle_Type", data = bikes_day_of_week, palette = "plasma")
plt.title("Motorbikes Accidents by Day of Week", size = 20)
plt.xlabel("Day of Week", size = 20)
plt.ylabel("Motorbikes Accidents", size = 20)
plt.xticks(size = 15)
plt.yticks(size = 15)
plt.savefig("MADW.png")
plt.show()


# **(c) For pedestrians involved in accidents, are there significant hours of the day, and days of the week, on which they are more likely to be involved?**

# padestrians involved in accidents by hours of the day

# In[16]:


padestrians_hour = df_a.groupby(["Hour", "Pedestrian_Crossing-Physical_Facilities"])["Pedestrian_Crossing-Physical_Facilities"].agg(["count"]).reset_index()
padestrians_hour


# In[134]:


plt.figure(figsize = (18, 7))
sns.barplot(x = "Hour", y = "count", hue = "Pedestrian_Crossing-Physical_Facilities", data = padestrians_hour)
plt.title("Padestrians involved in Accidents by Hour", size = 20)
plt.xlabel("Hour", size = 20)
plt.ylabel("Padestrians involved in Accidents", size = 20)
plt.xticks(size = 15)
plt.yticks(size = 15)
plt.savefig("PAH.png")
plt.show()


# padestrians involved in accidents by day of week

# In[18]:


padestrians_day = df_a.groupby(["Day_of_Week", "Pedestrian_Crossing-Physical_Facilities"])["Pedestrian_Crossing-Physical_Facilities"].agg(["count"]).reset_index()
padestrians_day


# In[161]:


plt.figure(figsize = (18, 7))
sns.barplot(x = "Day_of_Week", y = "count", hue = "Pedestrian_Crossing-Physical_Facilities", data = padestrians_day, palette = "hot")
plt.title("Padestrians Accidents by Day of Week", size = 20)
plt.xlabel("Day of Week", size = 20)
plt.ylabel("Padestrians Accidents", size = 20)
plt.xticks(size = 15)
plt.yticks(size = 15)
plt.savefig("PADW.png")
plt.show()


# **(d) What impact, if any, does daylight savings have on road traffic accidents in the week after it starts and stops?**

# In[20]:


def daylight(x):
    if x == 1:
        return "Datlight Starts"
    else:
        return "Daylight Stops"
    
df_a["daylight"] = df_a["Light_Conditions"].apply(daylight)


# In[21]:


daylight_week = df_a.groupby(["Day_of_Week", "daylight"])["daylight"].agg(["count"]).reset_index()
daylight_week


# In[162]:


plt.figure(figsize = (18, 7))
sns.barplot(x = "Day_of_Week", y = "count", hue = "daylight", data = daylight_week, palette = "magma_r")
plt.title("Road Accidents by Daylight in Day of Week", size = 20)
plt.xlabel("Day of Week", size = 20)
plt.ylabel("Road Accidents", size = 20)
plt.xticks(size = 15)
plt.yticks(size = 15)
plt.savefig("RADW.png")
plt.show()


# **(e)What impact, if any, does sunrise and sunset times have on road traffic accidents?**

# In[23]:


df_a["Date"] = pd.to_datetime(df_a["Date"])


# In[24]:


df_a["Month"] = df_a["Date"].dt.month


# In[25]:


df_a["Hour"].fillna(0, inplace = True)


# In[26]:


df_a["Hour"] = df_a["Hour"].astype("int")


# In[29]:


def sun(vec):
    month = vec[0]
    hour = vec[1]
    
    if month == 1 and (hour == 8 and hour == 7):
        return "sunrise"
    if month == 1 and hour == 16:
        return "sunset"
    
    if month == 2 and (hour == 7 and hour == 6):
        return "sunrise"
    if month == 2 and (hour == 17 and hour == 16):
        return "sunset"
    
    if month == 3 and (hour == 6 and hour == 5):
        return "sunrise"
    if month == 3 and (hour == 18 and hour == 17):
        return "sunset"
    
    if month == 4 and (hour == 6 and hour == 5):
        return "sunrise"
    if month == 4 and (hour == 20 and hour == 19):
        return "sunset"
    
    if month == 5 and (hour == 5 and hour == 4):
        return "sunrise"
    if month == 5 and (hour == 21 and hour == 20):
        return "sunset"
    
    if month == 6 and hour == 4:
        return "sunrise"
    if month == 6 and hour == 21:
        return "sunset"
    
    if month == 7 and (hour == 5 and hour == 4):
        return "sunrise"
    if month == 7 and (hour == 21 and hour == 20):
        return "sunset"
    
    if month == 8 and (hour == 6 and hour == 5):
        return "sunrise"
    if month == 8 and (hour == 20 and hour == 19):
        return "sunset"
    
    if month == 9 and hour == 6:
        return "sunrise"
    if month == 9 and (hour == 19 and hour == 18):
        return "sunset"
    
    if month == 10 and (hour == 6):
        return "sunrise"
    if month == 10 and (hour == 19 and hour == 18):
        return "sunset"
    
    if month == 11 and (hour == 7 and hour == 6):
        return "sunrise"
    if month == 11 and (hour == 16 and hour == 15):
        return "sunset"
    
    if month == 12 and (hour == 8 and hour == 7):
        return "sunrise"
    if month == 12 and hour == 15:
        return "sunset"
    
df_a["sun_rise_set"] = df_a[["Month", "Hour"]].apply(sun, axis = 1)


# In[34]:


accidents_sun = df_a["sun_rise_set"].value_counts().to_frame().reset_index()
accidents_sun.columns = ["Sun", "Accidents"]
accidents_sun


# In[163]:


plt.figure(figsize = (18, 7))
sns.barplot(x = "Sun", y = "Accidents", data = accidents_sun, palette = "hsv")
plt.title("Road Accidents by Sun Rise and Sun Set", size = 20)
plt.xlabel("Sun", size = 20)
plt.ylabel("Road Accidents", size = 20)
plt.xticks(size = 15)
plt.yticks(size = 15)
plt.savefig("rass.png")
plt.show()


# **(f) Are there particular types of vehicles that are more frequently involved in road traffic accidents?**

# Top 10 Most Number of Accidents By Age of Vehicle

# In[49]:


av = df["Age_of_Vehicle"].value_counts().to_frame().reset_index()
av.columns = ["Age of Vehicles", "Accidents"]
av = av.head(10)
av = av.sort_values(by = "Age of Vehicles")
av


# In[164]:


plt.figure(figsize = (18, 7))
sns.barplot(x = "Age of Vehicles", y = "Accidents", data = av, palette = "spring")
plt.title("Top 10 Most Number of Accidents By Age of Vehicle", size = 20)
plt.xlabel("Age of Vehicles", size = 20)
plt.ylabel("Road Accidents", size = 20)
plt.xticks(size = 15)
plt.yticks(size = 15)
plt.savefig("AAV.png")
plt.show()


# Road Accidents By Vehicle Propulsion

# In[55]:


vra = df["Propulsion_Code"].value_counts().sort_index().to_frame().reset_index()
vra.columns = ["Vehicle Propulsion", "Road Accidents"]
vra


# In[166]:


plt.figure(figsize = (18, 7))
sns.barplot(x = "Vehicle Propulsion", y = "Road Accidents", data = vra, palette = "plasma")
plt.title("Road Accidents by Vehicle Propulsion", size = 20)
plt.xlabel("Vehicle Propulsion", size = 20)
plt.ylabel("Road Accidents", size = 20)
plt.xticks(size = 15)
plt.yticks(size = 15)
plt.savefig("VP.png")
plt.show()


# **(g) Are there particular conditions (weather, geographic location, situations) that generate more road traffic accidents?**

# Road Accidents by Weather conditions

# In[60]:


wc = df_a["Weather_Conditions"].value_counts().sort_index().to_frame().reset_index()
wc.columns = ["Weather Condition", "Road Accidents"]
wc


# In[168]:


plt.figure(figsize = (18, 7))
sns.barplot(x = "Weather Condition", y = "Road Accidents", data = wc, palette = "flare")
plt.title("Road Accidents by Weather Condition", size = 20)
plt.xlabel("Weather Condition", size = 20)
plt.ylabel("Road Accidents", size = 20)
plt.xticks(size = 15)
plt.yticks(size = 15)
plt.savefig("RAWC.png")
plt.show()


# Road Accidents by Geographic Location

# In[64]:


ra_gl = df_a["Local_Authority_(District)"].value_counts().to_frame().reset_index()
ra_gl.columns = ["District", "Road Accidents"]
ra_gl = ra_gl.head(10)
ra_gl


# In[169]:


plt.figure(figsize = (18, 7))
sns.barplot(x = "District", y = "Road Accidents", data = ra_gl, palette = "icefire")
plt.title("Road Accidents by Districts", size = 20)
plt.xlabel("Districts", size = 20)
plt.ylabel("Road Accidents", size = 20)
plt.xticks(size = 15)
plt.yticks(size = 15)
plt.savefig("RAD.png")
plt.show()


# **Road Accidents by Road Type**

# In[69]:


rr = df_a["Road_Type"].value_counts().to_frame().reset_index()
rr.columns = ["Road Type", "Accidents"]
rr


# In[170]:


plt.figure(figsize = (18, 7))
sns.barplot(x = "Road Type", y = "Accidents", data = rr, palette = "hsv")
plt.title("Road Accidents by Road Type", size = 20)
plt.xlabel("Road Type", size = 20)
plt.ylabel("Road Accidents", size = 20)
plt.xticks(size = 15)
plt.yticks(size = 15)
plt.savefig("RART.png")
plt.show()


# **(h) How does driver related variables affect the outcome (e.g., age of the driver, and the purpose of the journey)?**

# In[77]:


da = df_v["Age_of_Driver"].value_counts().to_frame().reset_index()
da.columns = ["Age of Driver", "Accidents"]
da = da.head(10)
da.sort_values(by = "Age of Driver")


# In[171]:


plt.figure(figsize = (18, 7))
sns.barplot(x = "Age of Driver", y = "Accidents", data = da, palette = "plasma")
plt.title("Road Accidents by 10 Most Age of Driver", size = 20)
plt.xlabel("Age of Driver", size = 20)
plt.ylabel("Road Accidents", size = 20)
plt.xticks(size = 15)
plt.yticks(size = 15)
plt.savefig("RAAD.png")
plt.show()


# Road Accidents by Journey purpose of Driver

# In[80]:


raj = df_v["Journey_Purpose_of_Driver"].value_counts().to_frame().reset_index()
raj.columns = ["Journey Purpose", "Road Accidents"]
raj


# In[172]:


plt.figure(figsize = (18, 7))
sns.barplot(x = "Journey Purpose", y = "Road Accidents", data = raj, palette = "prism")
plt.title("Road Accidents by Journey Purpose of Driver", size = 20)
plt.xlabel("Journey Purpose", size = 20)
plt.ylabel("Road Accidents", size = 20)
plt.xticks(size = 15)
plt.yticks(size = 15)
plt.savefig("rajp.png")
plt.show()


# ### Prediction for When Accidents will occur

# Relevant features selected in a separate file

# In[92]:


w = pd.read_csv("when.csv")
w.head(2)


# In[93]:


w = w.dropna()


# In[94]:


w["Hour"] = w["Time"].str.split(":").str.get(0)


# In[95]:


w["Hour"] = w["Hour"].astype("int")


# In[96]:


del w["Time"]


# ### Random Forest

# In[97]:


X = w[w.columns.drop("Hour")]

y = w["Hour"]


# In[98]:


X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size = 0.2,
                                                    random_state = 4611)


# In[101]:


rfw = RandomForestClassifier()

rfw.fit(X_train, y_train)

rfw_pred = rfw.predict(X_test)
pd.DataFrame({"Actual": y_test,
              "Predicted": rfw_pred}).head(10)


# In[103]:


accuracy_score(y_test, rfw_pred)


# ### Predictions for Where Accidents will occur

# Relevant features selected in a separate file

# In[110]:


ur = pd.read_csv("UR.csv")
ur.head()


# In[106]:


X = ur[ur.columns.drop("Urban_or_Rural_Area")]

y = ur["Urban_or_Rural_Area"]


# In[107]:


X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size = 0.2,
                                                    random_state = 4611)


# In[108]:


rf_ur = RandomForestClassifier()

rf_ur.fit(X_train, y_train)

pred_rfur = rf_ur.predict(X_test)

pd.DataFrame({"Actual": y_test,
              "Predicted": pred_rfur}).head(10)


# In[109]:


accuracy_score(y_test, pred_rfur)


# ### Accident Severity Prediction

# In[111]:


X = ur[ur.columns.drop("Accident_Severity")]

y = ur["Accident_Severity"]


# In[113]:


X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size = 0.2,
                                                    random_state = 4611)


# In[114]:


rf_ur = RandomForestClassifier()

rf_ur.fit(X_train, y_train)

pred_rfs = rf_ur.predict(X_test)

pd.DataFrame({"Actual": y_test,
              "Predicted": pred_rfs}).head(10)


# In[115]:


accuracy_score(y_test, pred_rfs)

