# ACCIDENT-PROJECT
ROAD SAFETY ANALYSIS
This code portfolio showcases my skills in data analysis and machine learning using Python libraries like Pandas, Seaborn, Matplotlib, and scikit-learn. The purpose of this project is to analyze road safety data and identify patterns related to accident occurrences.

Data:

The project utilizes two datasets provided as CSV files:

Road Safety Data - Accidents 2019.csv: This dataset contains details about road accidents in 2019, including factors like time, day of week, weather conditions, vehicle types, etc.
Road Safety Data- Vehicles 2019.csv: This dataset provides information about vehicles involved in accidents, including age, type, purpose of the journey, and driver characteristics.
Code Structure:

The code is organized into well-defined sections with comments explaining each step:

Import Libraries: Necessary libraries for data manipulation, visualization, and machine learning are imported.
Data Import: Both datasets are loaded using pandas.read_csv.
Data Cleaning: Data is cleaned by handling missing values and converting data types when necessary.
Data Exploration and Visualization:
Questions (a) to (h) are addressed by analyzing the data and creating informative visualizations using Seaborn and Matplotlib. These visualizations explore various factors influencing road accidents, including:
Time of day and day of week
Vehicle types (motorbikes, pedestrians)
Daylight Saving Time impact
Sunrise and sunset times
Vehicle age, propulsion type
Weather conditions
Geographic location (district)
Road type
Driver age and journey purpose
Machine Learning Models (Optional):
The code includes sections for training Random Forest models to predict:
Time of Day for accident occurrence
Urban or Rural area of accident occurrence
Accident severity
The models are trained on a portion of the data and evaluated using accuracy scores.
Key Findings:

The visualizations and analyses reveal patterns in road accident occurrences. Some notable findings include:

Accidents are more frequent during certain hours of the day and days of the week.
Motorbike accidents have distinct patterns compared to overall accidents.
Pedestrian accidents show variations based on time of day and day of week.
Daylight Saving Time may have an impact on accidents.
Sunrise and sunset times might influence accident rates.
Certain vehicle types (e.g., motorbikes) and ages are more involved in accidents.
Weather conditions and geographic location play a role in accident occurrences.
Driver characteristics like age and journey purpose might be relevant factors.
Conclusion:

This data analysis provides insights into factors affecting road safety. These findings can be valuable for developing targeted interventions and preventative measures to reduce road accidents.

Future Work:

Explore more sophisticated machine learning models for improved prediction accuracy.
Integrate real-time data sources for accident prediction and prevention.
Develop interactive dashboards for data visualization and exploration.
I hope this Readme provides a clear overview of the project. Feel free to explore the code for further details and experimentation.
