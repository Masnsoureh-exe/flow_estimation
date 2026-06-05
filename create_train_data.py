import pandas as pd

''' PREPARE TRAIN DATA '''

# Read features and labels
features = pd.read_csv(
    "optical_flow_features.csv"
)

labels = pd.read_excel(
    "../data/labels/Validation_Numbers.ods",
    engine="odf"
).drop(columns=["Number"])

# Clean and modify the tables 
features["video_name"] = (
    features["video"]
    .str.replace(".avi", "", regex=False)
    .str.replace(".mp4", "", regex=False)
)
features= features.drop(columns=["video"])

labels= labels.rename(columns= {"Name":"video_name"})

# Create the training table
train_data = features.merge(
    labels,
    on="video_name"
)

# print(train_data.shape)
# print(train_data)

# Drop 2 rows with null values on Real(m/s)
train_data['Real (m/s)'] = pd.to_numeric(train_data['Real (m/s)'], errors= 'coerce')
train_data = train_data.dropna(subset= ['Real (m/s)'])

# print(train_data)