from src.data_preprocessing import load_data, clean_data, preprocess
from src.feature_engineering import create_features
from src.train_model import train




# load, Clean, preprocess
df =  preprocess("data/youtube_data.csv")

# Feature Engineering
df = create_features(df)

# Train
model = train(df)

print("✅ Model trained and saved")