import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
import joblib

# 1. Load your dataset
df = pd.read_csv("Crop_recommendation_expanded_12000.csv")

# 2. Separate Features and Target
X = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
y = df['label']

# 3. Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Initialize and fit the Scaler
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)

# 5. Train the Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# 6. SAVE THE FILES (This fixes your error)
joblib.dump(model, 'random_forest_crop_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

print("Success: 'random_forest_crop_model.pkl' and 'scaler.pkl' have been created!")