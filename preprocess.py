import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import pickle
import numpy as np

# Load processed data
df = pd.read_csv('processed_data.csv')

# Prediction dataset
pred_features = ['tenure_in_months', 'average_manager_score', 'average_engagement_score', 
                 'average_kpi_score', 'average_okr_score', 'team_name_encoded', 
                 'AppCount', 'ReprimandCount', 'sum_tardy', 'sum_absent', 'variableAmount', 
                 'score_trend', 'performance_stability']
X = df[pred_features]
y = df['average_weekly_score']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train XGBoost
xgb_model = xgb.XGBRegressor(n_estimators=150, learning_rate=0.05, max_depth=6, random_state=42)
xgb_model.fit(X_train, y_train)

# Calculate accuracy (RMSE)
y_pred = xgb_model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# Clustering
scaler = pickle.load(open('scaler.pkl', 'rb'))
cluster_features = ['average_weekly_score', 'average_engagement_score', 'sum_tardy', 'ReprimandCount']
cluster_data = scaler.transform(df[cluster_features])
kmeans = KMeans(n_clusters=4, random_state=42)
df['cluster'] = kmeans.fit_predict(cluster_data)

# Save models, accuracy, and updated data
xgb_model.save_model('xgb_model.json')
pickle.dump(kmeans, open('kmeans_model.pkl', 'wb'))
with open('model_accuracy.txt', 'w') as f:
    f.write(str(rmse))
df.to_csv('processed_data_with_clusters.csv', index=False)

print(f"Model training complete! RMSE: {rmse}")