import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import pickle
import numpy as np

# Load processed data (all quarters)
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

# Calculate metrics
y_pred = xgb_model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Clustering (all quarters)
scaler = pickle.load(open('scaler.pkl', 'rb'))
cluster_features = ['average_weekly_score', 'average_engagement_score', 'sum_tardy', 'ReprimandCount']
cluster_data = scaler.transform(df[cluster_features])
kmeans = KMeans(n_clusters=4, random_state=42)
df['cluster'] = kmeans.fit_predict(cluster_data)

# Assign labels based on centroids and feature analysis
centroids = scaler.inverse_transform(kmeans.cluster_centers_)
cluster_df = pd.DataFrame(centroids, columns=cluster_features)
cluster_df['cluster'] = range(4)

# Sort by average_weekly_score and assign meaningful labels
cluster_labels = {
    cluster_df['average_weekly_score'].idxmax(): "Top Performers",
    cluster_df['average_weekly_score'].idxmin(): "At Risk",
    cluster_df['average_engagement_score'].idxmin(): "Low Engagement",
}
remaining = set(range(4)) - set(cluster_labels.keys())
cluster_labels[list(remaining)[0]] = "Stable Performers"  # Assign remaining as Stable
df['cluster_label'] = df['cluster'].map(cluster_labels)

# Save models and metrics
xgb_model.save_model('xgb_model.json')
pickle.dump(kmeans, open('kmeans_model.pkl', 'wb'))
with open('model_metrics.txt', 'w') as f:
    f.write(f"RMSE: {rmse}\nMAE: {mae}\nR2: {r2}")
df.to_csv('processed_data_with_clusters.csv', index=False)

print(f"Model training complete!\nRMSE: {rmse:.2f}\nMAE: {mae:.2f}\nR²: {r2:.2f}")
print("Cluster Centroids:\n", cluster_df)
print("Cluster Labels:", cluster_labels)