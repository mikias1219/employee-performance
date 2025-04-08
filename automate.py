import pandas as pd

# Load data with clusters (all quarters)
df = pd.read_csv('processed_data_with_clusters.csv')

# Recommendation logic
def generate_recommendation(row):
    if row['cluster_label'] == "Top Performers":
        if row['variableAmount'] < 50:
            return f"Bonus recommended for {row['full_name']} (Top Performer, {row['quarter']})."
        return f"{row['full_name']} is a Top Performer ({row['quarter']})."
    elif row['cluster_label'] == "Low Engagement" and row['average_engagement_score'] < 50:
        return f"Training suggested for {row['full_name']} (Low Engagement, {row['quarter']})."
    elif row['cluster_label'] == "At Risk" and row['at_risk'] == 1:
        return f"Alert: {row['full_name']} is at risk—review urgently ({row['quarter']})."
    return f"Stable Performer: No action required for {row['full_name']} ({row['quarter']})."

df['recommendation'] = df.apply(generate_recommendation, axis=1)

# Fake email function with score-based logic
def send_email(to_email, subject, body, score):
    message = f"{body}\nAverage Weekly Score: {score:.2f}"
    print(f"Simulated Email:\nTo: {to_email}\nSubject: {subject}\nBody: {message}\n")
    return True

# Save updated data
df.to_csv('final_data.csv', index=False)

print("Automation data prepared!")