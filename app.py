import streamlit as st
import pandas as pd
import xgboost as xgb
import pickle
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from automate import send_email

# Load models and data
xgb_model = xgb.XGBRegressor()
xgb_model.load_model('xgb_model.json')
kmeans = pickle.load(open('kmeans_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))
df = pd.read_csv('final_data.csv')

# Load model metrics
with open('model_metrics.txt', 'r') as f:
    metrics = f.read().splitlines()
rmse = float(metrics[0].split(': ')[1])
mae = float(metrics[1].split(': ')[1])
r2 = float(metrics[2].split(': ')[1])

# Modern Styling
st.markdown("""
    <style>
    .main {padding: 20px; background-color: #F5F7FA;}
    .stButton>button {background-color: #007BFF; color: white; border-radius: 8px; padding: 10px;}
    .stSelectbox {margin-bottom: 20px; background-color: #FFFFFF; border-radius: 5px;}
    h1 {color: #1A3C34; font-family: 'Arial';}
    h2 {color: #2E5A50; font-family: 'Arial';}
    .sidebar .sidebar-content {background-color: #E8ECEF;}
    </style>
""", unsafe_allow_html=True)

# Sidebar Navigation
st.sidebar.title("Dashboard")
page = st.sidebar.radio("Navigate", ["Profile", "Insights", "Rankings", "Teams", "Segmentation", "Automation"])

# Prediction function
def get_prediction(employee_data):
    pred_features = ['tenure_in_months', 'average_manager_score', 'average_engagement_score', 
                     'average_kpi_score', 'average_okr_score', 'team_name_encoded', 
                     'AppCount', 'ReprimandCount', 'sum_tardy', 'sum_absent', 'variableAmount', 
                     'score_trend', 'performance_stability']
    latest_data = employee_data[pred_features].iloc[-1].values.reshape(1, -1)
    return xgb_model.predict(latest_data)[0]

# Page 1: Profile
if page == "Profile":
    st.header("Employee Profile")
    st.write("Explore individual employee performance across quarters.")

    col1, col2 = st.columns([1, 3])
    with col1:
        search_by = st.radio("Search by", ["ID", "Name"])
        if search_by == "ID":
            employee_id = st.selectbox("Select Employee ID", df['user_id'].unique())
            employee_data = df[df['user_id'] == employee_id]
        else:
            employee_name = st.selectbox("Select Employee Name", df['full_name'].unique())
            employee_data = df[df['full_name'] == employee_name]

    with col2:
        st.subheader(f"{employee_data['full_name'].iloc[0]}")
        st.dataframe(employee_data[['quarter', 'team_name', 'average_weekly_score', 'cluster_label', 'recommendation']],
                     use_container_width=True)
        prediction = get_prediction(employee_data)
        st.metric("Next Quarter Prediction", f"{prediction:.2f}", delta_color="normal")

    st.subheader("Trend")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(employee_data['quarter'], employee_data['average_weekly_score'], marker='o', color='#007BFF')
    ax.set_xlabel("Quarter")
    ax.set_ylabel("Weekly Score")
    st.pyplot(fig)

# Page 2: Insights
elif page == "Insights":
    st.header("Model Insights")
    st.write("Understand the AI model's predictive power.")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("RMSE", f"{rmse:.2f}", help="Average error in score units")
    with col2:
        st.metric("MAE", f"{mae:.2f}", help="Average absolute error")
    with col3:
        st.metric("R²", f"{r2:.2f}", help="Fit (0 to 1)")

# Page 3: Rankings
elif page == "Rankings":
    st.header("Performance Rankings")
    st.write("Latest quarter rankings per employee.")

    latest_df = df.groupby('user_id').apply(lambda x: x.iloc[-1]).reset_index(drop=True)
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Top 5")
        st.dataframe(latest_df.nlargest(5, 'average_weekly_score')[['full_name', 'quarter', 'average_weekly_score', 'cluster_label']],
                     use_container_width=True)
    with col2:
        st.subheader("Bottom 5")
        st.dataframe(latest_df.nsmallest(5, 'average_weekly_score')[['full_name', 'quarter', 'average_weekly_score', 'cluster_label']],
                     use_container_width=True)

# Page 4: Teams
elif page == "Teams":
    st.header("Team Performance")
    st.write("Analyze team performance by quarter.")

    quarter = st.selectbox("Select Quarter", df['quarter'].unique())
    team_performance = df[df['quarter'] == quarter].groupby('team_name')['average_weekly_score'].mean().sort_values(ascending=False)
    
    st.subheader(f"Team Rankings - {quarter}")
    st.dataframe(team_performance.reset_index(), use_container_width=True)

    fig, ax = plt.subplots(figsize=(8, 4))
    team_performance.plot(kind='bar', color='#007BFF', ax=ax)
    ax.set_xlabel("Team")
    ax.set_ylabel("Average Weekly Score")
    st.pyplot(fig)

# Page 5: Segmentation
elif page == "Segmentation":
    st.header("Employee Segmentation")
    st.write("View employee clusters based on performance metrics.")

    quarter = st.selectbox("Select Quarter", df['quarter'].unique())
    quarter_df = df[df['quarter'] == quarter]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(quarter_df['average_weekly_score'], quarter_df['average_engagement_score'], 
                         c=quarter_df['cluster'], cmap='viridis', s=100)
    plt.colorbar(scatter, label='Cluster')
    for label in quarter_df['cluster_label'].unique():
        ax.text(quarter_df[quarter_df['cluster_label'] == label]['average_weekly_score'].mean(), 
                quarter_df[quarter_df['cluster_label'] == label]['average_engagement_score'].mean(), 
                label, fontsize=12, ha='center', weight='bold')
    ax.set_xlabel("Weekly Score")
    ax.set_ylabel("Engagement Score")
    st.pyplot(fig)

# Page 6: Automation
elif page == "Automation":
    st.header("Automation")
    st.write("Send simulated emails based on weekly scores for a selected quarter.")

    col1, col2 = st.columns([1, 2])
    with col1:
        quarter = st.selectbox("Select Quarter", df['quarter'].unique())
        threshold = st.slider("Score Threshold for Alert", 0.0, 100.0, 50.0)
    
    with col2:
        quarter_df = df[df['quarter'] == quarter]
        low_scorers = quarter_df[quarter_df['average_weekly_score'] < threshold]
        if not low_scorers.empty:
            st.write(f"Employees below {threshold} in {quarter}:")
            st.dataframe(low_scorers[['full_name', 'average_weekly_score', 'recommendation']], use_container_width=True)
        else:
            st.info(f"No employees below {threshold} in {quarter}.")

    if st.button("Send Alerts"):
        for idx, row in low_scorers.iterrows():
            body = f"Review {row['full_name']} - Low performance in {quarter}."
            send_email("manager@example.com", "Performance Alert", body, row['average_weekly_score'])
        st.success(f"Sent {len(low_scorers)} simulated alerts!")

    if st.button("Export Report"):
        df.to_csv("employee_report.csv", index=False)
        st.download_button("Download Report", data=open("employee_report.csv", "rb"), file_name="employee_report.csv")

# Footer
st.sidebar.markdown("---")
st.sidebar.write("© 2025 Employee Performance AI")