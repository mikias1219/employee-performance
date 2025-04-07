import streamlit as st
import pandas as pd
import xgboost as xgb
import pickle
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from automate import send_email

# Load models and data (all quarters)
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

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Employee Profile", "Model Insights", "Rankings", "Segmentation", "Automation"])

# Modern Styling
st.markdown("""
    <style>
    .main {padding: 20px;}
    .stButton>button {background-color: #4CAF50; color: white; border-radius: 5px;}
    .stSelectbox {margin-bottom: 20px;}
    h1 {color: #2C3E50;}
    h2 {color: #34495E;}
    </style>
""", unsafe_allow_html=True)

# Prediction function
def get_prediction(employee_data):
    pred_features = ['tenure_in_months', 'average_manager_score', 'average_engagement_score', 
                     'average_kpi_score', 'average_okr_score', 'team_name_encoded', 
                     'AppCount', 'ReprimandCount', 'sum_tardy', 'sum_absent', 'variableAmount', 
                     'score_trend', 'performance_stability']
    latest_data = employee_data[pred_features].iloc[-1].values.reshape(1, -1)
    return xgb_model.predict(latest_data)[0]

# Page 1: Employee Profile
if page == "Employee Profile":
    st.header("Employee Profile")
    st.write("View detailed performance data and predictions for a specific employee.")

    col1, col2 = st.columns([1, 2])
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
        st.write(employee_data[['quarter', 'team_name', 'average_weekly_score', 'cluster_label', 'recommendation']])
        prediction = get_prediction(employee_data)
        st.metric("Predicted Next Quarter Score", f"{prediction:.2f}")

    st.subheader("Performance Trend")
    fig, ax = plt.subplots()
    ax.plot(employee_data['quarter'], employee_data['average_weekly_score'], marker='o', color='#2C3E50')
    ax.set_xlabel("Quarter")
    ax.set_ylabel("Average Weekly Score")
    st.pyplot(fig)

# Page 2: Model Insights
elif page == "Model Insights":
    st.header("Model Insights")
    st.write("Evaluate the effectiveness of the AI model predicting employee performance.")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Root Mean Squared Error (RMSE)", f"{rmse:.2f}", help="Lower is better")
    with col2:
        st.metric("Mean Absolute Error (MAE)", f"{mae:.2f}", help="Lower is better")
    with col3:
        st.metric("R-squared (R²)", f"{r2:.2f}", help="Closer to 1 is better")

    st.write("""
        - **RMSE**: Measures average prediction error in score units.
        - **MAE**: Average absolute error, less sensitive to outliers.
        - **R²**: Proportion of variance explained (0 to 1).
    """)

# Page 3: Rankings
elif page == "Rankings":
    st.header("Employee Rankings")
    st.write("See the top and bottom performers based on their latest quarter data.")

    latest_df = df.groupby('user_id').apply(lambda x: x.iloc[-1]).reset_index(drop=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Top 5 Performers")
        top_performers = latest_df.nlargest(5, 'average_weekly_score')[['full_name', 'quarter', 'average_weekly_score', 'cluster_label']]
        st.write(top_performers)
    
    with col2:
        st.subheader("Bottom 5 Performers")
        low_performers = latest_df.nsmallest(5, 'average_weekly_score')[['full_name', 'quarter', 'average_weekly_score', 'cluster_label']]
        st.write(low_performers)

# Page 4: Segmentation
elif page == "Segmentation":
    st.header("Employee Segmentation")
    st.write("Visualize how employees are grouped based on performance metrics.")

    fig, ax = plt.subplots()
    scatter = ax.scatter(df['average_weekly_score'], df['average_engagement_score'], c=df['cluster'], cmap='viridis')
    plt.colorbar(scatter, label='Cluster')
    for label in df['cluster_label'].unique():
        ax.text(df[df['cluster_label'] == label]['average_weekly_score'].mean(), 
                df[df['cluster_label'] == label]['average_engagement_score'].mean(), 
                label, fontsize=10, ha='center', color='black')
    ax.set_xlabel("Weekly Score")
    ax.set_ylabel("Engagement Score")
    st.pyplot(fig)

    st.write("""
        - **Top Performers**: High scores, strong performance.
        - **Stable Performers**: Average, no urgent needs.
        - **Low Engagement**: Lower engagement scores.
        - **At Risk**: High tardiness/reprimands.
    """)

# Page 5: Automation
elif page == "Automation":
    st.header("Automation")
    st.write("Simulate sending alerts to managers for at-risk employees.")

    search_by = st.radio("Search by", ["ID", "Name"])
    if search_by == "ID":
        employee_id = st.selectbox("Select Employee ID", df['user_id'].unique())
        employee_data = df[df['user_id'] == employee_id]
    else:
        employee_name = st.selectbox("Select Employee Name", df['full_name'].unique())
        employee_data = df[df['full_name'] == employee_name]

    st.write(employee_data[['quarter', 'recommendation']].iloc[-1:])  # Latest quarter
    if st.button("Send Manager Alert"):
        recommendation = employee_data['recommendation'].iloc[-1]
        if "Alert" in recommendation:
            success = send_email("manager@example.com", "Performance Alert", recommendation)
            if success:
                st.success("Alert sent (simulated for demo)!")
        else:
            st.info("No alert required for this employee.")

    if st.button("Export Full Report"):
        df.to_csv("employee_report.csv", index=False)
        st.download_button("Download Report", data=open("employee_report.csv", "rb"), file_name="employee_report.csv")

# Footer
st.sidebar.markdown("---")
st.sidebar.write("AI-Driven Employee Performance Dashboard © 2025")