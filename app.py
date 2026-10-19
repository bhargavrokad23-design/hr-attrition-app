import io
import time
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

st.set_page_config(
    page_title="Enterprise HR Attrition Suite",
    page_icon="🧑‍💼",
    layout="wide"
)

# ---------------- GLASSMORPHISM CSS ----------------
st.markdown("""
<style>
@property --p {
    syntax: '<integer>';
    initial-value: 0;
    inherits: true;
}

[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 35%, #312e81 70%, #4338ca 100%);
    background-attachment: fixed;
}
[data-testid="stHeader"] { background: transparent; }

/* Main card wrapper */
.block-container {
    background: rgba(255, 255, 255, 0.05);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid rgba(255, 255, 255, 0.15);
    border-radius: 20px;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
    padding: 2rem 2.5rem !important;
    margin-top: 1.5rem;
}

h1, h2, h3, h4, p, label, span, div, li, [data-testid="stMarkdownContainer"] {
    color: #f8fafc !important;
}

/* Badges & Pills */
.badge {
    display: inline-block;
    padding: 4px 14px;
    margin: 4px;
    border-radius: 999px;
    background: rgba(255, 255, 255, 0.12);
    border: 1px solid rgba(255, 255, 255, 0.25);
    font-size: 0.85rem;
}

/* Inputs styling */
div[data-baseweb="input"] > div,
div[data-baseweb="select"] > div,
[data-testid="stNumberInput"] input {
    background: rgba(255, 255, 255, 0.08) !important;
    border: 1px solid rgba(255, 255, 255, 0.22) !important;
    border-radius: 10px !important;
    color: #ffffff !important;
}

/* Tabs styling */
button[data-baseweb="tab"] {
    background: transparent !important;
    color: #cbd5e1 !important;
    font-weight: 600;
}
button[aria-selected="true"] {
    color: #38bdf8 !important;
    border-bottom: 2px solid #38bdf8 !important;
}

/* Gauge & Result Card */
.result-card {
    text-align: center;
    padding: 1.5rem;
    border-radius: 18px;
    background: rgba(255, 255, 255, 0.08);
    backdrop-filter: blur(20px);
    border: 1px solid var(--glow);
    box-shadow: 0 0 25px var(--glow);
    margin-top: 1rem;
}
.gauge {
    --p: var(--target);
    width: 150px; height: 150px;
    margin: 0 auto 1rem auto;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    background: conic-gradient(var(--ring) calc(var(--p) * 1%), rgba(255,255,255,0.1) 0);
    box-shadow: 0 0 20px var(--glow);
}
.gauge-inner {
    width: 115px; height: 115px;
    border-radius: 50%;
    background: rgba(15, 23, 42, 0.95);
    display: flex; flex-direction: column;
    align-items: center; justify-content: center;
}
.gauge-num {
    counter-reset: num var(--p);
    font-size: 2rem;
    font-weight: 800;
}
.gauge-num::after { content: counter(num) "%"; }
.gauge-label { font-size: 0.75rem; opacity: 0.8; }
.chip {
    display: inline-block;
    margin: 3px;
    padding: 4px 10px;
    border-radius: 999px;
    background: rgba(255, 255, 255, 0.12);
    border: 1px solid rgba(255, 255, 255, 0.2);
    font-size: 0.85rem;
}
</style>
""", unsafe_allow_html=True)

# ---------------- MODEL PIPELINE BUILDER ----------------
@st.cache_resource
def get_trained_pipeline():
    np.random.seed(42)
    n = 1200

    # Generating realistic enterprise feature distributions
    departments = np.random.choice(['Sales', 'Research & Development', 'Human Resources'], size=n, p=[0.45, 0.45, 0.10])
    age = np.random.randint(21, 58, size=n)
    monthly_income = np.random.randint(22000, 180000, size=n)
    years_at_company = np.random.randint(0, 18, size=n)
    job_satisfaction = np.random.randint(1, 5, size=n)
    work_life_balance = np.random.randint(1, 5, size=n)
    distance_from_home = np.random.randint(1, 35, size=n)
    overtime = np.random.choice([0, 1], size=n, p=[0.72, 0.28])

    # Realistic attrition weights
    logit = (
        0.55 * (overtime == 1)
        + 0.45 * (job_satisfaction <= 2)
        + 0.35 * (work_life_balance == 1)
        + 0.30 * (monthly_income < 45000)
        + 0.25 * (distance_from_home > 20)
        - 0.04 * (age - 21)
    )
    prob = 1 / (1 + np.exp(-logit + 1.2))
    attrition = (prob > 0.48).astype(int)

    df = pd.DataFrame({
        'Department': departments,
        'Age': age,
        'MonthlyIncome': monthly_income,
        'YearsAtCompany': years_at_company,
        'JobSatisfaction': job_satisfaction,
        'WorkLifeBalance': work_life_balance,
        'DistanceFromHome': distance_from_home,
        'OverTime': overtime,
        'Attrition': attrition
    })

    categorical_features = ['Department']
    numeric_features = ['Age', 'MonthlyIncome', 'YearsAtCompany', 'JobSatisfaction', 
                        'WorkLifeBalance', 'DistanceFromHome', 'OverTime']

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(drop='first'), categorical_features),
            ('num', StandardScaler(), numeric_features)
        ]
    )

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=150, max_depth=6, random_state=42))
    ])

    X = df.drop(columns=['Attrition'])
    y = df['Attrition']
    pipeline.fit(X, y)

    return pipeline, df

pipeline, training_df = get_trained_pipeline()

# ---------------- HEADER ----------------
st.title("🧑‍💼 Enterprise Employee Attrition Suite")
st.markdown(
    '<div style="text-align:center; margin-bottom: 1.5rem;">'
    '<span class="badge">🌲 Random Forest Classifier</span>'
    '<span class="badge">📂 Bulk CSV Processing</span>'
    '<span class="badge">📊 Executive EDA Analytics</span>'
    '</div>',
    unsafe_allow_html=True
)

tab1, tab2, tab3 = st.tabs(["🎯 Single Prediction", "📁 Batch Prediction (CSV)", "📈 HR Analytics Dashboard"])

# ---------------- TAB 1: SINGLE PREDICTION ----------------
with tab1:
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 👤 Employee Profile")
        dept = st.selectbox("Department", ['Sales', 'Research & Development', 'Human Resources'])
        age = st.slider("Age", min_value=18, max_value=65, value=29)
        income = st.number_input("Monthly Income (₹)", min_value=15000, max_value=250000, value=42000, step=5000)
        distance = st.slider("Distance From Home (KM)", min_value=1, max_value=40, value=12)

    with col2:
        st.markdown("#### 💼 Work Environment")
        years = st.slider("Years at Company", min_value=0, max_value=25, value=2)
        satisfaction = st.select_slider(
            "Job Satisfaction",
            options=[1, 2, 3, 4],
            format_func=lambda x: {1: "😞 Very Low", 2: "😕 Low", 3: "🙂 Medium", 4: "😄 High"}[x],
            value=2
        )
        wlb = st.select_slider(
            "Work-Life Balance",
            options=[1, 2, 3, 4],
            format_func=lambda x: {1: "⚠️ Bad", 2: "⚖️ Average", 3: "👍 Good", 4: "🌟 Excellent"}[x],
            value=2
        )
        overtime_opt = st.radio("Works Overtime?", ["No", "Yes"], horizontal=True)
        overtime_val = 1 if overtime_opt == "Yes" else 0

    if st.button("🚀 Analyze Individual Risk", type="primary", use_container_width=True):
        with st.spinner("Evaluating retention probabilities..."):
            time.sleep(0.5)

        sample = pd.DataFrame([{
            'Department': dept,
            'Age': age,
            'MonthlyIncome': income,
            'YearsAtCompany': years,
            'JobSatisfaction': satisfaction,
            'WorkLifeBalance': wlb,
            'DistanceFromHome': distance,
            'OverTime': overtime_val
        }])

        prob_attrition = pipeline.predict_proba(sample)[0][1] * 100
        pred = pipeline.predict(sample)[0]

        # Risk drivers
        drivers = []
        if overtime_val == 1:
            drivers.append("⏰ Overtime Burden")
        if satisfaction <= 2:
            drivers.append("😞 Low Job Satisfaction")
        if income < 45000:
            drivers.append("💸 Below Median Compensation")
        if distance > 20:
            drivers.append("🚗 Long Commute Distance")
        if wlb <= 2:
            drivers.append("⚖️ Poor Work-Life Balance")
        if not drivers:
            drivers.append("🌟 Strong Retention Anchors")

        chips = "".join([f'<span class="chip">{d}</span>' for d in drivers])
        val = int(round(prob_attrition))

        if pred == 1:
            glow = "rgba(239, 68, 68, 0.75)"
            ring = "#ef4444"
            status = "High Risk of Resignation"
            reco = "Immediate check-in recommended regarding compensation and workload rebalancing."
        else:
            glow = "rgba(34, 197, 94, 0.75)"
            ring = "#22c55e"
            status = "Low Risk / Likely to Retain"
            reco = "Employee stability is high. Maintain regular career progression reviews."

        st.markdown(f"""
        <div class="result-card" style="--glow:{glow}; --ring:{ring}; --target:{val};">
            <h3>{status}</h3>
            <div class="gauge">
                <div class="gauge-inner">
                    <div class="gauge-num"></div>
                    <div class="gauge-label">Resignation Risk</div>
                </div>
            </div>
            <p><strong>Driver Summary:</strong></p>
            <div>{chips}</div>
            <p style="margin-top:10px; opacity:0.9;">💡 {reco}</p>
        </div>
        """, unsafe_allow_html=True)

# ---------------- TAB 2: BATCH PREDICTION ----------------
with tab2:
    st.markdown("#### 📂 Upload Employee Data for Batch Inference")
    st.caption("Upload a `.csv` file containing: Department, Age, MonthlyIncome, YearsAtCompany, JobSatisfaction, WorkLifeBalance, DistanceFromHome, OverTime")

    # Downloadable template
    template_df = pd.DataFrame([{
        'Department': 'Sales', 'Age': 32, 'MonthlyIncome': 50000, 'YearsAtCompany': 3,
        'JobSatisfaction': 2, 'WorkLifeBalance': 2, 'DistanceFromHome': 15, 'OverTime': 1
    }])
    csv_buffer = io.StringIO()
    template_df.to_csv(csv_buffer, index=False)
    st.download_button("📥 Download CSV Template", data=csv_buffer.getvalue(), file_name="employee_template.csv", mime="text/csv")

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)
        required_cols = {'Department', 'Age', 'MonthlyIncome', 'YearsAtCompany', 'JobSatisfaction', 'WorkLifeBalance', 'DistanceFromHome', 'OverTime'}

        if required_cols.issubset(batch_df.columns):
            st.success("File verified! Processing predictions...")

            probs = pipeline.predict_proba(batch_df[list(required_cols)])[:, 1]
            batch_df['Risk_Score_%'] = np.round(probs * 100, 1)
            batch_df['Predicted_Attrition'] = np.where(batch_df['Risk_Score_%'] > 50, "Likely to Leave", "Likely to Stay")

            # Metrics
            total = len(batch_df)
            high_risk = len(batch_df[batch_df['Predicted_Attrition'] == "Likely to Leave"])
            rate = round((high_risk / total) * 100, 1)

            m1, m2, m3 = st.columns(3)
            m1.metric("Total Records", total)
            m2.metric("Flagged High Risk", high_risk)
            m3.metric("Projected Attrition Rate", f"{rate}%")

            st.dataframe(batch_df, use_container_width=True)

            out_csv = io.StringIO()
            batch_df.to_csv(out_csv, index=False)
            st.download_button(
                "📥 Download Scored Report",
                data=out_csv.getvalue(),
                file_name="attrition_predictions_report.csv",
                mime="text/csv"
            )
        else:
            missing = required_cols - set(batch_df.columns)
            st.error(f"Missing required columns in CSV: {', '.join(missing)}")

# ---------------- TAB 3: HR ANALYTICS DASHBOARD ----------------
with tab3:
    st.markdown("#### 📊 Internal Workforce Exploration")
    c1, c2 = st.columns(2)

    with c1:
        # Attrition by Department
        dept_summary = training_df.groupby(['Department', 'Attrition']).size().reset_index(name='Count')
        dept_summary['Attrition_Label'] = dept_summary['Attrition'].map({0: 'Retained', 1: 'Left'})
        fig_dept = px.bar(
            dept_summary,
            x='Department',
            y='Count',
            color='Attrition_Label',
            barmode='group',
            title="Attrition Distribution by Department",
            color_discrete_map={'Retained': '#38bdf8', 'Left': '#f43f5e'}
        )
        fig_dept.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color="#f8fafc")
        st.plotly_chart(fig_dept, use_container_width=True)

    with c2:
        # Salary vs Attrition Box Plot
        training_df['Status'] = training_df['Attrition'].map({0: 'Retained', 1: 'Left'})
        fig_income = px.box(
            training_df,
            x='Status',
            y='MonthlyIncome',
            color='Status',
            title="Compensation Range vs Retention Status",
            color_discrete_map={'Retained': '#34d399', 'Left': '#fb7185'}
        )
        fig_income.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color="#f8fafc")
        st.plotly_chart(fig_income, use_container_width=True)

    # Overtime vs Job Satisfaction Heatmap / Bubble
    fig_scatter = px.scatter(
        training_df,
        x='DistanceFromHome',
        y='MonthlyIncome',
        color='Status',
        size='YearsAtCompany',
        hover_data=['Department', 'JobSatisfaction'],
        title="Income vs Commute Distance (Bubble Size: Tenure)",
        color_discrete_map={'Retained': '#60a5fa', 'Left': '#f87171'}
    )
    fig_scatter.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color="#f8fafc")
    st.plotly_chart(fig_scatter, use_container_width=True)
