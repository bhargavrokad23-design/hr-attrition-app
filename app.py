import time
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LogisticRegression

st.set_page_config(
    page_title="Employee Attrition Predictor",
    page_icon="🧑‍💼",
    layout="centered"
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
    background: linear-gradient(135deg, #1e1b4b 0%, #4338ca 35%, #7c3aed 65%, #db2777 100%);
    background-attachment: fixed;
}
[data-testid="stHeader"] { background: transparent; }

[data-testid="stAppViewContainer"]::before,
[data-testid="stAppViewContainer"]::after {
    content: "";
    position: fixed;
    border-radius: 50%;
    filter: blur(90px);
    z-index: 0;
    pointer-events: none;
}
[data-testid="stAppViewContainer"]::before {
    width: 380px; height: 380px;
    background: rgba(56, 189, 248, 0.45);
    top: -80px; left: -80px;
}
[data-testid="stAppViewContainer"]::after {
    width: 420px; height: 420px;
    background: rgba(244, 114, 182, 0.40);
    bottom: -100px; right: -100px;
}

.block-container {
    position: relative;
    z-index: 1;
    background: rgba(255, 255, 255, 0.10);
    backdrop-filter: blur(18px);
    -webkit-backdrop-filter: blur(18px);
    border: 1px solid rgba(255, 255, 255, 0.25);
    border-radius: 24px;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.35);
    padding: 2.5rem 2.5rem 2rem 2.5rem !important;
    margin-top: 2rem;
    max-width: 850px;
}

h1, h2, h3, h4, p, label, span, div, li,
[data-testid="stMarkdownContainer"] {
    color: #f8fafc !important;
}
h1 { text-align: center; text-shadow: 0 2px 12px rgba(0,0,0,0.3); }
h3 { text-shadow: 0 2px 10px rgba(0,0,0,0.25); }
hr { border-color: rgba(255, 255, 255, 0.25) !important; }

.subtitle {
    text-align: center;
    font-size: 1.05rem;
    opacity: 0.9;
    margin-bottom: 0.5rem;
}

.badge {
    display: inline-block;
    padding: 4px 12px;
    margin: 4px 6px 4px 0;
    border-radius: 999px;
    background: rgba(255, 255, 255, 0.15);
    border: 1px solid rgba(255, 255, 255, 0.30);
    backdrop-filter: blur(8px);
    font-size: 0.9rem;
}

div[data-baseweb="input"] > div,
div[data-baseweb="select"] > div,
[data-testid="stNumberInput"] input {
    background: rgba(255, 255, 255, 0.12) !important;
    border: 1px solid rgba(255, 255, 255, 0.30) !important;
    border-radius: 12px !important;
    color: #ffffff !important;
    backdrop-filter: blur(8px);
}
[data-testid="stNumberInput"] button {
    background: rgba(255, 255, 255, 0.12) !important;
    border: none !important;
}

ul[role="listbox"] {
    background: rgba(30, 27, 75, 0.85) !important;
    backdrop-filter: blur(14px);
    border-radius: 12px;
}

[data-testid="stSlider"] [role="slider"] {
    background: #ffffff !important;
    box-shadow: 0 0 10px rgba(255, 255, 255, 0.8);
}

.stButton > button {
    background: rgba(255, 255, 255, 0.18) !important;
    border: 1px solid rgba(255, 255, 255, 0.40) !important;
    border-radius: 14px !important;
    color: #ffffff !important;
    font-weight: 600;
    backdrop-filter: blur(10px);
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
    transition: all 0.25s ease;
}
.stButton > button:hover {
    background: rgba(255, 255, 255, 0.30) !important;
    transform: translateY(-2px);
    box-shadow: 0 8px 28px rgba(0, 0, 0, 0.35);
}

[data-testid="stTabs"] button {
    color: #f8fafc !important;
    font-weight: 600;
}

@keyframes fadeUp {
    from { opacity: 0; transform: translateY(40px) scale(0.96); }
    to   { opacity: 1; transform: translateY(0) scale(1); }
}
@keyframes fillRing {
    from { --p: 0; }
    to   { --p: var(--target); }
}
@keyframes pulseGlow {
    0%, 100% { box-shadow: 0 0 20px var(--glow), 0 8px 32px rgba(0,0,0,0.35); }
    50%      { box-shadow: 0 0 45px var(--glow), 0 8px 32px rgba(0,0,0,0.35); }
}
@keyframes popIn {
    from { opacity: 0; transform: scale(0.6); }
    to   { opacity: 1; transform: scale(1); }
}

.result-card {
    text-align: center;
    padding: 2rem 1.5rem;
    margin-top: 0.5rem;
    border-radius: 24px;
    background: rgba(255, 255, 255, 0.12);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border: 1px solid var(--glow);
    animation: fadeUp 0.8s cubic-bezier(.2,.8,.2,1) both,
               pulseGlow 2.5s ease-in-out 0.8s infinite;
}
.result-title {
    font-size: 1.6rem;
    font-weight: 700;
    margin-bottom: 1.2rem;
    animation: popIn 0.6s ease 0.3s both;
}
.gauge {
    --p: var(--target);
    width: 170px; height: 170px;
    margin: 0 auto 1.2rem auto;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    background: conic-gradient(var(--ring) calc(var(--p) * 1%), rgba(255,255,255,0.15) 0);
    animation: fillRing 1.8s ease-out 0.3s both;
    box-shadow: 0 0 25px var(--glow);
}
.gauge-inner {
    width: 135px; height: 135px;
    border-radius: 50%;
    background: rgba(30, 27, 75, 0.85);
    display: flex; flex-direction: column;
    align-items: center; justify-content: center;
}
.gauge-num {
    counter-reset: num var(--p);
    font-size: 2.3rem;
    font-weight: 800;
}
.gauge-num::after { content: counter(num) "%"; }
.gauge-label { font-size: 0.8rem; opacity: 0.85; }
.level-pill {
    display: inline-block;
    padding: 6px 18px;
    border-radius: 999px;
    font-weight: 600;
    background: rgba(255,255,255,0.15);
    border: 1px solid var(--glow);
    animation: popIn 0.6s ease 1s both;
}
.reco {
    margin-top: 1.2rem;
    padding: 12px 16px;
    border-radius: 14px;
    background: rgba(255,255,255,0.10);
    border: 1px solid rgba(255,255,255,0.25);
    animation: fadeUp 0.7s ease 1.2s both;
}
.chips { margin-top: 1rem; animation: fadeUp 0.7s ease 1.4s both; }
.chip {
    display: inline-block;
    margin: 4px;
    padding: 6px 14px;
    border-radius: 999px;
    background: rgba(255,255,255,0.14);
    border: 1px solid rgba(255,255,255,0.28);
    font-size: 0.9rem;
}
</style>
""", unsafe_allow_html=True)

st.title("🧑‍💼 Employee Attrition Prediction System")
st.markdown(
    '<p class="subtitle">🔮 Enter the employee details to find out the risk of resignation 📊</p>',
    unsafe_allow_html=True
)
st.markdown(
    '<div style="text-align:center;">'
    '<span class="badge">🤖 ML Powered</span>'
    '<span class="badge">⚡ Instant Result</span>'
    '<span class="badge">🎯 Data Driven</span>'
    '</div>',
    unsafe_allow_html=True
)

@st.cache_resource
def load_and_train_model():
    np.random.seed(42)
    n_samples = 600

    age = np.random.randint(22, 60, size=n_samples)
    monthly_income = np.random.randint(20000, 150000, size=n_samples)
    years_at_company = np.random.randint(1, 15, size=n_samples)
    job_satisfaction = np.random.randint(1, 5, size=n_samples)
    overtime = np.random.choice([0, 1], size=n_samples, p=[0.7, 0.3])
    departments = np.random.choice(['Human Resources', 'Research & Development', 'Sales'], size=n_samples, p=[0.2, 0.5, 0.3])
    commute_distance = np.random.randint(1, 30, size=n_samples)

    attrition_prob = (
        0.4 * (overtime == 1) +
        0.3 * (job_satisfaction <= 2) +
        0.2 * (monthly_income < 40000)
    )
    attrition = (attrition_prob > 0.4).astype(int)

    df = pd.DataFrame({
        'Age': age,
        'MonthlyIncome': monthly_income,
        'YearsAtCompany': years_at_company,
        'JobSatisfaction': job_satisfaction,
        'OverTime': overtime,
        'Department': departments,
        'CommuteDistance': commute_distance,
        'Attrition': attrition
    })

    X = df[['Age', 'MonthlyIncome', 'YearsAtCompany', 'JobSatisfaction', 'OverTime']]
    y = df['Attrition']

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X, y)
    return clf, df

model, sample_df = load_and_train_model()

st.markdown("---")

tab1, tab2, tab3 = st.tabs(["🎯 Single Prediction", "📁 Batch Prediction (CSV)", "📈 HR Analytics Dashboard"])

with tab1:
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🧑 Personal Details")
        age = st.slider("🎂 Employee Age", min_value=20, max_value=65, value=28)
        income = st.number_input("💰 Monthly Income (₹)", min_value=15000, max_value=200000, value=35000, step=5000)

    with col2:
        st.markdown("### 💼 Work Details")
        years = st.slider("🏢 Years at Company", min_value=0, max_value=20, value=2)
        satisfaction = st.selectbox(
            "😊 Job Satisfaction Rating",
            options=[1, 2, 3, 4],
            format_func=lambda x: {
                1: "😞 1 - Very Low",
                2: "😕 2 - Low",
                3: "🙂 3 - Medium",
                4: "😄 4 - High"
            }[x],
            index=1
        )

    overtime_text = st.radio(
        "⏰ Works Overtime?",
        ["🙅 No", "🔥 Yes"],
        horizontal=True
    )
    overtime = 1 if "Yes" in overtime_text else 0

    st.markdown("---")

    if st.button("🔍 Predict Attrition Status", type="primary", use_container_width=True):

        with st.spinner("🧠 Analyzing employee data..."):
            time.sleep(1.2)

        input_data = pd.DataFrame([[age, income, years, satisfaction, overtime]],
                                  columns=['Age', 'MonthlyIncome', 'YearsAtCompany', 'JobSatisfaction', 'OverTime'])

        prediction = model.predict(input_data)[0]
        probabilities = model.predict_proba(input_data)[0]
        attrition_risk = probabilities[1] * 100
        retention_chance = probabilities[0] * 100

        if attrition_risk < 30:
            level = "🟢 Low Risk"
        elif attrition_risk < 60:
            level = "🟡 Medium Risk"
        else:
            level = "🔴 High Risk"

        factors = []
        if overtime == 1:
            factors.append("⏰ Works overtime")
        if satisfaction <= 2:
            factors.append("😞 Low job satisfaction")
        if income < 40000:
            factors.append("💸 Low income")
        if not factors:
            factors.append("🌟 No major risk factors")
        chips_html = "".join(f'<span class="chip">{f}</span>' for f in factors)

        if prediction == 1:
            title = "⚠️ Employee is likely to leave! 🚪"
            value = int(round(attrition_risk))
            label = "Resignation Risk"
            glow = "rgba(248, 113, 113, 0.85)"
            ring = "#f87171"
            reco = "💡 Review compensation 💸 and workload/overtime balance ⚖️."
            st.toast("High attrition risk detected!", icon="🚨")
        else:
            title = "✅ Employee is likely to stay! 🎉"
            value = int(round(retention_chance))
            label = "Retention Chance"
            glow = "rgba(52, 211, 153, 0.85)"
            ring = "#34d399"
            reco = "💡 Engagement parameters are stable, keep it up 👍."
            st.balloons()

        card = (
            f'<div class="result-card" style="--glow:{glow}; --ring:{ring}; --target:{value};">'
            f'<div class="result-title">{title}</div>'
            f'<div class="gauge"><div class="gauge-inner">'
            f'<div class="gauge-num"></div><div class="gauge-label">{label}</div>'
            f'</div></div>'
            f'<div class="level-pill">{level}</div>'
            f'<div class="reco">{reco}</div>'
            f'<div class="chips">{chips_html}</div>'
            f'</div>'
        )
        st.markdown(card, unsafe_allow_html=True)

with tab2:
    st.markdown("### 📁 Batch Prediction via CSV")
    st.write("Upload a CSV file containing employee details to predict attrition for multiple employees at once.")
    
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)
        st.success("CSV file uploaded successfully!")
        st.dataframe(batch_df.head())
        
        if st.button("Run Batch Prediction"):
            required_cols = ['Age', 'MonthlyIncome', 'YearsAtCompany', 'JobSatisfaction', 'OverTime']
            if all(col in batch_df.columns for col in required_cols):
                preds = model.predict(batch_df[required_cols])
                probs = model.predict_proba(batch_df[required_cols])[:, 1] * 100
                batch_df['Attrition_Prediction'] = ["Leave" if p == 1 else "Stay" for p in preds]
                batch_df['Attrition_Risk_%'] = np.round(probs, 2)
                
                st.write("### 📊 Prediction Results")
                st.dataframe(batch_df)
                
                csv_data = batch_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Results CSV",
                    data=csv_data,
                    file_name='attrition_predictions.csv',
                    mime='text/csv'
                )
            else:
                st.error(f"CSV must contain the following columns: {required_cols}")
    else:
        st.info("Tip: Your CSV should include columns: Age, MonthlyIncome, YearsAtCompany, JobSatisfaction, OverTime")

with tab3:
    st.markdown("### 📈 HR Analytics & Insights Dashboard")
    st.write("Interactive Overview of company employee distribution, compensation range, and commute analysis.")
    
    col_m1, col_m2, col_m3 = st.columns(3)
    col_m1.metric("Total Employees", len(sample_df))
    col_m2.metric("Overall Attrition Rate", f"{round(sample_df['Attrition'].mean() * 100, 1)}%")
    col_m3.metric("Avg Monthly Income", f"₹{int(sample_df['MonthlyIncome'].mean())}")

    st.markdown("---")
    
    # Prepare data labels for Plotly
    sample_df['Attrition_Label'] = sample_df['Attrition'].map({0: 'Retained', 1: 'Left'})

    # 1. Attrition Distribution by Department (Plotly Bar Chart)
    st.markdown("#### 🏢 Attrition Distribution by Department")
    fig1 = px.histogram(
        sample_df, 
        x='Department', 
        color='Attrition_Label', 
        barmode='group',
        color_discrete_map={'Retained': '#38bdf8', 'Left': '#f472b6'},
        labels={'Department': 'Department', 'count': 'Employee Count'}
    )
    fig1.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='white'
    )
    st.plotly_chart(fig1, use_container_width=True)

    st.markdown("---")

    # 2. Compensation Range vs Retention Status (Plotly Box Plot)
    st.markdown("#### 📦 Compensation Range vs Retention Status")
    fig2 = px.box(
        sample_df, 
        x='Attrition_Label', 
        y='MonthlyIncome', 
        color='Attrition_Label',
        color_discrete_map={'Retained': '#38bdf8', 'Left': '#f472b6'},
        labels={'Attrition_Label': 'Retention Status', 'MonthlyIncome': 'Monthly Income (₹)'}
    )
    fig2.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='white',
        showlegend=False
    )
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")

    # 3. Income vs Commute Distance (Plotly Scatter/Bubble Plot)
    st.markdown("#### 🔵 Income vs Commute Distance (Bubble Size: Tenure)")
    fig3 = px.scatter(
        sample_df,
        x='CommuteDistance',
        y='MonthlyIncome',
        size='YearsAtCompany',
        color='Attrition_Label',
        color_discrete_map={'Retained': '#38bdf8', 'Left': '#f472b6'},
        hover_data=['Age', 'JobSatisfaction'],
        labels={'CommuteDistance': 'Commute Distance (KM)', 'MonthlyIncome': 'Monthly Income (₹)'}
    )
    fig3.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='white'
    )
    st.plotly_chart(fig3, use_container_width=True)

st.markdown("---")
st.caption("🛠️ Built with Streamlit & Plotly  |  🧪 Demo model on synthetic data")
