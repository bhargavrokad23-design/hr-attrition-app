## 🌐 Live Demo
👉 **Click here to open the live app:** [Employee Attrition Prediction App](https://hr-attrition-app-7bgthchjifeqnsqgqyhqyg.streamlit.app)

# 🧑‍💼 Employee Attrition Prediction System

> **🔮 An interactive, ML-powered HR Analytics & Resignation Risk Prediction Web Application built with Streamlit, Scikit-Learn, and Plotly.**

---

## 🌟 Project Highlights (Main Points)

* **🤖 Machine Learning Powered:** Uses a **Logistic Regression** model (`scikit-learn`) to accurately calculate **Resignation Risk (%)** and **Retention Chance (%)** in real time.
* **🎨 Modern Glassmorphism UI:** Designed with custom CSS featuring **animated circular gauge meters**, glowing risk cards, floating background effects, and smart recommendation chips.
* **⚡ 3-in-1 Interactive Suite:** Seamlessly switch between **Single Prediction**, **Batch CSV Processing**, and an **Interactive HR Analytics Dashboard**.
* **📊 Advanced Plotly Visualizations:** Interactive charts with hover tooltips to analyze department-wise attrition, salary ranges, commute distance, and employee tenure.

---

## 🚀 Key Features

### 1️⃣ 🎯 Single Employee Prediction
Enter individual employee details through interactive sliders and dropdowns:
* **🧑 Personal Details:** Employee Age & Monthly Income (₹)
* **💼 Work Details:** Years at Company, Job Satisfaction Rating (1 to 4), and Overtime Status (`Yes` / `No`)
* **📌 Instant Output:** 
  * Dynamic Risk Level Badge (**🟢 Low Risk**, **🟡 Medium Risk**, **🔴 High Risk**)
  * Animated Percentage Ring Gauge
  * Key Risk Factor Detection (e.g., *⏰ Works overtime*, *😞 Low job satisfaction*, *💸 Low income*)
  * Actionable HR Recommendations

### 2️⃣ 📁 Batch Prediction (CSV Upload)
* Upload a **CSV file** containing multiple employee records (`Age`, `MonthlyIncome`, `YearsAtCompany`, `JobSatisfaction`, `OverTime`).
* Automatically generates predictions (**`Leave`** or **`Stay`**) along with **`Attrition_Risk_%`** for the entire dataset.
* **📥 One-Click Export:** Download the complete prediction results directly as `attrition_predictions.csv`.

### 3️⃣ 📈 HR Analytics & Insights Dashboard
Powered by **Plotly Express** for deep-dive visual exploration:
* **📌 Key HR Metrics:** Displays **Total Employees**, **Overall Attrition Rate (%)**, and **Average Monthly Income (₹)**.
* **🏢 Attrition Distribution by Department:** Grouped bar chart comparing *Retained* vs *Left* employees across *Human Resources*, *Research & Development*, and *Sales*.
* **📦 Compensation Range vs Retention Status:** Interactive box plot analyzing monthly income distribution and median salary gaps.
* **🔵 Income vs Commute Distance (Bubble Chart):** Multi-dimensional scatter plot where bubble size represents **Years at Company (Tenure)** and color indicates **Retention Status**.

---

## 🛠️ Tech Stack & Libraries

* **Frontend & Web Framework:** `Streamlit`, Custom `HTML5 / CSS3` (Glassmorphism & Keyframe Animations)
* **Machine Learning:** `Scikit-Learn` (`LogisticRegression`)
* **Data Processing:** `Pandas`, `NumPy`
* **Data Visualization:** `Plotly Express`

---

## 📂 Project Structure

```text
📦 hr-attrition-app
 ┣ 📜 app.py              # Main Streamlit application & ML pipeline
 ┣ 📜 requirements.txt    # Required Python dependencies for deployment
 ┗ 📜 README.md           # Project documentation
