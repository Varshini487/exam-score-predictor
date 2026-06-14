import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import shap

st.set_page_config(page_title="📚 Exam Score Predictor", layout="wide")
st.title("📚 Exam Score Predictor")
st.markdown("Predict student performance based on study habits and demographics")

@st.cache_data
def generate_student_data(n=500):
    np.random.seed(42)
    df = pd.DataFrame({
        "hours_studied": np.random.uniform(5, 20, n),
        "prev_gpa": np.random.uniform(2.0, 4.0, n),
        "attendance_percent": np.random.uniform(70, 100, n),
        "family_income": np.random.choice([1, 2, 3, 4, 5], n),  # 1=low, 5=high
        "sleep_hours": np.random.uniform(5, 9, n),
        "extracurricular": np.random.choice([0, 1], n),  # 0=no, 1=yes
        "school_type": np.random.choice([0, 1, 2], n),  # 0=public, 1=private, 2=charter
    })
    # Realistic target: exam_score influenced by features
    df["exam_score"] = (
        50 +
        2.5 * df["hours_studied"] +
        8 * df["prev_gpa"] +
        0.3 * df["attendance_percent"] +
        1.5 * df["family_income"] +
        1 * df["sleep_hours"] +
        2 * df["extracurricular"] +
        np.random.normal(0, 5, n)
    )
    df["exam_score"] = np.clip(df["exam_score"], 0, 100)
    return df

df = generate_student_data()

tab1, tab2, tab3 = st.tabs(["📊 EDA", "🤖 Model", "🔮 Predict"])

with tab1:
    st.subheader("Dataset Overview")
    st.dataframe(df.head(20))
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Students", len(df))
    col2.metric("Avg Score", f"{df['exam_score'].mean():.1f}")
    col3.metric("Std Dev", f"{df['exam_score'].std():.1f}")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0, 0].scatter(df["hours_studied"], df["exam_score"], alpha=0.5); axes[0, 0].set_title("Hours Studied vs Score")
    axes[0, 1].scatter(df["prev_gpa"], df["exam_score"], alpha=0.5, color="orange"); axes[0, 1].set_title("Prev GPA vs Score")
    axes[1, 0].scatter(df["attendance_percent"], df["exam_score"], alpha=0.5, color="green"); axes[1, 0].set_title("Attendance vs Score")
    axes[1, 1].scatter(df["sleep_hours"], df["exam_score"], alpha=0.5, color="red"); axes[1, 1].set_title("Sleep Hours vs Score")
    st.pyplot(fig)

with tab2:
    if st.button("🚀 Train Model"):
        X = df.drop("exam_score", axis=1)
        y = df["exam_score"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        st.success(f"✅ Model Trained! R² = {r2:.3f}, RMSE = {rmse:.2f}")
        st.metric("Mean Absolute Error", f"{np.mean(np.abs(y_test - y_pred)):.2f}")
        
        fig, ax = plt.subplots()
        ax.scatter(y_test, y_pred, alpha=0.5); ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        ax.set_xlabel("Actual"); ax.set_ylabel("Predicted")
        st.pyplot(fig)
        
        st.session_state["model"] = model
        st.session_state["X_train"] = X_train
        st.session_state["feature_names"] = X.columns.tolist()

with tab3:
    st.subheader("Predict Score for a Student")
    c1, c2 = st.columns(2)
    hours = c1.slider("Hours Studied", 0.0, 20.0, 10.0)
    gpa = c2.slider("Previous GPA", 2.0, 4.0, 3.5)
    attend = c1.slider("Attendance %", 70, 100, 85)
    income = c2.selectbox("Family Income Level", [1, 2, 3, 4, 5])
    sleep = c1.slider("Sleep Hours", 5, 9, 7)
    extra = c2.checkbox("Extracurricular Activities", value=True)
    school = c1.selectbox("School Type", ["Public", "Private", "Charter"])
    
    if st.button("📈 Predict Score") and "model" in st.session_state:
        school_map = {"Public": 0, "Private": 1, "Charter": 2}
        inp = np.array([[hours, gpa, attend, income, sleep, int(extra), school_map[school]]])
        pred = st.session_state["model"].predict(inp)[0]
        
        st.success(f"**Predicted Exam Score: {pred:.1f}/100**")
        
        if pred >= 80:
            st.balloons()
            st.success("🌟 Excellent! This student is well-prepared.")
        elif pred >= 70:
            st.info("👍 Good performance expected. Some review recommended.")
        else:
            st.warning("⚠️ At-risk. Recommend additional tutoring or study sessions.")
