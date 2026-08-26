#testing Github
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import io
import base64
import seaborn as sns
import os

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import r2_score, accuracy_score, confusion_matrix
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4

st.set_page_config(layout="wide", page_title="AI Analytics Studio")

# ================= BACKGROUND & STYLING =================
def get_base64(file):
    try:
        with open(file, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except FileNotFoundError:
        return ""

bg_image = get_base64("purple_bg.jpg")
robot_image = get_base64("robot.png")

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&family=Roboto:wght@300;400;700&display=swap');

.stApp {{
    background-image:
        linear-gradient(rgba(0,0,0,0.8), rgba(0,0,0,0.85)),
        url("data:image/jpg;base64,{bg_image}");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
    color:white;
    font-family:'Roboto',sans-serif;
}}

.block-container {{ padding:2rem 6rem; }}
#MainMenu, footer, header {{visibility:hidden;}}

.hero-title {{
    font-family: 'Orbitron', sans-serif;
    font-size:90px;
    font-weight:700;
    margin-top:40px;
    line-height: 0.9;
    letter-spacing: 2px;
    text-shadow: 0 0 20px rgba(138, 43, 226, 0.8);
}}

.hero-sub {{
    opacity:0.8;
    width:700px;
    font-size: 18px;
    margin-bottom: 30px;
}}

/* Login Page Robot Image - Fixed Right Side Background */
.login-robot-container {{
    position: fixed;
    right: 0;
    top: 0;
    width: 45%;
    height: 100vh;
    background-image: url("data:image/png;base64,{robot_image}");
    background-size: contain;
    background-position: center right;
    background-repeat: no-repeat;
    z-index: 0;
    pointer-events: none;
    opacity: 0.9;
}}

.glass-card {{
    background: rgba(255,255,255,0.05);
    backdrop-filter: blur(20px);
    border-radius:20px;
    padding:25px;
    border: 1px solid rgba(255,255,255,0.1);
    transition: transform 0.3s;
    height: 100%;
}}
.glass-card:hover {{
    transform: translateY(-5px);
    background: rgba(255,255,255,0.1);
}}

.stButton > button {{
    width: 100%;
    border-radius: 10px;
    font-weight: 600;
    background: rgba(255,255,255,0.1);
    color: white;
    border: 1px solid rgba(255,255,255,0.2);
}}
.stButton > button:hover {{
    background: rgba(138, 43, 226, 0.6);
    border-color: rgba(138, 43, 226, 1);
}}

.section-header {{
    font-family: 'Orbitron', sans-serif;
    font-size: 32px;
    margin-top: 40px;
    margin-bottom: 20px;
    border-bottom: 2px solid rgba(138, 43, 226, 0.5);
    padding-bottom: 10px;
}}

.quote-box {{
    background: linear-gradient(135deg, rgba(138, 43, 226, 0.3), rgba(75, 0, 130, 0.3));
    border-radius: 20px;
    padding: 50px;
    border: 2px solid rgba(138, 43, 226, 0.5);
    text-align: center;
    margin: 30px 0;
}}

.quote-text {{
    font-size: 42px;
    font-weight: 300;
    font-style: italic;
    color: #E0B0FF;
    margin-bottom: 30px;
}}

.data-info-box {{
    background: rgba(138, 43, 226, 0.2);
    border-radius: 10px;
    padding: 15px;
    margin: 10px 0;
    border-left: 4px solid rgba(138, 43, 226, 0.8);
}}

.suggestion-box {{
    background: rgba(75, 0, 130, 0.3);
    border-radius: 10px;
    padding: 20px;
    margin: 10px 0;
    border: 1px solid rgba(138, 43, 226, 0.4);
}}

.analysis-box {{
    background: rgba(255,255,255,0.08);
    border-radius: 15px;
    padding: 25px;
    margin: 15px 0;
    border: 1px solid rgba(138, 43, 226, 0.3);
}}

.login-input {{
    width: 100%;
    margin-bottom: 15px;
}}

.feature-box {{
    background: rgba(255,255,255,0.05);
    backdrop-filter: blur(20px);
    border-radius:20px;
    padding:25px;
    border: 1px solid rgba(255,255,255,0.1);
    height: 280px;
    display: flex;
    flex-direction: column;
    justify-content: flex-start;
}}

/* Login Form Container - Higher z-index than robot */
.login-form-container {{
    position: relative;
    z-index: 10;
    width: 55%;
    padding: 50px;
    background: transparent;
}}

/* Input fields styling */
.stTextInput > div > div > input {{
    background: rgba(255,255,255,0.1) !important;
    color: white !important;
    border: 1px solid rgba(138, 43, 226, 0.5) !important;
}}

.stTextInput > label {{
    color: white !important;
}}
</style>
""", unsafe_allow_html=True)

# ================= SESSION =================
if "logged_in" not in st.session_state:
    st.session_state.logged_in=False
if "menu" not in st.session_state:
    st.session_state.menu="Home"

# ================= LOGIN =================
if not st.session_state.logged_in:
    # Check if file exists
    image_path = "robot.png"
    image_exists = os.path.exists(image_path)
    
    # Add robot image as fixed background on right (BEHIND everything)
    if image_exists:
        st.markdown('<div class="login-robot-container"></div>', unsafe_allow_html=True)
    
    # Login Form on Left Side (IN FRONT of robot)
    st.markdown('<div class="login-form-container">', unsafe_allow_html=True)
    
    st.markdown('<div class="hero-title">AI<br>ANALYTICS<br>STUDIO</div>', unsafe_allow_html=True)
    st.markdown("<p class='hero-sub'>\"Empowering decisions through intelligent data synthesis. Unlock the hidden patterns within your noise.\"</p>", unsafe_allow_html=True)
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    c1, c2, c3 = st.columns([2, 3, 2])
    with c2:
        u = st.text_input("Username", key="login_user")
        p = st.text_input("Password", type="password", key="login_pass")

        if st.button("Login", use_container_width=True):
            if u=="admin" and p=="1234":
                st.session_state.logged_in=True
                st.rerun()
            else:
                st.error("Invalid login")
    
    st.markdown('</div>', unsafe_allow_html=True)

# ================= MAIN =================
else:
    col1, col2, col3 = st.columns([2, 6, 2])

    with col1:
        st.markdown("### 🤖 AI Studio")

    with col2:
        b1, b2, b3, b4, b5, b6 = st.columns(6)
        if b1.button("Home", use_container_width=True): st.session_state.menu="Home"
        if b2.button("Data Lab", use_container_width=True): st.session_state.menu="Data Lab"
        if b3.button("Visual Studio", use_container_width=True): st.session_state.menu="Visual Studio"
        if b4.button("AutoML", use_container_width=True): st.session_state.menu="AutoML"
        if b5.button("Insights", use_container_width=True): st.session_state.menu="Insights"
        if b6.button("Reports", use_container_width=True): st.session_state.menu="Reports"

    with col3:
        if st.button("Logout", use_container_width=True):
            st.session_state.clear()
            st.rerun()

    menu = st.session_state.menu
    df = st.session_state.get("clean", pd.DataFrame())

# ================= HOME =================
    if menu == "Home":
        st.markdown('<div class="hero-title">THE FUTURE OF DATA INTELLIGENCE</div>', unsafe_allow_html=True)

        st.markdown("""
        <p class="hero-sub">
        Welcome to AI Analytics Studio. This platform allows you to prepare data,
        train automated machine learning models, interpret performance,
        and export comprehensive professional reports.
        </p>
        """, unsafe_allow_html=True)

        st.markdown("## Overview")
        st.write("""
        AI Analytics Studio is an end‑to‑end machine learning platform designed for:
        • Structured data preparation  
        • Automated model detection (classification or regression)  
        • Ensemble-based training using Random Forest  
        • Cross-validation for performance stability  
        • Insight generation and downloadable reports  
        """)

        st.markdown("## How It Works")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown('<div class="glass-card"><h4>📁 Step 1: Data Lab</h4>Upload your dataset. Clean missing values and encode categorical variables.</div>', unsafe_allow_html=True)
        with c2:
            st.markdown('<div class="glass-card"><h4>🤖 Step 2: AutoML</h4>Select target and features. The system detects the problem type and trains a Random Forest model.</div>', unsafe_allow_html=True)
        with c3:
            st.markdown('<div class="glass-card"><h4>📊 Step 3: Insights & Reports</h4>Evaluate performance, detect overfitting, and download a full analytical PDF report.</div>', unsafe_allow_html=True)

        # --- Key Features in Boxes with Same Size/Dimension ---
        st.markdown('<div class="section-header">Key Features</div>', unsafe_allow_html=True)
        f1, f2, f3, f4 = st.columns(4)
        with f1:
            st.markdown('''
            <div class="feature-box">
                <h4>⚡ Auto-Cleaning</h4>
                <p>Intelligent missing value imputation using mean/median strategies. Automatically detects data types and applies appropriate cleaning methods for optimal model performance.</p>
            </div>
            ''', unsafe_allow_html=True)
        with f2:
            st.markdown('''
            <div class="feature-box">
                <h4>🧠 Smart Modeling</h4>
                <p>Auto-detects Regression or Classification problems. Uses Random Forest ensemble with 5-fold cross-validation for robust and reliable predictions.</p>
            </div>
            ''', unsafe_allow_html=True)
        with f3:
            st.markdown('''
            <div class="feature-box">
                <h4>📈 Visual Insights</h4>
                <p>AI-powered chart suggestions based on data types. Detailed correlation analysis and activity explanations for every visualization created.</p>
            </div>
            ''', unsafe_allow_html=True)
        with f4:
            st.markdown('''
            <div class="feature-box">
                <h4>📄 Pro Reports</h4>
                <p>Executive PDF summaries with correlation heatmaps, actual vs predicted plots, residual analysis, and confusion matrices for classification.</p>
            </div>
            ''', unsafe_allow_html=True)

        # --- Quote in Highlighted Box with Bigger Font + Button Inside ---
        st.markdown('''
        <div class="quote-box">
            <p class="quote-text">"Data is the new oil, but only if refined."</p>
        </div>
        ''', unsafe_allow_html=True)
        
        # Button within quote box area
        col_nav1, col_nav2, col_nav3 = st.columns([4, 1, 4])
        with col_nav2:
            if st.button("➤ Proceed to Data Lab", use_container_width=True):
                st.session_state.menu = "Data Lab"
                st.rerun()

# ================= DATA LAB =================
    if menu == "Data Lab":
        st.markdown('<div class="section-header">Data Laboratory</div>', unsafe_allow_html=True)
        
        file = st.file_uploader("Upload CSV")

        if file:
            df = pd.read_csv(file)
            st.session_state["raw"] = df

        if "raw" in st.session_state:
            df = st.session_state["raw"].copy()
            
            # KPIs
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Total Rows", df.shape[0])
            k2.metric("Total Columns", df.shape[1])
            k3.metric("Missing Values", df.isnull().sum().sum())
            k4.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

            st.markdown("### Preprocessing Actions")
            
            # Data Appending Explanation
            st.markdown('''
            <div class="data-info-box">
                <h5>📋 Data Transformation Log</h5>
                <p><strong>Original Shape:</strong> {} rows × {} columns</p>
                <p><strong>After Processing:</strong> Shape may change based on encoding (new columns added for categories)</p>
                <p><strong>Missing Value Strategy:</strong> Numeric columns filled with mean, categorical columns preserved</p>
                <p><strong>Encoding Method:</strong> One-Hot Encoding with drop_first=True to avoid multicollinearity</p>
            </div>
            '''.format(df.shape[0], df.shape[1]), unsafe_allow_html=True)
            
            c_chk1, c_exp1 = st.columns([1, 4])
            with c_chk1:
                fill_missing = st.checkbox("Fill Missing")
            with c_exp1:
                if fill_missing:
                    st.info("ℹ️ **Action:** Replaces NaN values with the mean of numeric columns. Ensures model compatibility.")
                    df = df.fillna(df.mean(numeric_only=True))
                    try:
                        st.toast("✅ Missing values filled successfully!", icon="🎉")
                    except:
                        st.success("✅ Missing values filled successfully!")

            c_chk2, c_exp2 = st.columns([1, 4])
            with c_chk2:
                encode_data = st.checkbox("Encode Categoricals")
            with c_exp2:
                if encode_data:
                    st.info("ℹ️ **Action:** Converts text categories into numbers (One-Hot Encoding). Required for mathematical modeling.")
                    df = pd.get_dummies(df, drop_first=True)
                    try:
                        st.toast("✅ Data encoded successfully!", icon="🎉")
                    except:
                        st.success("✅ Data encoded successfully!")

            st.session_state["clean"] = df
            st.markdown("### Processed Data Preview")
            st.dataframe(df)

            # Navigation Button to Next Page
            col_nav1, col_nav2, col_nav3 = st.columns([4, 1, 4])
            with col_nav2:
                if st.button("➤ Proceed to Visual Studio", use_container_width=True):
                    st.session_state.menu = "Visual Studio"
                    st.rerun()

# ================= VISUAL STUDIO =================
    if menu == "Visual Studio":
        st.markdown('<div class="section-header">Visual Studio</div>', unsafe_allow_html=True)

        if not df.empty:
            # --- List of All Suggestions with Actual Column Names ---
            st.markdown("### 💡 AI Suggestions")
            num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            
            suggestions = []
            
            if len(cat_cols) > 0 and len(num_cols) > 0:
                suggestions.append({
                    "Chart": "Bar Chart",
                    "X-Axis": f"{cat_cols[0]}",
                    "Y-Axis": f"{num_cols[0]}",
                    "Reason": "Best for comparing values across categories"
                })
            
            if len(num_cols) >= 2:
                suggestions.append({
                    "Chart": "Scatter Plot",
                    "X-Axis": f"{num_cols[0]}",
                    "Y-Axis": f"{num_cols[1]}",
                    "Reason": "Best for identifying correlations between numeric variables"
                })
                suggestions.append({
                    "Chart": "Line Chart",
                    "X-Axis": f"{num_cols[0]}",
                    "Y-Axis": f"{num_cols[1]}",
                    "Reason": "Best for visualizing trends over sequence/time"
                })
            
            if len(cat_cols) >= 1 and len(num_cols) >= 1:
                suggestions.append({
                    "Chart": "Pie Chart",
                    "X-Axis": f"{cat_cols[0]}",
                    "Y-Axis": f"{num_cols[0]}",
                    "Reason": "Best for showing proportional distribution"
                })
            
            # Display Suggestions in Boxes
            for i, sug in enumerate(suggestions[:4]):
                st.markdown(f'''
                <div class="suggestion-box">
                    <strong>Suggestion {i+1}:</strong> {sug['Chart']}<br>
                    <strong>X-Axis:</strong> {sug['X-Axis']}<br>
                    <strong>Y-Axis:</strong> {sug['Y-Axis']}<br>
                    <strong>Why:</strong> {sug['Reason']}
                </div>
                ''', unsafe_allow_html=True)

            st.markdown("### Chart Configuration")
            chart = st.selectbox("Chart Type", ["Bar", "Line", "Scatter", "Pie"])
            x = st.selectbox("X Axis", df.columns)
            y = st.selectbox("Y Axis", df.columns)

            try:
                if chart == "Scatter":
                    fig = px.scatter(df, x=x, y=y)
                elif chart == "Bar":
                    fig = px.bar(df, x=x, y=y)
                elif chart == "Line":
                    fig = px.line(df, x=x, y=y)
                elif chart == "Pie":
                    fig = px.pie(df, names=x, values=y)
                
                st.plotly_chart(fig, use_container_width=True)

                # --- Detailed Activity Analysis ---
                st.markdown("### 📊 Detailed Activity Analysis")
                
                if chart in ["Scatter", "Line"]:
                    if x in num_cols and y in num_cols:
                        corr = df[x].corr(df[y])
                        st.write(f"**Correlation Coefficient:** {corr:.4f}")
                        
                        if abs(corr) > 0.8:
                            st.success("🟢 **Very Strong Relationship:** 80%+ linear association. Changes in X strongly predict changes in Y.")
                        elif abs(corr) > 0.6:
                            st.success("🟢 **Strong Relationship:** 60-80% linear association. Good predictive power.")
                        elif abs(corr) > 0.4:
                            st.warning("🟡 **Moderate Relationship:** 40-60% linear association. Some predictive value exists.")
                        elif abs(corr) > 0.2:
                            st.warning("🟡 **Weak Relationship:** 20-40% linear association. Limited predictive value.")
                        else:
                            st.error("🔴 **Very Weak/No Relationship:** Below 20%. Variables are largely independent.")
                        
                        r_squared = corr ** 2
                        st.write(f"**R² Value:** {r_squared:.4f}")
                        st.write(f"**Explanation:** Approximately {r_squared*100:.1f}% of the variance in **{y}** can be explained by **{x}**.")
                        
                        st.write(f"**{x} Statistics:** Mean={df[x].mean():.2f}, Std={df[x].std():.2f}, Min={df[x].min():.2f}, Max={df[x].max():.2f}")
                        st.write(f"**{y} Statistics:** Mean={df[y].mean():.2f}, Std={df[y].std():.2f}, Min={df[y].min():.2f}, Max={df[y].max():.2f}")
                        
                    else:
                        st.warning("⚠️ **Improper Chart Selection:** Scatter/Line plots work best with two numeric axes. Try using a Bar chart for categorical data.")
                
                elif chart == "Bar":
                    if x in cat_cols:
                        unique_count = df[x].nunique()
                        st.write(f"**Categories in {x}:** {unique_count}")
                        st.write(f"**Average {y} per Category:** {df.groupby(x)[y].mean().round(2).to_dict()}")
                        st.write(f"**Explanation:** This bar chart shows how **{y}** varies across different categories of **{x}**. Higher bars indicate greater values.")
                    else:
                        st.write(f"**Explanation:** This bar chart displays **{y}** values grouped by **{x}**. Useful for comparing magnitudes across different groups.")
                
                elif chart == "Pie":
                    if x in cat_cols:
                        total = df[y].sum()
                        percentages = df.groupby(x)[y].sum() / total * 100
                        st.write(f"**Total Value:** {total:.2f}")
                        st.write(f"**Distribution:** {percentages.round(2).to_dict()}%")
                        st.write(f"**Explanation:** This pie chart shows the proportional share of **{y}** for each category in **{x}**.")
                
            except Exception as e:
                st.error(f"⚠️ **Chart Improper:** {str(e)}. Suggested Improvement: Ensure data types match the chart type.")

            # Navigation Button to Next Page
            col_nav1, col_nav2, col_nav3 = st.columns([4, 1, 4])
            with col_nav2:
                if st.button("➤ Proceed to AutoML", use_container_width=True):
                    st.session_state.menu = "AutoML"
                    st.rerun()

# ================= AUTOML =================
    if menu == "AutoML":
        st.markdown('<div class="section-header">AutoML Engine</div>', unsafe_allow_html=True)

        if not df.empty:
            # --- Auto Convert Non-Numeric to Numeric ---
            st.markdown("### 🔧 Data Preparation")
            auto_convert = st.checkbox("Auto-convert non-numeric columns to numeric", value=True)
            
            if auto_convert and "raw" in st.session_state:
                df_prep = st.session_state["raw"].copy()
                # Convert non-numeric columns using Label Encoding
                for col in df_prep.columns:
                    if df_prep[col].dtype == 'object' or df_prep[col].dtype == 'category':
                        le = LabelEncoder()
                        df_prep[col] = le.fit_transform(df_prep[col].astype(str))
                st.session_state["clean"] = df_prep
                df = df_prep
                st.success("✅ Non-numeric columns converted to numeric values")
            
            task_choice = st.selectbox("Model Task", ["Auto-Detect", "Classification", "Regression"])
            
            target = st.selectbox("Target", df.columns)
            features = st.multiselect("Features", df.columns)

            if st.button("Run Model"):
                if len(features) == 0:
                    st.error("Please select at least one feature.")
                else:
                    X = StandardScaler().fit_transform(df[features])
                    y = df[target]
                    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)

                    mtype = ""
                    if task_choice == "Classification" or (task_choice == "Auto-Detect" and y.nunique() < 10):
                        model = RandomForestClassifier()
                        model.fit(Xtr, ytr)
                        preds = model.predict(Xte)
                        score = accuracy_score(yte, preds)
                        mtype = "Classification"
                    else:
                        model = RandomForestRegressor()
                        model.fit(Xtr, ytr)
                        preds = model.predict(Xte)
                        score = r2_score(yte, preds)
                        mtype = "Regression"

                    cv = cross_val_score(model, X, y, cv=5).mean()

                    st.session_state.update({
                        "score": score,
                        "cv": cv,
                        "type": mtype,
                        "preds": preds,
                        "y_test": yte,
                        "features": features,
                        "model_obj": model,
                        "X_cols": features
                    })

                    st.success("Model Trained Successfully")
                    st.write("Model Type:", mtype)
                    st.write("Score:", round(score, 3))
                    st.write("CV:", round(cv, 3))

                    # Model Interpretation
                    st.markdown("### 🧠 Model Interpretation")
                    if score > 0.8:
                        st.success("🚀 **High Performance:** The model has learned the patterns very well. The CV score confirms stability.")
                    elif score > 0.5:
                        st.warning("⚙️ **Moderate Performance:** The model captures some trends but may need more data or feature engineering.")
                    else:
                        st.error("📉 **Low Performance:** The model struggles to predict. Check for noise or irrelevant features.")
                    
                    st.markdown(f"**Why this score?** The {mtype} model achieved {round(score, 2)} because the features selected have a {'strong' if score>0.7 else 'weak'} predictive relationship with the target.")

                    # --- Explain Negative CV ---
                    if cv < 0:
                        st.error("⚠️ **Negative CV Score Detected**")
                        st.markdown('''
                        <div class="data-info-box">
                            <h5>Why is CV Negative?</h5>
                            <p><strong>1. Model Worse Than Baseline:</strong> The model performs worse than simply predicting the mean value.</p>
                            <p><strong>2. Overfitting:</strong> The model memorized training data but fails on new folds.</p>
                            <p><strong>3. Data Issues:</strong> Possible data leakage, incorrect target, or insufficient features.</p>
                            <p><strong>4. High Variance:</strong> Model predictions vary wildly across different data folds.</p>
                            <p><strong>Recommendation:</strong> Try feature selection, reduce model complexity, or collect more data.</p>
                        </div>
                        ''', unsafe_allow_html=True)
                    else:
                        st.success(f"✅ **Positive CV Score ({round(cv, 3)}):** Model generalizes well across different data folds.")

                    # Feature Importance
                    if hasattr(model, 'feature_importances_'):
                        st.markdown("### Feature Importance")
                        fi_df = pd.DataFrame({
                            "Feature": features,
                            "Importance": model.feature_importances_
                        }).sort_values(by="Importance", ascending=False)
                        
                        fig_fi = px.bar(fi_df, x="Importance", y="Feature", orientation="h", title="Which Features Mattered Most?")
                        st.plotly_chart(fig_fi, use_container_width=True)

            # Navigation Button to Next Page
            col_nav1, col_nav2, col_nav3 = st.columns([4, 1, 4])
            with col_nav2:
                if st.button("➤ Proceed to Insights", use_container_width=True):
                    st.session_state.menu = "Insights"
                    st.rerun()

# ================= INSIGHTS =================
    if menu == "Insights":
        st.markdown('<div class="section-header">Performance Insights</div>', unsafe_allow_html=True)
        
        if "score" in st.session_state:
            score = st.session_state["score"]
            cv = st.session_state["cv"]

            c1, c2 = st.columns(2)
            with c1:
                st.metric("Model Accuracy/Score", f"{round(score * 100, 2)}%")
            with c2:
                st.metric("Stability (CV)", f"{round(cv * 100, 2)}%")

            st.write("Model Type:", st.session_state["type"])
            
            # --- Quantitative Analysis in Boxes ---
            st.markdown("### 📈 Quantitative Performance Analysis")
            progress = int(score * 100)
            st.progress(progress)
            
            # Performance Level Box
            st.markdown(f'''
            <div class="analysis-box">
                <h4>📊 Performance Level</h4>
                <p><strong>Score:</strong> {progress}%</p>
                <p><strong>Rating:</strong> {'Excellent' if progress > 80 else 'Good' if progress > 60 else 'Fair' if progress > 40 else 'Needs Improvement'}</p>
            </div>
            ''', unsafe_allow_html=True)

            # CV Score Analysis Box
            st.markdown("### 🔍 Cross-Validation Score Analysis")
            cv_percentage = cv * 100
            if cv > 0:
                st.markdown(f'''
                <div class="analysis-box">
                    <h4>✅ CV Score: {round(cv_percentage, 2)}%</h4>
                    <p><strong>Interpretation:</strong> On average, the model explains {round(cv_percentage, 2)}% of variance across 5 different data folds.</p>
                    <p><strong>Train-Test Gap:</strong> {round((score - cv) * 100, 2)}% difference indicates {'minimal overfitting' if abs(score-cv) < 0.1 else 'potential overfitting'}</p>
                </div>
                ''', unsafe_allow_html=True)
            elif cv < 0:
                st.markdown(f'''
                <div class="analysis-box">
                    <h4>❌ CV Score: {round(cv_percentage, 2)}%</h4>
                    <p><strong>Interpretation:</strong> Negative score means model performs worse than baseline (predicting mean).</p>
                    <p><strong>Performance Gap:</strong> Model is {round(abs(cv_percentage), 2)}% worse than simple mean prediction.</p>
                    <p><strong>Recommendation:</strong> Reduce model complexity or improve feature quality.</p>
                </div>
                ''', unsafe_allow_html=True)
            else:
                st.markdown(f'''
                <div class="analysis-box">
                    <h4>⚠️ CV Score: 0%</h4>
                    <p><strong>Interpretation:</strong> Model performs equally to baseline prediction.</p>
                </div>
                ''', unsafe_allow_html=True)

            # Model Accuracy Analysis Box
            st.markdown("### 🎯 Model Accuracy Analysis")
            if st.session_state["type"] == "Classification":
                st.markdown(f'''
                <div class="analysis-box">
                    <h4>Classification Metrics</h4>
                    <p><strong>Accuracy:</strong> {round(score * 100, 2)}% of predictions were correct</p>
                    <p><strong>Error Rate:</strong> {round((1 - score) * 100, 2)}% of predictions were incorrect</p>
                ''', unsafe_allow_html=True)
                if "y_test" in st.session_state:
                    y_test_data = st.session_state['y_test']
                    unique_classes = len(y_test_data.unique()) if len(y_test_data) > 0 else 1
                    baseline_accuracy = round(100 / unique_classes, 1)
                    st.markdown(f'''
                    <p><strong>Baseline Comparison:</strong> Random guessing would achieve ~{baseline_accuracy}% accuracy</p>
                </div>
                    ''', unsafe_allow_html=True)
                else:
                    st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'''
                <div class="analysis-box">
                    <h4>Regression Metrics</h4>
                    <p><strong>R² Score:</strong> {round(score * 100, 2)}% of target variance is explained by features</p>
                    <p><strong>Unexplained Variance:</strong> {round((1 - score) * 100, 2)}% remains unexplained</p>
                    <p><strong>Prediction Quality:</strong> {'Excellent' if score > 0.8 else 'Good' if score > 0.6 else 'Fair' if score > 0.4 else 'Poor'}</p>
                </div>
                ''', unsafe_allow_html=True)

            # Why This Performance Box
            st.markdown("### 🔍 Why This Performance?")
            if score > 0.8:
                st.markdown('''
                <div class="analysis-box">
                    <h4>✅ Strengths</h4>
                    <p>Data quality is high, features are relevant, and the model generalizes well.</p>
                </div>
                ''', unsafe_allow_html=True)
            elif score > 0.5:
                st.markdown('''
                <div class="analysis-box">
                    <h4>⚠️ Observations</h4>
                    <p>Some noise exists in the data. The model captures general trends but misses specifics.</p>
                </div>
                ''', unsafe_allow_html=True)
            else:
                st.markdown('''
                <div class="analysis-box">
                    <h4>❌ Issues</h4>
                    <p>High variance or bias. Features may not correlate strongly with the target.</p>
                </div>
                ''', unsafe_allow_html=True)

            # Scope for Improvement Box
            st.markdown("### 🚀 Scope for Improvement")
            st.markdown('''
            <div class="analysis-box">
                <h4>Recommendations</h4>
                <ul>
                    <li><strong>Data Collection:</strong> Gather more samples to reduce overfitting.</li>
                    <li><strong>Feature Engineering:</strong> Create new interaction features or remove low-importance ones.</li>
                    <li><strong>Hyperparameter Tuning:</strong> Optimize tree depth and estimator count for better fit.</li>
                </ul>
            </div>
            ''', unsafe_allow_html=True)

            # Navigation Button to Next Page
            col_nav1, col_nav2, col_nav3 = st.columns([4, 1, 4])
            with col_nav2:
                if st.button("➤ Proceed to Reports", use_container_width=True):
                    st.session_state.menu = "Reports"
                    st.rerun()

        else:
            st.warning("Run a model first.")

# ================= REPORTS =================
    if menu == "Reports":
        st.markdown('<div class="section-header">Executive Reports</div>', unsafe_allow_html=True)
        
        if "score" in st.session_state:
            st.success("🎉 Thank you for using AI Analytics Studio! Your analysis is ready.")

            # --- Preview Before Download ---
            st.markdown("### 📋 Report Preview")
            st.markdown('''
            <div class="glass-card">
                <h4>What You Will Download:</h4>
                <ul>
                    <li><strong>Executive Summary:</strong> Overview of analysis and key findings</li>
                    <li><strong>Model Details:</strong> Type, Accuracy Score, Cross-Validation Score</li>
                    <li><strong>Correlation Heatmap:</strong> Visual representation of feature relationships</li>
                    <li><strong>Actual vs Predicted Plot:</strong> Model performance visualization</li>
                    <li><strong>Residual Plot:</strong> Error distribution analysis</li>
                    <li><strong>Confusion Matrix:</strong> (For Classification) Prediction accuracy breakdown</li>
                    <li><strong>Thank You Message:</strong> Closing statement</li>
                </ul>
                <p><strong>File Format:</strong> PDF (A4 Size)</p>
                <p><strong>File Name:</strong> AI_Executive_Report.pdf</p>
            </div>
            ''', unsafe_allow_html=True)

            # Report Summary Preview
            st.markdown("### 📝 Summary Preview")
            st.info(f"""
            **Model Type:** {st.session_state['type']}  
            **Performance Score:** {round(st.session_state['score']*100, 2)}%  
            **Cross-Validation:** {round(st.session_state['cv']*100, 2)}%  
            **Features Used:** {len(st.session_state.get('features', []))}  
            **Data Quality Assessment:** {'High' if st.session_state['score'] > 0.7 else 'Moderate' if st.session_state['score'] > 0.5 else 'Needs Improvement'}
            """)

            def pdf():
                buffer = io.BytesIO()
                doc = SimpleDocTemplate(buffer, pagesize=A4)
                styles = getSampleStyleSheet()
                elems = []

                elems.append(Paragraph("AI Analytics Studio - Full Executive Report", styles['Title']))
                elems.append(Spacer(1, 20))
                
                elems.append(Paragraph("📝 Executive Summary", styles['Heading2']))
                summary_text = f"""
                We analyzed the dataset using a {st.session_state['type']} model. 
                The model achieved a performance score of {round(st.session_state['score']*100, 2)}%. 
                Key learnings indicate that the selected features have a 
                {'strong' if st.session_state['score'] > 0.7 else 'moderate'} impact on the target variable.
                """
                elems.append(Paragraph(summary_text, styles['Normal']))
                elems.append(Spacer(1, 20))

                elems.append(Paragraph(f"Model Type: {st.session_state['type']}", styles['Normal']))
                elems.append(Paragraph(f"Score: {round(st.session_state['score'],3)}", styles['Normal']))
                elems.append(Paragraph(f"Cross Validation: {round(st.session_state['cv'],3)}", styles['Normal']))
                elems.append(Spacer(1, 20))

                if "clean" in st.session_state:
                    df_rep = st.session_state["clean"]
                    corr = df_rep.corr(numeric_only=True)

                    fig_corr, ax_corr = plt.subplots(figsize=(6,5))
                    cax = ax_corr.matshow(corr)
                    fig_corr.colorbar(cax)

                    ax_corr.set_title("Correlation Heatmap", pad=20)
                    ax_corr.set_xticks(range(len(corr.columns)))
                    ax_corr.set_yticks(range(len(corr.columns)))
                    ax_corr.set_xticklabels(corr.columns, rotation=90, fontsize=6)
                    ax_corr.set_yticklabels(corr.columns, fontsize=6)

                    buf_corr = io.BytesIO()
                    fig_corr.tight_layout()
                    fig_corr.savefig(buf_corr, format="png")
                    buf_corr.seek(0)

                    elems.append(Image(buf_corr, width=400, height=300))
                    elems.append(Spacer(1, 20))

                if "preds" in st.session_state:
                    y_test = st.session_state["y_test"]
                    preds = st.session_state["preds"]

                    fig1, ax1 = plt.subplots(figsize=(6,4))
                    ax1.scatter(y_test, preds, label="Predictions", alpha=0.7)
                    ax1.set_title("Actual vs Predicted")
                    ax1.set_xlabel("Actual Values")
                    ax1.set_ylabel("Predicted Values")
                    ax1.legend()

                    buf1 = io.BytesIO()
                    fig1.tight_layout()
                    fig1.savefig(buf1, format="png")
                    buf1.seek(0)

                    elems.append(Image(buf1, width=400, height=250))
                    elems.append(Spacer(1, 20))

                    residuals = y_test - preds
                    fig2, ax2 = plt.subplots(figsize=(6,4))
                    ax2.scatter(preds, residuals, label="Residuals", alpha=0.7)
                    ax2.axhline(0, linestyle="--", color="red", label="Zero Error Line")
                    ax2.set_title("Residual Plot")
                    ax2.set_xlabel("Predicted Values")
                    ax2.set_ylabel("Residual Error")
                    ax2.legend()

                    buf2 = io.BytesIO()
                    fig2.tight_layout()
                    fig2.savefig(buf2, format="png")
                    buf2.seek(0)

                    elems.append(Image(buf2, width=400, height=250))
                    elems.append(Spacer(1, 20))

                if st.session_state["type"] == "Classification":
                    cm = confusion_matrix(st.session_state["y_test"], st.session_state["preds"])

                    fig_cm, ax_cm = plt.subplots(figsize=(5,4))
                    sns.heatmap(cm, annot=True, fmt="d", cmap="Purples", ax=ax_cm)
                    ax_cm.set_title("Confusion Matrix")
                    ax_cm.set_xlabel("Predicted Label")
                    ax_cm.set_ylabel("True Label")

                    buf_cm = io.BytesIO()
                    fig_cm.tight_layout()
                    fig_cm.savefig(buf_cm, format="png")
                    buf_cm.seek(0)

                    elems.append(Image(buf_cm, width=400, height=250))
                    elems.append(Spacer(1, 20))

                elems.append(Spacer(1, 50))
                elems.append(Paragraph("Thank you for trusting AI Analytics Studio.", styles['Heading3']))
                elems.append(Paragraph("Generated automatically by your AI Assistant.", styles['Normal']))

                doc.build(elems)
                buffer.seek(0)
                return buffer

            st.download_button("Download Full Executive Report", pdf(), "AI_Executive_Report.pdf", use_container_width=True)

            # Navigation Button to Home
            col_nav1, col_nav2, col_nav3 = st.columns([4, 1, 4])
            with col_nav2:
                if st.button("🏠 Return to Home", use_container_width=True):
                    st.session_state.menu = "Home"
                    st.rerun()

        else:
            st.warning("Run a model first.")
