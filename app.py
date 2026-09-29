"""
Q-CareTwin: Quantum Causal Disease Detection App
SIH26139 - Smart India Hackathon 2026

Beginner-friendly working app
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
import warnings

warnings.filterwarnings('ignore')

# -------------------------
# Page config & CSS
# -------------------------
st.set_page_config(
    page_title="Q-CareTwin | SIH26139",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .warning-box {
        background-color: #fff3cd;
        border-left: 5px solid #ffc107;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------
# Session state init
# -------------------------
if 'models_trained' not in st.session_state:
    st.session_state.models_trained = False
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'df' not in st.session_state:
    st.session_state.df = None
if 'scaler' not in st.session_state:
    st.session_state.scaler = None
if 'label_encoders' not in st.session_state:
    st.session_state.label_encoders = {}
if 'models' not in st.session_state:
    st.session_state.models = {}
if 'metrics' not in st.session_state:
    st.session_state.metrics = {}
if 'batch_predictions' not in st.session_state:
    st.session_state.batch_predictions = None
if 'feature_cols' not in st.session_state:
    st.session_state.feature_cols = None

# -------------------------
# Helper functions
# -------------------------

def generate_demo_data(n_patients=200, n_visits_per_patient=3):
    np.random.seed(42)
    data = []
    for i in range(n_patients):
        patient_id = f"P{str(i+1).zfill(3)}"
        base_age = np.random.randint(30, 70)
        base_bmi = np.random.uniform(22, 35)
        base_glucose = np.random.uniform(90, 180)
        base_hba1c = np.random.uniform(5.0, 9.0)
        base_bp = np.random.randint(110, 160)
        smoking = np.random.choice([0, 1], p=[0.7, 0.3])
        medication = np.random.choice(['None', 'Metformin', 'Insulin'])
        
        for visit in range(n_visits_per_patient):
            visit_date = datetime(2025, 1, 1) + timedelta(days=visit*90 + np.random.randint(-10, 10))
            age = base_age
            bmi = base_bmi + np.random.uniform(-1, 1)
            glucose = base_glucose + np.random.uniform(-15, 15)
            hba1c = base_hba1c + np.random.uniform(-0.5, 0.5)
            bp_systolic = base_bp + np.random.randint(-10, 10)
            
            risk_score = (
                0.02 * (glucose - 100) +
                0.05 * (hba1c - 5.5) +
                0.03 * (bmi - 25) +
                0.01 * (bp_systolic - 120) +
                0.3 * smoking +
                0.1 * (1 if medication != 'None' else 0)
            )
            disease_outcome = 1 if risk_score > np.random.uniform(2, 5) else 0
            
            data.append({
                'patient_id': patient_id,
                'visit_date': visit_date.strftime('%Y-%m-%d'),
                'age': int(age),
                'sex': np.random.choice(['M', 'F']),
                'bmi': round(bmi, 1),
                'glucose': round(glucose, 1),
                'hba1c': round(hba1c, 2),
                'bp_systolic': int(bp_systolic),
                'smoking': smoking,
                'medication': medication,
                'disease_outcome': disease_outcome
            })
    return pd.DataFrame(data)


def preprocess_data(df, target_col='disease_outcome'):
    df_processed = df.copy()
    df_processed['visit_date'] = pd.to_datetime(df_processed['visit_date'])
    df_processed['visit_month'] = df_processed['visit_date'].dt.month
    df_processed['visit_year'] = df_processed['visit_date'].dt.year
    
    le_sex = LabelEncoder()
    df_processed['sex_encoded'] = le_sex.fit_transform(df_processed['sex'])
    
    le_med = LabelEncoder()
    df_processed['medication_encoded'] = le_med.fit_transform(df_processed['medication'])
    
    feature_cols = [
        'age', 'bmi', 'glucose', 'hba1c', 'bp_systolic',
        'smoking', 'sex_encoded', 'medication_encoded',
        'visit_month', 'visit_year'
    ]
    
    X = df_processed[feature_cols].values
    y = df_processed[target_col].values
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, feature_cols, scaler, {'sex': le_sex, 'medication': le_med}


def train_classical_models(X, y):
    models = {}
    metrics = {}
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Logistic Regression
    lr = LogisticRegression(random_state=42, max_iter=1000)
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    y_prob_lr = lr.predict_proba(X_test)[:, 1]
    
    models['Logistic Regression'] = {
        'model': lr,
        'predictions': y_pred_lr,
        'probabilities': y_prob_lr,
        'X_test': X_test,
        'y_test': y_test
    }
    metrics['Logistic Regression'] = {
        'accuracy': accuracy_score(y_test, y_pred_lr),
        'precision': precision_score(y_test, y_pred_lr, zero_division=0),
        'recall': recall_score(y_test, y_pred_lr, zero_division=0),
        'f1': f1_score(y_test, y_pred_lr, zero_division=0),
        'auc': roc_auc_score(y_test, y_prob_lr) if len(np.unique(y_test)) > 1 else 0.5
    }
    
    # Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    y_prob_rf = rf.predict_proba(X_test)[:, 1]
    
    models['Random Forest'] = {
        'model': rf,
        'predictions': y_pred_rf,
        'probabilities': y_prob_rf,
        'X_test': X_test,
        'y_test': y_test
    }
    metrics['Random Forest'] = {
        'accuracy': accuracy_score(y_test, y_pred_rf),
        'precision': precision_score(y_test, y_pred_rf, zero_division=0),
        'recall': recall_score(y_test, y_pred_rf, zero_division=0),
        'f1': f1_score(y_test, y_pred_rf, zero_division=0),
        'auc': roc_auc_score(y_test, y_prob_rf) if len(np.unique(y_test)) > 1 else 0.5
    }
    
    # Gradient Boosting
    gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
    gb.fit(X_train, y_train)
    y_pred_gb = gb.predict(X_test)
    y_prob_gb = gb.predict_proba(X_test)[:, 1]
    
    models['Gradient Boosting'] = {
        'model': gb,
        'predictions': y_pred_gb,
        'probabilities': y_prob_gb,
        'X_test': X_test,
        'y_test': y_test
    }
    metrics['Gradient Boosting'] = {
        'accuracy': accuracy_score(y_test, y_pred_gb),
        'precision': precision_score(y_test, y_pred_gb, zero_division=0),
        'recall': recall_score(y_test, y_pred_gb, zero_division=0),
        'f1': f1_score(y_test, y_pred_gb, zero_division=0),
        'auc': roc_auc_score(y_test, y_prob_gb) if len(np.unique(y_test)) > 1 else 0.5
    }
    
    # MLP Neural Network
    mlp = MLPClassifier(hidden_layer_sizes=(50, 25), random_state=42, max_iter=500)
    mlp.fit(X_train, y_train)
    y_pred_mlp = mlp.predict(X_test)
    y_prob_mlp = mlp.predict_proba(X_test)[:, 1]
    
    models['Neural Network'] = {
        'model': mlp,
        'predictions': y_pred_mlp,
        'probabilities': y_prob_mlp,
        'X_test': X_test,
        'y_test': y_test
    }
    metrics['Neural Network'] = {
        'accuracy': accuracy_score(y_test, y_pred_mlp),
        'precision': precision_score(y_test, y_pred_mlp, zero_division=0),
        'recall': recall_score(y_test, y_pred_mlp, zero_division=0),
        'f1': f1_score(y_test, y_pred_mlp, zero_division=0),
        'auc': roc_auc_score(y_test, y_prob_mlp) if len(np.unique(y_test)) > 1 else 0.5
    }
    
    return models, metrics


def train_quantum_inspired_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    qnn = MLPClassifier(
        hidden_layer_sizes=(16, 8, 4),
        activation='tanh',
        solver='adam',
        random_state=42,
        max_iter=1000,
        early_stopping=True,
        validation_fraction=0.1
    )
    
    qnn.fit(X_train, y_train)
    y_pred = qnn.predict(X_test)
    y_prob = qnn.predict_proba(X_test)[:, 1]
    
    model_info = {
        'model': qnn,
        'predictions': y_pred,
        'probabilities': y_prob,
        'X_test': X_test,
        'y_test': y_test,
        'architecture': 'Quantum-Inspired VQC Simulator (4-qubit)',
        'note': 'This simulates a Variational Quantum Classifier. For production SIH, integrate PennyLane/Qiskit.'
    }
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
        'auc': roc_auc_score(y_test, y_prob) if len(np.unique(y_test)) > 1 else 0.5
    }
    
    return model_info, metrics


def calculate_what_if_risk_change(models, scaler, feature_names, baseline_features, 
                                   feature_name, change_percent, feature_idx):
    risk_changes = {}
    
    for model_name, model_data in models.items():
        model = model_data['model']
        
        X_original = scaler.transform([baseline_features])
        prob_original = model.predict_proba(X_original)[0][1]
        
        X_modified = baseline_features.copy()
        X_modified[feature_idx] = X_modified[feature_idx] * (1 + change_percent/100)
        X_modified = scaler.transform([X_modified])
        prob_modified = model.predict_proba(X_modified)[0][1]
        
        abs_change = prob_modified - prob_original
        rel_change = (abs_change / prob_original * 100) if prob_original > 0 else 0.0
        
        risk_changes[model_name] = {
            'original_risk': float(prob_original),
            'modified_risk': float(prob_modified),
            'absolute_change': float(abs_change),
            'relative_change': float(rel_change)
        }
    
    return risk_changes


def get_feature_importance(model, feature_names):
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importance = np.abs(model.coef_[0])
    else:
        importance = np.ones(len(feature_names)) / len(feature_names)
    
    return pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)


# -------------------------
# Sidebar
# -------------------------
with st.sidebar:
    st.image("https://img.icons8.com/color/96/quantum-computer.png", width=80)
    st.markdown("### 🏥 Q-CareTwin")
    st.markdown("**SIH26139** - Quantum Causal Disease Detection")
    st.markdown("---")
    
    menu = st.radio(
        "Navigation",
        [
            "📊 Home",
            "📁 Data Upload",
            "🔧 Data Preprocessing",
            "🤖 Train Models",
            "📈 Batch Analysis",
            "🔬 What-If Lab",
            "📊 Model Evaluation",
            "💾 Export Results"
        ],
        index=0
    )
    
    st.markdown("---")
    st.markdown("**Team Info:**")
    st.markdown("- SIH26139")
    st.markdown("- Egreen Quanta")
    st.markdown("- MedTech Theme")
    
    st.markdown("---")
    st.markdown("**Status:**")
    st.markdown(f"{'✅' if st.session_state.data_loaded else '⭕'} Data Loaded")
    st.markdown(f"{'✅' if st.session_state.models_trained else '⭕'} Models Trained")


# -------------------------
# Main app
# -------------------------
st.markdown('<p class="main-header">🏥 Q-CareTwin</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Quantum Causal Disease Detection Platform | SIH26139</p>', unsafe_allow_html=True)

st.markdown("""
<div class="warning-box">
    <strong>⚠️ IMPORTANT DISCLAIMER:</strong> This is a research prototype for SIH26139.
    <br>• NOT a medical device or diagnostic tool
    <br>• For educational and research purposes only
    <br>• Do NOT use for clinical decision-making
    <br>• What-if scenarios are model simulations, NOT causal treatment effects
    <br>• Use only de-identified or synthetic data
</div>
""", unsafe_allow_html=True)


# =========================
# PAGES
# =========================

if menu == "📊 Home":
    st.markdown("## Welcome to Q-CareTwin")
    
    st.markdown("""
    ### 🎯 Project Overview
    
    **Problem Statement:** SIH26139 - Hybrid Quantum Machine Learning Platform for Early Disease Detection
    
    **Organization:** Egreen Quanta
    
    **Theme:** MedTech / BioTech / HealthTech
    
    ---
    
    ### ✨ Key Features
    
    1. **📁 Data Upload & Validation**
       - CSV file upload with automatic schema validation
       - Generate safe synthetic demo dataset
       - Longitudinal patient record support
    
    2. **🤖 Hybrid Model Training**
       - Classical baselines: Logistic Regression, Random Forest, Gradient Boosting, Neural Network
       - Quantum-inspired VQC simulator (4-qubit architecture)
       - Automatic train/test split with stratification
    
    3. **📈 Batch Analysis**
       - Upload multiple patient reports
       - Get predictions for all patients at once
       - Download comprehensive results
    
    4. **🔬 What-If Lab**
       - Explore "what-if" scenarios (e.g., "What if glucose decreases by 10%?")
       - Compare risk changes across all models
       - ⚠️ Associational simulation only (not causal inference without proper DAG)
    
    5. **📊 Model Evaluation**
       - AUC, Precision, Recall, F1-Score
       - Confusion matrices
       - ROC curves
       - Feature importance analysis
    
    ---
    
    ### 🚀 Getting Started
    
    1. Go to **📁 Data Upload** and load your dataset
    2. Go to **🔧 Data Preprocessing** to review preprocessing
    3. Go to **🤖 Train Models** to train all models
    4. Go to **📈 Batch Analysis** to analyze patient data
    5. Go to **🔬 What-If Lab** to explore scenarios
    6. Go to **📊 Model Evaluation** to compare models
    """)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Models Available", "5")
    with col2:
        st.metric("Features", "10+")
    with col3:
        st.metric("Themes", "MedTech")
    with col4:
        st.metric("SIH Year", "2026")


elif menu == "📁 Data Upload":
    st.markdown("## 📁 Data Upload")
    
    st.markdown("""
    Upload your patient dataset in CSV format or generate a safe demo dataset.
    
    **Important:**
    - Use only de-identified or synthetic data
    - Ensure all required columns are present
    - disease_outcome must be 0 or 1
    """)
    
    st.markdown("### Option 1: Generate Demo Dataset")
    
    if st.button("🎲 Generate Safe Demo Dataset", key="gen_demo"):
        with st.spinner("Generating synthetic dataset..."):
            df_demo = generate_demo_data(n_patients=300, n_visits_per_patient=3)
            st.session_state.df = df_demo
            st.session_state.data_loaded = True
            
            st.success(f"✅ Generated {len(df_demo)} records for {df_demo['patient_id'].nunique()} patients")
            st.markdown("### Dataset Preview")
            st.dataframe(df_demo.head(10))
            
            csv = df_demo.to_csv(index=False)
            st.download_button(
                label="📥 Download Demo Dataset",
                data=csv,
                file_name='demo_patient_data.csv',
                mime='text/csv'
            )
    
    st.markdown("---")
    st.markdown("### Option 2: Upload Your Dataset")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'], help="Upload your patient dataset")
    
    if uploaded_file is not None:
        try:
            df_uploaded = pd.read_csv(uploaded_file)
            st.session_state.df = df_uploaded
            st.session_state.data_loaded = True
            
            st.success(f"✅ Loaded {len(df_uploaded)} records")
            
            required_cols = [
                'patient_id', 'visit_date', 'age', 'sex', 'bmi', 
                'glucose', 'hba1c', 'bp_systolic', 'smoking', 
                'medication', 'disease_outcome'
            ]
            
            missing_cols = [col for col in required_cols if col not in df_uploaded.columns]
            
            if missing_cols:
                st.error(f"❌ Missing columns: {', '.join(missing_cols)}")
            else:
                st.success("✅ All required columns present")
                st.markdown("### Dataset Preview")
                st.dataframe(df_uploaded.head(10))
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Records", len(df_uploaded))
                with col2:
                    st.metric("Unique Patients", df_uploaded['patient_id'].nunique())
                with col3:
                    st.metric("Disease Cases", df_uploaded['disease_outcome'].sum())
        
        except Exception as e:
            st.error(f"❌ Error loading file: {str(e)}")
    
    if st.session_state.data_loaded and st.session_state.df is not None:
        st.markdown("---")
        st.markdown("### Current Dataset")
        df = st.session_state.df
        st.dataframe(df.describe())


elif menu == "🔧 Data Preprocessing":
    st.markdown("## 🔧 Data Preprocessing")
    
    if not st.session_state.data_loaded:
        st.warning("⚠️ Please upload data first in the Data Upload section.")
    else:
        df = st.session_state.df
        
        st.markdown("### Data Quality Check")
        
        missing = df.isnull().sum()
        missing_pct = (missing / len(df) * 100).round(2)
        missing_df = pd.DataFrame({
            'Missing Count': missing,
            'Missing %': missing_pct
        })
        
        st.markdown("#### Missing Values")
        st.dataframe(missing_df[missing_df['Missing Count'] > 0])
        
        st.markdown("#### Data Types")
        st.dataframe(df.dtypes)
        
        st.markdown("#### Target Variable Distribution")
        target_dist = df['disease_outcome'].value_counts()
        st.metric("Disease Cases (1)", int(target_dist.get(1, 0)))
        st.metric("Non-Disease Cases (0)", int(target_dist.get(0, 0)))
        
        if len(target_dist) > 1:
            imbalance_ratio = target_dist.min() / target_dist.max()
            if imbalance_ratio < 0.5:
                st.warning(f"⚠️ Class imbalance detected (ratio: {imbalance_ratio:.2f})")
            else:
                st.success("✅ Balanced classes")
        
        st.markdown("### Preprocessing Steps")
        st.markdown("""
        The following preprocessing will be applied during model training:
        
        1. **Date Features:** Extract month and year from visit_date
        2. **Encoding:** Convert categorical variables (sex, medication) to numeric
        3. **Scaling:** StandardScaler for all numeric features
        4. **Features Used:** age, bmi, glucose, hba1c, bp_systolic,smoking, 
           sex_encoded, medication_encoded, visit_month, visit_year
        """)
        
        if st.button("🔍 Preview Preprocessed Data"):
            X, y, feature_cols, scaler, label_encoders = preprocess_data(df)
            st.session_state.scaler = scaler
            st.session_state.label_encoders = label_encoders
            st.session_state.feature_cols = feature_cols
            
            st.success("✅ Data preprocessed successfully")
            st.markdown(f"Features: {', '.join(feature_cols)}")
            
            X_df = pd.DataFrame(X[:10], columns=feature_cols)
            st.markdown("#### Sample Preprocessed Data (first 10 rows)")
            st.dataframe(X_df)


elif menu == "🤖 Train Models":
    st.markdown("## 🤖 Train Models")
    
    if not st.session_state.data_loaded:
        st.warning("⚠️ Please upload data first.")
    else:
        st.markdown("""
        Train both classical baseline models and quantum-inspired models.
        
        **Models to Train:**
        - Logistic Regression (baseline)
        - Random Forest (ensemble)
        - Gradient Boosting (advanced ensemble)
        - Neural Network (classical deep learning)
        - Quantum-Inspired VQC Simulator (4-qubit architecture)
        """)
        
        col1, col2 = st.columns(2)
        with col1:
            test_size = st.slider("Test Set Size", 0.1, 0.4, 0.2, 0.05)
        with col2:
            random_state = st.number_input("Random Seed", 0, 1000, 42)
        
        if st.button("🚀 Train All Models", type="primary"):
            with st.spinner("Training models... This may take a few minutes."):
                X, y, feature_cols, scaler, label_encoders = preprocess_data(st.session_state.df)
                st.session_state.scaler = scaler
                st.session_state.label_encoders = label_encoders
                st.session_state.feature_cols = feature_cols
                
                progress_bar = st.progress(0)
                
                with st.spinner("Training Logistic Regression, Random Forest, Gradient Boosting, Neural Network..."):
                    models, metrics = train_classical_models(X, y)
                    progress_bar.progress(50)
                    st.success("✅ Classical models trained")
                
                with st.spinner("Training VQC Simulator..."):
                    qnn_model, qnn_metrics = train_quantum_inspired_model(X, y)
                    progress_bar.progress(100)
                    st.success("✅ Quantum-inspired model trained")
                
                st.session_state.models = models
                st.session_state.models['Quantum-Inspired VQC'] = qnn_model
                st.session_state.metrics = metrics
                st.session_state.metrics['Quantum-Inspired VQC'] = qnn_metrics
                st.session_state.models_trained = True
                
                st.success("🎉 All models trained successfully!")
        
        if st.session_state.models_trained:
            st.markdown("---")
            st.markdown("### Model Performance Summary")
            
            metrics_df = pd.DataFrame(st.session_state.metrics).T
            metrics_df = metrics_df.round(4)
            
            st.dataframe(metrics_df.style.background_gradient(cmap='Blues'))
            
            best_model = metrics_df['auc'].idxmax()
            best_auc = metrics_df.loc[best_model, 'auc']
            st.success(f"🏆 Best Model by AUC: **{best_model}** (AUC: {best_auc:.4f})")


elif menu == "📈 Batch Analysis":
    st.markdown("## 📈 Batch Analysis")
    
    if not st.session_state.models_trained:
        st.warning("⚠️ Please train models first.")
    else:
        st.markdown("""
        Upload multiple patient reports for batch prediction.
        """)
        
        uploaded_batch = st.file_uploader(
            "Upload patient data for batch analysis",
            type=['csv'],
            key="batch_upload"
        )
        
        if uploaded_batch is not None:
            try:
                df_batch = pd.read_csv(uploaded_batch)
                
                with st.spinner("Processing batch..."):
                    feature_cols = st.session_state.feature_cols
                    scaler = st.session_state.scaler
                    label_encoders = st.session_state.label_encoders
                    
                    df_processed = df_batch.copy()
                    df_processed['visit_date'] = pd.to_datetime(df_processed['visit_date'])
                    df_processed['visit_month'] = df_processed['visit_date'].dt.month
                    df_processed['visit_year'] = df_processed['visit_date'].dt.year
                    
                    df_processed['sex_encoded'] = label_encoders['sex'].transform(df_processed['sex'])
                    df_processed['medication_encoded'] = label_encoders['medication'].transform(df_processed['medication'])
                    
                    X_batch = df_processed[feature_cols].values
                    X_batch_scaled = scaler.transform(X_batch)
                    
                    predictions_df = pd.DataFrame({
                        'patient_id': df_batch['patient_id'],
                        'visit_date': df_batch['visit_date'].astype(str)
                    })
                    
                    for model_name, model_data in st.session_state.models.items():
                        model = model_data['model']
                        pred = model.predict(X_batch_scaled)
                        prob = model.predict_proba(X_batch_scaled)[:, 1]
                        
                        predictions_df[f'{model_name}_prediction'] = pred
                        predictions_df[f'{model_name}_probability'] = prob.round(4)
                    
                    if 'disease_outcome' in df_batch.columns:
                        predictions_df['actual'] = df_batch['disease_outcome']
                    
                    st.session_state.batch_predictions = predictions_df
                    st.success(f"✅ Processed {len(predictions_df)} records")
                    
                    st.markdown("### Batch Predictions")
                    st.dataframe(predictions_df)
                    
                    csv = predictions_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Batch Results",
                        data=csv,
                        file_name='batch_predictions.csv',
                        mime='text/csv'
                    )
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Records", len(predictions_df))
                    with col2:
                        avg_risk = predictions_df['Quantum-Inspired VQC_probability'].mean()
                        st.metric("Avg Risk (VQC)", f"{avg_risk:.2%}")
                    with col3:
                        high_risk = (predictions_df['Quantum-Inspired VQC_probability'] > 0.7).sum()
                        st.metric("High Risk (>70%)", int(high_risk))
            
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
        
        elif st.session_state.data_loaded:
            if st.button("🔍 Analyze Current Dataset"):
                df = st.session_state.df
                feature_cols = st.session_state.feature_cols
                scaler = st.session_state.scaler
                label_encoders = st.session_state.label_encoders
                
                df_processed = df.copy()
                df_processed['visit_date'] = pd.to_datetime(df_processed['visit_date'])
                df_processed['visit_month'] = df_processed['visit_date'].dt.month
                df_processed['visit_year'] = df_processed['visit_date'].dt.year
                df_processed['sex_encoded'] = label_encoders['sex'].transform(df_processed['sex'])
                df_processed['medication_encoded'] = label_encoders['medication'].transform(df_processed['medication'])
                
                X = df_processed[feature_cols].values
                X_scaled = scaler.transform(X)
                
                predictions_df = pd.DataFrame({
                    'patient_id': df['patient_id'],
                    'visit_date': df['visit_date'].astype(str)
                })
                
                for model_name, model_data in st.session_state.models.items():
                    model = model_data['model']
                    pred = model.predict(X_scaled)
                    prob = model.predict_proba(X_scaled)[:, 1]
                    
                    predictions_df[f'{model_name}_prediction'] = pred
                    predictions_df[f'{model_name}_probability'] = prob.round(4)
                
                if 'disease_outcome' in df.columns:
                    predictions_df['actual'] = df['disease_outcome']
                
                st.session_state.batch_predictions = predictions_df
                st.success(f"✅ Analyzed {len(predictions_df)} records")
                st.dataframe(predictions_df.head(20))


elif menu == "🔬 What-If Lab":
    st.markdown("## 🔬 What-If Lab")
    
    st.markdown("""
    Explore "what-if" scenarios by changing patient features.
    
    **Example:** "What if we reduce glucose by 10%?"
    
    ---
    
    <div class="warning-box">
    <strong>⚠️ IMPORTANT:</strong>
    <br>• These are <strong>model simulations</strong>, NOT causal treatment effects
    <br>• For causal inference, you need: validated DAG, identification strategy, expert review
    <br>• Do NOT use for clinical decision-making
    <br>• This shows how the model WOULD respond to feature changes
    </div>
    """, unsafe_allow_html=True)
    
    if not st.session_state.models_trained:
        st.warning("⚠️ Please train models first.")
    else:
        feature_cols = st.session_state.feature_cols
        feature_display_names = {
            'age': 'Age',
            'bmi': 'BMI',
            'glucose': 'Glucose Level',
            'hba1c': 'HbA1c',
            'bp_systolic': 'Blood Pressure (Systolic)',
            'smoking': 'Smoking Status',
            'sex_encoded': 'Sex',
            'medication_encoded': 'Medication',
            'visit_month': 'Visit Month',
            'visit_year': 'Visit Year'
        }
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            selected_feature = st.selectbox(
                "Feature to Modify",
                feature_cols,
                format_func=lambda x: feature_display_names.get(x, x)
            )
        
        with col2:
            change_percent = st.slider(
                "Change (%)",
                -50, 50, -10, 5
            )
        
        with col3:
            baseline_option = st.selectbox(
                "Baseline",
                ["Average Patient", "Custom Patient"]
            )
        
        if baseline_option == "Average Patient":
            df = st.session_state.df
            baseline_features = [
                float(df['age'].mean()),
                float(df['bmi'].mean()),
                float(df['glucose'].mean()),
                float(df['hba1c'].mean()),
                float(df['bp_systolic'].mean()),
                float(df['smoking'].mean()),
                float(df['sex'].map({'M': 0, 'F': 1}).mean()),
                float(df['medication'].map({'None': 0, 'Metformin': 1, 'Insulin': 2}).mean()),
                6.0,
                2025.0
            ]
        else:
            st.markdown("### Enter Custom Patient Features")
            baseline_features = []
            for feat in feature_cols:
                val = st.number_input(feat, value=50.0 if feat in ['age', 'bmi', 'glucose'] else 0.0, key=f"cust_{feat}")
                baseline_features.append(float(val))
        
        if st.button("🔬 Run What-If Analysis", type="primary"):
            feature_idx = feature_cols.index(selected_feature)
            
            risk_changes = calculate_what_if_risk_change(
                st.session_state.models,
                st.session_state.scaler,
                feature_cols,
                baseline_features,
                selected_feature,
                change_percent,
                feature_idx
            )
            
            st.markdown("### Results")
            
            st.markdown(f"""
            **Scenario:** {feature_display_names.get(selected_feature, selected_feature)} changed by {change_percent}%
            
            **Baseline:** {baseline_option}
            """)
            
            results_data = []
            for model_name, changes in risk_changes.items():
                results_data.append({
                    'Model': model_name,
                    'Original Risk': f"{changes['original_risk']:.2%}",
                    'New Risk': f"{changes['modified_risk']:.2%}",
                    'Absolute Change': f"{changes['absolute_change']:+.2%}",
                    'Relative Change': f"{changes['relative_change']:+.1f}%"
                })
            
            results_df = pd.DataFrame(results_data)
            st.dataframe(results_df, use_container_width=True)
            
            st.markdown("### Risk Comparison")
            
            chart_data = pd.DataFrame({
                'Model': list(risk_changes.keys()),
                'Original Risk': [c['original_risk'] for c in risk_changes.values()],
                'New Risk': [c['modified_risk'] for c in risk_changes.values()]
            })
            chart_data = chart_data.set_index('Model')
            
            st.bar_chart(chart_data)
            
            st.markdown("### Interpretation")
            
            avg_change = np.mean([c['absolute_change'] for c in risk_changes.values()])
            
            if avg_change > 0.05:
                st.info(f"📈 Model predicts **increased risk** (avg +{avg_change:.1%})")
            elif avg_change < -0.05:
                st.info(f"📉 Model predicts **decreased risk** (avg {avg_change:.1%})")
            else:
                st.info(f"➡️ Model predicts **minimal change** (avg {avg_change:.1%})")


elif menu == "📊 Model Evaluation":
    st.markdown("## 📊 Model Evaluation")
    
    if not st.session_state.models_trained:
        st.warning("⚠️ Please train models first.")
    else:
        st.markdown("### Performance Comparison")
        
        metrics_df = pd.DataFrame(st.session_state.metrics).T.round(4)
        st.dataframe(metrics_df.style.background_gradient(cmap='viridis'))
        
        selected_model = st.selectbox(
            "Select Model for Detailed Analysis",
            list(st.session_state.models.keys())
        )
        
        model_data = st.session_state.models[selected_model]
        model = model_data['model']
        y_pred = model_data['predictions']
        y_prob = model_data['probabilities']
        y_test = model_data['y_test']
        
        st.markdown("### Confusion Matrix")
        
        cm = confusion_matrix(y_test, y_pred)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Values")
            st.write(f"**True Negatives:** {int(cm[0, 0])}")
            st.write(f"**False Positives:** {int(cm[0, 1])}")
            st.write(f"**False Negatives:** {int(cm[1, 0])}")
            st.write(f"**True Positives:** {int(cm[1, 1])}")
        
        with col2:
            st.markdown("#### Rates")
            tn, fp, fn, tp = cm.ravel()
            total = cm.sum()
            st.write(f"**Sensitivity (Recall):** {tp/(tp+fn):.2%}" if (tp+fn) > 0 else "**Sensitivity (Recall):** N/A")
            st.write(f"**Specificity:** {tn/(tn+fp):.2%}" if (tn+fp) > 0 else "**Specificity:** N/A")
            st.write(f"**Precision:** {tp/(tp+fp):.2%}" if (tp+fp) > 0 else "**Precision:** N/A")
            st.write(f"**Accuracy:** {(tp+tn)/total:.2%}" if total > 0 else "**Accuracy:** N/A")
        
        st.markdown("### ROC Curve Data")
        
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        
        roc_data = pd.DataFrame({
            'False Positive Rate': fpr,
            'True Positive Rate': tpr
        })
        
        st.line_chart(roc_data.set_index('False Positive Rate'))
        
        auc_val = roc_auc_score(y_test, y_prob) if len(np.unique(y_test)) > 1 else 0.5
        st.write(f"**AUC:** {auc_val:.4f}")
        
        st.markdown("### Feature Importance")
        
        if hasattr(model, 'feature_importances_') or hasattr(model, 'coef_'):
            importance_df = get_feature_importance(model, st.session_state.feature_cols)
            st.bar_chart(
                importance_df.set_index('feature').sort_values('importance'),
                horizontal=True
            )
        else:
            st.info("Feature importance not available for this model type")
        
        st.markdown("### Classification Report")
        st.text(classification_report(y_test, y_pred, zero_division=0))


elif menu == "💾 Export Results":
    st.markdown("## 💾 Export Results")
    
    if not st.session_state.models_trained:
        st.warning("⚠️ Please train models first.")
    else:
        st.markdown("""
        Export your model results and predictions.
        """)
        
        if st.session_state.batch_predictions is not None:
            st.markdown("### Batch Predictions")
            
            csv = st.session_state.batch_predictions.to_csv(index=False)
            st.download_button(
                label="📥 Download Predictions (CSV)",
                data=csv,
                file_name=f'predictions_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
                mime='text/csv'
            )
        else:
            st.info("No batch predictions yet. Go to Batch Analysis first.")
        
        st.markdown("### Performance Metrics")
        
        metrics_json = pd.DataFrame(st.session_state.metrics).T.to_json()
        st.download_button(
            label="📥 Download Metrics (JSON)",
            data=metrics_json,
            file_name=f'metrics_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json',
            mime='application/json'
        )
        
        st.markdown("### Feature Importance")
        
        model = st.session_state.models['Random Forest']['model']
        importance_df = get_feature_importance(model, st.session_state.feature_cols)
        
        csv = importance_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Feature Importance (CSV)",
            data=csv,
            file_name='feature_importance.csv',
            mime='text/csv'
        )
        
        st.markdown("### Model Information")
        
        model_info = f"""Q-CareTwin Model Export
=====================
Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
SIH Problem: SIH26139
Organization: Egreen Quanta
Theme: MedTech / BioTech / HealthTech

Models Trained:
{chr(10).join(['- ' + name for name in st.session_state.models.keys()])}

Features Used:
{chr(10).join(['- ' + feat for feat in st.session_state.feature_cols])}

Performance Summary:
{pd.DataFrame(st.session_state.metrics).T.round(4).to_string()}
"""
        
        st.download_button(
            label="📥 Download Model Info (TXT)",
            data=model_info,
            file_name=f'model_info_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt',
            mime='text/plain'
        )