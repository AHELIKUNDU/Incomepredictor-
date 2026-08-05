import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Salary Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional minimal design
st.markdown("""
    <style>
    /* Reset default spacing */
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
        max-width: 1200px;
    }
    
    /* Professional header */
    .app-header {
        background: #ffffff;
        padding: 1.2rem 2rem;
        border-bottom: 2px solid #f0f2f5;
        margin-bottom: 1.5rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    
    .app-header h1 {
        margin: 0;
        font-size: 1.6rem;
        font-weight: 600;
        color: #1a1a2e;
        letter-spacing: -0.5px;
    }
    
    .app-header .subtitle {
        font-size: 0.85rem;
        color: #6c757d;
        font-weight: 400;
        margin-top: 0.1rem;
    }
    
    .app-header .badge {
        background: #f8f9fa;
        padding: 0.3rem 1rem;
        border-radius: 20px;
        font-size: 0.75rem;
        color: #495057;
        border: 1px solid #e9ecef;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: #1a1a2e;
        border-right: 1px solid #2d2d44;
    }
    
    /* Change sidebar text colors to white */
    .css-1d391kg .stSelectbox label,
    .css-1d391kg .stNumberInput label,
    .css-1d391kg .stSlider label,
    .css-1d391kg .stMarkdown,
    .css-1d391kg p,
    .css-1d391kg h1,
    .css-1d391kg h2,
    .css-1d391kg h3,
    .css-1d391kg h4,
    .css-1d391kg h5,
    .css-1d391kg h6,
    .css-1d391kg span {
        color: #FFFFFF !important;
    }
    
    /* Sidebar input text color */
    .css-1d391kg .stNumberInput input,
    .css-1d391kg .stSelectbox select,
    .css-1d391kg .stSelectbox div,
    .css-1d391kg input {
        color: #FFFFFF !important;
    }
    
    /* Sidebar header */
    .sidebar-header {
        padding: 1rem 0 0.5rem 0;
        border-bottom: 1px solid rgba(255,255,255,0.1);
        margin-bottom: 1rem;
    }
    
    .sidebar-header h3 {
        font-size: 1rem;
        font-weight: 600;
        color: #FFFFFF !important;
        margin: 0;
        letter-spacing: 0.3px;
    }
    
    .sidebar-header p {
        font-size: 0.75rem;
        color: rgba(255,255,255,0.7) !important;
        margin: 0.2rem 0 0 0;
    }
    
    /* Sidebar input labels - WHITE */
    .css-1d391kg label {
        color: #FFFFFF !important;
        font-weight: 500;
    }
    
    /* Sidebar input values - WHITE */
    .css-1d391kg .stNumberInput input,
    .css-1d391kg .stSelectbox div,
    .css-1d391kg .stSelectbox select {
        color: #FFFFFF !important;
        background: rgba(255,255,255,0.05);
        border-color: rgba(255,255,255,0.2);
    }
    
    /* Sidebar divider */
    .css-1d391kg hr {
        border-color: rgba(255,255,255,0.1);
    }
    
    /* Card styling - minimal */
    .card {
        background: #ffffff;
        padding: 1.5rem;
        border-radius: 8px;
        border: 1px solid #e9ecef;
        margin-bottom: 1rem;
        transition: box-shadow 0.2s ease;
    }
    
    .card:hover {
        box-shadow: 0 2px 12px rgba(0,0,0,0.04);
    }
    
    .card-title {
        font-size: 0.85rem;
        font-weight: 600;
        color: #495057;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 1.2rem;
        padding-bottom: 0.6rem;
        border-bottom: 1px solid #f0f2f5;
    }
    
    /* Result display - clean */
    .result-primary {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 8px;
        text-align: center;
        border: 1px solid #e9ecef;
        margin-bottom: 1rem;
    }
    
    .result-primary .label {
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        color: #6c757d;
        font-weight: 500;
    }
    
    .result-primary .amount {
        font-size: 2.8rem;
        font-weight: 700;
        color: #1a1a2e;
        margin: 0.2rem 0;
        letter-spacing: -1px;
    }
    
    .result-primary .currency {
        font-size: 1.2rem;
        color: #6c757d;
        font-weight: 400;
    }
    
    /* Detail items */
    .detail-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 0.8rem;
        margin-top: 0.5rem;
    }
    
    .detail-item {
        background: #f8f9fa;
        padding: 0.8rem 1rem;
        border-radius: 6px;
        border: 1px solid #f0f2f5;
    }
    
    .detail-item .label {
        font-size: 0.65rem;
        text-transform: uppercase;
        letter-spacing: 0.3px;
        color: #6c757d;
        font-weight: 500;
    }
    
    .detail-item .value {
        font-size: 0.95rem;
        font-weight: 600;
        color: #1a1a2e;
        margin-top: 0.1rem;
    }
    
    /* Badges */
    .badge-exp {
        display: inline-block;
        padding: 0.2rem 0.8rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    
    .badge-exp.entry { background: #e8f5e9; color: #2e7d32; }
    .badge-exp.junior { background: #e3f2fd; color: #0d47a1; }
    .badge-exp.mid { background: #fff3e0; color: #e65100; }
    .badge-exp.senior { background: #f3e5f5; color: #4a148c; }
    .badge-exp.executive { background: #fce4ec; color: #b71c1c; }
    
    .badge-hr {
        display: inline-block;
        padding: 0.2rem 0.8rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    
    .badge-hr.premium { background: #e8f5e9; color: #2e7d32; }
    .badge-hr.standard { background: #fff3e0; color: #e65100; }
    .badge-hr.review { background: #fce4ec; color: #b71c1c; }
    
    /* Button */
    .stButton > button {
        width: 100%;
        background: #1a1a2e;
        color: white;
        font-weight: 500;
        border-radius: 6px;
        padding: 0.6rem;
        font-size: 0.9rem;
        border: none;
        transition: all 0.2s;
        letter-spacing: 0.3px;
    }
    
    .stButton > button:hover {
        background: #2d2d44;
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(26,26,46,0.15);
    }
    
    /* Form inputs - professional */
    .stSelectbox label, .stNumberInput label, .stSlider label {
        font-size: 0.75rem;
        font-weight: 500;
        color: #FFFFFF !important;
        text-transform: uppercase;
        letter-spacing: 0.3px;
    }
    
    .stSelectbox, .stNumberInput, .stSlider {
        margin-bottom: 0.3rem;
    }
    
    /* Sidebar form spacing */
    .sidebar-input {
        margin-bottom: 0.8rem;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 1rem 0 0.3rem 0;
        color: #adb5bd;
        font-size: 0.7rem;
        border-top: 1px solid #f0f2f5;
        margin-top: 1.5rem;
        letter-spacing: 0.3px;
    }
    
    .footer span {
        margin: 0 0.3rem;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Responsive */
    @media (max-width: 768px) {
        .detail-grid {
            grid-template-columns: 1fr;
        }
    }
    </style>
""", unsafe_allow_html=True)

# Load model function
@st.cache_resource
def load_artifacts():
    try:
        model = joblib.load('model.pkl')
        scaler = joblib.load('scaler.pkl')
        feature_columns = joblib.load('feature_columns.pkl')
        
        encoders = {}
        encoder_files = ['gender', 'education', 'job']
        for key in encoder_files:
            try:
                encoders[key] = joblib.load(f'le_{key}.pkl')
            except:
                encoders[key] = None
        
        try:
            df = pd.read_csv('Salary_Data.csv')
        except:
            df = pd.read_csv('Salary Data.csv')
        
        return model, scaler, feature_columns, encoders, df
    except Exception as e:
        return None, None, None, None, None

def get_experience_level(experience):
    if experience < 2:
        return "Entry Level", "entry"
    elif experience < 5:
        return "Junior", "junior"
    elif experience < 10:
        return "Mid-Level", "mid"
    elif experience < 20:
        return "Senior", "senior"
    else:
        return "Executive", "executive"

def get_hr_decision(salary, avg_salary):
    if salary > avg_salary * 1.3:
        return "Premium Package", "premium", "★ ★ ★ ★ ★"
    elif salary > avg_salary * 1.05:
        return "Standard Package", "standard", "★ ★ ★"
    else:
        return "Review Required", "review", "★ ★"

def main():
    # Header with dark background and white text
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("""
            <div style="padding: 0.5rem 0;">
                <h1 style="font-size: 1.6rem; font-weight: 600; color: #FFFFFF; margin: 0; letter-spacing: -0.5px;">
                    💼 Salary Predictor
                </h1>
                <div style="font-size: 0.85rem; color: rgba(255,255,255,0.7); margin-top: 0.1rem;">
                    Employee salary estimation using machine learning
                </div>
            </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
            <div style="text-align: right; padding: 0.5rem 0;">
                <span style="background: rgba(255,255,255,0.15); padding: 0.3rem 1rem; border-radius: 20px; font-size: 0.7rem; color: #FFFFFF; border: 1px solid rgba(255,255,255,0.3);">
                    v1.0
                </span>
            </div>
        """, unsafe_allow_html=True)

    # Load artifacts
    model, scaler, feature_columns, encoders, df = load_artifacts()
    
    if model is None:
        st.error("⚠️ Model files not found. Please run trainmodel.py first.")
        return
    
    # Sidebar - Input Section with WHITE text
    with st.sidebar:
        st.markdown("""
            <div class="sidebar-header">
                <h3>👤 Candidate Profile</h3>
                <p style="color: rgba(255,255,255,0.7) !important;">Enter the details below</p>
            </div>
        """, unsafe_allow_html=True)
        
        # Input fields in sidebar - Labels will be white via CSS
        age = st.number_input(
            "Age",
            min_value=18,
            max_value=65,
            value=30,
            step=1,
            help="Candidate's age in years"
        )
        
        gender = st.selectbox(
            "Gender",
            options=['Male', 'Female', 'Other'],
            index=0,
            help="Select gender"
        )
        
        education = st.selectbox(
            "Education Level",
            options=["Bachelor's", "Master's", "PhD", "Associate", "High School"],
            index=0,
            help="Highest education level achieved"
        )
        
        experience = st.number_input(
            "Years of Experience",
            min_value=0.0,
            max_value=50.0,
            value=2.0,
            step=0.5,
            help="Total years of professional experience"
        )
        
        job_title = st.selectbox(
            "Job Title",
            options=[
                'Account Manager', 'Software Engineer', 'Data Scientist', 
                'Manager', 'Analyst', 'Developer', 'Designer', 
                'Marketing Specialist', 'Sales Representative', 
                'HR Manager', 'Financial Analyst'
            ],
            index=0,
            help="Current job title"
        )
        
        st.markdown("---")
        
        predict_clicked = st.button("🔮 Predict Salary", use_container_width=True)
    
    # Main content area
    if predict_clicked:
        # Prepare input
        input_data = {
            'YearsExperience': [experience],
            'Age': [age]
        }
        
        if 'Gender' in feature_columns:
            input_data['Gender'] = [gender]
        if 'Education' in feature_columns:
            input_data['Education'] = [education]
        if 'JobTitle' in feature_columns:
            input_data['JobTitle'] = [job_title]
        
        input_df = pd.DataFrame(input_data)
        
        # Encode categorical
        for col in ['Gender', 'Education', 'JobTitle']:
            if col in input_df.columns and encoders.get(col.lower()) is not None:
                try:
                    input_df[col] = encoders[col.lower()].transform(input_df[col])
                except:
                    pass
        
        # Ensure all features present
        for col in feature_columns:
            if col not in input_df.columns:
                input_df[col] = 0
        
        input_df = input_df[feature_columns]
        input_scaled = scaler.transform(input_df)
        predicted_salary = model.predict(input_scaled)[0]
        
        # Get experience level
        exp_level, exp_class = get_experience_level(experience)
        
        # Get HR decision
        if df is not None and 'Salary' in df.columns:
            avg_salary = df['Salary'].mean()
            hr_text, hr_class, hr_stars = get_hr_decision(predicted_salary, avg_salary)
        else:
            hr_text, hr_class, hr_stars = "Standard Package", "standard", "★ ★ ★"
        
        # Display Results
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # Salary Display
            st.markdown(f"""
                <div class="result-primary">
                    <div class="label">Predicted Annual Salary</div>
                    <div class="amount"><span class="currency">$</span>{predicted_salary:,.2f}</div>
                </div>
            """, unsafe_allow_html=True)
        
        with col2:
            # Quick summary
            st.markdown(f"""
                <div style="background: #f8f9fa; padding: 1.5rem; border-radius: 8px; border: 1px solid #e9ecef; height: 100%; display: flex; flex-direction: column; justify-content: center;">
                    <div style="display: flex; justify-content: space-between; padding: 0.3rem 0;">
                        <span style="font-size: 0.75rem; color: #6c757d; text-transform: uppercase; letter-spacing: 0.3px;">Experience</span>
                        <span class="badge-exp {exp_class}">{exp_level}</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; padding: 0.3rem 0; border-top: 1px solid #e9ecef; margin-top: 0.3rem; padding-top: 0.5rem;">
                        <span style="font-size: 0.75rem; color: #6c757d; text-transform: uppercase; letter-spacing: 0.3px;">HR Decision</span>
                        <span class="badge-hr {hr_class}">{hr_text}</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; padding: 0.3rem 0; border-top: 1px solid #e9ecef; margin-top: 0.3rem; padding-top: 0.5rem;">
                        <span style="font-size: 0.75rem; color: #6c757d; text-transform: uppercase; letter-spacing: 0.3px;">Recommendation</span>
                        <span style="font-size: 0.9rem; color: #f59f00;">{hr_stars}</span>
                    </div>
                </div>
            """, unsafe_allow_html=True)
        
        # Detailed breakdown
        st.markdown("""
            <div style="margin-top: 1rem;">
                <div class="card-title">📋 Salary Breakdown</div>
            </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
                <div class="detail-item">
                    <div class="label">Predicted Salary</div>
                    <div class="value">${predicted_salary:,.0f}</div>
                </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
                <div class="detail-item">
                    <div class="label">Experience Category</div>
                    <div class="value">{exp_level}</div>
                </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
                <div class="detail-item">
                    <div class="label">HR Decision</div>
                    <div class="value">{hr_text}</div>
                </div>
            """, unsafe_allow_html=True)
        
        with col4:
            if df is not None and 'Salary' in df.columns:
                percentile = (df['Salary'] < predicted_salary).mean() * 100
                st.markdown(f"""
                    <div class="detail-item">
                        <div class="label">Market Percentile</div>
                        <div class="value">Top {100 - percentile:.0f}%</div>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                    <div class="detail-item">
                        <div class="label">Status</div>
                        <div class="value">Active</div>
                    </div>
                """, unsafe_allow_html=True)
        
        # Performance indicator
        if df is not None and 'Salary' in df.columns:
            avg_salary = df['Salary'].mean()
            diff_percent = ((predicted_salary - avg_salary) / avg_salary) * 100
            
            st.markdown("""
                <div style="margin-top: 1rem; background: #f8f9fa; padding: 1rem; border-radius: 8px; border: 1px solid #e9ecef;">
            """, unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns([1, 2, 1])
            
            with col2:
                if diff_percent > 20:
                    color = "#2e7d32"
                    icon = "📈"
                    text = "Above Market Average"
                elif diff_percent > -20:
                    color = "#e65100"
                    icon = "📊"
                    text = "Market Competitive"
                else:
                    color = "#b71c1c"
                    icon = "📉"
                    text = "Below Market Average"
                
                st.markdown(f"""
                    <div style="text-align: center; padding: 0.5rem;">
                        <span style="font-size: 1.5rem;">{icon}</span>
                        <div style="font-size: 0.9rem; font-weight: 500; color: {color}; margin-top: 0.2rem;">
                            {text}
                        </div>
                        <div style="font-size: 0.75rem; color: #6c757d; margin-top: 0.1rem;">
                            {diff_percent:+.1f}% vs market average
                        </div>
                    </div>
                """, unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
    
    else:
        # Welcome state - "Ready to Predict" in WHITE
        st.markdown("""
            <div style="text-align: center; padding: 3rem 1rem;">
                <div style="font-size: 4rem; margin-bottom: 1rem;">💼</div>
                <h2 style="font-weight: 400; color: #FFFFFF; margin-bottom: 0.3rem;">
                    Ready to Predict
                </h2>
                <p style="color: rgba(255,255,255,0.8); font-size: 0.95rem; max-width: 400px; margin: 0 auto;">
                    Enter candidate details in the sidebar and click 
                    <strong style="color: #FFFFFF;">"Predict Salary"</strong> to get started
                </p>
                <div style="margin-top: 1.5rem; display: flex; gap: 1rem; justify-content: center; flex-wrap: wrap;">
                    <span style="background: rgba(255,255,255,0.15); padding: 0.3rem 1rem; border-radius: 20px; font-size: 0.75rem; color: #FFFFFF; border: 1px solid rgba(255,255,255,0.3);">Age</span>
                    <span style="background: rgba(255,255,255,0.15); padding: 0.3rem 1rem; border-radius: 20px; font-size: 0.75rem; color: #FFFFFF; border: 1px solid rgba(255,255,255,0.3);">Gender</span>
                    <span style="background: rgba(255,255,255,0.15); padding: 0.3rem 1rem; border-radius: 20px; font-size: 0.75rem; color: #FFFFFF; border: 1px solid rgba(255,255,255,0.3);">Education</span>
                    <span style="background: rgba(255,255,255,0.15); padding: 0.3rem 1rem; border-radius: 20px; font-size: 0.75rem; color: #FFFFFF; border: 1px solid rgba(255,255,255,0.3);">Experience</span>
                    <span style="background: rgba(255,255,255,0.15); padding: 0.3rem 1rem; border-radius: 20px; font-size: 0.75rem; color: #FFFFFF; border: 1px solid rgba(255,255,255,0.3);">Job Title</span>
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("""
        <div class="footer">
            <div style="display: flex; justify-content: center; align-items: center; gap: 1.5rem; flex-wrap: wrap;">
                <span style="color: #6c757d;">© 2026 Salary Predictor</span>
                <span style="color: #dee2e6;">|</span>
                <span style="color: #6c757d;">Version 1.0</span>
                <span style="color: #dee2e6;">|</span>
                <span style="color: #6c757d;">Built with</span>
                <span style="color: #1a1a2e; font-weight: 500;">Streamlit</span>
                <span style="color: #6c757d;">&</span>
                <span style="color: #1a1a2e; font-weight: 500;">Scikit-Learn</span>
                <span style="color: #dee2e6;">|</span>
                <span style="color: #6c757d;">
                    <span style="color: #4caf50;">●</span> Online
                </span>
            </div>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()