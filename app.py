import streamlit as st
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModelForSequenceClassification, pipeline
import os

# Page configuration
st.set_page_config(
    page_title="SaluLink Chronic Treatment App",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2c3e50;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .result-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    """Load and cache the ML models"""
    with st.spinner("Loading ClinicalBERT models... This may take a few minutes on first run."):
        # Load NER model
        ner_pipeline = pipeline(
            "ner", 
            model="samrawal/bert-base-uncased_clinical-ner", 
            aggregation_strategy="simple"
        )
        
        # Load classification model
        model_name_classification = "bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12"
        conditions = [
            'Cardiac Failure',
            'Hypertension',
            'Diabetes Insipidus',
            'Diabetes Mellitus Type 1',
            'Diabetes Mellitus Type 2'
        ]
        id2label = {i: condition for i, condition in enumerate(conditions)}
        label2id = {condition: i for i, condition in enumerate(conditions)}
        
        tokenizer_classification = AutoTokenizer.from_pretrained(model_name_classification)
        model_classification = AutoModelForSequenceClassification.from_pretrained(
            model_name_classification,
            num_labels=len(conditions),
            id2label=id2label,
            label2id=label2id
        )
        model_classification.eval()
        
    return ner_pipeline, tokenizer_classification, model_classification, conditions

@st.cache_data
def load_data():
    """Load CSV datasets"""
    try:
        conditions_df = pd.read_csv('Cardiovascular and Endocrine Conditions.csv')
        medicine_df = pd.read_csv('Cardiovascular and Endocrine Medicine.csv')
        treatments_df = pd.read_csv('Cardiovascular and Endocrine Treatments.csv')
        return conditions_df, medicine_df, treatments_df
    except FileNotFoundError as e:
        st.error(f"Error loading CSV files: {e}")
        st.info("Please ensure all CSV files are in the same directory as the app.")
        return None, None, None

def extract_medical_terms(text, ner_pipeline):
    """Extract medical terms from clinical note"""
    try:
        entities = ner_pipeline(text)
        return entities
    except Exception as e:
        st.error(f"Error extracting medical terms: {e}")
        return []

def classify_condition(text, tokenizer, model):
    """Classify patient condition"""
    try:
        inputs = tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True, 
            padding=True,
            max_length=512
        )
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
        
        predicted_label_id = torch.argmax(logits, dim=1).item()
        predicted_condition = model.config.id2label[predicted_label_id]
        
        # Get confidence scores
        probabilities = torch.softmax(logits, dim=1)[0]
        
        return predicted_condition, probabilities
    except Exception as e:
        st.error(f"Error classifying condition: {e}")
        return None, None

def get_treatments(condition, treatments_df):
    """Get treatments for a condition"""
    if treatments_df is None:
        return []
    
    try:
        # Filter by condition (case-insensitive)
        filtered = treatments_df[
            treatments_df['CONDITION'].str.contains(condition, case=False, na=False)
        ]
        
        treatments = []
        # Get diagnostic basket procedures
        if 'DIAGNOSTIC BASKET' in filtered.columns:
            diag_cols = [col for col in filtered.columns if 'DIAGNOSTIC BASKET' in col]
            for col in diag_cols:
                treatments.extend(filtered[col].dropna().unique().tolist())
        
        # Get ongoing management basket procedures
        if 'ONGOING MANAGEMENT BASKET' in str(filtered.columns):
            mgmt_cols = [col for col in filtered.columns if 'ONGOING MANAGEMENT BASKET' in col]
            for col in mgmt_cols:
                treatments.extend(filtered[col].dropna().unique().tolist())
        
        return list(set(treatments)) if treatments else ["No specific treatments found for this condition."]
    except Exception as e:
        st.error(f"Error retrieving treatments: {e}")
        return []

def get_medicines(condition, medicine_df):
    """Get medicines for a condition"""
    if medicine_df is None:
        return []
    
    try:
        # Filter by condition (case-insensitive)
        filtered = medicine_df[
            medicine_df['CHRONIC DISEASE LIST CONDITION'].str.contains(condition, case=False, na=False)
        ]
        
        if filtered.empty:
            return []
        
        # Get relevant columns
        medicines = []
        if 'MEDICINE NAME AND STRENGTH' in filtered.columns:
            medicines = filtered['MEDICINE NAME AND STRENGTH'].dropna().unique().tolist()
        
        return medicines[:20]  # Limit to first 20 for display
    except Exception as e:
        st.error(f"Error retrieving medicines: {e}")
        return []

def get_icd_codes(condition, conditions_df):
    """Get ICD codes for a condition"""
    if conditions_df is None:
        return []
    
    try:
        filtered = conditions_df[
            conditions_df['CHRONIC CONDITIONS'].str.contains(condition, case=False, na=False)
        ]
        
        if filtered.empty:
            return []
        
        icd_data = []
        for _, row in filtered.iterrows():
            icd_code = row.get('ICD-C0DE', '') or row.get('ICD-CODE', '')
            description = row.get('ICD-CODE DESCRIPTION', '')
            if icd_code:
                icd_data.append({
                    'code': icd_code,
                    'description': description
                })
        
        return icd_data
    except Exception as e:
        st.error(f"Error retrieving ICD codes: {e}")
        return []

# Main App
def main():
    st.markdown('<h1 class="main-header">🏥 SaluLink Chronic Treatment App</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Load data
    conditions_df, medicine_df, treatments_df = load_data()
    
    if conditions_df is None or medicine_df is None or treatments_df is None:
        st.stop()
    
    # Load models
    ner_pipeline, tokenizer_classification, model_classification, conditions = load_models()
    
    # Sidebar
    with st.sidebar:
        st.header("📋 Navigation")
        st.markdown("""
        This app analyzes clinical notes to:
        - Extract medical terms and entities
        - Classify chronic conditions
        - Recommend treatments and medicines
        """)
        
        st.markdown("---")
        st.header("📊 Available Conditions")
        for condition in conditions:
            st.write(f"• {condition}")
    
    # Main content area
    tab1, tab2, tab3 = st.tabs(["🔍 Clinical Note Analysis", "📚 Database Explorer", "ℹ️ About"])
    
    with tab1:
        st.markdown('<h2 class="sub-header">Enter Clinical Note</h2>', unsafe_allow_html=True)
        
        # Sample clinical note
        sample_note = """Patient: John Doe
DOB: 01/15/1970
Chief Complaint: Chest pain and shortness of breath.
HPI: Mr. Doe, a 54-year-old male, presents to the emergency department with substernal chest pain that started approximately 3 hours ago. The pain is described as a pressure-like sensation, radiating to his left arm, and is associated with shortness of breath and diaphoresis. He denies fever, cough, or recent trauma. He has a history of hypertension and hyperlipidemia.
Physical Exam: BP 150/90, HR 98, RR 20, SpO2 96% on room air. Lungs clear to auscultation bilaterally. Cardiac exam reveals a regular rate and rhythm, no murmurs. Extremities warm and well-perfused, no edema. No tenderness to palpation of the chest wall.
Assessment: Acute coronary syndrome, differential includes unstable angina or NSTEMI.
Plan: Start aspirin, nitrates, oxygen, and obtain serial ECGs and cardiac enzymes. Consult Cardiology."""
        
        clinical_note = st.text_area(
            "Clinical Note",
            value=sample_note,
            height=300,
            help="Enter or paste the clinical note here"
        )
        
        col1, col2 = st.columns([1, 1])
        with col1:
            analyze_button = st.button("🔍 Analyze Clinical Note", type="primary", use_container_width=True)
        with col2:
            clear_button = st.button("🗑️ Clear", use_container_width=True)
        
        if clear_button:
            st.rerun()
        
        if analyze_button and clinical_note:
            with st.spinner("Analyzing clinical note... This may take a moment."):
                # Extract medical terms
                st.markdown('<h2 class="sub-header">📋 Extracted Medical Terms</h2>', unsafe_allow_html=True)
                extracted_terms = extract_medical_terms(clinical_note, ner_pipeline)
                
                if extracted_terms:
                    # Group terms by entity type
                    problems = [t for t in extracted_terms if t.get('entity_group') == 'problem']
                    tests = [t for t in extracted_terms if t.get('entity_group') == 'test']
                    treatments = [t for t in extracted_terms if t.get('entity_group') == 'treatment']
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown("**🔴 Problems**")
                        for term in problems[:10]:
                            st.write(f"• {term['word']} ({term['score']:.2f})")
                    
                    with col2:
                        st.markdown("**🔵 Tests**")
                        for term in tests[:10]:
                            st.write(f"• {term['word']} ({term['score']:.2f})")
                    
                    with col3:
                        st.markdown("**🟢 Treatments**")
                        for term in treatments[:10]:
                            st.write(f"• {term['word']} ({term['score']:.2f})")
                else:
                    st.info("No medical terms extracted.")
                
                # Classify condition
                st.markdown('<h2 class="sub-header">🏥 Condition Classification</h2>', unsafe_allow_html=True)
                predicted_condition, probabilities = classify_condition(
                    clinical_note, 
                    tokenizer_classification, 
                    model_classification
                )
                
                if predicted_condition:
                    st.success(f"**Predicted Condition: {predicted_condition}**")
                    
                    # Show confidence scores
                    if probabilities is not None:
                        st.markdown("**Confidence Scores:**")
                        conf_data = {
                            'Condition': conditions,
                            'Confidence': [probabilities[i].item() for i in range(len(conditions))]
                        }
                        conf_df = pd.DataFrame(conf_data)
                        conf_df = conf_df.sort_values('Confidence', ascending=False)
                        st.bar_chart(conf_df.set_index('Condition'))
                    
                    # Get recommendations
                    st.markdown('<h2 class="sub-header">💊 Recommendations</h2>', unsafe_allow_html=True)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**🔬 Diagnostic Procedures**")
                        treatments_list = get_treatments(predicted_condition, treatments_df)
                        if treatments_list:
                            for treatment in treatments_list[:10]:
                                if treatment and str(treatment) != 'nan':
                                    st.write(f"• {treatment}")
                        else:
                            st.info("No diagnostic procedures found.")
                    
                    with col2:
                        st.markdown("**💉 Recommended Medicines**")
                        medicines_list = get_medicines(predicted_condition, medicine_df)
                        if medicines_list:
                            for medicine in medicines_list[:10]:
                                if medicine and str(medicine) != 'nan':
                                    st.write(f"• {medicine}")
                        else:
                            st.info("No medicines found.")
                    
                    # ICD Codes
                    st.markdown('<h2 class="sub-header">📑 ICD Codes</h2>', unsafe_allow_html=True)
                    icd_codes = get_icd_codes(predicted_condition, conditions_df)
                    if icd_codes:
                        icd_df = pd.DataFrame(icd_codes)
                        st.dataframe(icd_df, use_container_width=True, hide_index=True)
                    else:
                        st.info("No ICD codes found.")
                else:
                    st.error("Failed to classify condition.")
        
        elif analyze_button:
            st.warning("Please enter a clinical note before analyzing.")
    
    with tab2:
        st.markdown('<h2 class="sub-header">📚 Database Explorer</h2>', unsafe_allow_html=True)
        
        st.markdown("### Conditions Database")
        if conditions_df is not None:
            st.dataframe(conditions_df.head(50), use_container_width=True)
        
        st.markdown("### Medicines Database")
        if medicine_df is not None:
            st.dataframe(medicine_df.head(50), use_container_width=True)
        
        st.markdown("### Treatments Database")
        if treatments_df is not None:
            st.dataframe(treatments_df.head(50), use_container_width=True)
    
    with tab3:
        st.markdown('<h2 class="sub-header">ℹ️ About SaluLink Chronic Treatment App</h2>', unsafe_allow_html=True)
        
        st.markdown("""
        ### Overview
        The SaluLink Chronic Treatment App is an AI-powered clinical decision support system that analyzes 
        clinical notes to identify chronic conditions and recommend appropriate treatments and medications.
        
        ### Features
        - **Medical Term Extraction**: Uses ClinicalBERT NER model to identify medical entities (problems, tests, treatments)
        - **Condition Classification**: Classifies patient conditions into predefined categories using ClinicalBERT
        - **Treatment Recommendations**: Suggests diagnostic procedures and medications based on classified conditions
        - **ICD Code Mapping**: Provides relevant ICD codes for identified conditions
        
        ### Supported Conditions
        - Cardiac Failure
        - Hypertension
        - Diabetes Insipidus
        - Diabetes Mellitus Type 1
        - Diabetes Mellitus Type 2
        
        ### Technology Stack
        - **Streamlit**: Web application framework
        - **Transformers**: Hugging Face transformers library
        - **PyTorch**: Deep learning framework
        - **ClinicalBERT Models**: Pre-trained clinical language models
        
        ### Disclaimer
        This application is for educational and research purposes only. It should not be used as a substitute 
        for professional medical advice, diagnosis, or treatment. Always consult with qualified healthcare providers 
        for medical decisions.
        """)

if __name__ == "__main__":
    main()

