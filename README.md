# SaluLink Chronic Treatment App

An AI-powered clinical decision support system that analyzes clinical notes to identify chronic conditions and recommend appropriate treatments and medications.

## Features

- **Medical Term Extraction**: Uses ClinicalBERT NER model to identify medical entities (problems, tests, treatments)
- **Condition Classification**: Classifies patient conditions into predefined categories using ClinicalBERT
- **Treatment Recommendations**: Suggests diagnostic procedures and medications based on classified conditions
- **ICD Code Mapping**: Provides relevant ICD codes for identified conditions

## Supported Conditions

- Cardiac Failure
- Hypertension
- Diabetes Insipidus
- Diabetes Mellitus Type 1
- Diabetes Mellitus Type 2

## Installation

1. Clone or download this repository
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

3. Ensure all CSV files are in the same directory as `app.py`:
   - `Cardiovascular and Endocrine Conditions.csv`
   - `Cardiovascular and Endocrine Medicine.csv`
   - `Cardiovascular and Endocrine Treatments.csv`

## Usage

Run the Streamlit app:

```bash
streamlit run app.py
```

The app will open in your default web browser. On first run, the models will be downloaded automatically (this may take several minutes).

## How to Use

1. **Clinical Note Analysis Tab**:
   - Enter or paste a clinical note in the text area
   - Click "Analyze Clinical Note" to process
   - View extracted medical terms, classified condition, and recommendations

2. **Database Explorer Tab**:
   - Browse the conditions, medicines, and treatments databases

3. **About Tab**:
   - Learn more about the app and its features

## Technology Stack

- **Streamlit**: Web application framework
- **Transformers**: Hugging Face transformers library
- **PyTorch**: Deep learning framework
- **ClinicalBERT Models**: Pre-trained clinical language models
  - `samrawal/bert-base-uncased_clinical-ner` for NER
  - `bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12` for classification

## Important Notes

- **First Run**: The app will download ML models on first run (several GB), which may take 10-15 minutes depending on your internet connection
- **Performance**: Model inference may take 30-60 seconds per analysis
- **Hardware**: GPU is not required but will speed up processing if available

## Disclaimer

This application is for educational and research purposes only. It should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult with qualified healthcare providers for medical decisions.

## File Structure

```
.
├── app.py                                    # Main Streamlit application
├── requirements.txt                          # Python dependencies
├── README.md                                 # This file
├── Cardiovascular and Endocrine Conditions.csv
├── Cardiovascular and Endocrine Medicine.csv
└── Cardiovascular and Endocrine Treatments.csv
```

## Troubleshooting

- **Model Download Issues**: If models fail to download, ensure you have a stable internet connection and sufficient disk space (~2GB)
- **CSV File Errors**: Ensure all CSV files are in the same directory as `app.py`
- **Memory Issues**: If you encounter memory errors, try closing other applications or using a machine with more RAM

