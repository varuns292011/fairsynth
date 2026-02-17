# FairSynth: Healthcare Cost Prediction Fairness Analysis

## Overview
This project evaluates machine learning fairness across age demographics in healthcare cost prediction. It identifies performance disparities between age groups and will assess whether LLM-generated synthetic data preserves or eliminates these gaps.

## Current Results

**Model Performance by Age Group:**
- **Young patients (18-30):** R² = 0.66, Error = $3,857
- **Middle-aged (31-50):** R² = 0.84, Error = $4,111  
- **Older patients (50+):** R² = 0.67, Error = $4,769

**Key Finding:** The model systematically underperforms for older patients, who face the highest prediction errors.

## Dataset
- Medical cost data with 1,338 patients
- Features: age, sex, BMI, smoking status, region
- Target: Insurance charges

## Next Steps
- Generate synthetic data using LLM
- Retrain model on synthetic data
- Compare fairness metrics
- Evaluate privacy-utility tradeoffs

## Tech Stack
- Python
- scikit-learn
- pandas
- LangChain (planned)
- Docker (planned)

## Usage
```bash
python3 fairness_analysis.py
```