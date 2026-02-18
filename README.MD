\# FairSynth
Evaluating fairness and utility preservation in LLM-generated synthetic healthcare data.

## What This Project Does
Detects inequality in healthcare AI predictions across age groups, generates synthetic data using LLMs, and tests whether that synthetic data is trustworthy for research.

## Key Findings
- Real data model R²: 0.78 overall
- Older patients have worst predictions (R²=0.67)
- Synthetic data model R²: 0.54 (significantly worse)
- Discriminator detects synthetic data with 95% accuracy

## Conclusion
LLM-generated synthetic data is not reliable enough for privacy-preserving healthcare ML research.

## Pipeline
Real Data → Fairness Analysis → Generate Synthetic → Compare Results → Discriminator Test

## Tech Stack
Python, scikit-learn, pandas, LangChain, Groq LLM, Git

## How To Run
```bash
python3 pipeline.py
```
