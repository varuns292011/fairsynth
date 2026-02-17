FROM python:3.8-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["python3", "fairness_analysis.py"]
```

Now create `requirements.txt` and paste this:
```
pandas
scikit-learn
groq
langchain
langchain-groq
joblib