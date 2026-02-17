import groq
import pandas as pd
import json
import os
import time

client = groq.Groq(api_key=os.environ.get("GROQ_API_KEY"))

real_data = pd.read_csv("insurance.csv")
sample = real_data.head(5).to_string()

def generate_batch(n):
    prompt = f"""Here are 5 rows from a healthcare insurance dataset:

{sample}

Generate exactly {n} new realistic rows as a valid JSON array with these exact keys: age, sex, bmi, children, smoker, region, charges.

Rules:
- age: integer between 18-64
- sex: "male" or "female"
- bmi: float between 15-50
- children: integer between 0-5
- smoker: "yes" or "no"
- region: "southwest", "southeast", "northwest", or "northeast"
- charges: realistic float

Return ONLY the JSON array. No explanation. No extra text."""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        max_tokens=2000,
        messages=[{"role": "user", "content": prompt}]
    )
    raw = response.choices[0].message.content
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw)

all_data = []
for i in range(5):
    print(f"Generating batch {i+1}...")
    batch = generate_batch(20)
    all_data.extend(batch)
    time.sleep(1)

df = pd.DataFrame(all_data)
df.to_csv("synthetic_data.csv", index=False)
print(f"Done! Generated {len(df)} synthetic rows")
print(df.head())