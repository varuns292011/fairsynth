import os
import json
import time
import pandas as pd
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate

def generate_synthetic_langchain(csv_path, num_rows=100):
    # Load dataset
    data = pd.read_csv(csv_path)
    columns = list(data.columns)
    sample = data.head(5).to_string()
    
    # Setup LangChain with Groq
    llm = ChatGroq(
        api_key=os.environ.get("GROQ_API_KEY"),
        model="llama-3.3-70b-versatile",
        max_tokens=2000
    )
    
    # Dynamic prompt that works for ANY dataset
    template = """Here are 5 rows from a dataset:

{sample}

Generate exactly 20 realistic rows as a valid JSON array.
Use these exact column names: {columns}
Match the data types and value ranges from the examples.
Return ONLY the JSON array, nothing else."""

    prompt = PromptTemplate(
        input_variables=["sample", "columns"],
        template=template
    )
    
    # Generate in batches
    all_data = []
    batches = num_rows // 20
    
    for i in range(batches):
        print(f"Generating batch {i+1} of {batches}...")
        chain = prompt | llm
        response = chain.invoke({
            "sample": sample,
            "columns": str(columns)
        })
        
        raw = response.content.strip()
        if "```" in raw:
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        
        batch = json.loads(raw)
        all_data.extend(batch)
        time.sleep(1)
    
    df = pd.DataFrame(all_data)
    output_path = csv_path.replace(".csv", "_synthetic.csv")
    df.to_csv(output_path, index=False)
    print(f"\nDone! Generated {len(df)} synthetic rows")
    print(f"Saved to: {output_path}")
    return output_path

# Run it
generate_synthetic_langchain("insurance.csv")