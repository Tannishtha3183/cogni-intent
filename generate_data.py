import pandas as pd
import uuid
from datetime import datetime, timedelta

# Set a base time for our simulation (e.g., 6 months ago)
base_time = datetime.now() - timedelta(days=180)

# We are building specific scenarios designed to trigger our MLOps drift detector.
data = [
    # --- PATIENT 1: The Symbicort Contradiction (Our Primary Test Case) ---
    {"patient_id": "P-001", "days_offset": 1, "text": "Patient reports mild asthma symptoms. Prescribed Albuterol inhaler."},
    {"patient_id": "P-001", "days_offset": 15, "text": "Patient states Albuterol is ineffective. Asthma symptoms persisting."},
    {"patient_id": "P-001", "days_offset": 16, "text": "Upgraded prescription to Symbicort for better asthma control."},
    {"patient_id": "P-001", "days_offset": 40, "text": "Patient reports severe heart palpitations after using Symbicort."},
    {"patient_id": "P-001", "days_offset": 41, "text": "Patient is intolerant to Symbicort. Medication discontinued. Prescribed Flovent."},
    
    # --- PATIENT 2: The Contact Info Change (Entity Drift) ---
    {"patient_id": "P-002", "days_offset": 5, "text": "Patient emergency contact listed as wife, Sarah (555-0100)."},
    {"patient_id": "P-002", "days_offset": 60, "text": "Patient updated emergency contact to brother, David (555-0299)."},
    
    # --- PATIENT 3: The Allergy Discovery (Safety Drift) ---
    {"patient_id": "P-003", "days_offset": 2, "text": "Patient has no known drug allergies (NKDA)."},
    {"patient_id": "P-003", "days_offset": 10, "text": "Patient prescribed Amoxicillin for sinus infection."},
    {"patient_id": "P-003", "days_offset": 12, "text": "Patient developed severe hives from Amoxicillin. Added Penicillin allergy to file."}
]

records = []
for row in data:
    event_time = base_time + timedelta(days=row["days_offset"])
    records.append({
        "id": str(uuid.uuid4()),
        "patient_id": row["patient_id"],
        "timestamp": event_time.timestamp(),
        "date_readable": event_time.strftime("%Y-%m-%d"),
        "text": row["text"]
    })

df = pd.DataFrame(records)
output_path = "data/synthetic_logs.csv"
df.to_csv(output_path, index=False)
print(f"Successfully generated {len(df)} synthetic medical records at {output_path}")