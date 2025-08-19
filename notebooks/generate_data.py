# notebooks/generate_data.py
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import uuid, random

# Sample Rwanda districts, sectors, and cells (for demo purposes)
RWANDA_DISTRICTS = {
    "Gasabo": {
        "Kimironko": ["Kibagabaga", "Bibare", "Nyagatovu"],
        "Remera": ["Rukiri I", "Rukiri II", "Nyabisindu"],
    },
    "Kicukiro": {
        "Kagarama": ["Kigarama", "Nyarurama", "Gatenga"],
        "Nyarugunga": ["Karama", "Kabeza", "Busanza"],
    },
    "Nyarugenge": {
        "Nyamirambo": ["Rugarama", "Kimisagara", "Mumena"],
        "Kigali": ["Biryogo", "Agatare", "Kiyovu"],
    },
    "Musanze": {
        "Muhoza": ["Kabeza", "Cyabagarura", "Ruhengeri"],
        "Cyuve": ["Gashangiro", "Kabeza", "Kirwa"],
    },
    "Huye": {
        "Ngoma": ["Tumba", "Rukira", "Ngoma"],
        "Tumba": ["Butare", "Ruhashya", "Mbazi"],
    }
}

DEPARTMENTS = [
    "Health and services", "water&sanitation", "Education", "public safety", "Social services"
]
STATUSES = ["Pending", "In Progress", "Resolved"]
ESCALATIONS = ["yes", "no"]
ESCALATED_TO = ["cell", "sector", "district"]
PRIORITIES = ["Low", "Medium", "High"]

DESCRIPTIONS = [
    "Water leakage reported near the main road.",
    "Broken streetlight causing darkness at night.",
    "Uncollected trash piling up in the neighborhood.",
    "Noise disturbance from nearby construction.",
    "Blocked drainage causing flooding during rain.",
    "Request for more security patrols in the area.",
    "Damaged public bench in the park.",
    "Overflowing sewage detected.",
    "Request for additional classroom furniture.",
    "Health hazard due to open waste.",
]

def random_rwanda_location():
    district = random.choice(list(RWANDA_DISTRICTS.keys()))
    sector = random.choice(list(RWANDA_DISTRICTS[district].keys()))
    cell = random.choice(RWANDA_DISTRICTS[district][sector])
    return district, sector, cell

def make_fake_issues(n=500, seed=42):
    random.seed(seed)
    np.random.seed(seed)
    rows = []
    for _ in range(n):
        time_stamp = datetime.now() - timedelta(days=random.randint(0, 90), hours=random.randint(0, 23), minutes=random.randint(0, 59))
        status = random.choices(STATUSES, weights=[0.2, 0.5, 0.3])[0]
        time_solved = (time_stamp + timedelta(days=random.randint(1, 20), hours=random.randint(0, 23), minutes=random.randint(0, 59))
                       if status == "Resolved" else pd.NaT)
        district, sector, cell = random_rwanda_location()
        description = random.choice(DESCRIPTIONS)
        department = random.choice(DEPARTMENTS)
        escalation = random.choice(ESCALATIONS)
        escalated_to = random.choice(ESCALATED_TO) if escalation == "yes" else ""
        priority = random.choice(PRIORITIES)
        rows.append({
            "Time_Stamp": time_stamp,
            "Time_Solved": time_solved,
            "District": district,
            "Sector": sector,
            "Cell": cell,
            "Status": status,
            "Description": description,
            "Department": department,
            "Escalation": escalation,
            "Escalated_to": escalated_to,
            "Priority": priority
        })
    return pd.DataFrame(rows)

if __name__ == "__main__":
    out_dir = Path(__file__).resolve().parents[1] / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    df = make_fake_issues(500)
    df.to_csv(out_dir / "issues_sample.csv", index=False)
    print(f"Saved {len(df)} rows to {out_dir / 'issues_sample.csv'}")
