import pandas as pd
import random
from faker import Faker
from datetime import datetime, timedelta

fake = Faker()

# Rwanda location data
DISTRICTS = {
    "Gasabo": ["Bumbogo", "Gisozi"],
    "Kicukiro": ["Gahanga", "Masaka"],
    "Nyarugenge": ["Muhima", "Nyarugenge"],
    "Bugesera": ["Nyamata", "Ruhuha"],
    "Gisagara": ["Gikonko", "Muganza"]
}

CELLS = {
    "Bumbogo": ["Cell_A1", "Cell_A2"],
    "Gisozi": ["Cell_B1", "Cell_B2"],
    "Gahanga": ["Cell_C1", "Cell_C2"],
    "Masaka": ["Cell_C3", "Cell_C4"],
    "Muhima": ["Cell_D1", "Cell_D2"],
    "Nyarugenge": ["Cell_D3", "Cell_D4"],
    "Nyamata": ["Cell_E1", "Cell_E2"],
    "Ruhuha": ["Cell_E3", "Cell_E4"],
    "Gikonko": ["Cell_F1", "Cell_F2"],
    "Muganza": ["Cell_F3", "Cell_F4"]
}

DEPARTMENTS = [
    "Health and services", "Water & sanitation", "Education",
    "Public safety", "Social services"
]

STATUSES = ["Pending", "In Progress", "Resolved"]
ESCALATION_TO = ["cell", "sector", "district"]
PRIORITIES = ["Low", "Medium", "High"]

# --- Problem description components ---
issue_types = [
    "Water shortage in the area",
    "Road full of potholes",
    "Broken streetlight",
    "Garbage collection delay",
    "School lacks enough desks",
    "Health post has no medicines",
    "Public toilet not functioning",
    "Blocked drainage causing floods",
    "Illegal dumping of waste",
    "Damaged bridge making crossing difficult",
    "Electricity blackout in the neighborhood",
    "Unfinished construction causing hazards",
    "Open sewage causing bad smell",
    "Poor internet connectivity in public offices",
    "Overgrown bushes blocking footpaths",
    "Public transport shortage during rush hours",
    "Unsafe pedestrian crossing without signs",
    "Noise pollution from nearby bars",
    "Flooding in homes after rain",
    "Lack of waste bins in public areas"
]

time_frames = [
    "for 3 days",
    "for 1 week",
    "for over a month",
    "since last week",
    "since yesterday",
    "for 2 months now",
    "since this morning",
    "for several weeks",
    "for almost a year",
    "since the rainy season began"
]

locations = [
    "near the market",
    "next to the bus stop",
    "at the main road",
    "in the village center",
    "near the health post",
    "by the primary school",
    "behind the local office",
    "near the river",
    "at the football field",
    "in the residential area"
]

def generate_description():
    return f"{random.choice(issue_types)} {random.choice(locations)} {random.choice(time_frames)}."

# --- Data generation ---
records = []
start_date = datetime.now() - timedelta(days=60)

for i in range(1, 501):
    district = random.choice(list(DISTRICTS.keys()))
    sector = random.choice(DISTRICTS[district])
    cell = random.choice(CELLS[sector])
    
    timestamp = start_date + timedelta(days=random.randint(0, 60), hours=random.randint(0,23))
    if random.random() < 0.7:  # 70% chance resolved
        time_solved = timestamp + timedelta(days=random.randint(1,14))
    else:
        time_solved = None
    
    status = random.choice(STATUSES)
    description = generate_description()
    department = random.choice(DEPARTMENTS)
    escalation = random.choice(["yes", "no"])
    escalated_to = random.choice(ESCALATION_TO) if escalation == "yes" else ""
    priority = random.choice(PRIORITIES)
    
    records.append({
        "ID": i,
        "Time_Stamp": timestamp,
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

df = pd.DataFrame(records)
df.to_excel("dummy_issues_data.xlsx", index=False)
