import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# ---- Parameters ----
num_records = 500
random.seed(42)
np.random.seed(42)

# ---- Sample location hierarchy ----
districts = {
    "Gasabo": {
        "Remera": ["Nyabisindu", "Rukiri I", "Rukiri II"],
        "Kimironko": ["Bibare", "Nyagatovu", "Kibagabaga"]
    },
    "Kicukiro": {
        "Kigarama": ["Kigarama I", "Kigarama II", "Kigarama III"],
        "Gikondo": ["Kanserege", "Rwimbogo", "Gikondo"]
    },
    "Nyarugenge": {
        "Nyamirambo": ["Mumena", "Rwezamenyo I", "Rwezamenyo II"],
        "Kigali": ["Kiyovu", "Nyarugenge", "Muhima"]
    }
}

categories = ["Water Supply", "Road Maintenance", "Electricity", "Waste Management", "Health Services"]

departments = ["Health and Services ", "Education", "Water & sanitation", "public Safety", "Social services"]

levels = ["Cell", "Sector", "District"]

statuses = ["Open", "In Progress", "Resolved", "Escalated"]

feedback_comments = [
    "Service was quick and effective.",
    "Took longer than expected.",
    "Very satisfied with the outcome.",
    "The issue is still not fully fixed.",
    "Good communication throughout.",
    "Poor handling, not happy."
]

# ---- Generate dummy records ----
data = []

start_date = datetime(2025, 1, 1)
today = datetime(2025, 8, 12)

for i in range(num_records):
    district = random.choice(list(districts.keys()))
    sector = random.choice(list(districts[district].keys()))
    cell = random.choice(districts[district][sector])
    
    category = random.choice(categories)
    dept = random.choice(departments)
    assigned_level = random.choice(levels)
    
    date_reported = start_date + timedelta(days=random.randint(0, 220))
    
    # Simulate resolution date or missing if still open
    if random.random() > 0.2:  # 80% resolved
        resolution_days = random.randint(1, 30)
        date_resolved = date_reported + timedelta(days=resolution_days)
        status = "Resolved"
    else:
        date_resolved = pd.NaT
        status = random.choice(["Open", "In Progress", "Escalated"])
    
    # Escalation logic
    escalated = "Yes" if status == "Escalated" else "No"
    
    # Overdue logic: threshold = 14 days
    if pd.isna(date_resolved):
        days_open = (today - date_reported).days
        is_overdue = "Yes" if days_open > 14 else "No"
    else:
        days_open = (date_resolved - date_reported).days
        is_overdue = "Yes" if days_open > 14 else "No"
    
    # Feedback only for resolved issues
    if status == "Resolved":
        # Better ratings for faster fixes
        if days_open <= 7:
            rating = random.randint(4, 5)
        elif days_open <= 14:
            rating = random.randint(3, 5)
        else:
            rating = random.randint(1, 4)
        comment = random.choice(feedback_comments)
    else:
        rating = None
        comment = None
    
    data.append([
        f"ISS-{1000+i}",
        category,
        district,
        sector,
        cell,
        date_reported.date(),
        date_resolved.date() if pd.notna(date_resolved) else None,
        dept,
        assigned_level,
        status,
        escalated,
        is_overdue,
        rating,
        comment
    ])

# ---- Create DataFrame ----
columns = [
    "issue_id", "issue_category", "district", "sector", "cell",
    "date_reported", "date_resolved", "assigned_department",
    "assigned_level", "status", "escalated", "is_overdue",
    "feedback_rating", "feedback_comment"
]

df = pd.DataFrame(data, columns=columns)

# ---- Preview ----
print(df.head())

# ---- Save to CSV ----
df.to_csv("data_with_feedback.csv", index=False)
print("Dummy data saved to dummy_issues_data_with_feedback.csv")
