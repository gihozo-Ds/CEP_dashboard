import pandas as pd
import random
from datetime import datetime, timedelta
import numpy as np

# Rwanda administrative structure with REAL cells from official government data
rwanda_admin = {
    'Kicukiro': {
        'sectors': ['Masaka', 'Gahanga', 'Niboye', 'Kagarama', 'Gatenga'],
        'cells': {
            'Masaka': ['Kabeza', 'Kagarama', 'Kimisagara', 'Masaka', 'Nyakabanda'],
            'Gahanga': ['Batsinda', 'Gahanga', 'Kabuga', 'Kimisagara', 'Nyanza'],
            'Niboye': ['Kanombe', 'Karama', 'Niboye'],
            'Kagarama': ['Kagarama', 'Kanombe', 'Karama', 'Nyarugunga'],
            'Gatenga': ['Gatenga', 'Kabuye', 'Nyanza']
        }
    },
    'Nyarugenge': {
        'sectors': ['Nyarugenge', 'Muhima', 'Rwezamenyo', 'Gitega', 'Kigali'],
        'cells': {
            'Nyarugenge': ['Biryogo', 'Gitega', 'Muhima', 'Nyamirambo'],
            'Muhima': ['Cyivugiza', 'Muhima', 'Rugenge'],
            'Rwezamenyo': ['Kanyinya', 'Mageragere', 'Nyakabanda', 'Rwezamenyo'],
            'Gitega': ['Gitega', 'Muhima'],
            'Kigali': ['Gisozi', 'Kacyiru', 'Kigali']
        }
    },
    'Gasabo': {
        'sectors': ['Remera', 'Kimisagara', 'Gikomero', 'Gisozi', 'Jabana', 'Bumbogo', 'Gatsata', 'Jali', 'Kacyiru', 'Kimihurura', 'Ndera', 'Nduba', 'Rusororo', 'Rutunga'],
        'cells': {
            'Remera': ['Gisozi', 'Kimihurura', 'Nyarutarama', 'Remera'],
            'Kimisagara': ['Kimisagara', 'Nyamirambo', 'Ubumwe'],
            'Gikomero': ['Cyahafi', 'Gasanze', 'Gikomero', 'Murambi', 'Ruli'],
            'Gisozi': ['Gisozi', 'Rwampara'],
            'Jabana': ['Jabana', 'Mpanga', 'Rusarabuye'],
            'Bumbogo': ['Bumbogo', 'Gitarama', 'Kinyami', 'Mpazi', 'Nduba'],
            'Gatsata': ['Gasogi', 'Gatsata', 'Gishushu'],
            'Jali': ['Jali', 'Mahiga', 'Munyiginya'],
            'Kacyiru': ['Kamatamu', 'Kacyiru', 'Kamutwa'],
            'Kimihurura': ['Kimihurura', 'Kisimenti'],
            'Ndera': ['Busanza', 'Ndera', 'Rushashi'],
            'Nduba': ['Cyahafi', 'Kinyami', 'Nduba'],
            'Rusororo': ['Kigabiro', 'Rusororo', 'Taba'],
            'Rutunga': ['Gasabo', 'Rutunga', 'Rwimbogo']
        }
    },
    # Real cells from Rutsiro district (from government data)
    'Rutsiro': {
        'sectors': ['Boneza', 'Gihango', 'Kigeyo', 'Kivumu', 'Manihira', 'Mukura', 'Murunda', 'Musasa', 'Mushonyi', 'Mushubati', 'Nyabirasi', 'Ruhango', 'Rusebeya'],
        'cells': {
            'Boneza': ['Bushaka', 'Kabihogo', 'Nkira', 'Remera'],
            'Gihango': ['Bugina', 'Congo-nil', 'Mataba', 'Murambi', 'Ruhingo', 'Shyembe', 'Teba'],
            'Kigeyo': ['Buhindure', 'Nkora', 'Nyagahinika', 'Rukaragata'],
            'Kivumu': ['Bunyoni', 'Bunyunju', 'Kabere', 'Kabujenje', 'Karambi', 'Nganzo'],
            'Manihira': ['Haniro', 'Muyira', 'Tangabo'],
            'Mukura': ['Kabuga', 'Kagano', 'Kageyo', 'Kagusa', 'Karambo', 'Mwendo'],
            'Murunda': ['Kirwa', 'Mburamazi', 'Rugeyo', 'Twabugezi'],
            'Musasa': ['Gabiro', 'Gisiza', 'Murambi', 'Nyarubuye'],
            'Mushonyi': ['Biruyi', 'Kaguriro', 'Magaba', 'Rurara'],
            'Mushubati': ['Bumba', 'Cyarusera', 'Gitwa', 'Mageragere', 'Sure'],
            'Nyabirasi': ['Busuku', 'Cyivugiza', 'Mubuga', 'Ngoma', 'Terimbere'],
            'Ruhango': ['Gatare', 'Gihira', 'Kavumu', 'Nyakarera', 'Rugasa', 'Rundoyi'],
            'Rusebeya': ['Kabona', 'Mberi', 'Remera', 'Ruronde']
        }
    },
    # Real cells from Rusizi district (from government data) 
    'Rusizi': {
        'sectors': ['Bugarama', 'Gihundwe', 'Kamembe', 'Muganza', 'Mururu', 'Nyakabuye', 'Rwimbogo'],
        'cells': {
            'Bugarama': ['Nyange', 'Pera', 'Ryankana'],
            'Gihundwe': ['Burunga', 'Kamatita', 'Shagasha'],
            'Kamembe': ['Cyangugu', 'Gihundwe', 'Kamashangi', 'Kamurera', 'Ruganda'],
            'Muganza': ['Cyarukara', 'Gakoni', 'Shara'],
            'Mururu': ['Gahinga', 'Tara'],
            'Nyakabuye': ['Gasebeya', 'Gaseke', 'Kamanu', 'Kiziho', 'Mashyuza', 'Nyabintare'],
            'Rwimbogo': ['Karenge', 'Muhehwe', 'Mushaka']
        }
    }
}

# Issue types and descriptions
issue_templates = {
    'Water & sanitation': [
        'Water shortage in the community near the {location} for {duration}.',
        'Blocked drainage causing floods at the {location} for {duration}.',
        'Contaminated water supply affecting households near {location} since {duration}.',
        'Sewage overflow at {location} for {duration}.',
        'Broken water pipes disrupting supply near {location} since {duration}.'
    ],
    'Health and services': [
        'Public transport shortage during rush hours by the {location} since {duration}.',
        'Garbage collection delay near the {location} since {duration}.',
        'Medical supplies shortage at {location} for {duration}.',
        'Ambulance service delay affecting {location} since {duration}.',
        'Health center overcrowding at {location} for {duration}.'
    ],
    'Education': [
        'School building damage at {location} affecting classes for {duration}.',
        'Teacher shortage at {location} since {duration}.',
        'Lack of textbooks at school near {location} for {duration}.',
        'School feeding program interruption at {location} since {duration}.',
        'Classroom overcrowding at {location} for {duration}.'
    ],
    'Public safety': [
        'Street lighting failure near {location} since {duration}.',
        'Road damage affecting traffic at {location} for {duration}.',
        'Security concerns at {location} since {duration}.',
        'Flooding in homes after rain near {location} for {duration}.',
        'Bridge damage blocking access to {location} since {duration}.'
    ],
    'Infrastructure': [
        'Electricity blackout in the neighborhood behind {location} since {duration}.',
        'Road construction delays at {location} for {duration}.',
        'Internet connectivity issues at {location} since {duration}.',
        'Building collapse risk at {location} for {duration}.',
        'Phone network outage affecting {location} since {duration}.'
    ]
}

# Random locations and durations
locations = [
    'the primary school', 'the local office', 'the market', 'the health center',
    'the football field', 'the river', 'the main road', 'the community center',
    'the church', 'the mosque', 'the cooperative office', 'the bus station'
]

durations = [
    'yesterday', 'last week', 'the rainy season began', '1 week', 'over a month',
    'two days', 'three weeks', 'the beginning of the month', 'last Monday',
    'several days', 'the weekend', 'this morning'
]

departments = list(issue_templates.keys())
statuses = ['Pending', 'In Progress', 'Resolved']
priorities = ['Low', 'Medium', 'High']
escalation_levels = ['cell', 'sector', 'district']

# Staff members for assignment
staff_members = [
    'John Uwimana', 'Marie Mukamana', 'Paul Nzeyimana', 'Grace Uwimana',
    'Peter Habimana', 'Alice Mukamazimpaka', 'David Nkurunziza', 'Sarah Nyirabeza',
    'Emmanuel Gasana', 'Josephine Uwamahoro', 'Robert Bizimana', 'Agnes Mukaneza'
]

def generate_dummy_data(num_records=100):
    """Generate dummy data for Rwanda issue tracking system"""
    
    data = []
    
    for i in range(1, num_records + 1):
        # Select random district, sector, and cell
        district = random.choice(list(rwanda_admin.keys()))
        sector = random.choice(rwanda_admin[district]['sectors'])
        cell = random.choice(rwanda_admin[district]['cells'][sector])
        
        # Generate timestamps
        start_date = datetime(2025, 1, 1)
        end_date = datetime(2025, 8, 12)
        time_stamp = start_date + timedelta(
            seconds=random.randint(0, int((end_date - start_date).total_seconds()))
        )
        
        # Generate status and time_solved
        status = random.choice(statuses)
        if status == 'Resolved':
            # Add 1-10 days to timestamp for resolution
            time_solved = time_stamp + timedelta(days=random.randint(1, 10))
        else:
            time_solved = None
        
        # Generate department and description
        department = random.choice(departments)
        template = random.choice(issue_templates[department])
        location = random.choice(locations)
        duration = random.choice(durations)
        description = template.format(location=location, duration=duration)
        
        # Escalation logic
        escalation = random.choice([True, False])
        if escalation:
            escalated_to = random.choice(escalation_levels)
        else:
            escalated_to = 'no'
        
        # Priority tends to be higher if escalated
        if escalation:
            priority = random.choices(priorities, weights=[0.2, 0.3, 0.5])[0]
        else:
            priority = random.choices(priorities, weights=[0.5, 0.3, 0.2])[0]
        
        # Assignment
        assigned_to = random.choice(staff_members) if random.random() > 0.1 else None
        
        record = {
            'ID': i,
            'Time_Stamp': time_stamp.strftime('%Y-%m-%d %H:%M:%S'),
            'Time_Solved': time_solved.strftime('%Y-%m-%d %H:%M:%S') if time_solved else None,
            'District': district,
            'Sector': sector,
            'Cell': cell,
            'Status': status,
            'Description': description,
            'Department': department,
            'Escalation': 'yes' if escalation else 'no',
            'Escalated_to': escalated_to,
            'Priority': priority,
            'Assigned_To': assigned_to
        }
        
        data.append(record)
    
    return pd.DataFrame(data)

# Generate the data
df = generate_dummy_data(100)

# Export to Excel file
excel_filename = 'rwanda_issues_tracking.xlsx'
df.to_excel(excel_filename, index=False, sheet_name='Issues')

# Display summary
print("Generated Rwanda Issue Tracking Data:")
print("="*50)
print(f"Total records generated: {len(df)}")
print(f"Data exported to: {excel_filename}")
print("\nData Summary:")
print(f"Status distribution: {df['Status'].value_counts().to_dict()}")
print(f"District distribution: {df['District'].value_counts().to_dict()}")
print(f"Priority distribution: {df['Priority'].value_counts().to_dict()}")
print(f"Escalation rate: {(df['Escalation'] == 'yes').sum()}/{len(df)} ({(df['Escalation'] == 'yes').mean()*100:.1f}%)")

# Display first few rows as preview
print("\nFirst 5 records preview:")
print(df.head().to_string(index=False, max_colwidth=50))