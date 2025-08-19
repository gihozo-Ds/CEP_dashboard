# CitizenConnect | Officials Dashboard

A Dash-powered dashboard for visualizing and analyzing citizen-reported issues, feedback, and operational metrics for officials.

## Features

- Overview of reported issues by department and administrative level
- Time-based filtering (All, Today, Last 7 Days)
- Location-based filtering (District, Sector, Cell)
- Insights: resolution times, trends, and more
- Sentiment analysis 
- Predictive model 

## Requirements

- Python 3.7+
- All dependencies listed in `requirements.txt`

Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Place your data file as `final_dataset.csv` in the project directory.
2. Run the app:
    ```bash
    python app.py
    ```
3. Open your browser at [http://127.0.0.1:8050](http://127.0.0.1:8050)

## File Structure

- `app.py` - Main dashboard application
- `final_dataset.csv` - Data source (must be present)
- `README.md` - Project documentation

## Customization

- Update `final_dataset.csv` with your latest data.
- Extend the dashboard by adding new pages or visualizations in `app.py`.


## Author

Gihozo Christian.
## Author

Gihozo Christian.
