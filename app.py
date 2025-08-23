import logging
import dash
from dash import dcc, html, Input, Output, callback_context
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import geopandas as gpd
import json
from textblob import TextBlob
from wordcloud import WordCloud
import base64
from io import BytesIO
import os

# ---------------- Load & prep data ----------------
DATA_PATH = "C:/Users/gihoz/OneDrive/Desktop/CEPdash/final_dataset.csv"
df = pd.read_csv(DATA_PATH)

# parse dates (DD/MM/YYYY) with coercion
df['date_reported'] = pd.to_datetime(df.get('date_reported'), format='%d/%m/%Y', errors='coerce')
df['date_resolved'] = pd.to_datetime(df.get('date_resolved'), format='%d/%m/%Y', errors='coerce')

# normalize text columns safely
text_cols = ['status', 'is_overdue', 'assigned_level', 'assigned_department', 'district', 'sector', 'cell', 'leaders', 'priority']
for col in text_cols:
    if col in df.columns:
        df[col] = df[col].astype(str).str.strip()
    else:
        df[col] = ""

# ensure feedback_rating exists (numeric)
if 'feedback_rating' not in df.columns:
    df['feedback_rating'] = np.nan
else:
    df['feedback_rating'] = pd.to_numeric(df['feedback_rating'], errors='coerce')

# ---- sentiment preprocessing (robust) ----
def classify_sentiment(text):
    if not isinstance(text, str) or text.strip() == "":
        return "neutral"
    polarity = TextBlob(text).sentiment.polarity
    if polarity > 0.1:
        return "positive"
    elif polarity < -0.1:
        return "negative"
    else:
        return "neutral"

if 'feedback_comment' in df.columns:
    df['sentiment'] = df['feedback_comment'].apply(classify_sentiment)
else:
    df['sentiment'] = ['neutral'] * len(df)

# ---------------- Preload & simplify GeoJSON (districts) ----------------
_GEOJSON_DISTRICT_PATH = "C:/Users/gihoz/OneDrive/Desktop/CEPdash/geoBoundaries-RWA-ADM2 (1).geojson"
try:
    with open(_GEOJSON_DISTRICT_PATH, "r", encoding="utf-8") as f:
        _rwanda_geojson_raw = json.load(f)
    _gdf_districts = gpd.GeoDataFrame.from_features(_rwanda_geojson_raw["features"])
    _gdf_districts['geometry'] = _gdf_districts['geometry'].simplify(tolerance=0.005, preserve_topology=True)
    _rwanda_geojson_simpl = json.loads(_gdf_districts.to_json())
except Exception as e:
    print("Warning: failed to load/parse district GeoJSON:", e)
    _gdf_districts = gpd.GeoDataFrame()
    _rwanda_geojson_simpl = {}

# ---------------- Dash app ----------------
app = dash.Dash(__name__)
server = app.server
app.title = "CitizenConnect | Officials Dashboard"


# remove default body margins and ensure 100% height for vh usage
app.index_string = app.index_string.replace(
    "</head>",
    "<style>html,body{margin:0;height:100%;}*{box-sizing:border-box;} \
    /* Ensure dropdown selected value and options are black for better contrast */ \
    .Select-menu-outer, .Select-option, .Select-placeholder, .Select-value, \
    .css-1uccc91-singleValue, .css-1wa3eu0-placeholder, .css-1laao21, .css-1okebmr-indicatorSeparator { color: black !important; } \
    .Select-menu-outer { background-color: white !important; } \
    </style></head>"
)

# --- Global graph style (parent should control size; graphs set autosize) ---
GRAPH_STYLE = {'width': '100%', 'height': '100%', 'minHeight': '300px', 'margin': 0}

# ---- small card factory ----
def make_card(title: str, value, color: str):
    return html.Div([
        html.Div(title, style={'fontWeight': 600, 'color': '#555', 'fontSize': '13px', 'textAlign': 'center', 'marginBottom': '4px'}),
        html.Div(str(value), style={'fontWeight': 700, 'fontSize': '20px', 'textAlign': 'center', 'color': color})
    ], style={
        'backgroundColor': '#fff',
        'padding': '8px',
        'borderRadius': '8px',
        'boxShadow': '0 2px 6px rgba(0,0,0,0.06)',
        'flex': '1',
        'maxWidth': '115px',
        'height': '86px',
        'display': 'flex',
        'flexDirection': 'column',
        'justifyContent': 'center',
        'alignItems': 'center',
        'boxSizing': 'border-box'
    })

# ---- Sentiment visual creators ----
def create_sentiment_by_level(filtered_df):
    if filtered_df.empty or 'sentiment' not in filtered_df.columns:
        return go.Figure()
    sentiment_counts = (
        filtered_df.groupby(['assigned_level', 'sentiment'])
        .size()
        .reset_index(name='count')
    )
    fig = px.bar(
        sentiment_counts,
        x="assigned_level",
        y="count",
        color="sentiment",
        barmode="stack",
        title="Sentiment Distribution by Assigned Level"
    )
    fig.update_layout(xaxis_title="Assigned Level", yaxis_title="Count", title_x=0.5)
    return fig

def create_avg_rating_by_department(filtered_df):
    rated_df = filtered_df.dropna(subset=['feedback_rating'])
    if rated_df.empty:
        fig = go.Figure()
        fig.update_layout(title="Average Feedback Rating by Department (no ratings)")
        return fig

    dept_avg = rated_df.groupby("assigned_department")['feedback_rating'].mean().reset_index()
    dept_avg = dept_avg.sort_values('assigned_department')

    fig = px.bar(dept_avg, x="assigned_department", y="feedback_rating", title="Average Feedback Rating by Department")
    fig.update_traces(marker_color="#3498db", opacity=0.95, hovertemplate="%{x}: %{y:.2f}<extra></extra>")

    max_val = dept_avg['feedback_rating'].max() if not dept_avg['feedback_rating'].empty else 0
    y_padding = max(0.2, max_val * 0.12)

    fig.update_yaxes(range=[0, max_val + y_padding], title_text="Average Rating")
    fig.update_xaxes(title_text="Department")

    for _, row in dept_avg.iterrows():
        fig.add_annotation(
            x=row['assigned_department'],
            y=row['feedback_rating'] + (y_padding * 0.05) + 0.01,
            text=f"{row['feedback_rating']:.2f}",
            showarrow=False,
            font=dict(size=11, color="black"),
            xanchor='center',
            yanchor='bottom'
        )

    fig.update_layout(title_x=0.5, margin=dict(l=40, r=20, t=60, b=80))
    return fig

def create_wordcloud(filtered_df):
    if 'feedback_comment' not in filtered_df.columns or filtered_df['feedback_comment'].dropna().empty:
        return html.P("No feedback available for word cloud.")
    text = " ".join(str(comment) for comment in filtered_df['feedback_comment'].dropna())
    wc = WordCloud(width=800, height=400, background_color="white").generate(text)
    img = BytesIO()
    wc.to_image().save(img, format="PNG")
    img.seek(0)
    encoded = base64.b64encode(img.read()).decode("utf-8")
    return html.Img(src=f"data:image/png;base64,{encoded}", style={'width': '95%', 'height': '90%'})

# ---- Chart creators ----
def create_dept_bar(filtered_df):
    dept_counts = (
        filtered_df['assigned_department']
        .replace('', np.nan)
        .dropna()
        .value_counts()
        .rename_axis('Department')
        .reset_index(name='Number of Issues')
    )
    if dept_counts.empty:
        fig = go.Figure()
        fig.update_layout(title="Issues by Department")
        return fig
    fig = px.bar(dept_counts.sort_values('Number of Issues', ascending=True),
                 x='Number of Issues', y='Department', orientation='h', text='Number of Issues')
    fig.update_traces(textposition='outside')
    fig.update_layout(
        xaxis=dict(showticklabels=False),
        title={'text': 'Issues by Department', 'x': 0.5, 'xanchor': 'center'},
        xaxis_title="Number of Issues",
        yaxis_title="Department",
        margin=dict(l=40, r=40, t=50, b=40)
    )
    return fig

def create_level_bar(filtered_df):
    level_order = ['Cell', 'Sector', 'District']
    level_counts = (
        filtered_df['assigned_level']
        .replace('', np.nan)
        .dropna()
        .value_counts()
        .rename_axis('Assigned Level')
        .reset_index(name='count')
    )

    if level_counts.empty:
        fig = go.Figure()
        fig.update_layout(title="Issues by Assigned Level (no data)")
        return fig

    level_counts = level_counts[level_counts['Assigned Level'].isin(level_order)]
    level_counts['Assigned Level'] = pd.Categorical(level_counts['Assigned Level'], categories=level_order, ordered=True)
    level_counts = level_counts.sort_values('Assigned Level')

    fig = px.bar(level_counts, x='Assigned Level', y='count', title="Issues by Assigned Level")
    fig.update_traces(marker_color="#636EFA", opacity=0.95, hovertemplate="%{x}: %{y}<extra></extra>")

    max_count = level_counts['count'].max() if not level_counts['count'].empty else 0
    y_padding = max(5, max_count * 0.12)

    fig.update_yaxes(range=[0, max_count + y_padding], title_text="Number of Issues")
    fig.update_xaxes(title_text="Assigned Level")

    for _, row in level_counts.iterrows():
        fig.add_annotation(
            x=row['Assigned Level'],
            y=row['count'] + (y_padding * 0.02) + 0.5,
            text=str(int(row['count'])),
            showarrow=False,
            font=dict(size=12, color="black"),
            xanchor='center',
            yanchor='bottom'
        )

    fig.update_layout(
        title={'text': 'Issues by Assigned Level', 'x': 0.5},
        margin=dict(l=20, r=20, t=70, b=20),
        uniformtext_minsize=10
    )
    return fig


def create_resolution_donut(filtered_df):
    resolved_df = filtered_df[
        (filtered_df['status'].str.lower() == 'resolved') &
        (filtered_df['date_reported'].notna()) &
        (filtered_df['date_resolved'].notna())
    ].copy()
    if resolved_df.empty:
        fig = go.Figure()
        fig.update_layout(title="Average Resolution Time by Department (no resolved cases)")
        return fig
    resolved_df['resolution_time'] = (resolved_df['date_resolved'] - resolved_df['date_reported']).dt.days.clip(lower=0)
    department_avg = resolved_df.groupby('assigned_department')['resolution_time'].mean().sort_values(ascending=False)
    department_avg.index = department_avg.index.str.replace(' and ', ' & ').str.title()
    labels = [f"{dept}<br>{avg:.1f} days" for dept, avg in zip(department_avg.index, department_avg.values)]
    fig = px.pie(names=labels, values=department_avg.values, hole=0.5, title='Average Resolution Time by Department')
    total_avg = resolved_df['resolution_time'].mean()
    fig.update_traces(textinfo='label', textfont_size=12, hovertemplate="<b>%{label}</b><extra></extra>")
    fig.update_layout(annotations=[dict(text=f'Avg: {total_avg:.1f} days', x=0.5, y=0.5, font_size=14, showarrow=False)],
                      showlegend=False, margin=dict(l=10, r=10, t=60, b=20))
    return fig

def create_issues_over_time(filtered_df):
    time_series = (
        filtered_df
        .dropna(subset=['date_reported'])
        .groupby(filtered_df['date_reported'].dt.date)
        .size()
        .reset_index(name='Issue Count')
    )
    if time_series.empty:
        fig = go.Figure()
        fig.update_layout(title="Issues Over Time (no data)")
        return fig
    time_series['Smoothed'] = time_series['Issue Count'].rolling(window=7, min_periods=1).mean()
    fig = px.line(time_series, x='date_reported', y='Smoothed', title="Issues Reported Over Time")
    fig.update_traces(line=dict(width=2))
    fig.update_layout(xaxis_title="Date", yaxis_title="Number of Issues", margin=dict(l=20, r=20, t=50, b=20))
    return fig

# ---------------- Map creator (uses preloaded simplified GeoDataFrame) ----------------
def create_district_map(filtered_df):
    if _gdf_districts.empty or not isinstance(_rwanda_geojson_simpl, dict):
        fig = go.Figure()
        fig.update_layout(title="District map not available")
        return fig
    issue_counts = filtered_df.groupby('district').size().reset_index(name='issue_count')
    merged = _gdf_districts.merge(issue_counts, left_on='shapeName', right_on='district', how='left')
    merged['issue_count'] = merged['issue_count'].fillna(0)
    green_palette = ["#e8f4fa", "#74c0fc", "#3498db", "#2471a3", "#0b3c5d"]

    fig = px.choropleth_map(
        merged,
        geojson=_rwanda_geojson_simpl,
        locations="shapeName",
        featureidkey="properties.shapeName",
        color="issue_count",
        color_continuous_scale=green_palette,
        center={"lat": -1.9441, "lon": 30.0619},
        zoom=7.1,
        labels={"issue_count": "Issue Count"},
        hover_data={"shapeName": True, "issue_count": True}
    )
    fig.update_layout(mapbox_style="carto-positron",
                      title="Density of Reported Issues by District in Kigali, Rwanda",
                      title_x=0.5,
                      title_font=dict(size=20, family="Lato, sans-serif", color="#000000"),
                      margin={"r": 0, "t": 50, "l": 0, "b": 0},
                      coloraxis_colorbar_title="Issue Count")
    return fig

# ---- NEW: leaders top/bottom + priority visuals (REVISED) ----
def create_top_bottom_leaders(filtered_df):
    """
    Compute average feedback_rating per leader and return a horizontal bar
    with top 5 and bottom 5 combined, ordered ASC by mean so it reads low->high.
    """
    if 'leaders' not in filtered_df.columns or filtered_df['leaders'].dropna().empty:
        fig = go.Figure()
        fig.update_layout(title="Leaders by Average Rating (no data)")
        return fig

    rated = filtered_df.dropna(subset=['feedback_rating', 'leaders']).copy()
    if rated.empty:
        fig = go.Figure()
        fig.update_layout(title="Leaders by Average Rating (no ratings)")
        return fig

    leaders_avg = rated.groupby('leaders')['feedback_rating'].agg(['mean', 'count']).reset_index()
    leaders_avg = leaders_avg.sort_values('mean', ascending=False)

    top = leaders_avg.head(5)
    bottom = leaders_avg.tail(5)

    combined = pd.concat([top, bottom]).drop_duplicates(subset=['leaders'])
    # Now order combined by mean ascending (so low to high)
    combined = combined.sort_values('mean', ascending=True).reset_index(drop=True)
    combined['group'] = combined['mean'].rank(method='first', ascending=False)  # dummy to keep group column if needed
    # For color grouping use whether leader was in top originally
    combined['which'] = combined['leaders'].apply(lambda x: 'Top' if x in set(top['leaders']) else 'Bottom')

    order = list(combined['leaders'])
    combined['leaders'] = pd.Categorical(combined['leaders'], categories=order, ordered=True)

    # Reduced margins to avoid large empty white space and allow the chart to fill its container
    fig = px.bar(combined, x='mean', y='leaders', orientation='h', color='which',
                 labels={'mean': 'Avg Rating (1-5)', 'leaders': 'Leader'},
                 text=combined['mean'].round(2),
                 color_discrete_map={'Top': '#3498db', 'Bottom': '#d62728'})

    # place text outside and increase margins to fit, but smaller than before so it fills space
    fig.update_traces(textposition='outside', marker_line_width=0.5)
    fig.update_layout(
        title="Top & Bottom Leaders by Average Feedback Rating (low → high)",
        xaxis_title="Average Rating",
        yaxis_title="Leader",
        xaxis=dict(range=[0, 5], automargin=True),
        margin=dict(l=120, r=40, t=80, b=40),  # reduced left/right margins
        legend=dict(title='', orientation='h', y=-0.12, x=0.5, xanchor='center'),
        height=520
    )

    # Ensure annotations/text don't get clipped in certain browsers
    fig.update_layout(autosize=True)
    return fig

def create_issues_by_priority(filtered_df):
    """
    Count issues by priority. Order: Urgent, High, Medium, Low
    Changed to vertical bars with default colors and increased visibility.
    """
    if 'priority' not in filtered_df.columns:
        fig = go.Figure()
        fig.update_layout(title="Issues by Priority (not available)")
        return fig

    filtered_df['priority_norm'] = filtered_df['priority'].str.title().replace({'Urgent': 'Urgent', 'High': 'High', 'Medium': 'Medium', 'Low': 'Low'})
    order = ['Urgent', 'High', 'Medium', 'Low']
    counts = filtered_df['priority_norm'].value_counts().rename_axis('priority').reset_index(name='count')
    if counts.empty:
        fig = go.Figure()
        fig.update_layout(title="Issues by Priority (no data)")
        return fig

    counts = counts.set_index('priority').reindex(order).fillna(0).reset_index()

    # Vertical bar chart, default colors, bigger height and tighter margins
    fig = px.bar(counts, x='priority', y='count', text='count',
                 labels={'count': 'Number of Issues', 'priority': 'Priority'})
    fig.update_traces(textposition='outside', marker_line_width=0.5)
    fig.update_layout(
        title="Issues by Priority",
        xaxis_title="Priority",
        yaxis_title="Number of Issues",
        margin=dict(l=40, r=40, t=70, b=80),
        height=520,
        showlegend=False,
        bargap=0.2
    )
    fig.update_layout(autosize=True)
    return fig

# ---- Dashboard view (use compact height so it fits presentation better) ----
def dashboard_view(filtered_df: pd.DataFrame):
    status_clean = filtered_df['status'].str.lower()
    total_issues = len(filtered_df)
    resolved = (status_clean == 'resolved').sum()
    open_ = (status_clean == 'open').sum()
    inprogress = (status_clean == 'in progress').sum()
    escalated = (status_clean == 'escalated').sum()
    overdue = (filtered_df['is_overdue'].str.lower() == 'yes').sum() if 'is_overdue' in filtered_df.columns else 0

    if filtered_df['feedback_rating'].notna().any():
        avg_rating = filtered_df['feedback_rating'].mean()
        citizen_satisfaction = f"{round((avg_rating/5)*100, 2)}%"
    else:
        citizen_satisfaction = "N/A"

    cards_row = html.Div([
        make_card("Total Issues", total_issues, '#2980b9'),
        make_card("Resolved", resolved, '#27ae60'),
        make_card("Open", open_, '#c0392b'),
        make_card("In Progress", inprogress, '#f39c12'),
        make_card("Escalated", escalated, '#8e44ad'),
        make_card("Overdue", overdue, '#d35400'),
        make_card("Satisfaction Rate", citizen_satisfaction, '#16a085'),
    ], style={
        'display': 'flex',
        'justifyContent': 'space-between',
        'gap': '8px',
        'marginBottom': '14px',
        'flexWrap': 'nowrap'
    })

    fig = create_dept_bar(filtered_df)
    fig.update_layout(autosize=True, margin=dict(l=20, r=20, t=50, b=30))

    # slightly smaller compact height for presentation
    graph_container = html.Div(
        dcc.Graph(
            figure=fig,
            config={'displayModeBar': False, 'responsive': True},
            style={'width': '100%', 'height': '100%'}
        ),
        style={'flex': '1 1 auto', 'height': '360px', 'width': '100%', 'boxSizing': 'border-box', 'paddingTop': '8px'}
    )

    return html.Div([cards_row, graph_container], style={'display': 'flex', 'flexDirection': 'column', 'width': '100%'})

# ---------------- Layout ----------------
HEADER_HEIGHT = '60px'
SIDEBAR_WIDTH = '200px'

app.layout = html.Div([
    dcc.Store(id='current-page', data='dashboard'),

    # Fixed header (top)
    html.Div(
        html.H1("🏠 CitizenConnect | Officials Dashboard",
                style={'textAlign': 'center', 'color': '#ffffff', 'margin': 0, 'lineHeight': HEADER_HEIGHT}),
        style={
            'position': 'fixed',
            'top': 0,
            'left': 0,
            'right': 0,
            'height': HEADER_HEIGHT,
            'backgroundColor': '#34495e',
            'padding': '10px 20px',
            'boxShadow': '0 2px 6px rgba(0,0,0,0.12)',
            'zIndex': 1000
        }
    ),

    # Sidebar + Content wrapper
    html.Div([
        # Fixed sidebar (left)
        html.Div([
            html.H3("Select Location", style={'marginBottom': '18px', 'color': '#fff'}),
            html.Label("Select Time filter", style={'color': '#fff', 'marginTop': '6px'}),
            dcc.Dropdown(
                id='time-filter',
                options=[
                    {'label': 'All', 'value': 'all'},
                    {'label': 'Today', 'value': 'today'},
                    {'label': 'Last 7 Days', 'value': 'last7'}
                ],
                value='all',
                style={'marginBottom': '16px', 'color': 'black', 'backgroundColor': 'white'}
            ),

            html.Label("District", style={'color': '#fff'}),
            dcc.Dropdown(
                id='district-dropdown',
                options=[{'label': d, 'value': d} for d in sorted(df['district'].dropna().unique())],
                placeholder="Select District",
                style={'marginBottom': '16px', 'color': 'black', 'backgroundColor': 'white'}
            ),

            html.Label("Sector", style={'color': '#fff'}),
            dcc.Dropdown(id='sector-dropdown', placeholder="Select Sector", style={'marginBottom': '16px', 'color': 'black', 'backgroundColor': 'white'}),

            html.Label("Cell", style={'color': '#fff'}),
            dcc.Dropdown(id='cell-dropdown', placeholder="Select Cell", style={'marginBottom': '16px', 'color': 'black', 'backgroundColor': 'white'}),

            html.Hr(style={'borderColor': 'rgba(255,255,255,0.15)'}),
            html.Button('More Insights', id='insights-button', n_clicks=0, style={'width': '100%', 'marginBottom': '10px'}),
            html.Button('Sentiments', id='sentiments-button', n_clicks=0, style={'width': '100%', 'marginBottom': '10px'}),
            html.Button('Model', id='model-button', n_clicks=0, style={'width': '100%'}),
        ],
        style={
            'position': 'fixed',
            'top': HEADER_HEIGHT,
            'left': 0,
            'bottom': 0,
            'width': SIDEBAR_WIDTH,
            'padding': '20px',
            'backgroundColor': "#34495e",
            'color': '#fff',
            'overflowY': 'auto',
            'boxSizing': 'border-box'
        }),

        # Main content area (scrollable & flexible)
        html.Div(id='main-content', style={
            'width': f"calc(100% - {SIDEBAR_WIDTH})",
            'marginLeft': SIDEBAR_WIDTH,
            'marginTop': HEADER_HEIGHT,
            'padding': '20px',
            'display': 'flex',
            'flexDirection': 'column',
            'height': f"calc(100vh - {HEADER_HEIGHT})",
            'overflowY': 'auto',
            'boxSizing': 'border-box',
            'backgroundColor': '#f7f9fb'
        }, children=[
            # compact go back button
            html.Button('⬅ Go Back', id='go-back-button', n_clicks=0,
                        style={'display': 'none', 'marginBottom': '20px', 'alignSelf': 'flex-start', 'width': 'auto'}),

            # page content area: flexible and scrollable
            html.Div(id='page-content', children=[dashboard_view(df.copy())],
                     style={'width': '100%', 'display': 'flex', 'flexDirection': 'column', 'gap': '12px', 'flex': '1 1 auto', 'height': '100%', 'overflowY': 'auto', 'boxSizing': 'border-box'})
        ]),

    ], style={'display': 'flex', 'flexDirection': 'row', 'height': '100vh', 'width': '100%'} )
])

# ---------------- Callbacks ----------------
_MAP_CACHE = {}

def _map_cache_key(filtered_df, time_filter, selected_district, selected_sector, selected_cell):
    counts = filtered_df.groupby('district').size()
    sig = tuple(sorted((str(k), int(v)) for k, v in counts.items()))
    return ('map_v1', time_filter, selected_district or '', selected_sector or '', selected_cell or '', sig)

@app.callback(
    Output('sector-dropdown', 'options'),
    Input('district-dropdown', 'value')
)
def update_sector_options(selected_district):
    if selected_district is None:
        return [{'label': s, 'value': s} for s in sorted(df['sector'].dropna().unique())]
    opts = sorted(df.loc[df['district'] == selected_district, 'sector'].dropna().unique())
    return [{'label': s, 'value': s} for s in opts]

@app.callback(
    Output('cell-dropdown', 'options'),
    Input('sector-dropdown', 'value')
)
def update_cell_options(selected_sector):
    if selected_sector is None:
        return [{'label': c, 'value': c} for c in sorted(df['cell'].dropna().unique())]
    opts = sorted(df.loc[df['sector'] == selected_sector, 'cell'].dropna().unique())
    return [{'label': c, 'value': c} for c in opts]

@app.callback(
    Output('current-page', 'data'),
    Input('insights-button', 'n_clicks'),
    Input('sentiments-button', 'n_clicks'),
    Input('model-button', 'n_clicks'),
    Input('go-back-button', 'n_clicks'),
)
def update_current_page(insights_n, sentiments_n, model_n, back_n):
    ctx = callback_context
    if not ctx.triggered:
        return dash.no_update
    trigger = ctx.triggered[0]['prop_id'].split('.')[0]
    if trigger == 'insights-button':
        return 'insights'
    if trigger == 'sentiments-button':
        return 'sentiments'
    if trigger == 'model-button':
        return 'model'
    if trigger == 'go-back-button':
        return 'dashboard'
    return dash.no_update

@app.callback(
    [Output('page-content', 'children'),
     Output('go-back-button', 'style')],
    [Input('current-page', 'data'),
     Input('district-dropdown', 'value'),
     Input('sector-dropdown', 'value'),
     Input('cell-dropdown', 'value'),
     Input('time-filter', 'value')],
    prevent_initial_call=True
)
def render_page(current_page, selected_district, selected_sector, selected_cell, time_filter):
    # --- fast filter ---
    filtered_df = df.copy()

    if df['date_reported'].notna().any():
        today = df['date_reported'].max().normalize()
        if time_filter == 'today':
            filtered_df = filtered_df[filtered_df['date_reported'] == today]
        elif time_filter == 'last7':
            filtered_df = filtered_df[filtered_df['date_reported'] >= today - pd.Timedelta(days=7)]

    if selected_district:
        filtered_df = filtered_df[filtered_df['district'] == selected_district]
    if selected_sector:
        filtered_df = filtered_df[filtered_df['sector'] == selected_sector]
    if selected_cell:
        filtered_df = filtered_df[filtered_df['cell'] == selected_cell]

    # ----------------- SENTIMENTS PAGE -----------------
    if current_page == 'sentiments':
        fig_sent_level = create_sentiment_by_level(filtered_df)
        fig_sent_level.update_layout(autosize=True, margin=dict(l=30, r=30, t=50, b=30))

        fig_avg_rating = create_avg_rating_by_department(filtered_df)
        fig_avg_rating.update_layout(autosize=True, margin=dict(l=30, r=30, t=50, b=30))

        sentiments_layout = html.Div([
            # compact top charts stacked vertically with reduced heights
            html.Div(
                dcc.Graph(figure=fig_sent_level, config={'responsive': True, 'displayModeBar': False},
                          style={'width': '100%', 'height': '360px'}),
                style={'width': '100%', 'boxSizing': 'border-box', 'marginBottom': '12px'}
            ),
            html.Div(
                dcc.Graph(figure=fig_avg_rating, config={'responsive': True, 'displayModeBar': False},
                          style={'width': '100%', 'height': '360px'}),
                style={'width': '100%', 'boxSizing': 'border-box'}
            ),
            html.H3("Word Cloud of Citizens' Feedback", style={'marginTop': '20px'}),
            create_wordcloud(filtered_df)
        ], style={'width': '100%', 'display': 'flex', 'flexDirection': 'column', 'gap': '12px', 'height': '100%', 'overflowY': 'auto'})

        return sentiments_layout, {'display': 'inline-block', 'marginBottom': '20px', 'alignSelf': 'flex-start', 'width': 'auto'}

    # ----------------- INSIGHTS PAGE -----------------
    if current_page == 'insights':
        left_fig = create_level_bar(filtered_df)
        right_fig = create_resolution_donut(filtered_df)
        bottom_fig = create_issues_over_time(filtered_df)

        key = _map_cache_key(filtered_df, time_filter, selected_district, selected_sector, selected_cell)
        map_fig = _MAP_CACHE.get(key)
        if map_fig is None:
            map_fig = create_district_map(filtered_df)
            map_fig.update_layout(uirevision='insights-map', autosize=True)
            _MAP_CACHE[key] = map_fig

        # Set tidy margins; graphs below have explicit container heights (px)
        left_fig.update_layout(autosize=True, margin=dict(l=30, r=30, t=50, b=30))
        right_fig.update_layout(autosize=True, margin=dict(l=30, r=30, t=50, b=30))
        bottom_fig.update_layout(autosize=True, margin=dict(l=30, r=30, t=50, b=30))
        map_fig.update_layout(autosize=True, margin=dict(l=0, r=0, t=50, b=0))

        # --- New visuals (leaders + priority) placed below the trendline ---
        leaders_fig = create_top_bottom_leaders(filtered_df)
        priority_fig = create_issues_by_priority(filtered_df)

        insights_layout = html.Div([
            # top row: two charts each compact fixed height
            html.Div([
                html.Div(
                    dcc.Graph(figure=left_fig, config={'displayModeBar': False, 'responsive': True},
                              style={'width': '100%', 'height': '360px'}),
                    style={'flex': '1 1 50%', 'minWidth': '320px', 'height': '360px', 'boxSizing': 'border-box', 'paddingRight': '8px'}
                ),
                html.Div(
                    dcc.Graph(figure=right_fig, config={'displayModeBar': False, 'responsive': True},
                              style={'width': '100%', 'height': '360px'}),
                    style={'flex': '1 1 50%', 'minWidth': '320px', 'height': '360px', 'boxSizing': 'border-box', 'paddingLeft': '8px'}
                ),
            ], style={'display': 'flex', 'flexDirection': 'row', 'gap': '12px', 'width': '100%'}),

            # map (reduced to 520px)
            html.Div(
                dcc.Graph(figure=map_fig, config={'displayModeBar': True, 'responsive': True}, style={'width': '100%', 'height': '520px'}),
                style={'width': '100%', 'marginTop': '18px', 'boxSizing': 'border-box'}
            ),

            # bottom trendline (compact)
            html.Div(
                dcc.Graph(figure=bottom_fig, config={'displayModeBar': True, 'responsive': True}, style={'width': '100%', 'height': '360px'}),
                style={'width': '100%', 'marginTop': '18px', 'boxSizing': 'border-box'}
            ),

            # --- NEW ROW: Leaders (left) & Priority (right) BELOW the trendline ---
            html.Div([
                html.Div(
                    dcc.Graph(figure=leaders_fig, config={'displayModeBar': False, 'responsive': True},
                              style={'width': '100%', 'height': '520px'}),
                    style={'flex': '2 1 65%', 'minWidth': '420px', 'height': '520px', 'boxSizing': 'border-box', 'paddingRight': '8px'}
                ),
                html.Div(
                    dcc.Graph(figure=priority_fig, config={'displayModeBar': False, 'responsive': True},
                              style={'width': '100%', 'height': '520px'}),
                    style={'flex': '1 1 35%', 'minWidth': '300px', 'height': '520px', 'boxSizing': 'border-box', 'paddingLeft': '8px'}
                ),
            ], style={'display': 'flex', 'flexDirection': 'row', 'gap': '12px', 'width': '100%', 'marginTop': '18px'}),

        ], style={'paddingTop': '6px', 'width': '100%', 'boxSizing': 'border-box', 'height': '100%', 'overflowY': 'auto'})

        return insights_layout, {'display': 'inline-block', 'marginBottom': '20px', 'alignSelf': 'flex-start', 'width': 'auto'}

    # ----------------- MODEL PAGE -----------------
    if current_page == 'model':
        model_layout = html.Div([
            html.H2("🤖 Predictive Model", style={'color': '#2c3e50'}),
            html.P("Display model predictions, accuracy, or risk flags here.")
        ])
        return model_layout, {'display': 'inline-block', 'marginBottom': '20px', 'alignSelf': 'flex-start', 'width': 'auto'}

    # default: dashboard
    return dashboard_view(filtered_df), {'display': 'none', 'alignSelf': 'flex-start', 'width': 'auto'}


if __name__ == '__main__':
    app.run(debug=True,port=8078)
   
    
    
