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

# ---------------- Load & prep data ----------------
DATA_PATH = "C:/Users/gihoz/OneDrive/Desktop/CEPdash/final_dataset.csv"
df = pd.read_csv(DATA_PATH)

# parse dates (DD/MM/YYYY) with coercion
df['date_reported'] = pd.to_datetime(df.get('date_reported'), format='%d/%m/%Y', errors='coerce')
df['date_resolved'] = pd.to_datetime(df.get('date_resolved'), format='%d/%m/%Y', errors='coerce')

# normalize text columns safely
text_cols = ['status', 'is_overdue', 'assigned_level', 'assigned_department', 'district', 'sector', 'cell']
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

# ---- sentiment preprocessing ----
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

df['sentiment'] = df['feedback_comment'].apply(classify_sentiment) if 'feedback_comment' in df.columns else "neutral"

# ---------------- Dash app ----------------
app = dash.Dash(__name__)
app.title = "CitizenConnect | Officials Dashboard"

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
    fig.update_layout(
        xaxis_title="Assigned Level",
        yaxis_title="Count",
        title_x=0.5,
        height=400
    )
    return fig

def create_avg_rating_by_department(filtered_df):
    rated_df = filtered_df.dropna(subset=['feedback_rating'])
    if rated_df.empty:
        return go.Figure()

    dept_avg = rated_df.groupby("assigned_department")['feedback_rating'].mean().reset_index()

    fig = px.bar(
        dept_avg,
        x="assigned_department",
        y="feedback_rating",
        title="Average Feedback Rating by Department",
        text="feedback_rating"
    )
    fig.update_traces(texttemplate='%{text:.2f}', textposition="outside", marker_color="#3498db")
    fig.update_layout(
        xaxis_title="Department",
        yaxis_title="Average Rating",
        title_x=0.5,
        height=400
    )
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

    return html.Img(src=f"data:image/png;base64,{encoded}", style={'width': '100%', 'height': 'auto'})

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

    fig = px.bar(
        dept_counts.sort_values('Number of Issues', ascending=True),
        x='Number of Issues',
        y='Department',
        orientation='h',
        text='Number of Issues'
    )
    fig.update_traces(textposition='outside')
    fig.update_layout(
        xaxis=dict(showticklabels=False),
        title={'text': 'Issues by Department', 'x': 0.5, 'xanchor': 'center'},
        xaxis_title="Number of Issues",
        yaxis_title="Department",
        height=360,
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
    level_counts['Assigned Level'] = pd.Categorical(
        level_counts['Assigned Level'], 
        categories=level_order, 
        ordered=True
    )
    level_counts = level_counts.sort_values('Assigned Level')

    fig = px.bar(level_counts,
                 x='Assigned Level', y='count',
                 text='count')
    fig.update_traces(textposition='outside', marker_color="#636EFA")
    fig.update_layout(
        title={'text': 'Issues by Assigned Level', 'x': 0.5, 'xanchor': 'center'},
        xaxis_title="Assigned Level",
        yaxis_title="Number of Issues",
        height=400,
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis={'categoryorder': 'array', 'categoryarray': level_order}
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

    resolved_df['resolution_time'] = (
        resolved_df['date_resolved'] - resolved_df['date_reported']
    ).dt.days.clip(lower=0)

    department_avg = resolved_df.groupby('assigned_department')['resolution_time'].mean().sort_values(ascending=False)
    department_avg.index = department_avg.index.str.replace(' and ', ' & ').str.title()

    labels = [f"{dept}<br>{avg:.1f} days" for dept, avg in zip(department_avg.index, department_avg.values)]

    fig = px.pie(
        names=labels,
        values=department_avg.values,
        hole=0.5,
        title='Average Resolution Time by Department'
    )

    total_avg = resolved_df['resolution_time'].mean()

    fig.update_traces(
        textinfo='label',
        textfont_size=12,
        hovertemplate="<b>%{label}</b><extra></extra>"
    )

    fig.update_layout(
        annotations=[dict(
            text=f'Avg: {total_avg:.1f} days',
            x=0.5, y=0.5,
            font_size=14,
            showarrow=False
        )],
        showlegend=False,
        height=360,
        margin=dict(l=10, r=10, t=60, b=20)
    )

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

    # apply 7-day rolling average
    time_series['Smoothed'] = time_series['Issue Count'].rolling(window=7, min_periods=1).mean()

    fig = px.line(
        time_series,
        x='date_reported',
        y='Smoothed',
        title="Issues Reported Over Time"
    )
    fig.update_traces(line=dict(width=2))
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Number of Issues",
        height=350,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    return fig

def create_district_map(filtered_df):
    issue_counts = filtered_df.groupby('district').size().reset_index(name='issue_count')

    with open("C:/Users/gihoz/OneDrive/Desktop/CEPdash/geoBoundaries-RWA-ADM2 (1).geojson", "r", encoding="utf-8") as f:
        rwanda_geojson = json.load(f)

    gdf = gpd.GeoDataFrame.from_features(rwanda_geojson["features"])
    merged = gdf.merge(issue_counts, left_on='shapeName', right_on='district', how='left')
    merged['issue_count'] = merged['issue_count'].fillna(0)

    green_palette = ["#e8f8ec", "#7bed9f", "#2ecc71", "#27ae60", "#145214"]

    fig = px.choropleth_map(
        merged,
        geojson=rwanda_geojson,
        locations="shapeName",
        featureidkey="properties.shapeName",
        color="issue_count",
        color_continuous_scale=green_palette,
        center={"lat": -1.9441, "lon": 30.0619},
        zoom=7.1,
        labels={"issue_count": "Issue Count"},
        hover_data={"shapeName": True, "issue_count": True}
    )

    fig.update_layout(
        mapbox_style="carto-positron",
        title="Density of Reported Issues by District in Kigali, Rwanda",
        title_x=0.5,
        title_font=dict(size=20, family="Lato, sans-serif", color="#144d14"),
        margin={"r": 0, "t": 50, "l": 0, "b": 0},
        coloraxis_colorbar_title="Issue Count"
    )
    return fig

# ---- Dashboard view ----
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
    ], style={'display': 'flex', 'justifyContent': 'space-between', 'gap': '8px', 'marginBottom': '14px', 'flexWrap': 'nowrap'})

    fig = create_dept_bar(filtered_df)
    return html.Div([cards_row, dcc.Graph(figure=fig, style={'height': '60vh'})])

# ---------------- Layout ----------------
app.layout = html.Div([
    dcc.Store(id='current-page', data='dashboard'),

    html.Div(
        html.H1("🏠 CitizenConnect | Officials Dashboard", style={'textAlign': 'center', 'color': '#2c3e50', 'margin': 0}),
        style={'backgroundColor': '#ecf0f1', 'padding': '15px', 'borderBottom': '2px solid #bdc3c7'}
    ),

    html.Div([
        html.Div([
            html.H3("Select Location", style={'marginBottom': '18px'}),

            html.Label("Select Time filter"),
            dcc.Dropdown(
                id='time-filter',
                options=[
                    {'label': 'All', 'value': 'all'},
                    {'label': 'Today', 'value': 'today'},
                    {'label': 'Last 7 Days', 'value': 'last7'}
                ],
                value='all',
                style={'marginBottom': '16px'}
            ),

            html.Label("District"),
            dcc.Dropdown(
                id='district-dropdown',
                options=[{'label': d, 'value': d} for d in sorted(df['district'].dropna().unique())],
                placeholder="Select District",
                style={'marginBottom': '16px'}
            ),

            html.Label("Sector"),
            dcc.Dropdown(id='sector-dropdown', placeholder="Select Sector", style={'marginBottom': '16px'}),

            html.Label("Cell"),
            dcc.Dropdown(id='cell-dropdown', placeholder="Select Cell", style={'marginBottom': '16px'}),

            html.Hr(),
            html.Button('More Insights', id='insights-button', n_clicks=0, style={'width': '100%', 'marginBottom': '10px'}),
            html.Button('Sentiments', id='sentiments-button', n_clicks=0, style={'width': '100%', 'marginBottom': '10px'}),  # <-- added here
            html.Button('Model', id='model-button', n_clicks=0, style={'width': '100%'}),
        ], style={'width': '12%', 'minWidth': '220px', 'padding': '20px', 'backgroundColor': '#f8f9fa', 'borderRight': '2px solid #bdc3c7'}),

        html.Div(id='main-content', style={'width': '88%', 'padding': '20px'}, children=[
            html.Button('⬅ Go Back', id='go-back-button', n_clicks=0, style={'display': 'none', 'marginBottom': '20px'}),
            html.Div(id='page-content', children=[dashboard_view(df.copy())])
        ]),
    ], style={'display': 'flex', 'flexDirection': 'row', 'height': '90vh'})
])

# ---------------- Callbacks ----------------
# --- lightweight cache (in-memory) for heavy figures ---
_MAP_CACHE = {}

def _map_cache_key(filtered_df, time_filter, selected_district, selected_sector, selected_cell):
    # summarize to a small, hashable signature (district counts)
    counts = filtered_df.groupby('district').size()
    sig = tuple(sorted((str(k), int(v)) for k, v in counts.items()))
    # include filters so keys differ when user changes them
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
    Input('sentiments-button', 'n_clicks'),  # <-- added here
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
    if trigger == 'sentiments-button':  # <-- added here
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
    prevent_initial_call=True  # initial content is already in the layout
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

    if current_page == 'sentiments':
        sentiments_layout = html.Div([
            dcc.Graph(figure=create_sentiment_by_level(filtered_df)),
            dcc.Graph(figure=create_avg_rating_by_department(filtered_df)),
            html.H3("Word Cloud of Citizens' Feedback", style={'marginTop': '20px'}),
            create_wordcloud(filtered_df)
        ])
        return sentiments_layout, {'display': 'inline-block', 'marginBottom': '20px'}
    # --- pages ---
    if current_page == 'insights':
        # Light charts (recompute)
        left_fig = create_level_bar(filtered_df)
        right_fig = create_resolution_donut(filtered_df)
        bottom_fig = create_issues_over_time(filtered_df)

        # Heavy map: use cache
        key = _map_cache_key(filtered_df, time_filter, selected_district, selected_sector, selected_cell)
        map_fig = _MAP_CACHE.get(key)
        if map_fig is None:
            map_fig = create_district_map(filtered_df)
            # preserve user zoom/pan and avoid re-draws unless data signature changes
            map_fig.update_layout(uirevision='insights-map')
            _MAP_CACHE[key] = map_fig

        insights_layout = html.Div([
            html.Div([
                html.Div(dcc.Graph(figure=left_fig, config={'displayModeBar': False}), style={
                    'width': '48%', 'boxSizing': 'border-box', 'display': 'inline-block', 'verticalAlign': 'top'
                }),
                html.Div([
                    dcc.Graph(figure=right_fig, config={'displayModeBar': False})
                ], style={'width': '48%', 'boxSizing': 'border-box', 'display': 'inline-block', 'paddingLeft': '12px', 'verticalAlign': 'top'})
            ], style={'display': 'flex', 'flexDirection': 'row', 'justifyContent': 'space-between', 'flexWrap': 'nowrap', 'alignItems': 'flex-start'}),

            html.Div([
                # keep interactivity but make it lighter; uirevision avoids redraw
                dcc.Graph(
                    figure=map_fig,
                    config={'displayModeBar': True, 'scrollZoom': False, 'doubleClick': 'reset', 'modeBarButtonsToRemove': ['lasso2d', 'select2d']}
                )
            ], style={'marginTop': '20px'}),

            html.Div([
                dcc.Graph(figure=bottom_fig, config={'displayModeBar': True})
            ], style={'marginTop': '20px'})
        ], style={'paddingTop': '6px'})

        return insights_layout, {'display': 'inline-block', 'marginBottom': '20px'}

    if current_page == 'sentiments':  # <-- added here
        sentiments_layout = html.Div([
            html.H2("📝 Sentiment Analysis", style={'color': '#2c3e50'}),
            html.P("This section will display citizens' feedback sentiment insights once the visuals are ready.")
        ])
        return sentiments_layout, {'display': 'inline-block', 'marginBottom': '20px'}

    if current_page == 'model':
        model_layout = html.Div([
            html.H2("🤖 Predictive Model", style={'color': '#2c3e50'}),
            html.P("Display model predictions, accuracy, or risk flags here.")
        ])
        return model_layout, {'display': 'inline-block', 'marginBottom': '20px'}

    # dashboard
    return dashboard_view(filtered_df), {'display': 'none'}

if __name__ == '__main__':
    app.run(debug=True)
