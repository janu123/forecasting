import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import dash_table
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
import plotly.graph_objs as go
from datetime import date

# Load and preprocess data
file_path = 'TTL Dashboard Input (1)_Ason Utilization Report.xlsx'
df = pd.read_excel(file_path)

# Data Preprocessing
df = df.dropna()

# Convert 'Month' to datetime format
df['Month'] = pd.to_datetime(df['Month'], format='%b\'%y')

# Set 'Month' as index
df.set_index('Month', inplace=True)

# Initialize the Dash app
app = dash.Dash(__name__)

# Layout of the app
app.layout = html.Div([
    html.H1("Network Utilization Forecast Dashboard", style={'textAlign': 'center'}),
    
    # Input fields for selecting region, circle, town, link
    html.Div([
        html.Label('Select Region:'),
        dcc.Dropdown(
            id='region-dropdown',
            options=[{'label': r, 'value': r} for r in df['Region'].unique()],
            value=df['Region'].unique()[0]
        ),
        
        html.Br(),  # Line break to ensure space between elements
        
        html.Label('Select Circle:'),
        dcc.Dropdown(id='circle-dropdown'),

        html.Br(),  # Line break

        html.Label('Select Town:'),
        dcc.Dropdown(id='town-dropdown'),

        html.Br(),  # Line break

        html.Label('New Link Capacity:'),
        dcc.Input(id='new-link-capacity', type='number', value=0),

        html.Br(),  # Line break for new line

        html.Label('Select Month for New Link:'),
        dcc.DatePickerSingle(
            id='new-link-month',
            display_format='MMM YYYY',
            placeholder='Select a month',
            min_date_allowed=date(2022, 1, 1),  # Minimum allowed date: January 2022
            max_date_allowed=date(2022, 12, 31) # Maximum allowed date: December 2022
        ),

        html.Br(),  # Line break before button

        html.Button('Add New Link & Update Forecast', id='add-link-button', n_clicks=0)
    ], style={'width': '30%', 'display': 'inline-block', 'verticalAlign': 'top'}),
    
    # Output for graph
    html.Div([
        dcc.Graph(id='forecast-plot')
    ], style={'width': '50%', 'display': 'inline-block', 'textAlign': 'center'}),

    # Display the list of links in a table on the right
    html.Div([
        html.Label('List of Links'),
        dash_table.DataTable(
            id='link-table',
            columns=[{'name': 'Link Name', 'id': 'link_name'}, {'name': 'Capacity', 'id': 'capacity'}],
            data=[],
            style_table={'width': '100%'},
            style_cell={'textAlign': 'left'}
        )
    ], style={'width': '20%', 'display': 'inline-block', 'verticalAlign': 'top'})
])

# Callbacks to update dropdowns
@app.callback(
    Output('circle-dropdown', 'options'),
    Output('circle-dropdown', 'value'),
    Input('region-dropdown', 'value')
)
def update_circle_dropdown(selected_region):
    filtered_df = df[df['Region'] == selected_region]
    circle_options = [{'label': c, 'value': c} for c in sorted(filtered_df['Circle'].unique())]
    value = circle_options[0]['value'] if circle_options else None
    return circle_options, value

@app.callback(
    Output('town-dropdown', 'options'),
    Output('town-dropdown', 'value'),
    Input('circle-dropdown', 'value')
)
def update_town_dropdown(selected_circle):
    filtered_df = df[df['Circle'] == selected_circle]
    town_options = [{'label': t, 'value': t} for t in sorted(filtered_df['Town'].unique())]
    value = town_options[0]['value'] if town_options else None
    return town_options, value

# Callback to update the link table for the selected town and add new links dynamically
# Callback to update the link table for the selected town and add new links dynamically
@app.callback(
    Output('link-table', 'data'),
    Input('town-dropdown', 'value'),
    Input('add-link-button', 'n_clicks'),
    Input('new-link-capacity', 'value'),
    State('link-table', 'data')
)
def update_link_table(selected_town, n_clicks, new_link_capacity, current_data):
    filtered_df = df[df['Town'] == selected_town]
    links = [{'link_name': link, 'capacity': ''} for link in filtered_df['Link Name'].unique()]

    # Add new link at the top if button is clicked and capacity > 0
    if n_clicks > 0 and new_link_capacity and new_link_capacity > 0:
        new_link_name = f"Link {len(links) + 1}"
        new_link = {'link_name': new_link_name, 'capacity': new_link_capacity}
        links.insert(0, new_link)  # Insert new link at the top of the list
    
    return links

# Callback to update the forecast plot
@app.callback(
    Output('forecast-plot', 'figure'),
    [Input('add-link-button', 'n_clicks')],
    [Input('region-dropdown', 'value'),
     Input('circle-dropdown', 'value'),
     Input('town-dropdown', 'value'),
     Input('new-link-capacity', 'value'),
     Input('new-link-month', 'date')]
)
def update_forecast(n_clicks, region, circle, town, new_link_capacity, new_link_month):
    # Handle cases where capacity is None (i.e., user clears input)
    if new_link_capacity is None:
        new_link_capacity = 0  # Default capacity to 0 if no input

    # Filter data for the selected town (without filtering by a specific link)
    filtered_df = df[
        (df['Region'] == region) &
        (df['Circle'] == circle) &
        (df['Town'] == town)
    ]

    # Aggregate data for all links in the selected town
    monthly_data = filtered_df.resample('M').agg({
        'Total Resources': 'sum',
        'HO+LO': 'sum'
    })
    monthly_data['Utilization'] = monthly_data['HO+LO'] / monthly_data['Total Resources']

    # Train SARIMA model
    y = monthly_data['Utilization']
    model = SARIMAX(y, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
    model_fit = model.fit(disp=False)

    # Forecast without adding new link
    forecast = model_fit.get_forecast(steps=12)
    forecast_values = forecast.predicted_mean
    forecast_conf_int = forecast.conf_int()

    forecast_df = pd.DataFrame({
        'Utilization': forecast_values,
        'Lower CI': forecast_conf_int.iloc[:, 0],
        'Upper CI': forecast_conf_int.iloc[:, 1]
    }, index=pd.date_range(start=y.index[-1] + pd.DateOffset(months=1), periods=12, freq='M'))

    # If new link is added with valid capacity and month
    if n_clicks > 0 and new_link_capacity > 0:
        # Convert the selected date to a pandas datetime object, or use the first forecasted month if no month is selected
        if new_link_month:
            new_link_month = pd.to_datetime(new_link_month)
        else:
            new_link_month = forecast_df.index[0]  # Default to the first forecasted month

        # Modify forecast starting from the selected month
        for i, forecast_month in enumerate(forecast_df.index):
            if forecast_month >= new_link_month:
                total_resources_with_new_link = monthly_data['Total Resources'].iloc[-1] + new_link_capacity
                forecast_df.loc[forecast_month:, 'Utilization'] += new_link_capacity / total_resources_with_new_link
                break

    # Create Plotly figure
# Create Plotly figure
    fig = go.Figure()

# Historical data line
    fig.add_trace(go.Scatter(
        x=monthly_data.index, 
        y=monthly_data['Utilization'],
        mode='lines+markers',  # Add markers to historical data
        name='Historical Utilization', 
        line=dict(color='blue'),
        marker=dict(symbol='circle', size=8)  # Marker style for historical data
    ))

# Forecasted data line
    fig.add_trace(go.Scatter(
        x=forecast_df.index, 
        y=forecast_df['Utilization'],
        mode='lines+markers',  # Add markers to forecasted data
        name='Forecasted Utilization', 
        line=dict(color='orange'),
        marker=dict(
        color=['red' if val > 0.7 else 'green' for val in forecast_df['Utilization']],  # Points above threshold are red
        symbol='circle',
        size=8
        )
    ))

# Add threshold line for 70% (0.7) utilization
    fig.add_shape(
        type='line', 
        x0=forecast_df.index[0], 
        x1=forecast_df.index[-1], 
        y0=0.7, 
        y1=0.7,
        line=dict(color='green', dash='dash'), 
        name='Threshold 70%'
    )

# Add confidence intervals as a filled area
    fig.add_trace(go.Scatter(
        x=forecast_df.index, 
        y=forecast_df['Upper CI'], 
        mode='lines',
        line=dict(width=0), 
        showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=forecast_df.index, 
        y=forecast_df['Lower CI'], 
        mode='lines',
        fill='tonexty', 
        line=dict(width=0), 
        showlegend=False, 
        fillcolor='rgba(255,165,0,0.2)'  # Confidence interval fill color
    ))

# Customize layout
    fig.update_layout(
        title='Utilization Forecast with Updated Link Capacity and Threshold Line',
        xaxis_title='Date',
        yaxis_title='Utilization',
        yaxis=dict(range=[0, 1])  # Ensure the y-axis range is between 0 and 1 for threshold line visibility
    )

    return fig

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
