import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import pandas as pd
import plotly.graph_objects as go
from statsmodels.tsa.arima.model import ARIMA
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Load your dataset from an Excel file
file_path = 'Combined_dataset.xlsx'
logging.info(f'Loading dataset from {file_path}')
try:
    df = pd.read_excel(file_path)
except Exception as e:
    logging.error(f'Error loading dataset: {e}')
    raise

# Ensure correct data types
df['Circle'] = df['Circle'].astype(str).str.strip()
df['Month'] = df['Month'].astype(str).str.strip()

# Handle 'Utilization' column, converting from percentage strings to numeric
df['Utilization'] = df['Utilization'].replace('%', '', regex=True).astype(float)

# Drop rows with missing values in relevant columns
df = df.dropna(subset=['Circle', 'Month', 'Utilization'])

# Initialize Dash app
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1('Forecasting-ARIMA', style={'textAlign': 'center', 'margin-bottom': '20px'}),

    html.Div([
        html.Div([
            html.Label('Select Circle:', style={'margin-right': '20px', 'white-space': 'nowrap'}),
            dcc.Dropdown(
                id='circle-dropdown',
                options=[{'label': circle, 'value': circle} for circle in sorted(df['Circle'].unique())],
                value=df['Circle'].unique()[0],
                clearable=False,
                style={'width': '100%'},
                placeholder='Select Circle'
            )
        ], style={'flex': 1, 'margin-right': '10px'}),

        html.Div([
            html.Label('Select Link Name:', style={'margin-right': '20px', 'white-space': 'nowrap'}),
            dcc.Dropdown(
                id='link-dropdown',
                options=[],
                value=None,
                clearable=False,
                style={'width': '100%'},
                placeholder='Select Link Name'
            )
        ], style={'flex': 1}),
    ], style={'display': 'flex', 'align-items': 'center', 'margin-bottom': '30px', 'width': '100%'}),

    html.P('Utilization Forecast Analysis', style={'textAlign': 'center', 'margin-bottom': '0px'}),

    dcc.Graph(id='line-chart')
])

@app.callback(
    Output('link-dropdown', 'options'),
    Output('link-dropdown', 'value'),
    Input('circle-dropdown', 'value')
)
def update_link_dropdown(selected_circle):
    try:
        filtered_df = df[df['Circle'] == selected_circle]
        link_options = [{'label': link, 'value': link} for link in sorted(filtered_df['Link Name'].unique())]
        default_value = link_options[0]['value'] if link_options else None
        return link_options, default_value
    except Exception as e:
        logging.error(f'Error updating link dropdown: {e}')
        return [], None

@app.callback(
    Output('line-chart', 'figure'),
    [Input('circle-dropdown', 'value'),
     Input('link-dropdown', 'value')]
)
def update_line_chart(selected_circle, selected_link):
    try:
        filtered_df = df
        if selected_circle:
            filtered_df = filtered_df[filtered_df['Circle'] == selected_circle]
        if selected_link:
            filtered_df = filtered_df[filtered_df['Link Name'] == selected_link]

        # Aggregate historical data by Month and Link Name
        aggregated_df = filtered_df.groupby(['Month', 'Link Name'], as_index=False).agg({
            'Utilization': 'mean'
        })

        # Convert Month to datetime format explicitly
        aggregated_df['Month'] = pd.to_datetime(aggregated_df['Month'], format="%b'%y", errors='coerce')

        # Drop rows where conversion failed
        aggregated_df = aggregated_df.dropna(subset=['Month'])

        # Sort data by Month
        aggregated_df = aggregated_df.sort_values(by='Month')

        # Set Month as the index
        aggregated_df.set_index('Month', inplace=True)

        # Use the ARIMA model for forecasting
        model = ARIMA(aggregated_df['Utilization'], order=(5, 1, 0))  # (p, d, q) order, this can be tuned
        model_fit = model.fit()

        # Define the number of months to forecast (until December 2022)
        last_date = aggregated_df.index[-1]
        future_months = pd.date_range(start=last_date, end='2022-12-01', freq='MS')
        future_forecast = model_fit.forecast(steps=len(future_months))

        # Create DataFrame for future predictions
        future_df_utilized = pd.DataFrame({
            'Month': future_months,
            'Utilization': future_forecast
        })

        # Format the dates as "MMM'YY" (e.g., "Jan'25")
        aggregated_df.index = aggregated_df.index.strftime("%b'%y")
        future_df_utilized['Month'] = future_df_utilized['Month'].dt.strftime("%b'%y")

        # Create the figure with separate traces for historical and forecast data
        fig = go.Figure()

        # Add historical data
        fig.add_trace(go.Scatter(x=aggregated_df.index, y=aggregated_df['Utilization'],
                                 mode='lines+markers',
                                 name='Historical Utilization',
                                 line=dict(color='blue'),
                                 marker=dict(color='blue')))

        # Add forecast data
        fig.add_trace(go.Scatter(x=future_df_utilized['Month'], y=future_df_utilized['Utilization'],
                                 mode='lines+markers',
                                 name='Forecast Utilization',
                                 line=dict(color='red', dash='dash'),
                                 marker=dict(color='red')))

        return fig

    except Exception as e:
        logging.error(f'Error updating line chart: {e}')
        return go.Figure()

if __name__ == '__main__':
    logging.info('Starting Dash app')
    app.run_server(debug=True)
