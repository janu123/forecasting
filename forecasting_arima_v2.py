import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import pandas as pd
import plotly.graph_objects as go
from statsmodels.tsa.arima.model import ARIMA
import logging
import numpy as np

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Load your dataset from an Excel file
file_path = 'TTL Dashboard Input (1)_Ason Utilization Report.xlsx'
logging.info(f'Loading dataset from {file_path}')
try:
    df = pd.read_excel(file_path)
except Exception as e:
    logging.error(f'Error loading dataset: {e}')
    raise
# Step 2: Handle NaN and Inf values (Add the code here)
# Step 2: Check for numerical data types and handle conversions (Add the code here)
# Preprocessing
# Preprocessing
logging.info("Initial DataFrame Info:")
logging.info(df.info())

# Fill missing values for categorical columns with 'Unknown'
logging.info("Filling missing values for categorical columns")
categorical_cols = ['Region', 'Circle', 'Town', 'Month', 'Link Name']
for col in categorical_cols:
    if col in df.columns:
        df[col] = df[col].astype(str).fillna('Unknown').str.strip()

logging.info("Converting and handling numeric columns")
numerical_cols = ['Utilization','HO+LO','Total Resources']
for col in numerical_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Handle Utilization column
#df['Utilization'] = df['Utilization'].replace('%', '', regex=True).astype(float)
df.fillna(df.mean(numeric_only=True), inplace=True)  # Handle NaN values in numeric columns

logging.info("DataFrame after Preprocessing:")
logging.info(df.head())

# Ensure correct data types
df['Region'] = df['Region'].astype(str).str.strip()
df['Circle'] = df['Circle'].astype(str).str.strip()
df['Town'] = df['Town'].astype(str).str.strip()
df['Link Name'] = df['Link Name'].astype(str).str.strip()
df['Month'] = df['Month'].astype(str).str.strip()

# Handle 'Utilization' column, converting from percentage strings to numeric
#df['Utilization'] = df['Utilization'].replace('%', '', regex=True).astype(float)
df['Utilization']=df["HO+LO"]/df["Total Resources"]
# Drop rows with missing values in relevant columns
df = df.dropna(subset=['Region', 'Circle', 'Town', 'Link Name', 'Month', 'Utilization'])

# Initialize Dash app
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1('Forecasting-ARIMA with Rolling Statistics', style={'textAlign': 'center', 'margin-bottom': '20px'}),

    html.Div([
        html.Div([
            html.Label('Select Region:', style={'margin-right': '20px', 'white-space': 'nowrap'}),
            dcc.Dropdown(
                id='region-dropdown',
                options=[{'label': region, 'value': region} for region in sorted(df['Region'].unique())],
                value=df['Region'].unique()[0],
                clearable=False,
                style={'width': '100%'},
                placeholder='Select Region'
            )
        ], style={'flex': 1, 'margin-right': '10px'}),

        html.Div([
            html.Label('Select Circle:', style={'margin-right': '20px', 'white-space': 'nowrap'}),
            dcc.Dropdown(
                id='circle-dropdown',
                options=[],
                value=None,
                clearable=False,
                style={'width': '100%'},
                placeholder='Select Circle'
            )
        ], style={'flex': 1}),
        
        html.Div([
            html.Label('Select Town:', style={'margin-right': '20px', 'white-space': 'nowrap'}),
            dcc.Dropdown(
                id='town-dropdown',
                options=[],
                value=None,
                clearable=False,
                style={'width': '100%'},
                placeholder='Select Town'
            )
        ], style={'flex': 1, 'margin-left': '10px'}),

    ], style={'display': 'flex', 'align-items': 'center', 'margin-bottom': '30px', 'width': '100%'}),

    html.Div([
        html.Label('Select Link Name(s):', style={'margin-right': '20px', 'white-space': 'nowrap'}),
        dcc.Dropdown(
            id='link-dropdown',
            options=[],  # Empty initially
            value=[],    # No default selection
            multi=True,
            placeholder='Select Link Names'  # Placeholder text
        )
    ], style={'margin-bottom': '30px', 'width': '100%'}),

    html.P('Utilization Forecast Analysis with Rolling Mean & Std Dev', style={'textAlign': 'center', 'margin-bottom': '0px'}),

    dcc.Graph(id='line-chart')
])

@app.callback(
    Output('circle-dropdown', 'options'),
    Output('circle-dropdown', 'value'),
    Input('region-dropdown', 'value')
)
def update_circle_dropdown(selected_region):
    logging.debug(f"Updating circle dropdown with selected region: {selected_region}")
    try:
        filtered_df = df[df['Region'] == selected_region]
        circle_options = [{'label': circle, 'value': circle} for circle in sorted(filtered_df['Circle'].unique())]
        logging.debug(f"Circle options: {circle_options}")
        return circle_options, None  # Initially, no circle selected
    except Exception as e:
        logging.error(f'Error updating circle dropdown: {e}')
        return [], None

@app.callback(
    Output('town-dropdown', 'options'),
    Output('town-dropdown', 'value'),
    Input('circle-dropdown', 'value')
)
def update_town_dropdown(selected_circle):
    logging.debug(f"Updating town dropdown with selected circle: {selected_circle}")
    try:
        if selected_circle:
            filtered_df = df[df['Circle'] == selected_circle]
            town_options = [{'label': town, 'value': town} for town in sorted(filtered_df['Town'].unique())]
        else:
            town_options = []
        logging.debug(f"Town options: {town_options}")
        return town_options, None  # Initially, no town selected
    except Exception as e:
        logging.error(f'Error updating town dropdown: {e}')
        return [], None

@app.callback(
    Output('link-dropdown', 'options'),
    Output('link-dropdown', 'value'),
    Input('town-dropdown', 'value')
)
def update_link_dropdown(selected_town):
    logging.debug(f"Updating link dropdown with selected town: {selected_town}")
    try:
        if selected_town:
            filtered_df = df[df['Town'] == selected_town]
            link_options = [{'label': link, 'value': link} for link in sorted(filtered_df['Link Name'].unique())]
        else:
            link_options = []
        logging.debug(f"Link options: {link_options}")
        return link_options, []  # Empty selection for multiple links
    except Exception as e:
        logging.error(f'Error updating link dropdown: {e}')
        return [], []

@app.callback(
    Output('line-chart', 'figure'),
    [Input('region-dropdown', 'value'),
     Input('circle-dropdown', 'value'),
     Input('town-dropdown', 'value'),
     Input('link-dropdown', 'value')]
)
def update_line_chart(selected_region, selected_circle, selected_town, selected_links):
    try:
        # Initial filter based on selected region
        filtered_df = df[df['Region'] == selected_region]

        # Apply additional filters if a circle and/or town is selected
        if selected_circle:
            filtered_df = filtered_df[filtered_df['Circle'] == selected_circle]
        if selected_town:
            filtered_df = filtered_df[filtered_df['Town'] == selected_town]
        if selected_links:
            filtered_df = filtered_df[filtered_df['Link Name'].isin(selected_links)]

        # Handle the case if no data is left after filtering
        if filtered_df.empty:
            return go.Figure()  # Return an empty figure if there's no data

        # Aggregate and process data as per your logic
        aggregated_df = filtered_df.groupby('Month').agg({'Utilization': 'mean'}).reset_index()

        # Convert and clean the 'Month' data
        aggregated_df['Month'] = pd.to_datetime(aggregated_df['Month'], format="%b'%y", errors='coerce')
        aggregated_df = aggregated_df.dropna(subset=['Month']).sort_values(by='Month')
        aggregated_df.set_index('Month', inplace=True)

        # Rolling mean and std
        rolling_mean = aggregated_df['Utilization'].rolling(window=3).mean()
        rolling_std = aggregated_df['Utilization'].rolling(window=3).std()

        # ARIMA model for forecasting
        model = ARIMA(aggregated_df['Utilization'], order=(5, 1, 0))
        model_fit = model.fit()

        # Forecast future values
        last_date = aggregated_df.index[-1]
        future_months = pd.date_range(start=last_date, end='2022-12-01', freq='MS')
        future_forecast = model_fit.forecast(steps=len(future_months))

        # Prepare future forecast data
        future_df_utilized = pd.DataFrame({
            'Month': future_months,
            'Utilization': future_forecast
        })

        # Combine for rolling stats
        combined_utilization = pd.concat([aggregated_df['Utilization'], future_df_utilized.set_index('Month')['Utilization']])
        rolling_mean_combined = combined_utilization.rolling(window=3).mean()
        rolling_std_combined = combined_utilization.rolling(window=3).std()

        # Format dates for plotting
        aggregated_df.index = aggregated_df.index.strftime("%b'%y")
        future_df_utilized['Month'] = future_df_utilized['Month'].dt.strftime("%b'%y")

        # Plotting the figure
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=aggregated_df.index,
            y=aggregated_df['Utilization'],
            mode='lines+markers',
            name='Historical Utilization',
            line=dict(color='blue')
        ))

        fig.add_trace(go.Scatter(
            x=future_df_utilized['Month'],
            y=future_df_utilized['Utilization'],
            mode='lines+markers',
            name='Forecast Utilization',
            line=dict(color='red', dash='dash'),
            error_y=dict(type='data', array=rolling_std_combined[-len(future_months):], visible=True),
        ))

        fig.add_trace(go.Scatter(
            x=aggregated_df.index,
            y=rolling_mean,
            mode='lines',
            name='Rolling Mean',
            line=dict(color='green', width=2)
        ))

        fig.add_trace(go.Scatter(
            x=aggregated_df.index,
            y=rolling_std,
            mode='lines',
            name='Rolling Std Dev',
            line=dict(color='orange', width=2),
            yaxis='y2'
        ))

        # Adding the rolling mean and std dev for the forecast period
        fig.add_trace(go.Scatter(
            x=future_df_utilized['Month'],
            y=rolling_mean_combined[-len(future_months):],
            mode='lines',
            name='Forecast Rolling Mean',
            line=dict(color='green', dash='dash', width=2)
        ))

        fig.add_trace(go.Scatter(
            x=future_df_utilized['Month'],
            y=rolling_std_combined[-len(future_months):],
            mode='lines',
            name='Forecast Rolling Std Dev',
            line=dict(color='orange', dash='dash', width=2),
            yaxis='y2'
        ))

        # Update layout with secondary y-axis
        fig.update_layout(
            title='Utilization Forecast with Rolling Mean and Std Dev',
            xaxis_title='Month',
            yaxis_title='Utilization (%)',
            yaxis2=dict(
                title='Rolling Std Dev',
                overlaying='y',
                side='right',
                showgrid=False
            ),
            legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0)', bordercolor='rgba(255,255,255,0)')
        )

        return fig

    except Exception as e:
        logging.error(f'Error updating line chart: {e}')
        return go.Figure()  # Return an empty figure in case of error
# Run the Dash app
if __name__ == '__main__':
    app.run( port=8052, debug=True, threaded=True)
