import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import dash_table
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
import plotly.graph_objs as go
import numpy as np

#from pmdarima import auto_arima

# Load and preprocess data
file_path = 'updated_link_data - Copy - Copy.xlsx'
df = pd.read_excel(file_path)

# Data Preprocessing
df = df.dropna()

# Convert 'Month' to datetime format
df['Month'] = pd.to_datetime(df['Month'], format='%b\'%y')

# Set 'Month' as index
df.set_index('Month', inplace=True)

# Initialize the Dash app
app = dash.Dash(__name__)

# Extract unique regions from the dataframe for the dropdown
region_options = [{'label': region, 'value': region} for region in df['Region'].unique()]
# Layout of the app with filters, month dropdown, and inputs for adjustments
app.layout = html.Div([
    html.H1("Network Utilization Forecast Dashboard", style={'textAlign': 'center'}),

    # Filters for Region, Circle, Town, Links, and Priority in a horizontal layout
    html.Div([
        html.Div([
            html.Label('Select Region:'),
            dcc.Dropdown(
                id='region-dropdown',
                options=region_options,  # Set options dynamically here
                placeholder="All",
                multi=True  # Enable multiselect
            )
        ], style={'width': '19%', 'display': 'inline-block', 'padding': '0 10px','fontSize': '12px'}),

        html.Div([
            html.Label('Select Circle:'),
            dcc.Dropdown(id='circle-dropdown', placeholder="All", multi=True)  # Enable multiselect
        ], style={'width': '19%', 'display': 'inline-block', 'padding': '0 10px','fontSize': '12px'}),

        html.Div([
            html.Label('Select Town:'),
            dcc.Dropdown(id='town-dropdown', placeholder="All", multi=True)  # Enable multiselect
        ], style={'width': '19%', 'display': 'inline-block', 'padding': '0 10px','fontSize': '12px'}),

        html.Div([
            html.Label('Select Link(s):'),
            dcc.Dropdown(id='link-dropdown', placeholder="All", multi=True)  # Already multiselect
        ], style={'width': '19%', 'display': 'inline-block', 'padding': '0 10px','fontSize': '12px'}),

        html.Div([
            html.Label('Select Priority:'),
            dcc.Dropdown(id='priority-dropdown', placeholder="All", multi=True)  # Enable multiselect
        ], style={'width': '19%', 'display': 'inline-block', 'padding': '0 10px','fontSize': '12px'}),
    ], style={'width': '100%', 'display': 'flex', 'justify-content': 'space-between'}),  # Flex display for row layout

    # Section for Month, Utilized, and Capacity Input
    html.Div([
        html.Div([
            html.Label('Select Month:',style={'fontSize': '12px'}),
            dcc.Dropdown(id='month-dropdown', placeholder='Select a month to adjust', style={'fontSize': '12px', 'width': '75%'}),
            
            html.Label('Increase BW Used (Utilized):'),
            dcc.Input(
                id='utilized-input',
                type='number',
                placeholder='Enter new BW used value',
                value=None,
                style={'width': '15%'}
            ),
            
            html.Label('Increase Capacity (Total Resources):'),
            dcc.Input(
                id='capacity-input',
                type='number',
                placeholder='Enter new Total Resources value',
                value=None,
                style={'width': '15%'}
            ),
            
            html.Label('Unit Rate for 10G (in Rs):'),
            dcc.Input(
                id='unit-rate-input',
                type='number',
                placeholder='Enter unit rate for 10G',
                value=None,
                style={'width': '15%'}
            ),
            
            html.Button('Apply', id='apply-button', n_clicks=0, style={'margin-left': '10px'})
        ], style={'display': 'flex', 'align-items': 'center', 'gap': '10px', 'width': '100%'})
    ], style={'padding': '10px'}),

    # Display for Recommended Capacity and Total Cost
   # Display for Recommended Capacity and Total Cost in a single line
        html.Div([
    # Recommended Additional Capacity Section in a single line
            html.Div([
                html.Label('Recommended Additional Capacity (in Mbps):', 
                        style={'fontWeight': 'bold', 'fontSize': '16px', 'color': '#333'}),
                html.Div(id='recommended-capacity-output', 
                        style={'fontWeight': 'bold', 'fontSize': '16px', 'color': 'white', 'backgroundColor': '#1E90FF',
                                'display': 'inline-block', 'padding': '5px 10px', 'borderRadius': '5px', 'marginLeft': '10px'}),
            ], style={'display': 'inline-block', 'marginRight': '50px'}),  # marginRight creates space between sections

            # Total Cost Section in a single line
            html.Div([
                html.Label('Total Cost for Required Capacity (in Rs):', 
                        style={'fontWeight': 'bold', 'fontSize': '16px', 'color': '#333'}),
                html.Div(id='total-cost-output', 
                        style={'fontWeight': 'bold', 'fontSize': '16px', 'color': 'white', 'backgroundColor': '#FF4500',
                                'display': 'inline-block', 'padding': '5px 10px', 'borderRadius': '5px', 'marginLeft': '10px'}),
            ], style={'display': 'inline-block'})
        ], style={'padding': '10px', 'width': '100%'}),

    # Two vertically stacked graphs and the table beside them
    html.Div([
        html.Div([dcc.Graph(id='utilized-forecast-plot')], 
                 style={'width': '49%', 'display': 'inline-block', 'padding': '0 10px', 'verticalAlign': 'top'}),
        html.Div([dcc.Graph(id='forecast-plot')], 
                 style={'width': '49%', 'display': 'inline-block', 'padding': '0 10px', 'verticalAlign': 'top'})
    ], style={'width': '100%', 'display': 'flex', 'justify-content': 'space-between'}),

    # Separate div for the table of links
    html.Div([
    # Div to display dynamic count of links above 80% utilization
    html.Label(id='link-count', style={'fontWeight': 'bold', 'fontSize': '16px'}),  # Dynamic count label

    # Radio buttons for selecting between actual and forecasted utilization tables
html.Div([
    html.Label('Select Table:'),
    dcc.RadioItems(
        id='table-toggle',
        options=[
            {'label': 'Actual Utilization', 'value': 'actual'},
            {'label': 'Forecasted Utilization', 'value': 'forecasted'}
        ],
        value='actual',  # Default value
        labelStyle={'display': 'inline-block', 'margin-right': '10px'}
    ),
], style={'padding': '10px', 'fontSize': '16px'}),

# Table for listing links (either actual or forecasted, based on the radio button selection)
html.Div(id='actual-table-container', children=[
    dash_table.DataTable(
        id='link-table',
        columns=[
            {'name': 'Link Name', 'id': 'Link Name'},
            {'name': 'Region', 'id': 'Region'},
            {'name': 'Circle', 'id': 'Circle'},
            {'name': 'Month', 'id': 'Month'},
            {'name': 'Utilization', 'id': 'Utilization'}
        ],
        data=[],  # Will be dynamically updated by the callback
        row_selectable='single',  # Enable single-row selection
        selected_rows=[],  # Keep same row selection index
        style_cell={'textAlign': 'left'},  # Align text to the left
        style_cell_conditional=[
            {
                'if': {'column_id': 'row_id'},
                'textAlign': 'center'
            },
            {
                'if': {'column_id': 'Link Name'},
                'whiteSpace': 'normal',  # Allow text to wrap for long names
                'textAlign': 'left',
                'maxWidth': '150px',  # Set max width for long text
                'overflow': 'hidden',  # Hide overflowing text
                'textOverflow': 'ellipsis',
            },
            {
                'if': {'column_id': 'Region'},
                'textAlign': 'left',
                'maxWidth': '100px',
            },
            {
                'if': {'column_id': 'Circle'},
                'textAlign': 'left',
                'maxWidth': '100px',
            },
            {
                'if': {'column_id': 'Month'},
                'textAlign': 'center',
                'maxWidth': '80px',
            },
            {
                'if': {'column_id': 'Utilization'},
                'textAlign': 'right',
                'maxWidth': '80px',
            }
        ]
    )
]),

html.Div(id='forecast-table-container', children=[
    dash_table.DataTable(
        id='forecast-table',
        columns=[
            {'name': 'Link Name', 'id': 'Link Name'},
            {'name': 'Region', 'id': 'Region'},
            {'name': 'Circle', 'id': 'Circle'},
            {'name': 'Month', 'id': 'Month'},
            {'name': 'Forecasted Utilization', 'id': 'Utilization'}
        ],
        data=[],  # Will be dynamically updated by the callback
        row_selectable='single',
        selected_rows=[],
        style_cell={'textAlign': 'left'},
        style_cell_conditional=[
            {
                'if': {'column_id': 'row_id'},
                'textAlign': 'center'
            },
            {
                'if': {'column_id': 'Link Name'},
                'whiteSpace': 'normal',
                'textAlign': 'left',
                'maxWidth': '150px',
                'overflow': 'hidden',
                'textOverflow': 'ellipsis',
            },
            {
                'if': {'column_id': 'Region'},
                'textAlign': 'left',
                'maxWidth': '100px',
            },
            {
                'if': {'column_id': 'Circle'},
                'textAlign': 'left',
                'maxWidth': '100px',
            },
            {
                'if': {'column_id': 'Month'},
                'textAlign': 'center',
                'maxWidth': '80px',
            },
            {
                'if': {'column_id': 'Utilization'},
                'textAlign': 'right',
                'maxWidth': '80px',
            }
        ]
    )
], style={'display': 'none'})  # Hidden by default
], style={'padding': '0 10px', 'margin-top': '20px'})

])
@app.callback(
    Output('link-table', 'data', allow_duplicate=True),  # Keep allow_duplicate here
    [Input('table-toggle', 'value')],
    [State('link-table', 'data'), State('forecast-table', 'data')],
    prevent_initial_call=True  # Add this to prevent the callback from running on initial load
)
def toggle_table(selected_table, link_table_data, forecast_table_data):
    if selected_table == 'actual':
        return link_table_data  # Return actual utilization table data
    else:
        return forecast_table_data  # Return forecasted utilization table data
    
@app.callback(
    [Output('circle-dropdown', 'options'),
     Output('circle-dropdown', 'value'),
     Output('town-dropdown', 'options'),
     Output('town-dropdown', 'value'),
     Output('link-dropdown', 'options'),
     Output('link-dropdown', 'value'),
     Output('priority-dropdown', 'options'),
     Output('priority-dropdown', 'value')],
    [Input('region-dropdown', 'value'),
     Input('circle-dropdown', 'value'),
     Input('town-dropdown', 'value')]
)
def update_filters(selected_regions, selected_circles, selected_towns):
    # Region Filter
    region_df = df
    if selected_regions:
        region_df = region_df[region_df['Region'].isin(selected_regions)]  # Use isin for multiple values

    # Circle Filter (based on selected regions)
    circle_options = [{'label': circle, 'value': circle} for circle in region_df['Circle'].unique()]
    circle_value = selected_circles if selected_circles and all(circle in region_df['Circle'].unique() for circle in selected_circles) else None

    # Town Filter (based on selected circles)
    circle_df = region_df
    if selected_circles:
        circle_df = circle_df[circle_df['Circle'].isin(selected_circles)]
    town_options = [{'label': town, 'value': town} for town in circle_df['Town'].unique()]
    town_value = selected_towns if selected_towns and all(town in circle_df['Town'].unique() for town in selected_towns) else None

    # Link Filter (based on selected towns)
    town_df = circle_df
    if selected_towns:
        town_df = town_df[town_df['Town'].isin(selected_towns)]
    link_options = [{'label': link, 'value': link} for link in town_df['Link Name'].unique()]
    link_value = None  # No default value for link selection

    # Priority Filter
    priority_options = [{'label': priority, 'value': priority} for priority in df['Priority'].unique()]
    priority_value = None

    return circle_options, circle_value, town_options, town_value, link_options, link_value, priority_options, priority_value


@app.callback(
    [Output('link-table', 'data'),  # Actual high utilization table
     Output('forecast-table', 'data'),  # Forecasted utilization table
     Output('forecast-plot', 'figure'),
     Output('utilized-forecast-plot', 'figure'),
     Output('month-dropdown', 'options'),
     Output('month-dropdown', 'value'),
     Output('link-count', 'children'),
     Output('recommended-capacity-output', 'children'),
     Output('total-cost-output', 'children')],
    [Input('region-dropdown', 'value'),
     Input('circle-dropdown', 'value'),
     Input('town-dropdown', 'value'),
     Input('link-dropdown', 'value'),
     Input('priority-dropdown', 'value'),
     Input('link-table', 'derived_virtual_selected_rows'),
     Input('apply-button', 'n_clicks')],
    [State('utilized-input', 'value'),
     State('capacity-input', 'value'),
     State('month-dropdown', 'value'),
     State('link-table', 'data'),
     State('unit-rate-input', 'value')]
)
def update_forecast_and_table(region, circle, town, links, priority, selected_rows, n_clicks, new_utilized_value, new_capacity_value, selected_month, table_data, unit_rate):
    lower_threshold = 40
    upper_threshold = 80

    # Filter dataset based on selected values
    filtered_df = df

    if region:
        filtered_df = filtered_df[filtered_df['Region'].isin(region)]
    
    if circle:
        filtered_df = filtered_df[filtered_df['Circle'].isin(circle)]
    
    if town:
        filtered_df = filtered_df[filtered_df['Town'].isin(town)]

    if links:
        filtered_df = filtered_df[filtered_df['Link Name'].isin(links)]

    if priority:
        filtered_df = filtered_df[filtered_df['Priority'].isin(priority)]

    # Handle row selection from the table (selected_rows gives indices of selected rows)
    if selected_rows is not None and len(selected_rows) > 0:
        selected_link = table_data[selected_rows[0]]['Link Name']  # Get the link name from the selected row
        filtered_df = filtered_df[filtered_df['Link Name'] == selected_link]  # Filter data for selected link

    # Aggregate data for utilization and total resources
    # Aggregate data for utilization and total resources
    monthly_data = filtered_df.resample('ME').agg({
    'Total Resources': 'mean',
    'Utilized': 'mean',
    'Utilization': 'mean',
    'Link Name': 'first',  # Retain the first Link Name for each group
    'Region': 'first',  # Retain the first Region for each group
    'Circle': 'first'   # Retain the first Circle for each group
})
    
    #monthly_data['Utilization'] = (monthly_data['Utilized'] / monthly_data['Total Resources'])*100
    # Reset index to bring 'Month' back into the dataframe for filtering purposes
    filtered_df = filtered_df.reset_index()

        # Generate forecast for Total Resources (HO)
    y_total_resources = monthly_data['Total Resources']
    model_total_resources = SARIMAX(y_total_resources, order=(2, 1, 2), seasonal_order=(1, 1, 1, 12))
    model_total_resources_fit = model_total_resources.fit(disp=False)

    # Get forecast values for Total Resources HO
    forecast_total_resources = model_total_resources_fit.get_forecast(steps=12)
    forecast_values_total_resources = forecast_total_resources.predicted_mean

    # Create DataFrame for forecasted Total Resources data (without noise)
    forecast_df_total_resources = pd.DataFrame({
        'Total Resources': forecast_values_total_resources,
    })

    # Adjust the forecast index to start from the 1st day of the next month after historical data
    forecast_start_date = (y_total_resources.index[-1] + pd.DateOffset(days=1)).replace(day=1)
    forecast_df_total_resources.index = pd.date_range(start=forecast_start_date, periods=len(forecast_df_total_resources), freq='MS')


    # Generate forecast for utilization
    y_utilization = monthly_data['Utilization']
    model_utilization = SARIMAX(y_utilization, order=(2, 1, 2), seasonal_order=(1, 1, 1, 12))
    model_utilization_fit = model_utilization.fit(disp=False)

    # Get forecast values for Utilization (without confidence intervals)
    forecast_utilization = model_utilization_fit.get_forecast(steps=12)
    forecast_values_utilization = forecast_utilization.predicted_mean

    # Create DataFrame for forecasted data (without noise)
    forecast_df_utilization = pd.DataFrame({
        'Utilization': forecast_values_utilization,
    })

    # Adjust the forecast index to start from the 1st day of the next month after historical data
    forecast_start_date = (y_utilization.index[-1] + pd.DateOffset(days=1)).replace(day=1)
    forecast_df_utilization.index = pd.date_range(start=forecast_start_date, periods=len(forecast_df_utilization), freq='MS')

        # Generate forecast for each link
    forecast_list = []
    for link_name, link_group in monthly_data.groupby('Link Name'):
        # Forecasting utilization per link
        y_utilization = link_group['Utilization']
        model_utilization = SARIMAX(y_utilization, order=(2, 1, 2), seasonal_order=(1, 1, 1, 12))
        model_utilization_fit = model_utilization.fit(disp=False)

        # Get forecast values for Utilization
        forecast_utilization = model_utilization_fit.get_forecast(steps=12)
        forecast_values_utilization = forecast_utilization.predicted_mean

        # Create forecast dataframe
        forecast_df = pd.DataFrame({
            'Link Name': link_name,
            'Region': link_group['Region'].iloc[0],  # Assuming 'Region' is consistent within a link
            'Circle': link_group['Circle'].iloc[0],  # Assuming 'Circle' is consistent within a link
            'Month': pd.date_range(start=y_utilization.index[-1] + pd.DateOffset(months=1), periods=12, freq='MS'),
            'Utilization': forecast_values_utilization
        })

        # Append forecast for this link to the forecast list
        forecast_list.append(forecast_df)

    # Combine forecasts for all links into one DataFrame
    forecast_df_combined = pd.concat(forecast_list)

    # Filter forecasted utilization values >= 80%
    forecast_high_utilization_df = forecast_df_combined[forecast_df_combined['Utilization'] >= 80]
    forecast_high_utilization_df = forecast_high_utilization_df.sort_values(by=['Link Name', 'Utilization'], ascending=[True, False])

    # Prepare forecasted table data
    forecast_high_utilization_df['Month'] = forecast_high_utilization_df['Month'].dt.strftime('%b %Y')
    forecasted_table_data = forecast_high_utilization_df[['Link Name', 'Region', 'Circle', 'Month', 'Utilization']].to_dict('records')

        # Prepare actual high utilization table (for historical data)
    high_utilization_df = filtered_df[filtered_df['Utilization'] >= 80]

    if high_utilization_df.empty:
        # If no data is above the threshold, return empty table data
        latest_3_months_high_utilization_table_data = []
        link_count_text = "No links with utilization ≥ 80%"
    else:
        # If there is data above the threshold, process as usual
        high_utilization_df = high_utilization_df.sort_values(by=['Link Name', 'Month'], ascending=[True, False])
        latest_3_months_high_utilization_df = high_utilization_df.groupby('Link Name').head(3)
        latest_3_months_high_utilization_df['Month'] = latest_3_months_high_utilization_df['Month'].dt.strftime('%b %Y')
        
        # Include Region and Circle columns in the table data
        latest_3_months_high_utilization_table_data = latest_3_months_high_utilization_df[['Link Name', 'Region', 'Circle', 'Month', 'Utilization']].to_dict('records')
        
        link_count = len(high_utilization_df['Link Name'].unique())
        link_count_text = f"Total Links with Utilization ≥ 80%: {link_count}"

    # Generate forecast for utilized
    y_utilized = monthly_data['Utilized']
    model_utilized = SARIMAX(y_utilized, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
    model_utilized_fit = model_utilized.fit(disp=False)
    forecast_utilized = model_utilized_fit.get_forecast(steps=12)
    forecast_values_utilized = forecast_utilized.predicted_mean

    # Create DataFrame for forecasted utilized data (without noise)
    forecast_df_utilized = pd.DataFrame({
        'Utilized': forecast_values_utilized,
    }, index=pd.date_range(start=y_utilized.index[-1] + pd.DateOffset(months=1), periods=12, freq='M'))

    # **Create dropdown options for forecasted months**
    month_options = [{'label': month.strftime('%B %Y'), 'value': str(month)} for month in forecast_df_utilized.index]

        # Initialize the variables to avoid the UnboundLocalError
    recommended_capacity_text = "No additional capacity needed"  # Default value
    total_cost_text = "No cost applicable"  # Default value

    # Apply "What-If" Analysis for forecasted months
    # Apply "What-If" Analysis for forecasted months
    if selected_month is not None:
        selected_month_dt = pd.to_datetime(selected_month)

        # Check if the selected month is in the forecasted months
        if selected_month_dt in forecast_df_utilized.index and n_clicks > 0:
            # Adjust Total Resources if new_capacity_value is provided
            if new_capacity_value is not None:
                # Apply the new capacity value to the forecasted Total Resources for the selected and subsequent months
                forecast_df_total_resources.loc[forecast_df_total_resources.index >= selected_month_dt, 'Total Resources'] += new_capacity_value
            
            # Adjust Utilized if new_utilized_value is provided
            if new_utilized_value is not None:
                # Add the new utilized value to the forecasted Utilized for the selected and subsequent months
                forecast_df_utilized.loc[forecast_df_utilized.index >= selected_month_dt, 'Utilized'] += new_utilized_value

            # Recalculate Utilization for the selected and subsequent months based on both new_capacity_value and new_utilized_value
            for idx in range(forecast_df_utilized.index.get_loc(selected_month_dt), len(forecast_df_utilized)):
                forecasted_utilized_value = forecast_df_utilized.iloc[idx]['Utilized']
                forecasted_total_resources = forecast_df_total_resources.iloc[idx]['Total Resources']

                # Updated utilization formula with new capacity and/or utilized value
                updated_utilization_value = (forecasted_utilized_value / forecasted_total_resources) * 100

                # Apply the updated utilization to the forecasted DataFrame
                forecast_df_utilization.iloc[idx, forecast_df_utilization.columns.get_loc('Utilization')] = updated_utilization_value
            # --- NEW PART -- Determine when utilization exceeds the threshold ---
            critical_threshold = 80  # Set your critical threshold for utilization
            critical_month = None
            recommended_capacity = 0

            for i, utilization in enumerate(forecast_df_utilization['Utilization']):
                if utilization > critical_threshold:
                    critical_month = forecast_df_utilization.index[i]
                    break

            # If the utilization exceeds the threshold, calculate the needed capacity
            if critical_month is not None:
                target_utilization_percent = 60  # Target utilization we want to achieve
                current_utilized_value = forecast_df_utilized['Utilized'][i]  # Use the correct index for critical month
                current_capacity_value = forecast_df_total_resources['Total Resources'][i]

                # Capacity needed to bring utilization to 60%
                recommended_capacity = (current_utilized_value / (target_utilization_percent / 100)) - current_capacity_value

                # Ensure capacity isn't negative
                if recommended_capacity < 0:
                    recommended_capacity = 0

                # Calculate the total cost based on the unit rate provided
                if unit_rate is not None:
                    total_cost_for_capacity = (recommended_capacity / 10240) * unit_rate  # Assuming unit rate is per 10G
                else:
                    total_cost_for_capacity = 0

                # Assign the text values
                recommended_capacity_text = f"Capacity needed by {critical_month.strftime('%B %Y')}: {recommended_capacity:.2f} Mbps"
                total_cost_text = f"Cost to build capacity by {critical_month.strftime('%B %Y')}: Rs.{total_cost_for_capacity:.2f}"
            else:
                # If no critical month, use the selected month for display
                recommended_capacity_text = f"No additional capacity needed for the selected month: {selected_month}"
                total_cost_text = "No additional cost for the selected month."

    # Create Plotly figure for Utilization (with conditional point coloring)
    fig_utilization = go.Figure()

    # Historical data with detailed hovertemplate for Utilization
    fig_utilization.add_trace(go.Scatter(
        x=monthly_data.index, 
        y=monthly_data['Utilization'],
        mode='lines+markers',
        name='Historical Utilization',
        line=dict(color='blue'),
        marker=dict(
            color=['red' if val > 80 else 'green' for val in monthly_data['Utilization']],  # Red for points above 0.8
            symbol='circle',
            size=8
        ),
        hovertemplate=(
            'Month: %{x|%B %Y}<br>' +
            'Utilization: %{y:.2f}<br>' +
            'Num-Consumption (Utilized): %{customdata[0]:,.0f}<br>' +
            'Deno-Capacity (Total Resources): %{customdata[1]:,.0f}<extra></extra>'
        ),
        customdata=monthly_data[['Utilized', 'Total Resources']].values  # Adding consumption and capacity as custom data
    ))

    # Forecasted data with detailed hovertemplate for Utilization
    fig_utilization.add_trace(go.Scatter(
        x=forecast_df_utilization.index, 
        y=forecast_df_utilization['Utilization'],
        mode='lines+markers',
        name='Forecasted Utilization',
        line=dict(color='orange'),
        marker=dict(
            color=['red' if val > 80 else 'green' for val in forecast_df_utilization['Utilization']],  # Red for points above 0.8
            symbol='circle',
            size=8
        ),
        hovertemplate=(
            'Month: %{x|%B %Y}<br>' +
            'Forecasted Utilization: %{y:.2f}<br>'
        )
    ))

    # Add constant threshold lines with LWM and HWM annotations
    # Green dashed line for LWM (Lower Threshold)
    fig_utilization.add_shape(
        type='line',
        x0=monthly_data.index[0],  # Start of the line
        x1=forecast_df_utilization.index[-1],  # End of the line
        y0=lower_threshold,  # LWM value
        y1=lower_threshold,  # LWM value (same)
        line=dict(color='green', dash='dash'),
        name='Lower Threshold (LWM)'  # Adding label in the name
    )

# Annotation for LWM
    fig_utilization.add_annotation(
        x=forecast_df_utilization.index[-1],  # Position at the end of the plot
        y=lower_threshold,  # LWM threshold position
        text="LWM",  # Text label
        showarrow=False,  # No arrow, just text
        yshift=10  # Adjust position slightly
    )

    # Red dashed line for HWM (Upper Threshold)
    fig_utilization.add_shape(
        type='line',
        x0=monthly_data.index[0],  # Start of the line
        x1=forecast_df_utilization.index[-1],  # End of the line
        y0=upper_threshold,  # HWM value
        y1=upper_threshold,  # HWM value (same)
        line=dict(color='red', dash='dash'),
        name='Upper Threshold (HWM)'  # Adding label in the name
    )

    # Annotation for HWM
    fig_utilization.add_annotation(
        x=forecast_df_utilization.index[-1],  # Position at the end of the plot
        y=upper_threshold,  # HWM threshold position
        text="HWM",  # Text label
        showarrow=False,  # No arrow, just text
        yshift=10  # Adjust position slightly
    )

    fig_utilization.update_layout(
        title='Utilization(%) Forecast',
        xaxis_title='Date',
        yaxis_title='Utilization(%)',
        xaxis=dict(
            range=['2023-06-01', forecast_df_utilization.index[-1]],  # Set the x-axis range to start from June 2020
            dtick="M1",  # Ensures the ticks are aligned with each month
            tickformat="%b %Y"  # Display months in 'Jun 2020' format
        ),
        yaxis=dict(range=[0, 120])  # Adjust y-axis as needed for the utilization values
    )

    # Create Plotly figure for Utilized (with noise added)
    fig_utilized = go.Figure()

    # Historical data with detailed hovertemplate for Utilized
    fig_utilized.add_trace(go.Scatter(
        x=monthly_data.index, 
        y=monthly_data['Utilized'],
        mode='lines+markers',
        name='Historical Utilized',
        line=dict(color='blue'),
        hovertemplate=(
            'Month: %{x|%B %Y}<br>' +
            'Utilized: %{y:,.0f}<extra></extra>'
        )
    ))

    # Forecasted data with detailed hovertemplate for Utilized (with noise and "What-If" adjustments)
    fig_utilized.add_trace(go.Scatter(
        x=forecast_df_utilization.index, 
        y=forecast_df_utilized['Utilized'],
        mode='lines+markers',
        name='Forecasted Utilized',
        line=dict(color='orange'),
        hovertemplate=(
            'Forecasted Month: %{x|%B %Y}<br>' +
            'Forecasted Utilized: %{y:,.0f}<extra></extra>'
        )
    ))

    fig_utilized.update_layout(
        title='Utilized Forecast(Mbps)',
        xaxis_title='Date',
        yaxis_title='Utilized (Mbps)',
        xaxis=dict(
            range=['2023-06-01', forecast_df_utilized.index[-1]],  # Set the x-axis range to start from June 2020
            dtick="M1",  # Ensures the ticks are aligned with each month
            tickformat="%b %Y"  # Display months in 'Jun 2020' format
        ),
        yaxis=dict(range=[0, 6000])  # Ensure y-axis starts from 0
    )
    
    # Return either actual or forecasted table data based on radio button toggle
    # Update and return both the actual and forecasted table data along with other outputs
    return (latest_3_months_high_utilization_table_data,  # Actual utilization data for link-table
            forecasted_table_data,  # Forecasted utilization data for forecast-table
            # Other outputs like plots, month-dropdown, etc.
            fig_utilization, 
            fig_utilized, 
            month_options, 
            selected_month, 
            link_count_text, 
            recommended_capacity_text, 
            total_cost_text)
# Run the app
if __name__ == '__main__':
    app.run(port=8051, debug=True, threaded=True)

