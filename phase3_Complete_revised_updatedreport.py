import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import sqlite3
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from scipy.optimize import minimize
from PIL import Image
import plotly.colors as pc
from pyppeteer import launch
from PIL import Image
import nest_asyncio
import streamlit as st
from PIL import Image
from streamlit_modal import Modal
import streamlit as st
import pandas as pd
from datetime import datetime
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

nest_asyncio.apply()

st.set_page_config(layout="wide")
st.title("Racial Harm Portfolio Analyzer")
# Function to execute trades
def execute_trade(stock, recommended_ticker):
    # Placeholder for your trade execution logic
    st.success(f"Executed trade: {stock} → {recommended_ticker}")






# Initialize session state for storing the DataFrame
if 'portfolio_df' not in st.session_state:
    st.session_state.portfolio_df = pd.DataFrame(columns=[
        'Stock', 'Units', 'Purchase Date', 'Purchase Price ($)', 'Current Price ($)',
        'Initial Investment ($)', 'Current Value ($)', 'Gain/Loss ($)', 'Gain/Loss %', 
        'Portfolio Allocation', 'GICS Sector','Sector Harm Score' ,'Portfolio Harm Contribution'
    ])

if 'optimized_portfolio_df' not in st.session_state:
    st.session_state.optimized_portfolio_df = None

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from PIL import Image
from io import BytesIO

def generate_portfolio_pdf():
    """Generates a PDF report of the portfolio and returns a BytesIO buffer."""
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    # **Load the RFL logo**
    logo_path = "RFL.png"  # Ensure the file is in the correct path
    logo_width, logo_height = 80, 40  # Adjust size as needed

    try:
        c.drawImage(logo_path, 30, height - 50, width=logo_width, height=logo_height, mask='auto')
    except Exception as e:
        print(f"Error loading logo: {e}")

    # **Title**
    c.setFont("Helvetica-Bold", 16)
    c.drawString(200, height - 50, "Portfolio Report")

    # **Date**
    c.setFont("Helvetica", 10)
    current_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.drawString(200, height - 70, f"Generated on: {current_date}")

    y_position = height - 100  # Start position for text

    # **Function to draw portfolio summary**
    def draw_portfolio_summary(title, df, reference_df):
        nonlocal y_position
        c.setFont("Helvetica-Bold", 12)
        c.drawString(50, y_position, title)
        y_position -= 20

        if df is not None and not df.empty:
            portfolio_value = round(df["Current Value ($)"].astype(float).sum(), 2)
            total_gain_loss = round(df["Gain/Loss ($)"].astype(float).sum(), 2)

            c.setFont("Helvetica", 10)
            c.drawString(50, y_position, f"Total Portfolio Value: ${portfolio_value:,.2f}")
            y_position -= 15
            c.drawString(50, y_position, f"Total Portfolio Gain/Loss: ${total_gain_loss:,.2f}")
            y_position -= 30

            # **Table Headers**
            c.setFont("Helvetica-Bold", 10)
            headers = ["Stock", "Units", "Current Value ($)", "Gain/Loss ($)", "Gain/Loss %"]
            col_positions = [50, 150, 250, 350, 450]

            for i, header in enumerate(headers):
                c.drawString(col_positions[i], y_position, header)
            y_position -= 15
            c.line(50, y_position, 500, y_position)  # Table line separator
            y_position -= 10

            # **Draw each row**
            c.setFont("Helvetica", 9)
            for _, row in df.iterrows():
                if y_position < 50:  # Create a new page if running out of space
                    c.showPage()
                    y_position = height - 50

                stock = row["Stock"]
                gain_loss_percentage = 0.0

                if stock in reference_df["Stock"].values:
                    reference_row = reference_df[reference_df["Stock"] == stock].iloc[0]
                    gain_loss_percentage = (reference_row["Gain/Loss ($)"] / reference_row["Current Value ($)"]) * 100
                    gain_loss_percentage = round(gain_loss_percentage, 2)

                c.drawString(col_positions[0], y_position, str(row["Stock"]))
                c.drawString(col_positions[1], y_position, str(row["Units"]))
                c.drawString(col_positions[2], y_position, f"{round(row['Current Value ($)'], 2):,.2f}")
                c.drawString(col_positions[3], y_position, f"{round(row['Gain/Loss ($)'], 2):,.2f}")
                c.drawString(col_positions[4], y_position, f"{gain_loss_percentage:.2f}%")
                y_position -= 15
        else:
            c.setFont("Helvetica", 10)
            c.drawString(50, y_position, "No data available.")
            y_position -= 30

    # **Draw Original Portfolio Summary**
    draw_portfolio_summary("Original Portfolio Summary", st.session_state.portfolio_df, st.session_state.portfolio_df)

    # **Draw Optimized Portfolio Summary (if available)**
    draw_portfolio_summary("Optimized Portfolio Summary", st.session_state.optimized_portfolio_df, st.session_state.portfolio_df)

    c.save()
    buffer.seek(0)
    return buffer



def get_stock_colors(stock_list):
    """
    Assigns consistent colors to stocks using a predefined color palette.
    If more stocks are added dynamically, they get a new color from the palette.
    """
    color_palette = pc.qualitative.Set2  # Choose from: Set1, Set2, Dark2, Pastel, etc.
    stock_color_map = {stock: color_palette[i % len(color_palette)] for i, stock in enumerate(stock_list)}
    return stock_color_map



# Function to update portfolio allocation
def update_portfolio_allocation(df):
    if not df.empty:
        total_value = df['Current Value ($)'].astype(float).sum()
        df.loc[:, 'Portfolio Allocation'] = df['Current Value ($)'].astype(float) / total_value * 100
        df.loc[:, 'Portfolio Allocation'] = df['Portfolio Allocation'].apply(lambda x: f"{x:.2f}%")
    return df
def get_sector_harm_score(sector):
    conn = sqlite3.connect('nycprocurement.db')
    
    query = """
        SELECT Normalized_Score_2 FROM stockracialharm2 WHERE sector = ?
    """
    scores = pd.read_sql_query(query, conn, params=(sector,))
    conn.close()
    return scores.iloc[0]['Normalized_Score_2'] if not scores.empty else None

def optimize_portfolio(df, max_harm_score, min_stock_threshold):
    
    
    # Prepare data
    current_prices = df['Current Price ($)'].astype(float)
    returns = df['Gain/Loss %'].astype(float) / 100
    harm_scores = df['Sector Harm Score'].astype(float)
    current_values = df['Current Value ($)'].astype(float)
    current_units = df['Units'].astype(float)
    purchase_prices = df['Purchase Price ($)'].astype(float)
    
    total_portfolio_value = current_values.sum()
    initial_total_return = df['Gain/Loss ($)'].astype(float).sum()
    total_units = current_units.sum()
    
    
    initial_weights = current_values / total_portfolio_value
    
  
    def objective(weights):
        new_values = weights * total_portfolio_value * 1.1  
        return -np.sum(new_values)  
    
  
    def harm_score_constraint(weights):
        new_values = weights * total_portfolio_value * 1.1
        new_units = new_values / current_prices
        total_new_units = new_units.sum()
        weighted_harm = np.sum(harm_scores * new_units) / total_new_units
        return weighted_harm - max_harm_score  
    
 
    def units_constraint(weights):
        new_values = weights * total_portfolio_value * 1.1
        new_units = new_values / current_prices
        return new_units.sum() - total_units 
    

    def weight_sum_constraint(weights):
        return np.sum(weights) - 1.0
    
    def return_constraint(weights):
        new_values = weights * total_portfolio_value * 1.1
        new_units = new_values / current_prices
        new_initial_investment = new_units * purchase_prices
        new_total_return = np.sum(new_values - new_initial_investment)
        return new_total_return - initial_total_return
    
   
    min_weights = np.array((min_stock_threshold * current_prices) / (total_portfolio_value * 1.1))
    

    constraints = [
        {"type": "ineq", "fun": harm_score_constraint}, 
        {"type": "eq", "fun": units_constraint},         
        {"type": "eq", "fun": weight_sum_constraint},    
        {"type": "ineq", "fun": return_constraint}       
    ]
    

    for i in range(len(df)):
        constraints.append({
            "type": "ineq",
            "fun": lambda w, idx=i: w[idx] - min_weights[idx]
        })
    

    bounds = [(0, 1.1)] * len(df)
    
    # Run optimization
    result = minimize(
        fun=objective,
        x0=initial_weights,
        bounds=bounds,
        constraints=constraints,
        method='SLSQP'  
    )
    
    if result.success:
    
        optimized_weights = result.x
        new_values = optimized_weights * total_portfolio_value * 1.1
        new_units = new_values / current_prices
        new_initial_investment = new_units * purchase_prices
        new_total_return = np.sum(new_values - new_initial_investment)
        
    
        if abs(new_units.sum() - total_units) > 0.01:
            raise ValueError("Optimization failed: Total units constraint not met")
        
        total_new_units = new_units.sum()
        weighted_harm = np.sum(harm_scores * new_units) / total_new_units
        if weighted_harm < max_harm_score:
            raise ValueError("Optimization failed: Solution does not meet minimum harm score requirement")
        

        df['Portfolio Allocation'] = (optimized_weights / np.sum(optimized_weights)) * 100
        df['Current Value ($)'] = new_values
        df['Units'] = new_units.round()
        df['Initial Investment ($)'] = df['Units'] * purchase_prices
        df['Gain/Loss ($)'] = df['Current Value ($)'] - df['Initial Investment ($)']
        df['Gain/Loss %'] = (df['Gain/Loss ($)'] / df['Initial Investment ($)']) * 100
        
       
        total_harm_units = (df['Units'] * df['Sector Harm Score']).sum()
        df['Portfolio Harm Contribution'] = (df['Units'] * df['Sector Harm Score']) / total_harm_units * 100
        
        return df
    else:
        raise ValueError(f"Optimization failed: {result.message}")

def calculate_historical_portfolio_value(portfolio_df):
    """
    Calculate daily portfolio value from the earliest purchase date until today
    """
    if portfolio_df.empty:
        return pd.DataFrame()
    
    # Convert purchase dates to datetime
    portfolio_df['Purchase Date'] = pd.to_datetime(portfolio_df['Purchase Date']).dt.tz_localize(None)
    
    # Get the earliest purchase date and today's date
    start_date = portfolio_df['Purchase Date'].min()
    end_date = datetime.now()
    
    # Initialize DataFrame to store daily values
    date_range = pd.date_range(start=start_date, end=end_date, freq='B')  # 'B' for business days only
    portfolio_value_df = pd.DataFrame(index=date_range)
    portfolio_value_df['Total Value'] = 0.0
    
    # Calculate daily value for each stock
    for _, row in portfolio_df.iterrows():
        try:
            # Get historical data for the stock
            stock = yf.Ticker(row['Stock'])
            hist_data = stock.history(start=row['Purchase Date'], end=end_date)
            
            if hist_data.empty:
                continue
                
            # Make sure indices are comparable
            hist_data.index = hist_data.index.tz_localize(None)

            # Calculate daily value (price * units)
            units = float(row['Units'])
            daily_value = hist_data['Close'] * units
            
            # Only consider values from purchase date onwards
            daily_value = daily_value[daily_value.index >= row['Purchase Date']]
            
            # Reindex to match our portfolio date range (filling forward)
            daily_value = daily_value.reindex(portfolio_value_df.index, method='ffill')
            
            # Add to total portfolio value
            portfolio_value_df.loc[daily_value.index, 'Total Value'] += daily_value
            
        except Exception as e:
            st.warning(f"Error fetching historical data for {row['Stock']}: {str(e)}")
            continue
    
    # Drop rows with zero values (non-trading days)
    portfolio_value_df = portfolio_value_df[portfolio_value_df['Total Value'] > 0]
    
    # Fill any remaining NaN values with the previous day's value
    portfolio_value_df = portfolio_value_df.fillna(method='ffill')
    
    return portfolio_value_df



# Function to get GICS Sector
def get_gics_sector(ticker):
    try:
        stock = yf.Ticker(ticker)
        return stock.info.get('sector', 'N/A')
    except:
        return 'N/A'

# Function to get normalized scores from SQLite database based on sector



def calculate_portfolio_stats(df):
    """
    Calculate comprehensive summary statistics for the portfolio including trading metrics
    """
    if df.empty:
        return None
    
    stats = {}
    
    # Convert relevant columns to numeric
    df['Current Value ($)'] = pd.to_numeric(df['Current Value ($)'].astype(str).str.replace('$', ''), errors='coerce')
    df['Initial Investment ($)'] = pd.to_numeric(df['Initial Investment ($)'].astype(str).str.replace('$', ''), errors='coerce')
    df['Gain/Loss ($)'] = pd.to_numeric(df['Gain/Loss ($)'].astype(str).str.replace('$', ''), errors='coerce')
    df['Gain/Loss %'] = pd.to_numeric(df['Gain/Loss %'], errors='coerce')
    df['Portfolio Harm Contribution'] = pd.to_numeric(df['Portfolio Harm Contribution'], errors='coerce')
    
    # Basic Portfolio Statistics
    stats['Start Value'] = df['Initial Investment ($)'].sum()
    stats['End Value'] = df['Current Value ($)'].sum()
    stats['Total Return [%]'] = ((stats['End Value'] - stats['Start Value']) / stats['Start Value']) * 100
    
    # Trading Performance Metrics
    winning_trades = df[df['Gain/Loss %'] > 0]
    losing_trades = df[df['Gain/Loss %'] < 0]
    
    stats['Total Trades'] = len(df)
    stats['Total Closed Trades'] = len(df[df['Current Value ($)'] > 0])
    
    
    # Win/Loss Metrics
    stats['Best Trade [%]'] = df['Gain/Loss %'].max()
    stats['Worst Trade [%]'] = df['Gain/Loss %'].min()
    stats['Avg Winning Trade [%]'] = winning_trades['Gain/Loss %'].mean() if not winning_trades.empty else 0
    stats['Avg Losing Trade [%]'] = losing_trades['Gain/Loss %'].mean() if not losing_trades.empty else 0
    
    # Risk Metrics
    portfolio_value_series = df['Current Value ($)'].cumsum()
    peak = portfolio_value_series.expanding(min_periods=1).max()
    drawdown = (portfolio_value_series - peak) / peak * 100
    stats['Max Drawdown [%]'] = abs(drawdown.min()) if not drawdown.empty else 0
    
    # Calculate Profit Factor
    total_gains = winning_trades['Gain/Loss ($)'].sum() if not winning_trades.empty else 0
    total_losses = abs(losing_trades['Gain/Loss ($)'].sum()) if not losing_trades.empty else 0
    stats['Profit Factor'] = abs(total_gains / total_losses) if total_losses != 0 else float('inf')
    
    # Win Rate
    stats['Win Rate [%]'] = (len(winning_trades) / len(df)) * 100 if len(df) > 0 else 0
    
    return pd.DataFrame([stats]).transpose().round(3)

def get_normalized_score2(sector):
    
    conn = sqlite3.connect('nycprocurement.db')
    
    query = "SELECT Normalized_Score_1 FROM stockracialharm2 WHERE sector = ?"
    scores = pd.read_sql_query(query, conn, params=(sector,))
    conn.close()
    
    return scores.iloc[0]['Normalized_Score_1'] if not scores.empty else None

def get_normalized_score_graph1(sector):
    conn = sqlite3.connect('nycprocurement.db')
    
    query = "SELECT Normal_Score_Graph1 FROM stockracialharm2 WHERE sector = ?"
    scores = pd.read_sql_query(query, conn, params=(sector,))
    conn.close()
    
    return scores.iloc[0]['normal_score_graph1'] if not scores.empty else None
# Sidebar for adding stocks

# Load the logo

logo_path = "RFL.png"

logo = Image.open(logo_path)

# Sidebar with logo
with st.sidebar:
    st.image(logo, width=150)

with st.sidebar:
    st.header("Add Stock to Portfolio")
    with st.form("stock_form"):
        ticker = st.text_input("Enter stock ticker (e.g., AAPL, GOOGL)")
        units = st.number_input("Enter number of units", min_value=1, step=1)
        transaction_date = st.date_input("Select transaction date")
        submit_button = st.form_submit_button(label="Add to Portfolio")

# Main application logic
# Find the section where new_row is created in the submit_button logic and modify it:
if submit_button:
    if units == 0:
        st.error("Units cannot be zero. Please enter a valid number of units.")
    else:
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(start=transaction_date)

            if hist.empty:
                st.error("No data available for the selected date. Please choose a valid trading day.")
            else:
                purchase_price = hist.iloc[0]['Close']
                current_price = stock.info['currentPrice']

                initial_investment = purchase_price * units
                current_value = current_price * units
                gain_loss = current_value - initial_investment
                gain_loss_percentage = (gain_loss / initial_investment) * 100

                gics_sector = get_gics_sector(ticker)
                normalized_score = get_normalized_score2(gics_sector) or np.random.uniform(0, 1)
                normalized_score_graph = get_normalized_score_graph1(gics_sector) or np.random.uniform(0, 1)
                
                # Calculate total harm score units for existing portfolio
                existing_harm_units = 0
                if not st.session_state.portfolio_df.empty:
                    existing_harm_units = (
                        st.session_state.portfolio_df['Normalized Harm Score Graph'].astype(float) * 
                        st.session_state.portfolio_df['Units'].astype(float)
                    ).sum()
                
                # Add new stock's harm units
                new_harm_units = normalized_score_graph * units
                total_harm_units = existing_harm_units + new_harm_units
                
                # Calculate harm score contribution as percentage
                harm_score_contribution = (new_harm_units / total_harm_units * 100) if total_harm_units > 0 else 0

                sector_harm_score = get_sector_harm_score(gics_sector)
                new_row = pd.DataFrame({
                    'Stock': [ticker],
                    'Units': [units],
                    'Purchase Date': [transaction_date],
                    'Purchase Price ($)': [f"{purchase_price:.2f}"],
                    'Current Price ($)': [f"{current_price:.2f}"],
                    'Initial Investment ($)': [f"{initial_investment:.2f}"],
                    'Current Value ($)': [f"{current_value:.2f}"],
                    'Gain/Loss ($)': [f"{gain_loss:.2f}"],
                    'Gain/Loss %': [gain_loss_percentage],
                    'Portfolio Allocation': ["0.00%"],
                    'GICS Sector': [gics_sector],
                    'Sector Harm Score': [sector_harm_score],
                    'Portfolio Harm Contribution': [harm_score_contribution],  # Updated to use the new calculation
                    'Normalized Harm Score Graph': [normalized_score_graph]
                })
                
                # After adding the new row, recalculate harm score contributions for all stocks
                combined_df = pd.concat([st.session_state.portfolio_df, new_row], ignore_index=True)
                
                # Calculate total harm units for the entire portfolio
                total_portfolio_harm_units = (
                    combined_df['Normalized Harm Score Graph'].astype(float) * 
                    combined_df['Units'].astype(float)
                ).sum()
                
                # Update harm score contributions for all stocks
                combined_df['Portfolio Harm Contribution'] = (
                    combined_df['Normalized Harm Score Graph'].astype(float) * 
                    combined_df['Units'].astype(float) / 
                    total_portfolio_harm_units * 100
                )
                
                st.session_state.portfolio_df = combined_df
            
                st.session_state.portfolio_df = update_portfolio_allocation(st.session_state.portfolio_df)

                st.session_state.historical_value_df = calculate_historical_portfolio_value(st.session_state.portfolio_df)
                
                st.success(f"Added {ticker} to your portfolio.")

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")


company_ticker_dict = {
    "Amalgamated Bank": "AMAL",
    "PayPal": "PYPL",
    "The Walt Disney Company": "DIS",
    "Victoria's Secret": "VSCO",
    "American Electric Power": "AEP",
    "Philip Morris International": "PM",
    "Visa": "V",
    "Microsoft": "MSFT",
    "JLL": "JLL",
    "Albertsons Companies": "ACI",
    "NL Industries": "NL",
    "Orion": "OEC",  # Orion Engineered Carbons (You may need to verify this)
    "SunCoke Energy": "SXC",
    "CleanHarbors": "CLH",
    "Amplify Energy": "AMPY",
    "Olin": "OLN",
    "Mosaic": "MOS",
    "Huntsman": "HUN",
    "Westlake": "WLK",
    "Berkshire Hathaway": "BRK.B"  # Class B shares (BRK.A for Class A)
}



# Optimize portfolio if stocks are present and button is clicked
if not st.session_state.portfolio_df.empty:
    with st.sidebar:
        st.header("Optimize Portfolio")
            # New Minimum Stock Holding Threshold input field
        min_stock_threshold = st.number_input( 
            "Minimum Stock Holding Threshold (Units)",
            min_value=1,  # Minimum of 1 unit
            value=1,      # Default value
            step=1
        )

        max_harm_score = st.number_input( 
            "Set Maximum Harm Score Threshold",
            min_value=0.0,
            max_value=100.0,
            step=1.0,
            value=50.0,
            format="%.2f"
        )
        
        if st.button("Apply Optimization"):
            try:
                optimized_df = optimize_portfolio(
                    st.session_state.portfolio_df.copy(),
                    max_harm_score,
                    min_stock_threshold  # Pass the dynamic minimum stock threshold here
                )
                st.session_state.optimized_portfolio_df = optimized_df
                st.session_state.optimization_success = True
            except Exception as e:
                st.error(f"Optimization failed: {str(e)}")

def execute_trade(selected_ticker, recommended_ticker):
    """Executes a trade by replacing the selected stock with the recommended one."""
    try:

        updated_df = st.session_state.portfolio_df.copy()
        mask = updated_df['Stock'] == selected_ticker
    
        if mask.any():
            new_ticker = company_ticker_dict[recommended_ticker]
            new_stock = yf.Ticker(new_ticker)
            new_price = new_stock.info.get('currentPrice', 0)

            # Update portfolio
            updated_df.loc[mask, 'Stock'] = new_ticker
            updated_df.loc[mask, 'Current Price ($)'] = f"{new_price:.2f}"
            st.session_state.portfolio_df = updated_df
            st.session_state.portfolio_df = update_portfolio_allocation(st.session_state.portfolio_df)
            st.session_state.historical_value_df = calculate_historical_portfolio_value(st.session_state.portfolio_df)

            st.success(f"Replaced {selected_ticker} with {recommended_ticker}.")
        else:
            st.error(f"{selected_ticker} not found in portfolio.")

    except Exception as e:
        st.error(f"An error occurred in execute_trade: {str(e)}")
        print(f"Error in execute_trade: {e}")  # This will log the error in the terminal/console


def load_recommendations():
    """Fetch recommendations from the database for each stock in the portfolio."""
         
    conn = sqlite3.connect('nycprocurement.db')
    
    recommendations = {}
    for _, row in st.session_state.portfolio_df.iterrows():
        stock_ticker = row['Stock']
        sector = row['GICS Sector']

        # Query recommendation
        query = "SELECT * FROM asyousowrj WHERE Sector = ? AND Category = 'Leader'"
        cursor = conn.cursor()
        cursor.execute(query, (sector,))
        result = cursor.fetchone()

        # Store recommendations
        recommendations[stock_ticker] = {
            'recommended_ticker': result[0] if result else 'NA',
            'recommending_source': result[6] if result else 'NA',
            'distinction_signal': result[1] if result else 'NA'
        }

    conn.close()
    return recommendations

# Initialize session state for selected trades
if 'selected_trades' not in st.session_state:
    st.session_state.selected_trades = set()

# Initialize modal
modal = Modal(key="racial_harm_modal", title="Racial Harm Reduction Recommendations" ,  max_width=1200  )

if "selected_trades" not in st.session_state:
    st.session_state.selected_trades = set()

# Open modal when the button is clicked
if not st.session_state.portfolio_df.empty:
    with st.sidebar:
        if st.button("Racial Harm Reduction Recommendations"):
            modal.open()

# Display modal content
if modal.is_open():
    with modal.container():
        # CSS styling for table with column dividers
        st.markdown(
            """
            <style>
                /* Modal size and appearance */
                .modal-content {
                    width: 90% !important;
                    max-width: 1200px !important;
                    background-color: white !important;
                    color: #333 !important;
                }
                
                /* Title styling */
                .modal-title {
                    font-size: 20px;
                    font-weight: 600;
                    margin-bottom: 20px;
                    color: #333;
                }
                
                /* Button styling */
                .stButton button {
                    background-color: #f0f2f6 !important;
                    color: #333 !important;
                    font-weight: 500 !important;
                    border-radius: 4px !important;
                    border: 1px solid #ddd !important;
                    padding: 8px 16px !important;
                }
                
                .stButton button:hover {
                    background-color: #e6e9ef !important;
                }
                
                /* Table styling with column dividers */
                .simple-table {
                    border-collapse: collapse;
                    width: 100%;
                    margin-bottom: 20px;
                }
                
                .simple-table th, .simple-table td {
                    border: 1px solid #ddd;
                    padding: 8px 12px;
                     text-align: center; /* Center-aligns all values */
                    border-right: 1px solid #ddd;
                    border-left: 1px solid #ddd;
                }
                
                .simple-table th {
                    background-color: #f8f9fa;
                    font-weight: 500;
                    border-top: 1px solid #ddd;
                    border-bottom: 1px solid #ddd;
                }
                
                .simple-table tr:nth-child(even) {
                    background-color: #f9f9f9;
                }
                
                /* First column styling */
                .simple-table th:first-child, .simple-table td:first-child {
                    border-left: 1px solid #ddd;
                }
                
                /* Last column styling */
                .simple-table th:last-child, .simple-table td:last-child {
                    border-right: 1px solid #ddd;
                }
                
                /* Checkbox styling */
                .stCheckbox > label {
                    color: #333 !important;
                }
                
                /* Container for table */
                .table-container {
                    max-height: 500px;
                    overflow-y: auto;
                    margin-bottom: 20px;
                }
            </style>
            """,
            unsafe_allow_html=True
        )
        
        # Add title
        
        
        # Database query to get recommendations
        recommendations = load_recommendations()
        
        # Create HTML table
        table_html = """
        <div class="table-container">
            <table class="simple-table">
                <thead>
                    <tr>
                        <th>Rank</th>
                        <th>Selected Stock</th>
                        <th>Sector</th>
                        <th>Recommended Stock</th>
                        <th>Recommending Source</th>
                        <th>Distinction Signal</th>
                        <th>Execute</th>
                    </tr>
                </thead>
                <tbody>
        """
        
        # Close the table HTML opening to display it
        st.markdown(table_html, unsafe_allow_html=True)
        
        # Generate rows with data and checkboxes
        for i, (stock_ticker, recommendation) in enumerate(recommendations.items(), 1):
            sector = st.session_state.portfolio_df.loc[st.session_state.portfolio_df['Stock'] == stock_ticker, 'GICS Sector'].values[0]
            recommended_ticker = recommendation['recommended_ticker']
            recommending_source = recommendation['recommending_source']
            distinction_signal = recommendation['distinction_signal']
            
            # Create row with columns
            cols = st.columns([0.5, 1, 1, 1, 1.5, 1, 0.7])
            
            # Display values in columns
            cols[0].write(f" &nbsp;&nbsp;&nbsp; {str(i)}")
            cols[1].write(stock_ticker)
            cols[2].write(sector)
            cols[3].write(recommended_ticker)
            cols[4].write(recommending_source)
            cols[5].write(distinction_signal)
            
            # Add checkbox in the last column
            execute_key = f"execute_{stock_ticker}"
            if cols[6].checkbox("", key=execute_key):
                st.session_state.selected_trades.add(stock_ticker)
            else:
                st.session_state.selected_trades.discard(stock_ticker)
        
        # Close table HTML
        st.markdown('</tbody></table></div>', unsafe_allow_html=True)
        
        # Add execute button centered at the bottom
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("Execute Selected Trades", key="execute_all", use_container_width=True):
                if not st.session_state.selected_trades:
                    st.warning("No trades selected!")
                else:
                    with st.spinner("Executing trades..."):
                        for stock in list(st.session_state.selected_trades):
                            recommended_ticker = recommendations.get(stock, {}).get('recommended_ticker')
                            
                            if recommended_ticker and recommended_ticker != "NA":
                                st.success(f"Executing trade: {stock} → {recommended_ticker}")
                                execute_trade(stock, recommended_ticker)
                            else:
                                st.error(f"No valid recommendation for {stock}.")
                        
                        # Clear session state after execution
                        st.session_state.selected_trades.clear()
                        modal.close()  # Close modal after execution
                        st.rerun()  # Refresh UI


# Portfolio Summary and Analysis (Before Optimization)
st.subheader("Portfolio Summary and Analysis")
if not st.session_state.portfolio_df.empty:
    portfolio_value = st.session_state.portfolio_df['Current Value ($)'].astype(float).sum()
    total_gain_loss = st.session_state.portfolio_df['Gain/Loss ($)'].astype(float).sum()

    # Display total portfolio value and total gain/loss
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Portfolio Value", f"${portfolio_value:,.2f}")

    with col2:
        st.metric("Total Portfolio Gain/Loss", 
              f"${total_gain_loss:,.2f}")

else:
    st.info("Add stocks to your portfolio to see analysis.")

# Display the portfolio table and other visualizations
if not st.session_state.portfolio_df.empty:
    st.subheader("Public Equity Portfolio")
    df_display = st.session_state.portfolio_df.drop(columns=['Normalized Harm Score Graph']).reset_index(drop=True)
    df_display.index += 1  # Start numbering from 1
     # Add as first column
    
    st.dataframe(df_display, use_container_width=True)


    # Add this code after the portfolio summary section
if not st.session_state.portfolio_df.empty:
    st.subheader("Portfolio Value Over Time")
    
    # Calculate historical portfolio value
    historical_value_df = calculate_historical_portfolio_value(st.session_state.portfolio_df.copy())
    
    if not historical_value_df.empty:
        # Create line chart using plotly
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=historical_value_df.index,
            y=historical_value_df['Total Value'],
            mode='lines',
            name='Portfolio Value',
            line=dict(color='#1f77b4', width=2)
        ))
        
        fig.update_layout(
            title='Portfolio Value Over Time',
            xaxis_title='Date',
            yaxis_title='Portfolio Value ($)',
            hovermode='x unified',
            showlegend=True,
            template='plotly_white'
        )
        
        # Add dollar sign and comma formatting to y-axis
        fig.update_layout(
            yaxis=dict(
                tickformat="$,.2f"
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    else:
        st.warning("Unable to fetch historical data for the portfolio.")


    st.subheader("Portfolio Performance Analysis")
    stats_df = calculate_portfolio_stats(st.session_state.portfolio_df)
    if stats_df is not None:
        # Format the statistics table
        formatted_stats = stats_df.copy()
        formatted_stats.columns = ['Value']
        
        # Format monetary values
        monetary_metrics = ['Start Value', 'End Value']
        for metric in monetary_metrics:
            if metric in formatted_stats.index:
                formatted_stats.loc[metric, 'Value'] = f"${formatted_stats.loc[metric, 'Value']:,.2f}"
        
        # Format percentage values
        percentage_metrics = [
            'Total Return [%]', 'Max Drawdown [%]', 'Best Trade [%]', 
            'Worst Trade [%]', 'Avg Winning Trade [%]', 'Avg Losing Trade [%]',
            'Win Rate [%]'
        ]
        for metric in percentage_metrics:
            if metric in formatted_stats.index:
                formatted_stats.loc[metric, 'Value'] = f"{formatted_stats.loc[metric, 'Value']}%"
        
        # Display the formatted table
        st.dataframe(formatted_stats, use_container_width=True)
    # Create two columns for side-by-side layout for visualizations
    col1, col2 = st.columns(2)

    # Create a doughnut chart for normalized harm scores in the first column
    units_df = st.session_state.portfolio_df[['Stock', 'Units']]

    # Check if 'Units' column is not null and handle conversion
    if not units_df['Units'].isnull().all():
        units_df['Units'] = pd.to_numeric(units_df['Units'], errors='coerce')

        # Fill NaN values with 0
        units_df['Units'].fillna(0, inplace=True)

        # Calculate total units
        total_units = units_df['Units'].sum()

        # Create a new column for percentage
        units_df['Percentage'] = (units_df['Units'] / total_units) * 100
        stock_list = st.session_state.portfolio_df['Stock'].unique().tolist()
        stock_color_map = get_stock_colors(stock_list)
        # Create doughnut chart using the percentage
        fig1 = px.pie(
            units_df,
            names='Stock',
            values='Percentage',
            hole=0.4,
            title="Stock Portfolio Units as Percentage of Total",
            labels={'Percentage': 'Percentage of Total Units'},
            color='Stock',  # Assign colors based on stocks
            color_discrete_map=stock_color_map  # Use predefined colors
        )

        with col1:
            st.plotly_chart(fig1, key="original_units_chart")

    # Create a doughnut chart that calculates normalized harm score * number of units
    portfolio_df = st.session_state.portfolio_df[['Stock', 'Normalized Harm Score Graph', 'Units']].copy()

    # Convert columns to numeric and fill NaNs with 0
    portfolio_df['Normalized Harm Score Graph'] = pd.to_numeric(portfolio_df['Normalized Harm Score Graph'], errors='coerce').fillna(0)
    portfolio_df['Units'] = pd.to_numeric(portfolio_df['Units'], errors='coerce').fillna(0)

    # Calculate total harm score units
    total_harm_score_units = (portfolio_df['Normalized Harm Score Graph'] * portfolio_df['Units']).sum()

    # Create a new column for harm score contribution percentage
    portfolio_df['Harm Score Contribution (%)'] = (
        (portfolio_df['Normalized Harm Score Graph'] * portfolio_df['Units']) / total_harm_score_units * 100
    ).fillna(0)

    # Prepare data for harm score contribution chart
    contribution_data = portfolio_df[['Stock', 'Harm Score Contribution (%)']]

    # Create doughnut chart for harm score contribution
    if not contribution_data['Harm Score Contribution (%)'].isnull().all():
        contribution_data['Harm Score Contribution (%)'] = contribution_data['Harm Score Contribution (%)'].astype(float)

        fig2 = px.pie(
            contribution_data,
            names='Stock',
            values='Harm Score Contribution (%)',
            title="Portfolio Harm Contribution by Stock",
            labels={'Harm Score Contribution (%)': 'Contribution (%)'},
            hole=0.4,
            color='Stock',  # Assign colors based on stocks
            color_discrete_map=stock_color_map  # Use predefined colors
        ) 

        with col2:
            st.plotly_chart(fig2 ,  key="original_harm_chart")

# Display optimized portfolio
if st.session_state.optimized_portfolio_df is not None:
    st.subheader("Harm Reduction Optimized Portfolio")
    if hasattr(st.session_state, 'optimization_success') and st.session_state.optimization_success:
        st.success(
        "**Portfolio Optimization Insight:**\n"
        "This rebalanced portfolio allocation maximizes total financial return while "
        "minimizing the harm profile, or average weighted harm score, of the holdings "
        "in your portfolio to the user-selected maximum harm threshold and required holding per stock. "
        "The harm profile score generation is powered by the RFL Racial Harm Score. "
        "The Score is generated at the GICS sector level. Racial harm, or disparate "
        "impact, is measured across a proprietary harm typology mapped to the "
        "lifecycle activities of typical sector enterprises. Evidence of these disparate "
        "impacts are backed by deep learning methodologies and validated by expert "
        "human research resources. "
        )
    df_display1 = st.session_state.optimized_portfolio_df.drop(columns=['Normalized Harm Score Graph']).reset_index(drop=True)
    df_display1.index += 1  # Start numbering from 1


    st.dataframe(df_display1, use_container_width=True)

    if not st.session_state.optimized_portfolio_df.empty:
        st.subheader("Portfolio Value Over Time")
        
        # Calculate historical portfolio value
        historical_value_df = calculate_historical_portfolio_value(st.session_state.optimized_portfolio_df.copy())
        
        if not historical_value_df.empty:
            # Create line chart using plotly
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                x=historical_value_df.index,
                y=historical_value_df['Total Value'],
                mode='lines',
                name='Portfolio Value',
                line=dict(color='#1f77b4', width=2)
            ))
            
            fig2.update_layout(
                title='Optimized Portfolio Value Over Time',
                xaxis_title='Date',
                yaxis_title='Portfolio Value ($)',
                hovermode='x unified',
                showlegend=True,
                template='plotly_white'
            )
            
            # Add dollar sign and comma formatting to y-axis
            fig2.update_layout(
                yaxis=dict(
                    tickformat="$,.2f"
                )
            )
            
            st.plotly_chart(fig2, use_container_width=True)
            
        else:
            st.warning("Unable to fetch historical data for the portfolio.")
    st.subheader("Portfolio Performance Analysis")
    stats_df = calculate_portfolio_stats(st.session_state.optimized_portfolio_df)
       # Add this code after the portfolio summary section
    
    if stats_df is not None:
        # Format the statistics table
        formatted_stats = stats_df.copy()
        formatted_stats.columns = ['Value']
        
        # Format monetary values
        monetary_metrics = ['Start Value', 'End Value']
        for metric in monetary_metrics:
            if metric in formatted_stats.index:
                formatted_stats.loc[metric, 'Value'] = f"${formatted_stats.loc[metric, 'Value']:,.2f}"
        
        # Format percentage values
        percentage_metrics = [
            'Total Return [%]', 'Max Drawdown [%]', 'Best Trade [%]', 
            'Worst Trade [%]', 'Avg Winning Trade [%]', 'Avg Losing Trade [%]',
            'Win Rate [%]'
        ]
        for metric in percentage_metrics:
            if metric in formatted_stats.index:
                formatted_stats.loc[metric, 'Value'] = f"{formatted_stats.loc[metric, 'Value']}%"
        
        # Display the formatted table
    
        st.dataframe(formatted_stats, use_container_width=True)
    
    

    col3, col4 = st.columns(2)

    optimized_units_df = st.session_state.optimized_portfolio_df[['Stock', 'Units']]
    optimized_units_df['Units'] = pd.to_numeric(optimized_units_df['Units'], errors='coerce').fillna(0)
    optimized_units_df['Percentage'] = (optimized_units_df['Units'] / optimized_units_df['Units'].sum()) * 100
    fig3 = px.pie(optimized_units_df, names='Stock', values='Percentage', hole=0.4, title="Optimized Portfolio Units Distribution",color='Stock',  # Assign colors based on stocks
            color_discrete_map=stock_color_map  )
    with col3:
        st.plotly_chart(fig3 ,key = "optimized_units_chart")
    
    optimized_harm_scores_df = st.session_state.optimized_portfolio_df[['Stock', 'Normalized Harm Score Graph', 'Units']]

    optimized_harm_scores_df['Contribution'] = optimized_harm_scores_df['Normalized Harm Score Graph'] * optimized_harm_scores_df['Units']
    optimized_harm_scores_df['Percentage'] = optimized_harm_scores_df['Contribution'] / optimized_harm_scores_df['Contribution'].sum() * 100
    fig4 = px.pie(optimized_harm_scores_df, 
                  names='Stock', 
                  values='Percentage', 
                  hole=0.4, 
                  title="Optimized Portfolio Harm Contribution",
                  color='Stock',  # Assign colors based on stocks
            color_discrete_map=stock_color_map )
    with col4:
        st.plotly_chart(fig4, key="optimized_harm_chart")

    # Optimized Portfolio Summary
    optimized_portfolio_value = st.session_state.optimized_portfolio_df['Current Value ($)'].astype(float).sum()
    optimized_total_gain_loss = st.session_state.optimized_portfolio_df['Gain/Loss ($)'].astype(float).sum()
    total_gain_loss = st.session_state.portfolio_df['Gain/Loss ($)'].astype(float).sum()
    st.subheader("Portfolio Summary (Harm Reduction Optimized)")
    col1, col2 = st.columns(2)
    portfolio_change = optimized_portfolio_value - portfolio_value
    with col1:
        st.metric(
            "Total Portfolio Value", 
            f"${optimized_portfolio_value:,.2f}", 
            delta=f"${portfolio_change:,.2f}", 
            delta_color="normal" if portfolio_change > 0 else "inverse"
        )
            

    with col2:
        st.metric("Total Portfolio Gain/Loss", 
              f"${optimized_total_gain_loss:,.2f}", 
              delta=f"${optimized_total_gain_loss-total_gain_loss:,.2f}", 
              delta_color="inverse" if optimized_total_gain_loss-total_gain_loss < 0 else "normal")

    
    portfolio_df = st.session_state.portfolio_df[['Stock', 'Portfolio Harm Contribution', 'Units']].copy()

    # Convert columns to numeric and fill NaNs with 0
    portfolio_df['Portfolio Harm Contribution'] = pd.to_numeric(portfolio_df['Portfolio Harm Contribution'], errors='coerce').fillna(0)
    portfolio_df['Units'] = pd.to_numeric(portfolio_df['Units'], errors='coerce').fillna(0)

    # Calculate total harm score units
    total_harm_score_units = (portfolio_df['Portfolio Harm Contribution'] * portfolio_df['Units']).sum()

    # Create a new column for harm score contribution percentage
    portfolio_df['Harm Score Contribution (%)'] = (
        (portfolio_df['Portfolio Harm Contribution'] * portfolio_df['Units']) / total_harm_score_units * 100
    ).fillna(0)

    # Prepare data for harm score contribution chart
    contribution_data = portfolio_df[['Stock', 'Harm Score Contribution (%)']]



# Option to remove stocks from portfolio in sidebar
with st.sidebar:
    st.header("Remove Stocks from Portfolio")
    stocks_to_remove = st.multiselect("Select stocks to remove", 
                                      options=st.session_state.portfolio_df['Stock'].unique())

    if st.button("Remove Selected Stocks", key="remove_stocks_button"):
        st.session_state.portfolio_df = st.session_state.portfolio_df[
            ~st.session_state.portfolio_df['Stock'].isin(stocks_to_remove)]
        
        # Update portfolio allocation after removal
        st.session_state.portfolio_df = update_portfolio_allocation(st.session_state.portfolio_df)
        st.success("Selected stocks removed from portfolio.")

# Sidebar Button to Generate and Download Report
st.sidebar.header("Download Portfolio Report")
if st.sidebar.button("Generate Report"):
    if not st.session_state.portfolio_df.empty:
        pdf_buffer = generate_portfolio_pdf()
        st.sidebar.download_button(
            label="Download Portfolio Report (PDF)",
            data=pdf_buffer,
            file_name="portfolio_report.pdf",
            mime="application/pdf"
        )
    else:
        st.sidebar.warning("Please add stocks to your portfolio first.")
    
