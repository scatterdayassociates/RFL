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
        'Portfolio Allocation', 'GICS Sector','Sector Harm Score' ,'Portfolio Harm Contribution',
        'IndexAlign DEI Pro %', 'IndexAlign DEI Neutral %', 'IndexAlign DEI Against %'
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
        SELECT `Min-Max-Norm` FROM stockracialharm2 WHERE sector = ?
    """
    scores = pd.read_sql_query(query, conn, params=(sector,))
    conn.close()
    return scores.iloc[0]['Min-Max-Norm'] if not scores.empty else None

def optimize_portfolio(df, max_harm_score, min_stock_threshold):
    """
    Optimize portfolio to maximize Rate of Return (RoR) by reallocating units among stocks,
    subject to:
    - Minimum harm score constraint (optimized harm score must be >= max_harm_score)
    - Minimum stock holding threshold
    - Portfolio value must not decrease (must be >= original)
    - Gain/Loss must not decrease (must be >= original)
    """
    
    # Prepare data
    current_prices = df['Current Price ($)'].astype(float)
    returns = df['Gain/Loss %'].astype(float) / 100
    harm_scores = df['Sector Harm Score'].astype(float)
    current_values = df['Current Value ($)'].astype(float)
    current_units = df['Units'].astype(float)
    purchase_prices = df['Purchase Price ($)'].astype(float)
    
    total_portfolio_value = current_values.sum()
    initial_total_return = df['Gain/Loss ($)'].astype(float).sum()
    initial_total_investment = df['Initial Investment ($)'].astype(float).sum()
    initial_ror = (initial_total_return / initial_total_investment * 100) if initial_total_investment > 0 else 0
    total_units = current_units.sum()
    
    # Ensure we have valid data
    if total_units <= 0 or total_portfolio_value <= 0:
        raise ValueError("Invalid portfolio: total units or portfolio value must be positive")
    
    initial_weights = current_values / total_portfolio_value
    
    # Objective: Maximize Rate of Return (minimize negative RoR)
    def objective(weights):
        new_values = weights * total_portfolio_value
        new_units = new_values / current_prices
        # Ensure no negative units
        new_units = np.maximum(new_units, 0)
        new_initial_investment = new_units * purchase_prices
        new_total_return = np.sum(new_values - new_initial_investment)
        new_total_investment = np.sum(new_initial_investment)
        # Calculate RoR as percentage
        if new_total_investment > 0:
            ror = (new_total_return / new_total_investment) * 100
        else:
            ror = -1000  # Penalty for invalid solution
        # Minimize negative RoR (which maximizes RoR)
        return -ror
  
    # Constraint: Weighted harm score must be >= max_harm_score (minimum threshold)
    def harm_score_constraint(weights):
        new_values = weights * total_portfolio_value
        new_units = new_values / current_prices
        new_units = np.maximum(new_units, 0)  # Ensure non-negative
        total_new_units = new_units.sum()
        if total_new_units <= 0:
            return -1000  # Invalid solution
        weighted_harm = np.sum(harm_scores * new_units) / total_new_units
        # Constraint: weighted_harm >= max_harm_score (minimum threshold)
        # Return value should be >= 0 for constraint satisfaction
        return weighted_harm - max_harm_score
  
    # Constraint: Total units must equal original total
    def units_constraint(weights):
        new_values = weights * total_portfolio_value
        new_units = new_values / current_prices
        new_units = np.maximum(new_units, 0)  # Ensure non-negative
        return new_units.sum() - total_units 
    
    # Constraint: Weights must sum to 1.0
    def weight_sum_constraint(weights):
        return np.sum(weights) - 1.0
    
    # Constraint: Ensure portfolio value stays positive (all weights >= 0)
    def non_negative_weights_constraint(weights):
        return np.min(weights)
    
    # Constraint: Portfolio value must not decrease (must be >= original)
    def portfolio_value_constraint(weights):
        new_values = weights * total_portfolio_value
        new_units = new_values / current_prices
        new_units = np.maximum(new_units, 0)  # Ensure non-negative
        # Recalculate actual values with continuous units
        actual_values = new_units * current_prices
        actual_total = actual_values.sum()
        # Constraint: actual_total >= total_portfolio_value
        # Return value should be >= 0 for constraint satisfaction
        return actual_total - total_portfolio_value
    
    # Constraint: Gain/Loss must not decrease (must be >= original)
    def gain_loss_constraint(weights):
        new_values = weights * total_portfolio_value
        new_units = new_values / current_prices
        new_units = np.maximum(new_units, 0)  # Ensure non-negative
        # Recalculate actual values with continuous units
        actual_values = new_units * current_prices
        new_initial_investment = new_units * purchase_prices
        new_total_return = np.sum(actual_values - new_initial_investment)
        # Constraint: new_total_return >= initial_total_return
        # Return value should be >= 0 for constraint satisfaction
        return new_total_return - initial_total_return
    
    # Calculate minimum weights based on minimum stock threshold
    min_weights = np.array((min_stock_threshold * current_prices) / total_portfolio_value)
    # Ensure min_weights don't exceed 1.0
    min_weights = np.minimum(min_weights, 0.99)
    
    # Build constraints list
    constraints = [
        {"type": "ineq", "fun": harm_score_constraint},  # weighted_harm >= max_harm_score (minimum threshold)
        {"type": "eq", "fun": units_constraint},         # total units = original total
        {"type": "eq", "fun": weight_sum_constraint},    # weights sum to 1.0
        {"type": "ineq", "fun": non_negative_weights_constraint},  # all weights >= 0
        {"type": "ineq", "fun": portfolio_value_constraint},  # portfolio value >= original (must not decrease)
        {"type": "ineq", "fun": gain_loss_constraint}  # gain/loss >= original (must not decrease)
    ]
    
    # Add minimum weight constraints for each stock
    for i in range(len(df)):
        constraints.append({
            "type": "ineq",
            "fun": lambda w, idx=i: w[idx] - min_weights[idx]
        })
    
    # Bounds: weights between 0 and 1 (no need for 1.1 since we have weight sum constraint)
    bounds = [(0, 1)] * len(df)
    
    # Run optimization
    result = minimize(
        fun=objective,
        x0=initial_weights,
        bounds=bounds,
        constraints=constraints,
        method='SLSQP',
        options={'maxiter': 1000, 'ftol': 1e-6}
    )
    
    if result.success:
        optimized_weights = result.x
        new_values = optimized_weights * total_portfolio_value
        new_units = new_values / current_prices
        new_units = np.maximum(new_units, 0)  # Ensure non-negative
        
        # Adjust units to exactly match total_units (distribute rounding error)
        current_total = new_units.sum()
        if abs(current_total - total_units) > 0.01:
            # Scale to match exactly
            if current_total > 0:
                new_units = new_units * (total_units / current_total)
                # Recalculate values after scaling
                new_values = new_units * current_prices
            else:
                raise ValueError("Optimization failed: All units became zero")
        
        # Smart rounding: Round units to preserve total portfolio value
        # Use value-weighted rounding strategy
        
        # Calculate target values for each stock
        target_values = new_values.copy()
        
        # Round units using value-preserving strategy
        # For each stock, decide whether to round up or down based on value preservation
        new_units_rounded = np.zeros(len(new_units), dtype=int)
        total_rounded = 0
        
        # First pass: round each stock, prioritizing value preservation
        for i in range(len(new_units)):
            floor_units = int(np.floor(new_units[i]))
            ceil_units = int(np.ceil(new_units[i]))
            
            floor_value = floor_units * current_prices[i]
            ceil_value = ceil_units * current_prices[i]
            target_value = target_values[i]
            
            # Choose rounding direction that preserves value better
            floor_diff = abs(floor_value - target_value)
            ceil_diff = abs(ceil_value - target_value)
            
            if floor_diff <= ceil_diff:
                new_units_rounded[i] = floor_units
            else:
                new_units_rounded[i] = ceil_units
            
            total_rounded += new_units_rounded[i]
        
        # Adjust to match total units exactly
        rounding_diff = int(total_units - total_rounded)
        
        if rounding_diff != 0:
            # Calculate value impact of adding/removing one unit from each stock
            current_rounded_values = new_units_rounded * current_prices
            value_deviations = target_values - current_rounded_values
            
            if rounding_diff > 0:
                # Need to add units - add to stocks where it best preserves target value
                # Sort by how much adding a unit improves value match
                improvements = []
                for i in range(len(new_units_rounded)):
                    if current_prices[i] > 0:
                        # Value if we add one unit
                        new_value = (new_units_rounded[i] + 1) * current_prices[i]
                        improvement = abs(target_values[i] - new_value) - abs(value_deviations[i])
                        improvements.append((improvement, i))
                
                # Sort by improvement (best first) and add units
                improvements.sort(reverse=True)
                for _, idx in improvements[:rounding_diff]:
                    new_units_rounded[idx] += 1
                    
            else:
                # Need to remove units - remove from stocks where it least hurts value match
                # Sort by how much removing a unit worsens value match
                deteriorations = []
                for i in range(len(new_units_rounded)):
                    if new_units_rounded[i] > 0:
                        # Value if we remove one unit
                        new_value = (new_units_rounded[i] - 1) * current_prices[i]
                        deterioration = abs(target_values[i] - new_value) - abs(value_deviations[i])
                        deteriorations.append((deterioration, i))
                
                # Sort by deterioration (least bad first) and remove units
                deteriorations.sort()
                for _, idx in deteriorations[:abs(rounding_diff)]:
                    new_units_rounded[idx] = max(0, new_units_rounded[idx] - 1)
        
        new_units = new_units_rounded.astype(float)
        
        # Recalculate values with rounded units
        new_values = new_units * current_prices
        actual_total_value = new_values.sum()
        
        # If we still lost value, try one more optimization pass
        value_loss = total_portfolio_value - actual_total_value
        if value_loss > 0.01:  # Try to recover value loss
            # Try swapping units between stocks to recover value
            # Find stocks where we can gain value by adding a unit
            sorted_by_price_high = np.argsort(current_prices)[::-1]
            sorted_by_price_low = np.argsort(current_prices)
            
            max_iterations = 10  # Limit iterations
            iteration = 0
            while value_loss > 0.01 and iteration < max_iterations:
                iteration += 1
                best_swap = None
                best_gain = 0
                
                # Find best swap: remove from low-price stock, add to high-price stock
                for low_idx in sorted_by_price_low:
                    if new_units[low_idx] <= 0:
                        continue
                    for high_idx in sorted_by_price_high:
                        if high_idx == low_idx:
                            continue
                        # Value gain from swapping (removing from low, adding to high)
                        gain = current_prices[high_idx] - current_prices[low_idx]
                        if gain > best_gain:
                            best_gain = gain
                            best_swap = (low_idx, high_idx)
                
                if best_swap:
                    low_idx, high_idx = best_swap
                    new_units[low_idx] = max(0, new_units[low_idx] - 1)
                    new_units[high_idx] += 1
                    # Recalculate
                    new_values = new_units * current_prices
                    actual_total_value = new_values.sum()
                    value_loss = total_portfolio_value - actual_total_value
                else:
                    break
        
        # Final recalculation
        new_values = new_units * current_prices
        actual_total_value = new_values.sum()
        
        # Final check: if value is still below target, make one more adjustment
        # This ensures we don't lose value due to rounding
        if actual_total_value < total_portfolio_value - 0.01:
            # Calculate how much we need to recover
            needed_recovery = total_portfolio_value - actual_total_value
            
            # Try to recover by finding the best unit adjustments
            # Strategy: find stocks where we can gain value by adjusting units
            # while maintaining total units constraint
            
            # Calculate value per unit for each stock
            value_per_unit = current_prices
            
            # Find best opportunities: stocks with high value per unit that could use more units
            # and stocks with low value per unit that could give up units
            sorted_high = np.argsort(value_per_unit)[::-1]
            sorted_low = np.argsort(value_per_unit)
            
            # Try to make adjustments that recover value
            for attempt in range(5):  # Try up to 5 adjustments
                if actual_total_value >= total_portfolio_value - 0.01:
                    break
                    
                best_adjustment = None
                best_recovery = 0
                
                # Look for swaps that recover value
                for low_idx in sorted_low:
                    if new_units[low_idx] <= 0:
                        continue
                    for high_idx in sorted_high:
                        if high_idx == low_idx:
                            continue
                        # Recovery from swapping
                        recovery = value_per_unit[high_idx] - value_per_unit[low_idx]
                        if recovery > best_recovery and recovery <= needed_recovery + 0.01:
                            best_recovery = recovery
                            best_adjustment = (low_idx, high_idx)
                
                if best_adjustment:
                    low_idx, high_idx = best_adjustment
                    new_units[low_idx] = max(0, new_units[low_idx] - 1)
                    new_units[high_idx] += 1
                    # Recalculate
                    new_values = new_units * current_prices
                    actual_total_value = new_values.sum()
                    needed_recovery = total_portfolio_value - actual_total_value
                else:
                    break
        
        # Final recalculation after all adjustments
        new_values = new_units * current_prices
        actual_total_value = new_values.sum()
        
        new_initial_investment = new_units * purchase_prices
        new_total_return = np.sum(new_values - new_initial_investment)
        
        # Validation checks
        if abs(new_units.sum() - total_units) > 0.01:
            raise ValueError(f"Optimization failed: Total units constraint not met. Expected {total_units}, got {new_units.sum()}")
        
        if np.any(new_units < 0):
            raise ValueError("Optimization failed: Negative units detected")
        
        if np.any(new_values < 0):
            raise ValueError("Optimization failed: Negative portfolio values detected")
        
        if np.any(new_initial_investment < 0):
            raise ValueError("Optimization failed: Negative initial investment detected")
        
        # Ensure all values are positive
        new_values = np.maximum(new_values, 0)
        new_initial_investment = np.maximum(new_initial_investment, 0)
        
        # Final validation: ensure total value is reasonable and does not decrease
        final_total_value = new_values.sum()
        if final_total_value <= 0:
            raise ValueError(f"Optimization failed: Total portfolio value is non-positive: {final_total_value}")
        
        # Prevent optimization if value decreases (allow only tiny tolerance for floating point errors)
        value_decrease = total_portfolio_value - final_total_value
        if value_decrease > 0.01:  # If value decreased by more than 1 cent, reject optimization
            raise ValueError(
                f"Optimization rejected: Portfolio value would decrease from ${total_portfolio_value:,.2f} "
                f"to ${final_total_value:,.2f} (decrease of ${value_decrease:,.2f}). "
                f"Optimization cannot proceed if it results in value loss. "
                f"Please adjust your constraints (minimum harm score threshold or minimum stock holding) to allow for a solution that maintains or increases portfolio value."
            )
        
        # Prevent optimization if gain/loss decreases
        gain_loss_decrease = initial_total_return - new_total_return
        if gain_loss_decrease > 0.01:  # If gain/loss decreased by more than 1 cent, reject optimization
            raise ValueError(
                f"Optimization rejected: Portfolio gain/loss would decrease from ${initial_total_return:,.2f} "
                f"to ${new_total_return:,.2f} (decrease of ${gain_loss_decrease:,.2f}). "
                f"Optimization cannot proceed if it results in gain/loss reduction. "
                f"Please adjust your constraints (minimum harm score threshold or minimum stock holding) to allow for a solution that maintains or increases gain/loss."
            )
        
        total_new_units = new_units.sum()
        if total_new_units > 0:
            weighted_harm = np.sum(harm_scores * new_units) / total_new_units
            if weighted_harm < max_harm_score - 0.01:  # Small tolerance for floating point
                raise ValueError(
                    f"Optimization failed: Harm score {weighted_harm:.2f} is below minimum threshold {max_harm_score:.2f}. "
                    f"Optimized portfolio harm score must be >= {max_harm_score:.2f}."
                )
        else:
            raise ValueError("Optimization failed: Total units is zero")
        
        # Update dataframe
        df['Portfolio Allocation'] = (new_values / new_values.sum() * 100) if new_values.sum() > 0 else 0
        df['Units'] = new_units
        df['Current Value ($)'] = new_values
        df['Initial Investment ($)'] = new_initial_investment
        # Calculate individual stock gain/loss
        df['Gain/Loss ($)'] = new_values - new_initial_investment
        df['Gain/Loss %'] = (df['Gain/Loss ($)'] / df['Initial Investment ($)']) * 100
        df['Gain/Loss %'] = df['Gain/Loss %'].fillna(0)
        
        # Recalculate harm contribution
        total_harm_units = (df['Units'] * df['Sector Harm Score']).sum()
        if total_harm_units > 0:
            df['Portfolio Harm Contribution'] = (df['Units'] * df['Sector Harm Score']) / total_harm_units * 100
        else:
            df['Portfolio Harm Contribution'] = 0
        
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
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    portfolio_value_df = pd.DataFrame(index=date_range)
    portfolio_value_df['Total Value'] = 0.0
    
    # Calculate daily value for each stock
    for _, row in portfolio_df.iterrows():
        try:
            # Get historical data for the stock
            stock = yf.Ticker(row['Stock'])
            hist_data = stock.history(start=row['Purchase Date'], end=end_date)
            
            
            hist_data.index = hist_data.index.tz_localize(None)

            # Calculate daily value (price * units)
            units = float(row['Units'])
            daily_value = hist_data['Close'] * units
            
            # Only consider values from purchase date onwards
            daily_value = daily_value[daily_value.index >= row['Purchase Date']]
            
            # Add to total portfolio value
            portfolio_value_df.loc[daily_value.index, 'Total Value'] += daily_value
            
        except Exception as e:
            st.warning(f"Error fetching historical data for {row['Stock']}: {str(e)}")
            continue
    
    portfolio_value_df.iloc[1:] = portfolio_value_df.iloc[1:].replace(0, np.nan).ffill()
    
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

def get_indexalign_data(ticker):
    """Fetch IndexAlign DEI data from database by ticker"""
    conn = sqlite3.connect('nycprocurement.db')
    
    query = "SELECT Pro, Neutral, Against FROM indexaligncorporate WHERE Ticker = ?"
    result = pd.read_sql_query(query, conn, params=(ticker,))
    conn.close()
    
    if not result.empty:
        return {
            'pro': result.iloc[0]['Pro'] if pd.notna(result.iloc[0]['Pro']) else None,
            'neutral': result.iloc[0]['Neutral'] if pd.notna(result.iloc[0]['Neutral']) else None,
            'against': result.iloc[0]['Against'] if pd.notna(result.iloc[0]['Against']) else None
        }
    return {'pro': None, 'neutral': None, 'against': None}
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
                
                # Fetch IndexAlign DEI data
                indexalign_data = get_indexalign_data(ticker)
                
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
                    'Normalized Harm Score Graph': [normalized_score_graph],
                    'IndexAlign DEI Pro %': [indexalign_data['pro'] if indexalign_data['pro'] is not None else 'N/A'],
                    'IndexAlign DEI Neutral %': [indexalign_data['neutral'] if indexalign_data['neutral'] is not None else 'N/A'],
                    'IndexAlign DEI Against %': [indexalign_data['against'] if indexalign_data['against'] is not None else 'N/A']
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
            "Set Minimum Harm Score Threshold",
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
            # Check if recommended_ticker is a company name (needs conversion) or already a ticker
            if recommended_ticker in company_ticker_dict:
                # It's a company name from asyousowrj, convert to ticker
                new_ticker = company_ticker_dict[recommended_ticker]
            else:
                # It's already a ticker from IndexAlign, use it directly
                new_ticker = recommended_ticker
            
            # Get the row data to preserve Units and Purchase Date
            row_data = updated_df.loc[mask].iloc[0]
            units = float(row_data['Units'])
            purchase_date = row_data['Purchase Date']
            
            # Fetch new stock data
            new_stock = yf.Ticker(new_ticker)
            new_price = new_stock.info.get('currentPrice', 0)
            
            # Get purchase price for the new stock on the purchase date
            try:
                hist = new_stock.history(start=purchase_date)
                if not hist.empty:
                    purchase_price = hist.iloc[0]['Close']
                else:
                    purchase_price = new_price  # Fallback to current price
            except:
                purchase_price = new_price  # Fallback to current price
            
            # Calculate financial metrics
            initial_investment = purchase_price * units
            current_value = new_price * units
            gain_loss = current_value - initial_investment
            gain_loss_percentage = (gain_loss / initial_investment) * 100 if initial_investment > 0 else 0
            
            # Get sector and harm score data
            gics_sector = get_gics_sector(new_ticker)
            sector_harm_score = get_sector_harm_score(gics_sector)
            normalized_score_graph = get_normalized_score_graph1(gics_sector) or np.random.uniform(0, 1)
            
            # Fetch IndexAlign DEI data
            indexalign_data = get_indexalign_data(new_ticker)
            
            # Get the index of the row to update
            row_idx = updated_df[mask].index[0]
            
            # Update the row with all new data using .at for single cell updates
            updated_df.at[row_idx, 'Stock'] = new_ticker
            updated_df.at[row_idx, 'Current Price ($)'] = f"{new_price:.2f}"
            updated_df.at[row_idx, 'Purchase Price ($)'] = f"{purchase_price:.2f}"
            updated_df.at[row_idx, 'Initial Investment ($)'] = f"{initial_investment:.2f}"
            updated_df.at[row_idx, 'Current Value ($)'] = f"{current_value:.2f}"
            updated_df.at[row_idx, 'Gain/Loss ($)'] = f"{gain_loss:.2f}"
            updated_df.at[row_idx, 'Gain/Loss %'] = gain_loss_percentage
            updated_df.at[row_idx, 'GICS Sector'] = gics_sector
            updated_df.at[row_idx, 'Sector Harm Score'] = sector_harm_score
            updated_df.at[row_idx, 'Normalized Harm Score Graph'] = normalized_score_graph
            updated_df.at[row_idx, 'IndexAlign DEI Pro %'] = indexalign_data['pro'] if indexalign_data['pro'] is not None else 'N/A'
            updated_df.at[row_idx, 'IndexAlign DEI Neutral %'] = indexalign_data['neutral'] if indexalign_data['neutral'] is not None else 'N/A'
            updated_df.at[row_idx, 'IndexAlign DEI Against %'] = indexalign_data['against'] if indexalign_data['against'] is not None else 'N/A'
            
            # Ensure the dataframe is properly updated
            st.session_state.portfolio_df = updated_df.copy()
            
            # Recalculate harm score contributions for all stocks
            total_portfolio_harm_units = (
                st.session_state.portfolio_df['Normalized Harm Score Graph'].astype(float) * 
                st.session_state.portfolio_df['Units'].astype(float)
            ).sum()
            
            st.session_state.portfolio_df['Portfolio Harm Contribution'] = (
                st.session_state.portfolio_df['Normalized Harm Score Graph'].astype(float) * 
                st.session_state.portfolio_df['Units'].astype(float) / 
                total_portfolio_harm_units * 100
            )
            
            # Update portfolio allocation and historical data
            st.session_state.portfolio_df = update_portfolio_allocation(st.session_state.portfolio_df)
            st.session_state.historical_value_df = calculate_historical_portfolio_value(st.session_state.portfolio_df)
            
            # Clear optimized portfolio since it's now outdated
            st.session_state.optimized_portfolio_df = None

            st.success(f"Replaced {selected_ticker} with {new_ticker}.")
        else:
            st.error(f"{selected_ticker} not found in portfolio.")

    except Exception as e:
        st.error(f"An error occurred in execute_trade: {str(e)}")
        print(f"Error in execute_trade: {e}")  # This will log the error in the terminal/console


def load_recommendations():
    """Fetch recommendations from both asyousowrj and indexaligncorporate tables."""
         
    conn = sqlite3.connect('nycprocurement.db')
    
    recommendations = {}
    for _, row in st.session_state.portfolio_df.iterrows():
        stock_ticker = row['Stock']
        sector = row['GICS Sector']
        
        # Initialize recommendations list for this stock
        recommendations[stock_ticker] = []

        # Method 1: Query recommendation from asyousowrj table
        query = "SELECT * FROM asyousowrj WHERE Sector = ? AND Category = 'Leader'"
        cursor = conn.cursor()
        cursor.execute(query, (sector,))
        result = cursor.fetchone()

        # Store asyousowrj recommendation
        recommendations[stock_ticker].append({
            'recommended_ticker': result[0] if result else 'NA',
            'recommending_source': result[6] if result else 'NA',
            'distinction_signal': result[1] if result else 'NA'
        })
        
        # Method 2: Query recommendation from indexaligncorporate table
        # Step 1: Fetch sector from indexaligncorporate using ticker
        sector_query = "SELECT Sector FROM indexaligncorporate WHERE Ticker = ?"
        cursor.execute(sector_query, (stock_ticker,))
        sector_result = cursor.fetchone()
        
        if sector_result:
            indexalign_sector = sector_result[0]
            
            # Step 2: Find tickers in the same sector with Pro > 85%
            # Handle VARCHAR Pro column that may contain percentage values, spaces, commas, etc.
            recommendation_query = """
                SELECT Ticker, Name, Sector, Pro, Neutral, Against 
                FROM indexaligncorporate 
                WHERE Sector = ? 
                AND Ticker != ?
                AND CAST(REPLACE(REPLACE(REPLACE(Pro, '%', ''), ' ', ''), ',', '') AS REAL) > 85
                LIMIT 1
            """
            cursor.execute(recommendation_query, (indexalign_sector, stock_ticker))
            result = cursor.fetchone()
            
            # Store indexaligncorporate recommendation
            if result:
                recommendations[stock_ticker].append({
                    'recommended_ticker': result[0] if result[0] else 'NA',  # Ticker
                    'recommending_source': 'IndexAlign Corporate Political Giving',
                    'distinction_signal': 'Recipients > 85% Pro-DEI Voting'
                })
            else:
                recommendations[stock_ticker].append({
                    'recommended_ticker': 'NA',
                    'recommending_source': 'IndexAlign Corporate Political Giving',
                    'distinction_signal': 'NA'
                })
        else:
            # If ticker not found in indexaligncorporate, still add entry with NA
            recommendations[stock_ticker].append({
                'recommended_ticker': 'NA',
                'recommending_source': 'IndexAlign Corporate Political Giving',
                'distinction_signal': 'NA'
            })

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
        
        # Generate rows with data and checkboxes - display both recommendations as separate rows
        row_counter = 0
        for stock_ticker, recommendation_list in recommendations.items():
            sector = st.session_state.portfolio_df.loc[st.session_state.portfolio_df['Stock'] == stock_ticker, 'GICS Sector'].values[0]
            
            # Display each recommendation as a separate, independent row
            for rec_idx, recommendation in enumerate(recommendation_list):
                row_counter += 1
                recommended_ticker = recommendation['recommended_ticker']
                recommending_source = recommendation['recommending_source']
                distinction_signal = recommendation['distinction_signal']
                
                # Create row with columns (removed Rank column)
                cols = st.columns([1, 1, 1, 1.5, 1, 0.7])
                
                # Display values in columns - each row is independent
                cols[0].write(stock_ticker)
                cols[1].write(sector)
                cols[2].write(recommended_ticker)
                cols[3].write(recommending_source)
                cols[4].write(distinction_signal)
                
                # Add checkbox in the last column
                execute_key = f"execute_{stock_ticker}_{rec_idx}_{row_counter}"
                if cols[5].checkbox("", key=execute_key):
                    st.session_state.selected_trades.add((stock_ticker, recommended_ticker, rec_idx))
                else:
                    st.session_state.selected_trades.discard((stock_ticker, recommended_ticker, rec_idx))
        
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
                        for trade_info in list(st.session_state.selected_trades):
                            # trade_info is a tuple: (stock_ticker, recommended_ticker, rec_idx)
                            if isinstance(trade_info, tuple) and len(trade_info) == 3:
                                stock, recommended_ticker, rec_idx = trade_info
                            else:
                                # Handle old format for backward compatibility
                                stock = trade_info
                                stock_recommendations = recommendations.get(stock, [])
                                if stock_recommendations:
                                    recommended_ticker = stock_recommendations[0].get('recommended_ticker', 'NA')
                                else:
                                    recommended_ticker = 'NA'
                            
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
    # Convert to numeric, handling string values with dollar signs and commas
    portfolio_value = pd.to_numeric(
        st.session_state.portfolio_df['Current Value ($)'].astype(str).str.replace('$', '').str.replace(',', ''), 
        errors='coerce'
    ).fillna(0).sum()
    
    total_gain_loss = pd.to_numeric(
        st.session_state.portfolio_df['Gain/Loss ($)'].astype(str).str.replace('$', '').str.replace(',', ''), 
        errors='coerce'
    ).fillna(0).sum()
    
    # Calculate Weighted Average Harm Score
    # Multiply Units × Sector Harm Score for each stock, then sum and divide by total units
    units = st.session_state.portfolio_df['Units'].astype(float)
    sector_harm_scores = st.session_state.portfolio_df['Sector Harm Score'].astype(float)
    total_units = units.sum()
    
    if total_units > 0:
        weighted_avg_harm_score = (units * sector_harm_scores).sum() / total_units
    else:
        weighted_avg_harm_score = 0

    # Display total portfolio value, total gain/loss, and weighted average harm score
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Portfolio Value", f"${portfolio_value:,.2f}")

    with col2:
        st.metric("Total Portfolio Gain/Loss", 
              f"${total_gain_loss:,.2f}")
    
    with col3:
        st.metric("Weighted Average RFL Corporate Racial Justice Score", 
              f"{weighted_avg_harm_score:.2f}")

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

    # Create a new row for the two new doughnut charts
    col5, col6 = st.columns(2)
    
    # Chart 1: Portfolio Value Contribution by Stock
    value_contribution_df = st.session_state.portfolio_df[['Stock', 'Current Value ($)']].copy()
    value_contribution_df['Current Value ($)'] = pd.to_numeric(
        value_contribution_df['Current Value ($)'].astype(str).str.replace('$', '').str.replace(',', ''), 
        errors='coerce'
    ).fillna(0)
    
    total_value = value_contribution_df['Current Value ($)'].sum()
    if total_value > 0:
        value_contribution_df['Value Contribution %'] = (value_contribution_df['Current Value ($)'] / total_value) * 100
        
        fig3 = px.pie(
            value_contribution_df,
            names='Stock',
            values='Value Contribution %',
            hole=0.4,
            title="Portfolio Value Contribution by Stock",
            labels={'Value Contribution %': 'Contribution (%)'},
            color='Stock',
            color_discrete_map=stock_color_map
        )
        
        with col5:
            st.plotly_chart(fig3, key="original_value_contribution_chart")
    
    # Chart 2: IndexAlign Political Giving DEI Voting Record
    indexalign_df = st.session_state.portfolio_df[['Stock', 'IndexAlign DEI Pro %', 'IndexAlign DEI Neutral %', 'IndexAlign DEI Against %', 'Current Value ($)']].copy()
    
    # Convert Current Value to numeric for weighting
    indexalign_df['Current Value ($)'] = pd.to_numeric(
        indexalign_df['Current Value ($)'].astype(str).str.replace('$', '').str.replace(',', ''), 
        errors='coerce'
    ).fillna(0)
    
    # Convert to numeric, handling 'N/A' and string values
    def convert_to_numeric(val):
        if val == 'N/A' or pd.isna(val):
            return None
        # Remove % sign if present and convert to float
        if isinstance(val, str):
            val = val.replace('%', '').strip()
        try:
            return float(val)
        except:
            return None
    
    indexalign_df['Pro'] = indexalign_df['IndexAlign DEI Pro %'].apply(convert_to_numeric)
    indexalign_df['Neutral'] = indexalign_df['IndexAlign DEI Neutral %'].apply(convert_to_numeric)
    indexalign_df['Against'] = indexalign_df['IndexAlign DEI Against %'].apply(convert_to_numeric)
    
    # Calculate weighted average for Pro (weighted by Current Value)
    # Filter out rows where Pro is None/NaN
    pro_mask = indexalign_df['Pro'].notna()
    if pro_mask.any():
        total_value = indexalign_df.loc[pro_mask, 'Current Value ($)'].sum()
        if total_value > 0:
            weighted_avg_pro = (indexalign_df.loc[pro_mask, 'Pro'] * indexalign_df.loc[pro_mask, 'Current Value ($)']).sum() / total_value
        else:
            weighted_avg_pro = 0
    else:
        weighted_avg_pro = 0
    
    # Calculate simple averages for Neutral and Against (excluding None/NaN values)
    neutral_values = indexalign_df['Neutral'].dropna()
    against_values = indexalign_df['Against'].dropna()
    
    avg_neutral = neutral_values.mean() if len(neutral_values) > 0 else 0
    avg_against = against_values.mean() if len(against_values) > 0 else 0
    
    # Divide by 100 to convert percentage to decimal (as per requirement: AVERAGE(...)/100%)
    weighted_avg_pro_decimal = weighted_avg_pro / 100 if weighted_avg_pro > 0 else 0
    avg_neutral_decimal = avg_neutral / 100 if avg_neutral > 0 else 0
    avg_against_decimal = avg_against / 100 if avg_against > 0 else 0
    
    # Normalize to percentages for pie chart (ensure they sum to 100%)
    total_avg = weighted_avg_pro_decimal + avg_neutral_decimal + avg_against_decimal
    if total_avg > 0:
        avg_pro_pct = (weighted_avg_pro_decimal / total_avg) * 100
        avg_neutral_pct = (avg_neutral_decimal / total_avg) * 100
        avg_against_pct = (avg_against_decimal / total_avg) * 100
    else:
        avg_pro_pct = 0
        avg_neutral_pct = 0
        avg_against_pct = 0
    
    # Create data for the chart
    indexalign_chart_data = pd.DataFrame({
        'Category': ['Pro', 'Neutral', 'Against'],
        'Percentage': [avg_pro_pct, avg_neutral_pct, avg_against_pct]
    })
    
    # Define colors for the three categories
    indexalign_colors = {'Pro': '#2ecc71', 'Neutral': '#f39c12', 'Against': '#e74c3c'}
    
    if total_avg > 0:
        fig4 = px.pie(
            indexalign_chart_data,
            names='Category',
            values='Percentage',
            hole=0.4,
            title="IndexAlign Political Giving DEI Voting Record",
            labels={'Percentage': 'Average (%)'},
            color='Category',
            color_discrete_map=indexalign_colors
        )
        
        with col6:
            st.plotly_chart(fig4, key="original_indexalign_chart")

# Display optimized portfolio
if st.session_state.optimized_portfolio_df is not None:
    st.subheader("Harm Reduction Optimized Portfolio")
    if hasattr(st.session_state, 'optimization_success') and st.session_state.optimization_success:
        st.success(
        "**Portfolio Optimization Insight:**\n"
        "This rebalanced portfolio allocation maximizes total financial return while "
        "ensuring the harm profile, or average weighted harm score, of the holdings "
        "in your portfolio meets or exceeds the user-selected minimum harm threshold and required holding per stock. "
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

    # Create a new row for the two new doughnut charts (optimized portfolio)
    col7, col8 = st.columns(2)
    
    # Chart 1: Portfolio Value Contribution by Stock (Optimized)
    optimized_value_contribution_df = st.session_state.optimized_portfolio_df[['Stock', 'Current Value ($)']].copy()
    optimized_value_contribution_df['Current Value ($)'] = pd.to_numeric(
        optimized_value_contribution_df['Current Value ($)'].astype(str).str.replace('$', '').str.replace(',', ''), 
        errors='coerce'
    ).fillna(0)
    
    optimized_total_value = optimized_value_contribution_df['Current Value ($)'].sum()
    if optimized_total_value > 0:
        optimized_value_contribution_df['Value Contribution %'] = (optimized_value_contribution_df['Current Value ($)'] / optimized_total_value) * 100
        
        fig5 = px.pie(
            optimized_value_contribution_df,
            names='Stock',
            values='Value Contribution %',
            hole=0.4,
            title="Portfolio Value Contribution by Stock",
            labels={'Value Contribution %': 'Contribution (%)'},
            color='Stock',
            color_discrete_map=stock_color_map
        )
        
        with col7:
            st.plotly_chart(fig5, key="optimized_value_contribution_chart")
    
    # Chart 2: IndexAlign Political Giving DEI Voting Record (Optimized)
    optimized_indexalign_df = st.session_state.optimized_portfolio_df[['Stock', 'IndexAlign DEI Pro %', 'IndexAlign DEI Neutral %', 'IndexAlign DEI Against %', 'Current Value ($)']].copy()
    
    # Convert Current Value to numeric for weighting
    optimized_indexalign_df['Current Value ($)'] = pd.to_numeric(
        optimized_indexalign_df['Current Value ($)'].astype(str).str.replace('$', '').str.replace(',', ''), 
        errors='coerce'
    ).fillna(0)
    
    # Convert to numeric, handling 'N/A' and string values
    def convert_to_numeric_optimized(val):
        if val == 'N/A' or pd.isna(val):
            return None
        # Remove % sign if present and convert to float
        if isinstance(val, str):
            val = val.replace('%', '').strip()
        try:
            return float(val)
        except:
            return None
    
    optimized_indexalign_df['Pro'] = optimized_indexalign_df['IndexAlign DEI Pro %'].apply(convert_to_numeric_optimized)
    optimized_indexalign_df['Neutral'] = optimized_indexalign_df['IndexAlign DEI Neutral %'].apply(convert_to_numeric_optimized)
    optimized_indexalign_df['Against'] = optimized_indexalign_df['IndexAlign DEI Against %'].apply(convert_to_numeric_optimized)
    
    # Calculate weighted average for Pro (weighted by Current Value)
    # Filter out rows where Pro is None/NaN
    optimized_pro_mask = optimized_indexalign_df['Pro'].notna()
    if optimized_pro_mask.any():
        optimized_total_value = optimized_indexalign_df.loc[optimized_pro_mask, 'Current Value ($)'].sum()
        if optimized_total_value > 0:
            optimized_weighted_avg_pro = (optimized_indexalign_df.loc[optimized_pro_mask, 'Pro'] * optimized_indexalign_df.loc[optimized_pro_mask, 'Current Value ($)']).sum() / optimized_total_value
        else:
            optimized_weighted_avg_pro = 0
    else:
        optimized_weighted_avg_pro = 0
    
    # Calculate simple averages for Neutral and Against (excluding None/NaN values)
    optimized_neutral_values = optimized_indexalign_df['Neutral'].dropna()
    optimized_against_values = optimized_indexalign_df['Against'].dropna()
    
    optimized_avg_neutral = optimized_neutral_values.mean() if len(optimized_neutral_values) > 0 else 0
    optimized_avg_against = optimized_against_values.mean() if len(optimized_against_values) > 0 else 0
    
    # Divide by 100 to convert percentage to decimal (as per requirement: AVERAGE(...)/100%)
    optimized_weighted_avg_pro_decimal = optimized_weighted_avg_pro / 100 if optimized_weighted_avg_pro > 0 else 0
    optimized_avg_neutral_decimal = optimized_avg_neutral / 100 if optimized_avg_neutral > 0 else 0
    optimized_avg_against_decimal = optimized_avg_against / 100 if optimized_avg_against > 0 else 0
    
    # Normalize to percentages for pie chart (ensure they sum to 100%)
    optimized_total_avg = optimized_weighted_avg_pro_decimal + optimized_avg_neutral_decimal + optimized_avg_against_decimal
    if optimized_total_avg > 0:
        optimized_avg_pro_pct = (optimized_weighted_avg_pro_decimal / optimized_total_avg) * 100
        optimized_avg_neutral_pct = (optimized_avg_neutral_decimal / optimized_total_avg) * 100
        optimized_avg_against_pct = (optimized_avg_against_decimal / optimized_total_avg) * 100
    else:
        optimized_avg_pro_pct = 0
        optimized_avg_neutral_pct = 0
        optimized_avg_against_pct = 0
    
    # Create data for the chart
    optimized_indexalign_chart_data = pd.DataFrame({
        'Category': ['Pro', 'Neutral', 'Against'],
        'Percentage': [optimized_avg_pro_pct, optimized_avg_neutral_pct, optimized_avg_against_pct]
    })
    
    # Define colors for the three categories (same as unoptimized)
    optimized_indexalign_colors = {'Pro': '#2ecc71', 'Neutral': '#f39c12', 'Against': '#e74c3c'}
    
    if optimized_total_avg > 0:
        fig6 = px.pie(
            optimized_indexalign_chart_data,
            names='Category',
            values='Percentage',
            hole=0.4,
            title="IndexAlign Political Giving DEI Voting Record",
            labels={'Percentage': 'Average (%)'},
            color='Category',
            color_discrete_map=optimized_indexalign_colors
        )
        
        with col8:
            st.plotly_chart(fig6, key="optimized_indexalign_chart")

    # Optimized Portfolio Summary
    # Calculate optimized portfolio value directly from the optimized portfolio dataframe
    # Handle both numeric and string formats to ensure we get the exact values from the table
    try:
        # Try direct numeric conversion first (optimization function sets these as numeric)
        optimized_portfolio_value = st.session_state.optimized_portfolio_df['Current Value ($)'].astype(float).sum()
        optimized_total_gain_loss = st.session_state.optimized_portfolio_df['Gain/Loss ($)'].astype(float).sum()
    except (ValueError, TypeError):
        # Fallback to string parsing if values are formatted as strings
        optimized_portfolio_value = pd.to_numeric(
            st.session_state.optimized_portfolio_df['Current Value ($)'].astype(str).str.replace('$', '').str.replace(',', ''), 
            errors='coerce'
        ).fillna(0).sum()
        optimized_total_gain_loss = pd.to_numeric(
            st.session_state.optimized_portfolio_df['Gain/Loss ($)'].astype(str).str.replace('$', '').str.replace(',', ''), 
            errors='coerce'
        ).fillna(0).sum()
    
    # Calculate original portfolio value for comparison (handle string formats)
    total_gain_loss = pd.to_numeric(
        st.session_state.portfolio_df['Gain/Loss ($)'].astype(str).str.replace('$', '').str.replace(',', ''), 
        errors='coerce'
    ).fillna(0).sum()
    
    portfolio_value = pd.to_numeric(
        st.session_state.portfolio_df['Current Value ($)'].astype(str).str.replace('$', '').str.replace(',', ''), 
        errors='coerce'
    ).fillna(0).sum()
    
    st.subheader("Portfolio Summary (Harm Reduction Optimized)")
    
    # Add CSS to reduce vertical margins and set consistent font size for delta and percentage
    st.markdown("""
        <style>
        .portfolio-metric-container {
            margin-bottom: -10px !important;
        }
        .portfolio-percentage {
            margin-top: -15px !important;
            margin-bottom: 5px !important;
            display: block;
            font-size: 20px !important;
        }
        /* Set delta (increased/decreased) font size to match percentage */
        div[data-testid="stMetric"] div[data-testid="stMetricDelta"],
        div[data-testid="stMetric"] div[data-testid="stMetricDelta"] span,
        div[data-testid="stMetric"] div[data-testid="stMetricDelta"] div {
            font-size: 20px !important;
            font-weight: 600 !important;
        }
        div[data-testid="stMetric"] div[data-testid="stMetricDelta"] svg {
            width: 20px !important;
            height: 20px !important;
        }
        /* Alternative selector for delta values */
        .stMetric [data-testid="stMetricDelta"] {
            font-size: 20px !important;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Calculate Weighted Average Harm Score for optimized portfolio
    optimized_units = st.session_state.optimized_portfolio_df['Units'].astype(float)
    optimized_sector_harm_scores = st.session_state.optimized_portfolio_df['Sector Harm Score'].astype(float)
    optimized_total_units = optimized_units.sum()
    
    if optimized_total_units > 0:
        optimized_weighted_avg_harm_score = (optimized_units * optimized_sector_harm_scores).sum() / optimized_total_units
    else:
        optimized_weighted_avg_harm_score = 0
    
    # Calculate Weighted Average Harm Score for original portfolio (for comparison)
    original_units = st.session_state.portfolio_df['Units'].astype(float)
    original_sector_harm_scores = st.session_state.portfolio_df['Sector Harm Score'].astype(float)
    original_total_units = original_units.sum()
    
    if original_total_units > 0:
        original_weighted_avg_harm_score = (original_units * original_sector_harm_scores).sum() / original_total_units
    else:
        original_weighted_avg_harm_score = 0
    
    # Calculate harm score change
    harm_score_change = optimized_weighted_avg_harm_score - original_weighted_avg_harm_score
    harm_score_pct_change = ((optimized_weighted_avg_harm_score - original_weighted_avg_harm_score) / original_weighted_avg_harm_score * 100) if original_weighted_avg_harm_score > 0 else 0
    
    col1, col2, col3 = st.columns(3)
    portfolio_change = optimized_portfolio_value - portfolio_value
    
    # Calculate percentage changes
    portfolio_value_pct_change = ((optimized_portfolio_value - portfolio_value) / portfolio_value * 100) if portfolio_value > 0 else 0
    gain_loss_pct_change = ((optimized_total_gain_loss - total_gain_loss) / abs(total_gain_loss) * 100) if total_gain_loss != 0 else 0
    
    with col1:
        st.markdown('<div class="portfolio-metric-container">', unsafe_allow_html=True)
        st.metric(
            "Total Portfolio Value", 
            f"${optimized_portfolio_value:,.2f}", 
            delta=f"${portfolio_change:,.2f}", 
            delta_color="normal" if portfolio_change > 0 else "inverse"
        )
        # Display percentage change with arrow
        if portfolio_value_pct_change != 0:
            arrow = "↑" if portfolio_value_pct_change > 0 else "↓"
            color = "green" if portfolio_value_pct_change > 0 else "red"
            st.markdown(f'<span class="portfolio-percentage" style="color: {color}; font-size: 20px;">{arrow} {abs(portfolio_value_pct_change):.2f}%</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="portfolio-percentage" style="color: gray; font-size: 20px;">→ 0.00%</span>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="portfolio-metric-container">', unsafe_allow_html=True)
        st.metric("Total Portfolio Gain/Loss", 
              f"${optimized_total_gain_loss:,.2f}", 
              delta=f"${optimized_total_gain_loss-total_gain_loss:,.2f}", 
              delta_color="inverse" if optimized_total_gain_loss-total_gain_loss < 0 else "normal")
        # Display percentage change with arrow
        if gain_loss_pct_change != 0:
            arrow = "↑" if gain_loss_pct_change > 0 else "↓"
            color = "green" if gain_loss_pct_change > 0 else "red"
            st.markdown(f'<span class="portfolio-percentage" style="color: {color}; font-size: 20px;">{arrow} {abs(gain_loss_pct_change):.2f}%</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="portfolio-percentage" style="color: gray; font-size: 20px;">→ 0.00%</span>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="portfolio-metric-container">', unsafe_allow_html=True)
        st.metric("Weighted Average RFL Corporate Racial Justice Score", 
              f"{optimized_weighted_avg_harm_score:.2f}", 
              delta=f"{harm_score_change:.2f}", 
              delta_color="normal")
        # Display percentage change with arrow
        if harm_score_pct_change != 0:
            arrow = "↓" if harm_score_pct_change < 0 else "↑"  # Down arrow for decrease, up arrow for increase
            color = "red" if harm_score_pct_change < 0 else "green"  # Red for decrease, green for increase
            st.markdown(f'<span class="portfolio-percentage" style="color: {color}; font-size: 20px;">{arrow} {abs(harm_score_pct_change):.2f}%</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="portfolio-percentage" style="color: gray; font-size: 20px;">→ 0.00%</span>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    
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

# Helper function to read disclaimer from .docx file
def read_disclaimer_file(file_path):
    """Read content from a .docx file and return as HTML/markdown"""
    try:
        from docx import Document
        doc = Document(file_path)
        content = []
        for paragraph in doc.paragraphs:
            if paragraph.text.strip():
                content.append(paragraph.text)
        return "\n\n".join(content)
    except ImportError:
        st.warning("python-docx library not installed. Please install it using: pip install python-docx")
        return "Disclaimer file could not be loaded. Please ensure python-docx is installed."
    except FileNotFoundError:
        return f"Disclaimer file not found: {file_path}"
    except Exception as e:
        return f"Error reading disclaimer file: {str(e)}"

# Helper function to render PDF pages
def render_pdf_pages(pdf_path):
    """Render PDF pages in Streamlit"""
    try:
        import base64
        with open(pdf_path, "rb") as pdf_file:
            pdf_bytes = pdf_file.read()
            base64_pdf = base64.b64encode(pdf_bytes).decode('utf-8')
            
            # Provide download button
            st.download_button(
                label="📥 Download Methodology PDF",
                data=pdf_bytes,
                file_name=pdf_path,
                mime="application/pdf"
            )
            
            # Try to display PDF using iframe with base64 encoding
            pdf_display = f'''
            <iframe src="data:application/pdf;base64,{base64_pdf}" 
                    width="100%" 
                    height="800px" 
                    type="application/pdf"
                    style="border: 1px solid #ccc; margin-top: 10px;">
            </iframe>
            '''
            st.markdown(pdf_display, unsafe_allow_html=True)
            
            # Fallback: try to extract and display text using PyPDF2 if iframe doesn't work
            try:
                import PyPDF2
                with open(pdf_path, "rb") as pdf_file:
                    pdf_reader = PyPDF2.PdfReader(pdf_file)
                    with st.expander("📄 View PDF Text Content (Alternative)", expanded=False):
                        text_content = ""
                        for i, page in enumerate(pdf_reader.pages):
                            page_text = page.extract_text()
                            if page_text.strip():
                                text_content += f"**Page {i+1}**\n\n{page_text}\n\n---\n\n"
                        st.markdown(text_content)
            except ImportError:
                pass  # PyPDF2 not installed, skip text extraction
            except Exception:
                pass  # Text extraction failed, skip
                
    except FileNotFoundError:
        st.error(f"PDF file not found: {pdf_path}")
    except Exception as e:
        st.error(f"Error loading PDF: {str(e)}")
        # Try PyPDF2 as last resort
        try:
            import PyPDF2
            with open(pdf_path, "rb") as pdf_file:
                pdf_reader = PyPDF2.PdfReader(pdf_file)
                st.info("PDF viewer not available. Extracted text content:")
                text_content = ""
                for i, page in enumerate(pdf_reader.pages):
                    page_text = page.extract_text()
                    if page_text.strip():
                        text_content += f"**Page {i+1}**\n\n{page_text}\n\n---\n\n"
                st.markdown(text_content)
        except ImportError:
            st.info("💡 Tip: Install PyPDF2 for text extraction: `pip install PyPDF2`")
            st.info("📄 Use the download button above to view the full PDF document.")
        except Exception as e2:
            st.warning(f"Could not extract PDF text: {str(e2)}")
            st.info("📄 Please use the download button above to view the PDF document.")

# Legal Disclaimer and Score Methodology Section at the end
st.subheader("Legal Disclaimer and Score Methodology", divider="blue")
st.markdown(" ")
with st.expander("Legal Disclaimer", expanded=False):
    disclaimer_content = read_disclaimer_file("Kataly-Disclaimer.docx")
    st.markdown(disclaimer_content, unsafe_allow_html=True)

with st.expander("Score Methodology", expanded=False):
    render_pdf_pages("Corporate Racial Equity Score - Methodology Statement (1).pdf")
    
