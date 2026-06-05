import os  # Used to create output directories and build file paths
import requests  # Used to request data from public APIs
import numpy as np  # Used for numerical computation
import pandas as pd  # Used for tabular data processing
import matplotlib.pyplot as plt  # Used for plotting and saving charts

from sklearn.pipeline import make_pipeline  # Used to combine preprocessing and model steps
from sklearn.preprocessing import PolynomialFeatures  # Used to generate polynomial features
from sklearn.linear_model import LinearRegression  # Linear regression model
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error  # Model evaluation metrics


OUTPUT_DIR = "btc_forecast_output"  # Output directory for CSV files and chart images
os.makedirs(OUTPUT_DIR, exist_ok=True)  # Create the output directory if it does not exist


def fetch_btc_prices(days=365, vs_currency="usd"):  # Fetch BTC historical price data
    url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"  # CoinGecko market chart API

    params = {  # API request parameters
        "vs_currency": vs_currency,  # Quote currency, such as USD
        "days": days,  # Number of historical days to fetch
        "interval": "daily"  # Use daily price data
    }

    response = requests.get(url, params=params, timeout=30)  # Send GET request to the API
    response.raise_for_status()  # Raise an error if the request fails

    data = response.json()  # Parse API response as JSON

    df = pd.DataFrame(data["prices"], columns=["timestamp", "price"])  # Convert price data to DataFrame
    df["date"] = pd.to_datetime(df["timestamp"], unit="ms").dt.date  # Convert timestamp to date
    df["date"] = pd.to_datetime(df["date"])  # Convert date column to pandas datetime type

    df = (  # Clean and sort the dataset
        df[["date", "price"]]  # Keep only date and price columns
        .drop_duplicates("date")  # Remove duplicate dates
        .sort_values("date")  # Sort data by date
        .reset_index(drop=True)  # Reset row index
    )

    return df  # Return cleaned historical price data


def evaluate_model(y_true, y_pred):  # Calculate model performance metrics
    mae = mean_absolute_error(y_true, y_pred)  # Mean Absolute Error
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))  # Root Mean Squared Error
    r2 = r2_score(y_true, y_pred)  # R-squared score

    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100  # Mean Absolute Percentage Error

    return {  # Return all evaluation metrics
        "r2": r2,  # R-squared score
        "mae": mae,  # Mean Absolute Error
        "rmse": rmse,  # Root Mean Squared Error
        "mape": mape  # Mean Absolute Percentage Error
    }


def train_predict(df, train_days, forecast_days, degree=3):  # Train model and generate future forecast
    train_df = df.tail(train_days).copy().reset_index(drop=True)  # Use the latest train_days rows as training data

    X_train = np.arange(len(train_df)).reshape(-1, 1)  # Use time index as model feature
    y_train = train_df["price"].values  # Use BTC price as target variable

    model = make_pipeline(  # Build polynomial regression pipeline
        PolynomialFeatures(degree=degree),  # Generate polynomial features
        LinearRegression()  # Fit linear regression on polynomial features
    )

    model.fit(X_train, y_train)  # Train the model

    fitted_price = model.predict(X_train)  # Predict fitted values on training data

    metrics = evaluate_model(y_train, fitted_price)  # Evaluate in-sample model performance

    X_future = np.arange(  # Create future time index for forecasting
        len(train_df),  # Start from the next time index after training data
        len(train_df) + forecast_days  # End after forecast_days future points
    ).reshape(-1, 1)  # Convert to 2D array required by scikit-learn

    predicted_price = model.predict(X_future)  # Forecast future BTC prices

    future_dates = pd.date_range(  # Create future date range
        start=train_df["date"].iloc[-1] + pd.Timedelta(days=1),  # Start from the day after the last training date
        periods=forecast_days,  # Number of forecast days
        freq="D"  # Daily frequency
    )

    fitted_df = pd.DataFrame({  # Store actual and fitted values for the training period
        "date": train_df["date"],  # Historical date
        "actual_price": train_df["price"],  # Actual BTC price
        "fitted_price": fitted_price  # Model fitted BTC price
    })

    forecast_df = pd.DataFrame({  # Store future forecast values
        "date": future_dates,  # Future date
        "predicted_price": predicted_price  # Forecasted BTC price
    })

    return fitted_df, forecast_df, metrics  # Return fitted data, forecast data, and metrics


def print_report(name, metrics, forecast_df):  # Print model metrics and forecast summary
    print("\n" + "=" * 70)  # Print separator line
    print(name)  # Print experiment name
    print("=" * 70)  # Print separator line

    print(f"R²   : {metrics['r2']:.6f}")  # Print R-squared score
    print(f"MAE  : ${metrics['mae']:,.2f}")  # Print Mean Absolute Error
    print(f"RMSE : ${metrics['rmse']:,.2f}")  # Print Root Mean Squared Error
    print(f"MAPE : {metrics['mape']:.2f}%")  # Print Mean Absolute Percentage Error

    print("\nForecast Summary")  # Print forecast summary title
    print(f"Forecast Start Date    : {forecast_df['date'].iloc[0].date()}")  # Print forecast start date
    print(f"Forecast End Date      : {forecast_df['date'].iloc[-1].date()}")  # Print forecast end date
    print(f"Final Forecast Price   : ${forecast_df['predicted_price'].iloc[-1]:,.2f}")  # Print final forecast price
    print(f"Maximum Forecast Price : ${forecast_df['predicted_price'].max():,.2f}")  # Print maximum forecast price
    print(f"Minimum Forecast Price : ${forecast_df['predicted_price'].min():,.2f}")  # Print minimum forecast price

    print("\nFirst 10 Forecasted Days:")  # Print first 10 forecasted rows
    print(forecast_df.head(10).to_string(index=False))  # Display first 10 forecasted rows

    print("\nLast 10 Forecasted Days:")  # Print last 10 forecasted rows
    print(forecast_df.tail(10).to_string(index=False))  # Display last 10 forecasted rows


def plot_and_save(title, fitted_df, forecast_df, filename):  # Plot actual, fitted, and forecast prices, then save chart
    plt.figure(figsize=(14, 7))  # Create figure with fixed size

    plt.plot(  # Plot actual BTC price
        fitted_df["date"],  # X-axis: historical date
        fitted_df["actual_price"],  # Y-axis: actual price
        label="Actual Price"  # Legend label
    )

    plt.plot(  # Plot fitted BTC price
        fitted_df["date"],  # X-axis: historical date
        fitted_df["fitted_price"],  # Y-axis: fitted price
        label="Fitted Price"  # Legend label
    )

    plt.plot(  # Plot forecasted BTC price
        forecast_df["date"],  # X-axis: future date
        forecast_df["predicted_price"],  # Y-axis: forecasted price
        label="Forecast Price"  # Legend label
    )

    plt.title(title)  # Set chart title
    plt.xlabel("Date")  # Set X-axis label
    plt.ylabel("BTC Price USD")  # Set Y-axis label
    plt.legend()  # Show legend
    plt.grid(True)  # Show grid
    plt.tight_layout()  # Adjust layout automatically

    path = os.path.join(OUTPUT_DIR, filename)  # Build chart output path
    plt.savefig(path, dpi=200)  # Save chart as PNG file
    plt.show()  # Display chart

    print(f"Chart saved to: {path}")  # Print saved chart path


def save_csv(fitted_df, forecast_df, prefix):  # Save fitted data and forecast data as CSV files
    fitted_path = os.path.join(OUTPUT_DIR, f"{prefix}_fitted.csv")  # Build fitted CSV file path
    forecast_path = os.path.join(OUTPUT_DIR, f"{prefix}_forecast.csv")  # Build forecast CSV file path

    fitted_df.to_csv(fitted_path, index=False)  # Save fitted data to CSV
    forecast_df.to_csv(forecast_path, index=False)  # Save forecast data to CSV

    print(f"Fitted data saved to: {fitted_path}")  # Print fitted CSV path
    print(f"Forecast data saved to: {forecast_path}")  # Print forecast CSV path


if __name__ == "__main__":  # Program entry point
    degree = 3  # Polynomial degree

    df = fetch_btc_prices(days=365)  # Fetch BTC price data for the latest 365 days

    print("Dataset Summary")  # Print dataset summary title
    print(df.tail())  # Print latest rows of the dataset
    print("\nPrice Statistics")  # Print price statistics title
    print(df["price"].describe())  # Print descriptive statistics for BTC price

    experiments = [  # Define training and forecasting experiments
        {
            "name": "Train on Last 180 Days, Forecast Next 180 Days",  # Experiment name
            "train_days": 180,  # Number of training days
            "forecast_days": 180,  # Number of forecast days
            "prefix": "train_180_forecast_180",  # Output file prefix
            "image": "train_180_forecast_180.png"  # Output chart filename
        },
        {
            "name": "Train on Last 180 Days, Forecast Next 365 Days",  # Experiment name
            "train_days": 180,  # Number of training days
            "forecast_days": 365,  # Number of forecast days
            "prefix": "train_180_forecast_365",  # Output file prefix
            "image": "train_180_forecast_365.png"  # Output chart filename
        },
        {
            "name": "Train on Last 365 Days, Forecast Next 180 Days",  # Experiment name
            "train_days": 365,  # Number of training days
            "forecast_days": 180,  # Number of forecast days
            "prefix": "train_365_forecast_180",  # Output file prefix
            "image": "train_365_forecast_180.png"  # Output chart filename
        },
        {
            "name": "Train on Last 365 Days, Forecast Next 365 Days",  # Experiment name
            "train_days": 365,  # Number of training days
            "forecast_days": 365,  # Number of forecast days
            "prefix": "train_365_forecast_365",  # Output file prefix
            "image": "train_365_forecast_365.png"  # Output chart filename
        }
    ]

    for exp in experiments:  # Run each experiment
        fitted_df, forecast_df, metrics = train_predict(  # Train model and generate forecast
            df=df,  # Historical BTC price dataset
            train_days=exp["train_days"],  # Training window size
            forecast_days=exp["forecast_days"],  # Forecast horizon
            degree=degree  # Polynomial degree
        )

        print_report(exp["name"], metrics, forecast_df)  # Print model and forecast report

        save_csv(  # Save fitted and forecast data
            fitted_df=fitted_df,  # Fitted data
            forecast_df=forecast_df,  # Forecast data
            prefix=exp["prefix"]  # Output file prefix
        )

        plot_and_save(  # Plot and save chart
            title=f"{exp['name']} | Polynomial Degree {degree}",  # Chart title
            fitted_df=fitted_df,  # Fitted data
            forecast_df=forecast_df,  # Forecast data
            filename=exp["image"]  # Chart filename
        )
