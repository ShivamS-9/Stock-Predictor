# Stock Market Predictor

A deep learning-based stock price prediction application built with LSTM (Long Short-Term Memory) neural networks and deployed using Streamlit. This project predicts stock prices using historical data and provides interactive visualizations of price trends and moving averages.

## Features

- **Real-time Stock Data**: Fetches live stock data using Yahoo Finance API
- **LSTM Neural Network**: Deep learning model with multiple LSTM layers for accurate predictions
- **Interactive Visualizations**: 
  - Stock price trends
  - Moving averages (MA50, MA100, MA200)
  - Predicted vs actual price comparison
- **User-Friendly Interface**: Built with Streamlit for easy interaction
- **Customizable Date Range**: Select custom start and end dates for analysis
- **Multiple Stock Support**: Analyze any stock symbol available on Yahoo Finance

## Technologies Used

- **Python 3.10+**
- **TensorFlow/Keras**: Deep learning framework for LSTM model
- **Streamlit**: Web application framework
- **yfinance**: Yahoo Finance API wrapper
- **Pandas & NumPy**: Data manipulation and numerical computing
- **Matplotlib**: Data visualization
- **scikit-learn**: Data preprocessing and metrics

## Prerequisites

Before running this application, ensure you have Python 3.10 or higher installed on your system.

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/ShivamS-9/Stock-Predictor
   cd Stock-Predictor
   ```

2. **Install required packages**
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify model files**
   Ensure the following files are present in the project directory:
   - `stock_pred_model_v1.keras`
   - `scaler.save`

## Requirements

Create a `requirements.txt` file with the following dependencies:

```
numpy
pandas
matplotlib
yfinance
streamlit
tensorflow
keras
scikit-learn
joblib
```

## Usage

### Running the Streamlit App

1. **Start the application**
   ```bash
   streamlit run app.py
   ```

2. **Access the web interface**
   - The app will automatically open in your default browser
   - Default URL: `http://localhost:8501`

3. **Using the application**
   - Enter a stock symbol (e.g., GOOG, AAPL, TSLA)
   - Select start and end dates for analysis
   - View historical data, moving averages, and predictions

### Training Your Own Model

If you want to train the model from scratch:

1. **Open the training noteboo**
   - The repository includes the complete training code
   - Modify parameters like stock symbol, date range, epochs, etc.

2. **Run the training script:**
   Simply run the jupyter file
   

3. **Model will be saved as**
   - `stock_pred_model_v1.keras`
   - `scaler.save` (MinMaxScaler object)

## Model Architecture

The LSTM model consists of:

- **Input Layer**: Sequences of 100 previous days' stock prices
- **LSTM Layer 1**: 50 units with ReLU activation, return sequences
- **Dropout Layer 1**: 20% dropout for regularization
- **LSTM Layer 2**: 60 units with ReLU activation, return sequences
- **Dropout Layer 2**: 30% dropout
- **LSTM Layer 3**: 80 units with ReLU activation, return sequences
- **Dropout Layer 3**: 40% dropout
- **LSTM Layer 4**: 120 units with ReLU activation
- **Dropout Layer 4**: 50% dropout
- **Output Layer**: Dense layer with 1 unit

**Training Configuration**:
- Optimizer: Adam
- Loss Function: Mean Squared Error (MSE)
- Epochs: 50
- Batch Size: 32
- Train/Test Split: 80/20

## Model Performance

The model is evaluated using:
- **Mean Absolute Error (MAE)**
- **Root Mean Squared Error (RMSE)**
- **Mean Absolute Percentage Error (MAPE)**


## Visualizations

The app provides four main visualizations:

1. **Price vs MA50**: Close price with 50-day moving average
2. **Price vs MA50 vs MA100**: Close price with 50 and 100-day moving averages
3. **Price vs MA100 vs MA200**: Close price with 100 and 200-day moving averages
4. **Original vs Predicted Price**: Comparison of actual and predicted prices

## Project Structure

```
Stock-Predictor/
│
├── app.py                          # Streamlit web application
├── README.md                       # Project documentation
├── stock_pred_model_v1.keras       # Trained LSTM model
├── scaler.save                     # Fitted MinMaxScaler object
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
└── stock_pred_v1.ipynb             # Jupyter notebook for model training
```

## Customization

### Changing the Stock Symbol
In `app.py`, modify the default stock:
```python
stock = st.text_input('Enter Stock Symbol', 'AAPL')  # Change 'GOOG' to any symbol
```

### Adjusting Moving Average Windows
Modify the rolling window size:
```python
ma_50 = data.Close.rolling(50).mean()   # Change to desired window
ma_100 = data.Close.rolling(100).mean()
ma_200 = data.Close.rolling(200).mean()
```

### Model Hyperparameters
In the training script, adjust:
- Number of LSTM units
- Dropout rates
- Number of epochs
- Batch size
- Lookback window (default: 100 days)


## Acknowledgments

- Yahoo Finance for providing free stock data API
- TensorFlow/Keras team for the deep learning framework
- Streamlit for the amazing web framework
- The open-source community for inspiration and resources

## References

- [LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
- [Stock Price Prediction using Machine Learning](https://www.researchgate.net/)
- [Yahoo Finance API Documentation](https://pypi.org/project/yfinance/)
- [Streamlit Documentation](https://docs.streamlit.io/)

---

**⚡ Quick Start**: Clone the repo → Install dependencies → Run `streamlit run app.py` → Start predicting!

**💡 Tip**: Try different stocks and date ranges to see how the model performs across various market conditions!