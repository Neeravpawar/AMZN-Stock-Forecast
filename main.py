import os
from dotenv import load_dotenv
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
import numpy as np

# Load environment variables
load_dotenv()

# Get environment variables
DATA_FILE = os.getenv('DATA_FILE')
FIG_WIDTH = int(os.getenv('FIGURE_WIDTH', 12))
FIG_HEIGHT = int(os.getenv('FIGURE_HEIGHT', 8))

# Read and prepare stock data
df = pd.read_csv(DATA_FILE)
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# Apply log transformation to closing prices
df['Log_Close'] = np.log(df['Close'])

# Perform time series decomposition on log-transformed prices
decomposition = seasonal_decompose(df['Log_Close'], 
                                 period=252,  # 252 trading days in a year
                                 model='additive')  # additive on log scale = multiplicative on original scale

# Plot the decomposition
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(FIG_WIDTH, FIG_HEIGHT*2))

# Original (log scale)
ax1.plot(df.index, df['Log_Close'])
ax1.set_title('Original Time Series (Log Scale)')

# Trend
ax2.plot(df.index, decomposition.trend)
ax2.set_title('Trend (Log Scale)')

# Seasonal
ax3.plot(df.index, decomposition.seasonal)
ax3.set_title('Seasonal Component (Log Scale)')

# Residual
ax4.plot(df.index, decomposition.resid)
ax4.set_title('Residual (Log Scale)')

plt.tight_layout()
plt.show()
