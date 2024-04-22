# Stock Price Prediction using LSTM

## Overview
This project aims to predict stock prices using machine learning, specifically Long Short-Term Memory (LSTM) networks. We utilize historical stock data from Tata Global Beverages Limited, obtained from the National Stock Exchange of India (NSE), to train our model.

## Table of Contents
1. [Dependencies](#dependencies)
2. [Dataset](#dataset)
3. [Data Preprocessing](#data-preprocessing)
4. [Model Architecture](#model-architecture)
5. [Training](#training)
6. [Evaluation](#evaluation)
7. [Results](#results)
8. [Usage](#usage)
9. [Contributing](#contributing)
10. [License](#license)

## Dependencies
The following dependencies are required to run the code:
- Python 3.x
- pandas
- numpy
- matplotlib
- scikit-learn
- TensorFlow

You can install these dependencies using pip:

pip install pandas numpy matplotlib scikit-learn tensorflow

## Dataset
We use historical stock data of Tata Global Beverages Limited, obtained from the National Stock Exchange of India (NSE). The dataset contains information about the opening, closing, highest, and lowest prices of the stock, along with the trading volume, for each day.

Dataset Source: [Quandl - NSE/TATAGLOBAL](https://www.quandl.com/data/NSE/TATAGLOBAL-Tata-Global-Beverages-Limited)

## Data Preprocessing
- We start by analyzing the closing prices over time to understand the trends.
- Exponential smoothing is applied to make the data less rigid and easier for the model to learn from.
- The data is sorted in ascending order with respect to date values and transformed into a format suitable for training.

## Model Architecture
- We use a stacked LSTM architecture for our prediction model.
- The model consists of three LSTM layers followed by a dense layer for regression.
- Mean Squared Error (MSE) is used as the loss function, and Adam optimizer is used for training.

## Training
- The dataset is split into training and test sets.
- Input sequences of closing prices with a specified time step are created.
- The model is trained on the training set for a specified number of epochs and batch size.

## Evaluation
- After training, the model is evaluated using Root Mean Squared Error (RMSE) as the performance metric.
- RMSE is calculated for both the training and test sets to assess the model's performance.

## Results
- The trained model is used to make predictions on both the training and test sets.
- Predictions are inverse-transformed to the original scale for visualization.
- Baseline and predicted stock prices are plotted to visualize the model's performance.

## Usage
To use this code:
1. Clone this repository.
2. Download the dataset from the provided source.
3. Update the file path in the code to point to the downloaded dataset.
4. Run the Python notebook.


