import yfinance as yf
import numpy as np
from model import *
def conver_to_lstm_data(data,sequence_length):
    data =np.array(data)
    new_shape = [data.shape[0]-sequence_length]
    #data_shape = list(data.shape)
    
    new_shape.append(sequence_length)
    new_shape.append(data.shape[1])
    data_shape = tuple(new_shape)
    #print(data_shape)
    new_data = np.zeros(shape=data_shape)
    for i in range(len(data)-sequence_length):
        new_data[i]=data[i:i+sequence_length]
    
    return new_data


"""
 The 10 most heavily weighed securities in the Nasdaq Composite Index as of March 31, 2023, were:
    Apple (AAPL): 13.79%
    Microsoft (MSFT): 11.44%
    Amazon (AMZN): 6.04%
    NVIDIA (NVDA): 4.72%
    Tesla Inc (TSLA): 3.75%
    Alphabet Class A (GOOGL): 3.21%
    Alphabet Class C (GOOG): 3.21%
    Meta Platforms (META): 2.87%
    Broadcom (AVGO): 1.63%
    Pepsico (PEP): 1.15%3
"""
stocks = yf.download("^IXIC AAPL MSFT AMZN NVDA TSLA GOOGL GOOG META AVGO PEP",period = "6mo")

stocksClose = np.array(stocks["Adj Close"])
stocksVolume = np.array(stocks["Volume"])
stocksTime = np.array(stocks.index) 

xColumns = ["AAPL", "MSFT", "AMZN", "NVDA", "TSLA", "GOOGL","GOOG", "META", "AVGO", "PEP"]
y=["^IXIC"]
xClose = stocksClose[xColumns]
xVolume =stocksVolume[xColumns]
yClose =stocksClose[y]
yVolume =stocksVolume[y]

input_sequence_length = 5
output_sequence_length = 3

# lstm = LSTM(x.shape[1],hidden,layers,y.shape[1],input_sequence_length,output_sequence_length)

stocksClose = conver_to_lstm_data(stocksClose,input_sequence_length)
stocksVolume = conver_to_lstm_data(stocksVolume,input_sequence_length)
input_size = 10
hidden_size = 64
output_size = 1
ff_nn = NN(input_size,hidden_size,output_size)
ff_nn.train(100,stocksCloseX,stocksCloseY)
# time_lstm =conver_to_lstm_data(stocks.index,input_sequence_length)

# y_train = conver_to_lstm_data(y[0:stop],output_sequence_length)
# x_test = conver_to_lstm_data(x[stop:],input_sequence_length) 
# y_test = conver_to_lstm_data(y[stop:],output_sequence_length) 
    
# x_train = x_train[:-output_sequence_length]
# y_train = y_train[input_sequence_length:]
    
# x_test = x_test[:-output_sequence_length]
# y_test = y_test[input_sequence_length:]
