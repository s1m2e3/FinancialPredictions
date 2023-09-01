import yfinance as yf
import numpy as np
from model import *
import matplotlib.pyplot as plt
import torch
from scipy.signal import correlate

def conver_to_lstm_data(data,sequence_length):
    data =np.array(data)
    new_shape = [data.shape[0]-sequence_length]
    new_shape.append(sequence_length)
    for i in np.arange(1,len(data.shape)):
        new_shape.append(data.shape[i])
    
    data_shape = tuple(new_shape)
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
stocks = yf.download("^IXIC AAPL MSFT AMZN NVDA TSLA GOOGL GOOG META AVGO PEP",period = "24mo")
for col in stocks.columns:
    stocks[col]=(stocks[col]-stocks[col].min())/(stocks[col].max()-stocks[col].min())

stocksClose = np.array(stocks["Adj Close"])
stocksCloseX = np.array(stocks["Adj Close"])[:,1:]
stocksCloseY = np.array(stocks["Adj Close"])[:,0]

stocksClose.shape = (stocksClose.shape[0],stocksClose.shape[1],1)
stocksCloseX.shape = (stocksCloseX.shape[0],stocksCloseX.shape[1],1)
stocksCloseY.shape = (stocksCloseY.shape[0],1,1)

stocksVolume = np.array(stocks["Volume"])
stocksVolumeX = np.array(stocks["Volume"])[:,1:]
stocksVolumeY = np.array(stocks["Volume"])[:,0]

stocksVolume.shape = (stocksVolume.shape[0],stocksVolume.shape[1],1)
stocksVolumeX.shape = (stocksVolumeX.shape[0],stocksVolumeX.shape[1],1)
stocksVolumeY.shape = (stocksVolumeY.shape[0],1,1)
# stocksTime = np.array(stocks.index) 

stocksTotal = np.append(stocksClose,stocksVolume,axis=2)
stocksTotalY = np.append(stocksCloseY,stocksVolumeY,axis=2)
# xColumns = ["AAPL", "MSFT", "AMZN", "NVDA", "TSLA", "GOOGL","GOOG", "META", "AVGO", "PEP"]
# y=["^IXIC"]
# xClose = stocksClose[["AAPL", "MSFT", "AMZN", "NVDA", "TSLA", "GOOGL","GOOG", "META", "AVGO", "PEP"]]
# xVolume =stocksVolume[["AAPL", "MSFT", "AMZN", "NVDA", "TSLA", "GOOGL","GOOG", "META", "AVGO", "PEP"]]
# yClose =stocksClose["^IXIC"]
# yVolume =stocksVolume["^IXIC"]

input_sequence_length = 5
output_sequence_length = 3

# lstm = LSTM(x.shape[1],hidden,layers,y.shape[1],input_sequence_length,output_sequence_length)
# print(xClose.head())
stocksTotalLstmX = conver_to_lstm_data(stocksTotal,input_sequence_length)[:-output_sequence_length,:,:,:]
stocksTotalLstmY = conver_to_lstm_data(stocksTotalY,output_sequence_length)[input_sequence_length:,:,:,:]

stop = int(len(stocksTotalLstmX)*0.7)

stocksTotalLstmXTrain = stocksTotalLstmX[:stop,:,:,:]
stocksTotalLstmXTest = stocksTotalLstmX[stop:,:,:,:]
stocksTotalLstmYTrain = stocksTotalLstmY[:stop,:,:,:]
stocksTotalLstmYTest = stocksTotalLstmY[stop:,:,:,:]

input_size = 110
output_size = 6 
hidden_size = 40
# stocksClose = conver_to_lstm_data(xClose,input_sequence_length)
# stocksVolume = conver_to_lstm_data(xVolume,input_sequence_length)
# stocksTime = conver_to_lstm_data(yClose,input_sequence_length)
# stocksVolumeY = conver_to_lstm_data(yVolume,input_sequence_length)
# print(stocksCloseX.shape)
# print(stocksCloseY.shape)
# input_size = 10
# hidden_size = 64
# output_size = 1

lr = 0.01
iterations=100
fig, (ax1,ax2) = plt.subplots(2,1)
fig.set_figheight(15)
fig.set_figwidth(20)
data = {"autocorrelated":[],"not":[]}
for weight_decay in [0]:
    for autocorr in [True,False]:
        ff_nn = NN(input_size,hidden_size,output_size,lr=lr,weight_decay=weight_decay)
        ff_nn.train(iterations,stocksTotalLstmXTrain,stocksTotalLstmYTrain,autocorr=autocorr)
        for stream in range(len(stocksTotalLstmXTest)):
            # if stream % 3 == 0:    
            outputs = ff_nn.forward(torch.flatten(torch.tensor(stocksTotalLstmXTest[stream,:,:,:]))).cpu().detach().numpy().reshape(3,2)
            if autocorr:
                data["autocorrelated"].append(outputs)
            else:
                data["not"].append(outputs)                
count = 0
count0 = []
count1 = []
count2  = [] 
unfolded={"autocorrelated":{0:[],1:[],2:[]},"not":{0:[],1:[],2:[]}}
for stream in data["autocorrelated"]:
    count0.append(count)
    count1.append(count+1)
    count2.append(count+2)
    unfolded["autocorrelated"][0].append(stream[0,0])
    unfolded["autocorrelated"][1].append(stream[1,0])
    unfolded["autocorrelated"][2].append(stream[2,0])
    ax1.plot([count,count+1,count+2],stream[:,0])
    # ax1.scatter([count+1],stream[1,0],color="orange")
    # ax1.scatter([count+2],stream[2,0],color="purple")
    count +=1
# ax1.scatter(count0,unfolded["autocorrelated"][0],color="blue")
# ax1.scatter(count1,unfolded["autocorrelated"][1],color="orange")
# ax1.scatter(count2,unfolded["autocorrelated"][2],color="purple")
ax1.title.set_text('Autocorrelated')
ax1.set_xlim(0,150)
count = 0


for stream in data["not"]:
    unfolded["not"][0].append(stream[0,0])
    unfolded["not"][1].append(stream[1,0])
    unfolded["not"][2].append(stream[2,0])
    ax2.plot([count,count+1,count+2],stream[:,0])
    count +=1
ax2.title.set_text('Not Autocorrelated')
# ax2.scatter(count0,unfolded["not"][0],color="blue")
# ax2.scatter(count1,unfolded["not"][1],color="orange")
# ax2.scatter(count2,unfolded["not"][2],color="purple")
ax2.set_xlim(0,150)
ax1.plot(stocksTotalY[stop:,0,0],color="green")
ax2.plot(stocksTotalY[stop:,0,0],color="green")
plt.savefig("feedforward_autocorr.png")


plt.figure(figsize=(20,15))
plt.plot(stocksTotalY[stop:,0,0],color="green")
plt.plot(count0,unfolded["not"][0],color="red")
plt.plot(count0,unfolded["autocorrelated"][0],color="salmon")
plt.savefig("feedforward_autocorr_justones.png")
# time_lstm =conver_to_lstm_data(stocks.index,input_sequence_length)

# y_train = conver_to_lstm_data(y[0:stop],output_sequence_length)
# x_test = conver_to_lstm_data(x[stop:],input_sequence_length) 
# y_test = conver_to_lstm_data(y[stop:],output_sequence_length) 
    
# x_train = x_train[:-output_sequence_length]
# y_train = y_train[input_sequence_length:]
    
# x_test = x_test[:-output_sequence_length]
# y_test = y_test[input_sequence_length:]
