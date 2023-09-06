import yfinance as yf
import numpy as np
from model import *
import matplotlib.pyplot as plt
import torch
from scipy.signal import correlate
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from pmdarima import auto_arima
from statsmodels.tsa.seasonal import seasonal_decompose
import pandas as pd
import seaborn as sns

def predict_arima(params,data,diff,p,q):
    
    params_names = list(params)
    autorregresive_params = [param for param in params_names if "ar" in param]
    movingaverage_params = [param for param in params_names if "ma" in param]
    prediction = 0
    
    for i in range(diff):
        data = np.diff(data)
    for i in range(p+1):
        for j in autorregresive_params:
            if "L"+str(i) in j:
                prediction= params[j]*data[-i]+prediction
                
    for i in range(q+1):
        for j in movingaverage_params:
            if "L"+str(i) in j:
                prediction= params[j]*np.random.normal(scale=params["sigma2"])+prediction
                
    return prediction

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

sns.set_theme()
stocks = yf.download("^IXIC AAPL MSFT AMZN NVDA TSLA GOOGL GOOG META AVGO",period = "60mo")
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
stocksTime = np.array(stocks.index) 

stocksTotal = np.append(stocksClose,stocksVolume,axis=2)
stocksTotalY = np.append(stocksCloseY,stocksVolumeY,axis=2)
stop = int(len(stocksTotalY)*0.7)
stop_sarima = int(len(stocksTotalY)*0.3)

p_values=[]
for i in [0,1]:
    if i>0:
        data = np.diff(data)
        
    else:
        data=stocksTotalY[:stop,0,0]
        
    # Perform the ADF test
    result = adfuller(data)

    # Extract and print the test statistic and p-value
    test_statistic, p_value, _, _, _, _ = result
    p_values.append(p_value)
d=np.argmin(p_values)
q = 5
p = 4

df = pd.DataFrame(stocksTotal[:,:,0])
df.index = pd.to_datetime(stocksTime)
df = df.asfreq("D")
df = df.resample("D").mean().fillna(method="ffill")
data=np.array(df)

stocksTotal = data
stocksTotalY = data[:,0]

model = sm.tsa.SARIMAX(endog=stocksTotalY[:stop], order=(p, 1, q),seasonal_order=(1,1,1,7))# ARIMA(p,d,q) 
results = model.fit()

input_sequence_length = 5
output_sequence_length = 3

stocksTotalLstmX = conver_to_lstm_data(stocksTotal,input_sequence_length)[:-output_sequence_length,:,:]
stocksTotalLstmY = conver_to_lstm_data(stocksTotalY,output_sequence_length)[input_sequence_length:,:]

stocksTotalLstmXTrain = stocksTotalLstmX[:stop,:,:]
stocksTotalLstmXTest = stocksTotalLstmX[stop:,:,:]
stocksTotalLstmYTrain = stocksTotalLstmY[:stop,:]
stocksTotalLstmYTest = stocksTotalLstmY[stop:,:]

input_size = 10*input_sequence_length
output_size = output_sequence_length 
hidden_size = 512

lr = 0.01
weight_decay = 0

iterations=10000
fig, (ax1,ax2) = plt.subplots(2,1,sharex=True)

plt.xlabel('Day',fontsize="18",fontweight="bold")
plt.xticks(fontsize=18)
fig.set_figheight(15)
fig.set_figwidth(20)
data = {"autocorrelated":[],"not":[]}
losses={"autocorrelated":{0:[],1:[],2:[]},"not":{0:[],1:[],2:[]}}
for weight_decay in [0]:
    for autocorr in [True,False]:
        ff_nn = NN(input_size,hidden_size,output_size,lr=lr,weight_decay=weight_decay,sarimax=results)
        ff_nn.train(iterations,stocksTotalLstmXTrain,stocksTotalLstmYTrain,autocorr=autocorr)
        for stream in range(len(stocksTotalLstmXTest)):
            # if stream % 3 == 0:    
            outputs = ff_nn.forward(torch.flatten(torch.tensor(stocksTotalLstmXTest[stream,:,:]))).cpu().detach().numpy()
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
    unfolded["autocorrelated"][0].append(stream[0])
    unfolded["autocorrelated"][1].append(stream[1])
    unfolded["autocorrelated"][2].append(stream[2])
    
    
    
    # ax1.plot([count,count+1,count+2],stream[:])
    # ax1.scatter([count+1],stream[1,0],color="orange")
    # ax1.scatter([count+2],stream[2,0],color="purple")
    count +=1
ax1.scatter(count0,unfolded["autocorrelated"][0],color="blue",label = "One day Prediction" )
ax1.scatter(count1,unfolded["autocorrelated"][1],color="orange",label = "Two day Prediction" )
ax1.scatter(count2,unfolded["autocorrelated"][2],color="purple",label = "Three day Prediction" )
ax1.set_title('FeedForward Neural Network + SARIMAX')
ax1.title.set_fontsize(24)
ax1.title.set_fontweight("bold")
ax1.tick_params(axis="y",labelsize=16)
ax1.set_ylabel('NASDAQ Normalized Closing Price',fontsize="18")
ax1.yaxis.label.set_fontweight("bold")
ax1.legend()
ax1.set_xlim(0,150)

count = 0


for stream in data["not"]:
    unfolded["not"][0].append(stream[0])
    unfolded["not"][1].append(stream[1])
    unfolded["not"][2].append(stream[2])
    # ax2.plot([count,count+1,count+2],stream[:])
    count +=1
ax2.set_title('FeedForward Neural Network')
ax2.title.set_fontsize(24)
ax2.title.set_fontweight("bold")
ax2.tick_params(axis="y",labelsize=16)
ax2.set_ylabel('NASDAQ Normalized Closing Price',fontsize="18")
ax2.yaxis.label.set_fontweight("bold")
ax2.scatter(count0,unfolded["not"][0],color="blue",label = "One day Prediction" )
ax2.scatter(count1,unfolded["not"][1],color="orange",label = "Two day Prediction" )
ax2.scatter(count2,unfolded["not"][2],color="purple",label = "Three day Prediction" )

ax2.set_xlim(0,150)
ax2.legend()
ax1.plot(stocksTotalY[stop+3:-3],color="green")
ax2.plot(stocksTotalY[stop+3:-3],color="green")
plt.savefig("feedforward_autocorr.png")

plt.figure(figsize=(20,15))

plt.fill_between(x=np.arange(len(stocksTotalY[stop+3:-3])),y1=stocksTotalY[stop+3:-3],y2=0,color="lawngreen",label="Ground Truth",alpha=0.6)
plt.plot(count0,unfolded["not"][0],color="red",label="Regular FFN")
plt.title("Neural Networks Comparison",fontsize="24",fontweight="bold")
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.xlabel("Day",fontsize="18",fontweight="bold")
plt.xlim(0,len(stocksTotalY[stop+3:-3]))
plt.ylabel("NASDAQ Normalized Closing Price",fontsize="18",fontweight="bold")
plt.plot(count0,unfolded["autocorrelated"][0],color="salmon",label="FFN + SARIMAX")
plt.legend(loc="upper right")
plt.savefig("feedforward_autocorr_justones.png")


losses["autocorrelated"][1]=np.sum((stocksTotalLstmYTest[:,1]-np.array(unfolded["autocorrelated"][1]))**2/len(stocksTotalLstmYTest))
losses["autocorrelated"][0]=np.sum((stocksTotalLstmYTest[:,0]-np.array(unfolded["autocorrelated"][0]))**2/len(stocksTotalLstmYTest))
losses["autocorrelated"][2]=np.sum((stocksTotalLstmYTest[:,2]-np.array(unfolded["autocorrelated"][2]))**2/len(stocksTotalLstmYTest))
losses["autocorrelated"]["total"]=losses["autocorrelated"][2]+losses["autocorrelated"][1]+losses["autocorrelated"][0]
losses["not"][0]=np.sum((stocksTotalLstmYTest[:,0]-np.array(unfolded["not"][0]))**2/len(stocksTotalLstmYTest))
losses["not"][1]=np.sum((stocksTotalLstmYTest[:,1]-np.array(unfolded["not"][1]))**2/len(stocksTotalLstmYTest))
losses["not"][2]=np.sum((stocksTotalLstmYTest[:,2]-np.array(unfolded["not"][2]))**2/len(stocksTotalLstmYTest))
losses["not"]["total"]=losses["not"][2]+losses["not"][1]+losses["not"][0]
bar_colors = ["blue","orange","purple"]
bar_label = ["One day Prediction","Two day Prediction","Three day Prediction"]



fig = plt.figure(figsize=(20,15))

gs = fig.add_gridspec(2,2)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[1, :])

ax1.bar(["One day Prediction","Two day Prediction","Three day Prediction"],\
    [losses["autocorrelated"][0],losses["autocorrelated"][1],losses["autocorrelated"][2]],\
        label = bar_label ,color = bar_colors )
ax1.set_ylabel("Weighted Average Loss per Days Predicted",fontsize=18)
ax1.set_title("FFN + SARIMAX")
ax1.yaxis.label.set_fontweight("bold")
ax1.title.set_fontsize(24)
ax1.title.set_fontweight("bold")
ax1.tick_params(axis="y",labelsize=14)
ax1.tick_params(axis="x",labelsize=14)
ax1.legend()

ax2.bar(["One day Prediction","Two day Prediction","Three day Prediction"],\
    [losses["not"][0],losses["not"][1],losses["not"][2]],\
        label = bar_label ,color = bar_colors )
ax2.set_title("FFN")
ax2.title.set_fontsize(24)
ax2.title.set_fontweight("bold")
ax2.tick_params(axis="y",labelsize=14)
ax2.tick_params(axis="x",labelsize=14)
ax2.legend()

ax3.barh(["FFN+SARIMAX","FFN"],[losses["autocorrelated"]["total"],losses["not"]["total"]],\
    label=['FFN+SARIMAX','FFN'],color=["salmon",'red'])

ax3.set_xlabel("Weighted Average Loss per Days Predicted",fontsize=18)
ax3.xaxis.label.set_fontweight("bold")
ax3.set_title("FFN+SARIMAX Vs. FFN")
ax3.title.set_fontsize(24)
ax3.title.set_fontweight("bold")
ax3.tick_params(axis="y",labelsize=14)
ax3.tick_params(axis="x",labelsize=14)
ax3.legend()
plt.savefig("lossescomparison.png")

