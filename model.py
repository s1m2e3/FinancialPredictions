import torch
import numpy as np
import torch.nn as nn
# import pytorch_forecasting 


def reshape_tensor(x, batch_size):
    original_shape = x.size()
    num_elements = x.numel()
    new_shape = (batch_size,) + original_shape[1:]
    if num_elements != new_shape[0] * torch.tensor(new_shape[1:]).prod():
        raise ValueError("Number of elements in tensor does not match new shape")
    return x.view(new_shape)

class NN(nn.Module):
    def __init__(self, input_size1, hidden_size, output_size,lr,weight_decay,params):
        super(NN, self).__init__()
        
        if torch.cuda.is_available():
            dev = "cuda:0"
        else:
            dev = "cpu"
        self.number_of_nodes = hidden_size
        self.fc1 = nn.Linear(input_size1, output_size,dtype=torch.float).to(dev)
        self.relu = nn.ReLU().to(dev)
        self.fc2 = nn.Linear(output_size, hidden_size,dtype=torch.float).to(dev)
        self.relu = nn.ReLU().to(dev)
        self.fc3 = nn.Linear(hidden_size, output_size,dtype=torch.float).to(dev)
        self.optimizer = torch.optim.SGD(self.parameters(),lr=lr,weight_decay=weight_decay)
        self.criterion = nn.MSELoss()
    def forward(self, x):
        if torch.cuda.is_available():
            dev = "cuda:0"
            
        else:
            dev = "cpu"
        device = torch.device(dev)
        out = torch.tensor(x,dtype=torch.float).to(device)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)

        return out
    def forward_linear(self,x):
        if torch.cuda.is_available():
            dev = "cuda:0"
            
        else:
            dev = "cpu"
        device = torch.device(dev)
        out = torch.tensor(x,dtype=torch.float).to(device)
        out = self.fc1(out)
       
        return out

    def train(self,num_epochs,x_train_data,y_train_data,autocorr=False,lambda_=0.1):
        if torch.cuda.is_available():
            dev = "cuda:0"
        else:
            dev = "cpu"
        device = torch.device(dev)
        x_train_data = torch.tensor(np.array(x_train_data),dtype=torch.float).to(device)
        y_train_data = torch.tensor(np.array(y_train_data),dtype=torch.float).to(device)
        
        # print(self.forward(x_train_data)-y_train_data)
        # print(self.criterion(self.forward(x_train_data),y_train_data))
        for epoch in range(num_epochs):
            for stream in range(len(x_train_data)):    

                self.optimizer.zero_grad()
                outputs = self.forward(torch.flatten(x_train_data[stream,:,:,:]))
                loss = self.criterion(outputs, torch.flatten(y_train_data[stream,:,:,:]))
                if autocorr and stream<len(x_train_data)-4:
                    # outputs1 = outputs[0:3]
                    # outputs2 = self.forward(torch.flatten(x_train_data[stream+3,:,:,:]))[0:3]
                    # loss += ((outputs1[-1]-outputs2[0])**2).mean()*lambda_
                    # loss += (outputs1[0]-outputs1[1])**2*lambda_
                    # loss += (outputs1[1]-outputs1[2])**2*lambda_
                    outputs_arima = self.forward_linear(torch.flatten(x_train_data[stream,:,:,:]))
                    loss += self.criterion(outputs_arima, torch.flatten(y_train_data[stream,:,:,:]))*lambda_
                loss.backward()
                self.optimizer.step()
                
            # Print training statistics
            if (epoch+1) % 10 == 0:
                # print("Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}"
                    # .format(epoch+1, num_epochs, epoch+1, len(x_train_data), loss.item()))
                print("mse ",loss,"number of nodes:",self.number_of_nodes)

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size,input_sequence_length,output_sequence_length):
        super(LSTM, self).__init__()
        
        self.hidden_size = hidden_size
        print(self.hidden_size)
        self.num_layers = num_layers
        if torch.cuda.is_available():
            dev = "cuda:0"
            
        else:
            dev = "cpu"
        self.lstm = nn.LSTM(input_size,hidden_size, num_layers,batch_first=True,dtype=torch.float).to(dev)
        self.relu = nn.ReLU()
        self.output_sequence_length = output_sequence_length
        self.input_sequence_length = input_sequence_length
        self.fc1 = nn.Linear(hidden_size, output_size,dtype=torch.float).to(dev)
        # self.fc2 = nn.Linear(512, 512,dtype=torch.float).to(dev)
        # self.fc3 = nn.Linear(512, output_size,dtype=torch.float).to(dev)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.parameters(),lr=0.1)


    def forward(self, x):
        if torch.cuda.is_available():
            dev = "cuda:0"
            
        else:
            dev = "cpu"
        device = torch.device(dev)
        x = torch.tensor(x,dtype=torch.float).to(device)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size,dtype=torch.float).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size,dtype=torch.float).to(x.device)
        out, (h_n, c_n) = self.lstm(x, (h0, c0))
        out = self.relu(out[:,self.input_sequence_length-self.output_sequence_length:,:])
        out = self.fc1(out)
        # out = self.relu(out)
        # out = self.fc2(out)
        # out = self.relu(out)
        # out = self.fc3(out)
        
        return out

    def train(self,num_epochs,x_train_data,y_train_data):
        if torch.cuda.is_available():
            dev = "cuda:0"
            
        else:
            dev = "cpu"
        device = torch.device(dev)
        x_train_data = torch.tensor(x_train_data,dtype=torch.float).to(device)
        y_train_data = torch.tensor(y_train_data,dtype=torch.float).to(device)
        
        for epoch in range(num_epochs):
            
            self.optimizer.zero_grad()
            outputs = self.forward(x_train_data)
            
            loss = self.criterion(outputs, y_train_data)
            loss.backward()
            self.optimizer.step()
            

            # Print training statistics
            if (epoch+1) % 100 == 0:
                print(torch.cuda.get_device_name(0))
                print("Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}"
                    .format(epoch+1, num_epochs, epoch+1, len(x_train_data), loss.item()))
        print("mse ",loss,"number of nodes:",512)
        print("number of hidden:",self.hidden_size,"number of hidden layers:",self.num_layers)

# Define the RNN model
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.parameters(),lr=0.1)

    def forward(self, x, hidden):
        out, hidden = self.rnn(x, hidden)
        out = self.linear(out)
        return out, hidden
    def train(self,num_epochs,x_train_data,y_train_data):
        if torch.cuda.is_available():
            dev = "cuda:0"
            
        else:
            dev = "cpu"
        device = torch.device(dev)
        x_train_data = torch.tensor(x_train_data,dtype=torch.float).to(device)
        y_train_data = torch.tensor(y_train_data,dtype=torch.float).to(device)
        
        # Training loop
        for epoch in range(num_epochs):
            
            self.optimizer.zero_grad()
            outputs, hidden = self(x_train_data, hidden)
            
            loss = self.criterion(outputs, y_train_data)
            loss.backward()
            self.optimizer.step()
            
            # Print training statistics
            if (epoch+1) % 100 == 0:
                print(torch.cuda.get_device_name(0))
                print("Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}"
                    .format(epoch+1, num_epochs, epoch+1, len(x_train_data), loss.item()))
        print("mse ",loss,"number of nodes:",512)
        print("number of hidden:",self.hidden_size,"number of hidden layers:",self.num_layers)
