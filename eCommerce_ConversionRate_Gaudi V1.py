import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from habana_frameworks.torch.utils.library_loader import load_habana_module
load_habana_module()
import habana_frameworks.torch.core as htcore


import time

# Model Parameters
EPOCHS = 1428
BATCH_SIZE = 1200
LEARNING_RATE = 0.001
TRAIN_MODEL = 0

# Define Custom Dataloaders
class TrainData(Dataset):
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data

    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]

    def __len__(self):
        return len(self.X_data)

class ValidationData(Dataset):
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data

    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]

    def __len__(self):
        return len(self.X_data)

class TestData(Dataset):
    def __init__(self, X_data):
        self.X_data = X_data

    def __getitem__(self, index):
        return self.X_data[index]

    def __len__(self):
        return len(self.X_data)


class BinaryClassification(nn.Module):
    def __init__(self):
        super(BinaryClassification, self).__init__()
        # Number of input features is 46.
        self.layer_1 = nn.Linear(46, BATCH_SIZE)
        self.layer_2 = nn.Linear(BATCH_SIZE, BATCH_SIZE)
        self.layer_3 = nn.Linear(BATCH_SIZE, BATCH_SIZE)
        self.layer_out = nn.Linear(BATCH_SIZE, 1)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)
        self.batchnorm1 = nn.BatchNorm1d(BATCH_SIZE)
        self.batchnorm2 = nn.BatchNorm1d(BATCH_SIZE)
        self.batchnorm3 = nn.BatchNorm1d(BATCH_SIZE)

    def forward(self, inputs):
        x = self.relu(self.layer_1(inputs))
        x = self.batchnorm1(x)
        x = self.relu(self.layer_2(x))
        x = self.batchnorm2(x)
        x = self.relu(self.layer_3(x))
        x = self.batchnorm3(x)
        x = self.dropout(x)
        x = self.layer_out(x)

        return x 



def load_data():
    #Load Data
    df_train = pd.read_csv("/home/studio-lab-user/eCommerce Conversion Rate Improvement/DATA/CR NewRelic Combine Data.CSV")
    '''
    df_train = df_train.loc[:,['Conversion_Status', 'userAgentName','userAgentOS','deviceType','countryCode','avg_duration','avg_timeToDomComplete','avg_timeToDomContentLoadedEventEnd','avg_timeToDomContentLoadedEventStart','avg_timeToLoadEventEnd','Req_Count']]
    '''
    
    df_pred = pd.read_csv("/home/studio-lab-user/eCommerce Conversion Rate Improvement/DATA/Pred.csv")
    '''
    df_pred = df_pred.loc[:,['Conversion_Status', 'userAgentName', 'userAgentOS', 'deviceType','countryCode','avg_duration','avg_timeToDomComplete','avg_timeToDomContentLoadedEventEnd','avg_timeToDomContentLoadedEventStart','avg_timeToLoadEventEnd','Req_Count']]
    '''
    return df_train,df_pred

def data_preprocessing(df_train,df_pred):
    le = preprocessing.LabelEncoder()
    le.fit(df_train['userAgentName'])
    df_train['userAgentName'] = list(le.transform(df_train['userAgentName']))
    le.fit(df_train['userAgentOS'])
    df_train['userAgentOS'] = list(le.transform(df_train['userAgentOS']))
    le.fit(df_train['deviceType'])
    df_train['deviceType'] = list(le.transform(df_train['deviceType']))
    le.fit(df_train['countryCode'])
    df_train['countryCode'] = list(le.transform(df_train['countryCode']))

    le.fit(df_pred['userAgentName'])
    df_pred['userAgentName'] = list(le.transform(df_pred['userAgentName']))
    le.fit(df_pred['userAgentOS'])
    df_pred['userAgentOS'] = list(le.transform(df_pred['userAgentOS']))
    le.fit(df_pred['deviceType'])
    df_pred['deviceType'] = list(le.transform(df_pred['deviceType']))
    le.fit(df_pred['countryCode'])
    df_pred['countryCode'] = list(le.transform(df_pred['countryCode']))

    return df_train,df_pred

def train_test_pred_data(df_train,df_pred):
    X = df_train.iloc[:, 1:]
    y = df_train['Conversion_Status']
    X_pred = df_pred.iloc[:, 1:]
    y_pred = df_pred['Conversion_Status']

    # Train Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=44)

    # Standardization/normalization in neural nets here.
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    X_pred = scaler.transform(X_pred)

    #print(y_test)

    train_data = TrainData(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    test_data = ValidationData(torch.FloatTensor(X_test), torch.FloatTensor(y_test.to_numpy()))
    pred_data = TestData(torch.FloatTensor(X_pred))

    #Let’s initialize our dataloaders. We’ll use a batch_size = 1 for our test dataloader.
    train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
    validation_loader = DataLoader(dataset=test_data, batch_size=BATCH_SIZE)
    pred_loader = DataLoader(dataset=pred_data, batch_size=BATCH_SIZE)
    return train_loader,validation_loader,pred_loader,scaler


# BaseExceptionBinary accuracy functions
def binary_acc(y_pred, y_test):
    y_pred_tag = torch.round(torch.sigmoid(y_pred))

    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum / y_test.shape[0]
    acc = torch.round(acc * 100)

    return acc

def build_model(train_loader,validation_loader,pred_loader):
    # Check if GPU cuda is available to run model on
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    # Define model and set OPtimizer
    model = BinaryClassification()
    model.to(device)
    print(model)
    criterion = nn.BCEWithLogitsLoss()
    #optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)

    start_time = time.time()
    min_valid_loss = np.inf

    # Train the model
    model.train()
    for e in range(1, EPOCHS + 1):
        epoch_loss = 0
        epoch_acc = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()

            y_pred = model(X_batch)

            train_loss = criterion(y_pred, y_batch.unsqueeze(1))
            acc = binary_acc(y_pred, y_batch.unsqueeze(1))

            train_loss.backward()
            optimizer.step()
            htcore.mark_step()

            epoch_loss += train_loss.item()
            epoch_acc += acc.item()
            
            
            
        #New code  
        valid_loss = 0.0
        model.eval()     # Optional when not using Model Specific layer
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        for X_batch,y_batch in validation_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            target = model(X_batch)
            valid_loss = criterion(target, y_batch.unsqueeze(1))
            acc = binary_acc(target, y_batch.unsqueeze(1))

        print(f'Epoch {e+1} \t\t Training Loss: {train_loss / len(train_loader)} \t\t Validation Loss: {valid_loss / len(validation_loader)}')
        print(f'Epoch {e + 0:03}: | Loss: {epoch_loss / len(train_loader):.5f} | Acc: {epoch_acc / len(train_loader):.3f}')

        if min_valid_loss > valid_loss:
            print(f'Validation Loss Decreased({min_valid_loss:.6f}--->{valid_loss:.6f}) \t Saving The Model')
            min_valid_loss = valid_loss
            # Saving State Dict
            torch.save(model, '/home/studio-lab-user/eCommerce Conversion Rate Improvement/MODEL/eCommerce_ConversionRate_New')
        #print(f'Epoch {e + 0:03}: | Loss: {epoch_loss / len(train_loader):.5f} | Acc: {epoch_acc / len(train_loader):.3f}')
         
    #torch.save(model, '/home/studio-lab-user/eCommerce Conversion Rate Improvement/MODEL/eCommerce_ConversionRate_New')
    print("--- %s seconds ---" % (time.time() - start_time))
    return model,device

def read_predictionfile(filename):
    le = preprocessing.LabelEncoder()
    df_pred = pd.read_csv(filename)
    '''
    df_pred = df_pred.loc[:,['Conversion_Status', 'userAgentName', 'userAgentOS', 'deviceType','countryCode','avg_duration','avg_timeToDomComplete','avg_timeToDomContentLoadedEventEnd','avg_timeToDomContentLoadedEventStart','avg_timeToLoadEventEnd','Req_Count']]
    '''
    le.fit(df_pred['userAgentName'])
    df_pred['userAgentName'] = list(le.transform(df_pred['userAgentName']))
    le.fit(df_pred['userAgentOS'])
    df_pred['userAgentOS'] = list(le.transform(df_pred['userAgentOS']))
    le.fit(df_pred['deviceType'])
    df_pred['deviceType'] = list(le.transform(df_pred['deviceType']))
    le.fit(df_pred['countryCode'])
    df_pred['countryCode'] = list(le.transform(df_pred['countryCode']))
    #print(list(df_pred.columns[5:10]))
    return df_pred



def pred_model(device,scaler):
    model = torch.load('/home/studio-lab-user/eCommerce Conversion Rate Improvement/MODEL/eCommerce_ConversionRate_New')
    print(model)

    df_pred = read_predictionfile("/home/studio-lab-user/eCommerce Conversion Rate Improvement/DATA/Pred.csv")
    X_pred = df_pred.iloc[:, 1:]
    y_pred = df_pred['Conversion_Status']
    X_pred = scaler.fit_transform(X_pred)
    #X_train = scaler.fit_transform(X_train)

    pred_data = TestData(torch.FloatTensor(X_pred))
    pred_loader = DataLoader(dataset=pred_data, batch_size=1)

    # Model Prediction and Evaluation
    y_pred_list = []
    model.eval()
    with torch.no_grad():
        for X_batch in pred_loader:
            X_batch = X_batch.to(device)
            y_test_pred = model(X_batch)
            y_test_pred = torch.sigmoid(y_test_pred)
            y_pred_tag = torch.round(y_test_pred)
            y_pred_list.append(y_pred_tag.cpu().numpy())

    y_pred_list = [a.squeeze().tolist() for a in y_pred_list]
    No_Conversion_Pred_Count = y_pred_list.count(0.0)
    print(No_Conversion_Pred_Count)
    return No_Conversion_Pred_Count




def validation_model(device,scaler):
    model = torch.load('/home/studio-lab-user/eCommerce Conversion Rate Improvement/MODEL/eCommerce_ConversionRate_New')
    print(model)

    df_pred = read_predictionfile("/home/studio-lab-user/eCommerce Conversion Rate Improvement/DATA/Pred.csv")
    X_pred = df_pred.iloc[:, 1:]
    y_pred = df_pred['Conversion_Status']
    X_pred = scaler.fit_transform(X_pred)
    #X_train = scaler.fit_transform(X_train)

    pred_data = TestData(torch.FloatTensor(X_pred))
    pred_loader = DataLoader(dataset=pred_data, batch_size=1)

    # Model Prediction and Evaluation
    y_pred_list = []
    model.eval()
    with torch.no_grad():
        for X_batch in pred_loader:
            X_batch = X_batch.to(device)
            y_test_pred = model(X_batch)
            y_test_pred = torch.sigmoid(y_test_pred)
            y_pred_tag = torch.round(y_test_pred)
            y_pred_list.append(y_pred_tag.cpu().numpy())

    y_pred_list = [a.squeeze().tolist() for a in y_pred_list]
    No_Conversion_Pred_Count = y_pred_list.count(0.0)
    print(No_Conversion_Pred_Count)
    return No_Conversion_Pred_Count



def pred_conversion_rate_impact(device,scaler,No_Conversion_Pred_Count):
    model = torch.load('/home/studio-lab-user/eCommerce Conversion Rate Improvement/MODEL/eCommerce_ConversionRate_New')

    Performance_Gain = [0.00, 0.02, 0.03, 0.04, 0.05,0.06]
    #Performance_Gain = [0.00]
    df_pred = read_predictionfile("/home/studio-lab-user/eCommerce Conversion Rate Improvement/DATA/Pred.csv")
    actual_conversion = [float(i) for i in list(df_pred['Conversion_Status'])]
    '''
    Critical_KPIs = ['avg_duration','avg_timeToDomComplete','avg_timeToDomContentLoadedEventEnd','avg_timeToDomContentLoadedEventStart','avg_timeToLoadEventEnd','Req_Count']
    '''
    Critical_KPIs = list(df_pred.columns)[5:27]
    #Critical_KPIs = list(df_pred.columns)[5:6]

    improvement_dict = {}
    Data = pd.DataFrame()
    
    for col in Critical_KPIs:
        df_pred = read_predictionfile("/home/studio-lab-user/eCommerce Conversion Rate Improvement/DATA/Pred.csv")
        #print(f'KPI {col} shows Conversion Number Improvement')
        No_ConversionNumbers_Count_List = []
        Conversion_Rate_Improvement_List = []
        for pg in Performance_Gain:
            df_pred = read_predictionfile("/home/studio-lab-user/eCommerce Conversion Rate Improvement/DATA/Pred.csv")
            df_pred[col] = df_pred[col] * (1.00 - pg)
            X_pred = df_pred.iloc[:, 1:]
            y_pred = df_pred['Conversion_Status']    
            X_pred = scaler.transform(X_pred)

            pred_data = TestData(torch.FloatTensor(X_pred))
            pred_loader = DataLoader(dataset=pred_data, batch_size=1)

            # Model Prediction and Evaluation
            y_pred_list = []
            model.eval()
            with torch.no_grad():
                for X_batch in pred_loader:
                    # print(X_batch)
                    X_batch = X_batch.to(device)
                    y_test_pred = model(X_batch)
                    y_test_pred = torch.sigmoid(y_test_pred)
                    y_pred_tag = torch.round(y_test_pred)
                    y_pred_list.append(y_pred_tag.cpu().numpy())

            y_pred_list = [a.squeeze().tolist() for a in y_pred_list]
            #print(col)
            #print(confusion_matrix(actual_conversion, y_pred_list))
            No_ConversionNumbers_Count_List.append(y_pred_list.count(0.0))
            Conversion_Rate_Improvement_List.append(
                round((No_Conversion_Pred_Count - y_pred_list.count(0.0)) / No_Conversion_Pred_Count * 100, 2))

        x = np.array(["0%", "2%", "3%", "4%", "5%","6%"])
        max_improvement = max(Conversion_Rate_Improvement_List)
        min_improvement = min(Conversion_Rate_Improvement_List)
        if (min_improvement >= 0 and max_improvement > 0 ):
                improvement_dict[col] = Conversion_Rate_Improvement_List
                Data[col] = pd.Series(Conversion_Rate_Improvement_List)


    max_cr_dict = dict(Data.max())
    max_cr_kpis = sorted(max_cr_dict, key=max_cr_dict.get, reverse=True)[:3]
    
    improvement_dict = { select_max_kpis: improvement_dict[select_max_kpis] for select_max_kpis in max_cr_kpis }

    #print(improvement_dict)
    N = 6
    ind = np.arange(N) 
    width = 0.25
    
    fig = plt.figure(figsize=(12, 8))

    for kpi,values in improvement_dict.items():
        plt.bar(ind, values, width, label=kpi)
        ind = ind + width
    
    plt.ylabel('CR Improvement')
    plt.xticks((ind + width) + 0.15    , ('2%', '3%', '4%', '5%','6%',''))
    plt.legend(loc='best')
    #plt.show()
    plt.savefig('/home/studio-lab-user/eCommerce Conversion Rate Improvement/CHARTS/CR Improvement.PNG')


        
def main():
    if TRAIN_MODEL == 1:
        df_train,df_pred = load_data()
        df_train,df_pred = data_preprocessing(df_train,df_pred)
        train_loader,test_loader,pred_loader,scaler = train_test_pred_data(df_train,df_pred)
        model,device = build_model(train_loader,test_loader,pred_loader)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    scaler = StandardScaler()
    #No_Conversion_Pred_Count = validation_model(device,scaler)
    No_Conversion_Pred_Count = pred_model(device,scaler)
    pred_conversion_rate_impact(device,scaler,No_Conversion_Pred_Count)
        
if __name__ == "__main__":
   main()