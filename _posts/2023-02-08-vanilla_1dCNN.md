```python
from google.colab import drive
drive.mount('/content/drive')
```

    Mounted at /content/drive
    


```python
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from torch import optim
```


```python
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(device)
```

    cuda:0
    


```python
import pandas as pd
df = pd.read_csv('/content/drive/MyDrive/ETTh1.csv')
```


```python
df.head()
```





  <div id="df-c9778767-1ffb-4b14-87b1-794dd54d9a01">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>date</th>
      <th>HUFL</th>
      <th>HULL</th>
      <th>MUFL</th>
      <th>MULL</th>
      <th>LUFL</th>
      <th>LULL</th>
      <th>OT</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2016-07-01 00:00:00</td>
      <td>5.827</td>
      <td>2.009</td>
      <td>1.599</td>
      <td>0.462</td>
      <td>4.203</td>
      <td>1.340</td>
      <td>30.531000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2016-07-01 01:00:00</td>
      <td>5.693</td>
      <td>2.076</td>
      <td>1.492</td>
      <td>0.426</td>
      <td>4.142</td>
      <td>1.371</td>
      <td>27.787001</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2016-07-01 02:00:00</td>
      <td>5.157</td>
      <td>1.741</td>
      <td>1.279</td>
      <td>0.355</td>
      <td>3.777</td>
      <td>1.218</td>
      <td>27.787001</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2016-07-01 03:00:00</td>
      <td>5.090</td>
      <td>1.942</td>
      <td>1.279</td>
      <td>0.391</td>
      <td>3.807</td>
      <td>1.279</td>
      <td>25.044001</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2016-07-01 04:00:00</td>
      <td>5.358</td>
      <td>1.942</td>
      <td>1.492</td>
      <td>0.462</td>
      <td>3.868</td>
      <td>1.279</td>
      <td>21.948000</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-c9778767-1ffb-4b14-87b1-794dd54d9a01')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-c9778767-1ffb-4b14-87b1-794dd54d9a01 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-c9778767-1ffb-4b14-87b1-794dd54d9a01');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>





```python
import matplotlib.pyplot as plt
```


```python
df.plot()
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7fc5af5b4d30>




    
![png](output_6_1.png)
    



```python
plt.plot(ydata[-64*24:])
```




    [<matplotlib.lines.Line2D at 0x7fc4c576ebb0>]




    
![png](output_7_1.png)
    



```python
ydata = df['OT']
xdata = df[['HUFL','HULL','MUFL','MULL','LUFL','LULL','OT']]
```


```python
import numpy as np
def sequence(x,y,inp,oup,sliding_window):
  lst1=[]
  lst2=[]
  idx=0
  while True:
    xd = x[idx:idx+inp]
    yd = y[idx+inp:idx+inp+oup]

    lst1.append(xd)
    lst2.append(yd)
    idx+=sliding_window
    if idx+inp+oup >= len(x):
      break
  xdata = np.array(lst1)
  ydata = np.array(lst2)
  return xdata,ydata
```


```python
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(xdata)
xdata = scaler.transform(xdata)
```


```python
x,y = sequence(xdata,ydata,96,24,24)
```


```python
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(y)
y = scaler.transform(y)
```


```python
batch = 64
train_x = x[:-batch]
train_y = y[:-batch]
valid_x = x[-batch:]
valid_y = y[-batch:]

train_x = torch.FloatTensor(train_x).to(device)
train_y = torch.FloatTensor(train_y).to(device)
valid_x = torch.FloatTensor(valid_x).to(device)
valid_y = torch.FloatTensor(valid_y).to(device)

```


```python
train_set = TensorDataset(train_x,train_y)
valid_set = TensorDataset(valid_x,valid_y)
```


```python
train_loader = DataLoader(
    train_set,
    batch_size = batch,
    shuffle = True,
    drop_last=True
    )

valid_loader = DataLoader(
    valid_set,
    batch_size = batch,
    shuffle = False
    )
```


```python
class cnn(nn.Module):
  def __init__(self):
    super(cnn,self).__init__()
    self.cnn1 = nn.Sequential(
        nn.Conv1d(7,32,3,2,padding=1),
        nn.ReLU(),
        nn.MaxPool1d(3,stride=2,padding=1)
    )
    self.cnn2 = nn.Sequential(
        nn.Conv1d(32,128,3,1,padding=1),
        nn.ReLU()
    )
    self.dropout = nn.Dropout(0.2)

    self.fn = nn.Linear(64,1)
    self.tanh = nn.Tanh()
    self.lstm = nn.GRU(128,64,batch_first=True)

  def forward(self,x):
    batch = x.shape[0]
    sequence = x.shape[1]
    variable = x.shape[2]
    out_seq = 24
    x = x.view(batch,variable,sequence)

    x = self.cnn1(x)
    x = self.cnn2(x)
    x = self.dropout(x)

    x = x.transpose(1,2)
    x,_ = self.lstm(x)
    x = self.fn(x)
    x = self.tanh(x)

    x = x.view(batch,out_seq)
    return x
```


```python
model = cnn().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(),lr=1e-4,weight_decay=1e-6)
```


```python
def train_step(batch_item,epoch,batch,training):
  x = batch_item[0]
  y = batch_item[1]
  if training is True:
    model.train()
    optimizer.zero_grad()
    output = model(x)
    loss = criterion(output,y)
    loss.backward()
    optimizer.step()
    return loss
  else:
    model.eval()
    with torch.no_grad():
      output = model(x)
      loss = criterion(output,y)

    return loss,output,y
```


```python
from tqdm import tqdm
train_loss_plot,valid_loss_plot = [],[]
predict = []
real = []
epochs=300
for epoch in range(epochs):
  total_loss,val_loss = 0,0
  tqdm_dataset = tqdm(enumerate(train_loader))
  training = True
  for batch,batch_item in tqdm_dataset:
    batch_loss = train_step(batch_item,epoch,batch,training)
    total_loss += batch_loss

    tqdm_dataset.set_postfix({
        'Epoch': epoch+1,
        'Loss' : '{:06f}'.format(batch_loss.item()),
        'Total Loss' : '{:06f}'.format(total_loss/(batch+1))
    })
  train_loss_plot.append((total_loss/(batch+1)).cpu().detach().numpy())


  tqdm_dataset = tqdm(enumerate(valid_loader))
  training = False
  for batch,batch_item in tqdm_dataset:
    batch_loss,pred,true = train_step(batch_item,epoch,batch,training)
    val_loss += batch_loss

    tqdm_dataset.set_postfix({
        'Epoch': epoch+1,
        'Loss' : '{:06f}'.format(batch_loss.item()),
        'Val Loss' : '{:06f}'.format(val_loss/(batch+1))
    })
  valid_loss_plot.append((val_loss/(batch+1)).cpu().detach().numpy())
  if epoch+1 == epochs:
    predict.append(pred.cpu().detach().numpy())
    real.append(true.cpu().detach().numpy())

```


```python
np.ravel(predict[0], order='C')
```




    array([0.21121897, 0.24198611, 0.25637367, ..., 0.2889117 , 0.30024824,
           0.28892744], dtype=float32)




```python
predict[0].shape
```




    (64, 24)




```python
plt.figure(figsize=(12,12))
plt.plot(np.ravel(predict[0], order='C'),label='pred')
plt.plot(np.ravel(real[0], order='C'),label='True')
plt.legend()
```




    <matplotlib.legend.Legend at 0x7fc56d5a85e0>




    
![png](output_22_1.png)
    



```python
plt.figure(figsize=(12,8))
plt.plot(np.array(train_loss_plot),label='Train')
plt.plot(np.array(valid_loss_plot),label='Valid')
plt.legend()
```




    <matplotlib.legend.Legend at 0x7fc4ed125310>




    
![png](output_23_1.png)
    



```python

```
