# Simple Neural Network using Pytorch

**Objective :**
Train and Create 1 model baesd on IRIS dataset which can **predict** the type of type of flower based on **IRIS dataset** 

### Step 1: Necessary Import
- Torch is base module for pytorch
 - NN is for neural network
- F is responsible each forward function in pytorch

    ```bash
        import torch
        import torch.nn as nn
        import torch.nn. functional as F 
    ```
### Step 2: Create Model Class
Create model Class that inherits nn.Module
```bash
class Model(nn.Module):

    # Input layer ( 4 features of  the flower) --> 
    # Hidden layer1 (NUmber of neuron) --> 
    # Hidden layer2 (number of neurons) --> 
    # H2 (n) --> 
    # Output ( 3 classes of iris flowers)

    def __init__(self, in_features = 4, h1=8, h2=9, out_features=3):
        super().__init__()
        self.fc1 = nn.Linear(in_features, h1)           # Here fc stands for fully connected layer 
        self.fc2 = nn.Linear(h1, h2)                    # Note OUt of one layer will be input of second layer
        self.out = nn.Linear(h2, out_features)

    def forward(self, x):
        x = F.relu(self.fc1(x))                         # relu stands for Rectified Linear Units
        x = F.relu(self.fc2(x))              
        x = self.out(x)

        return x
```
### Step 3 : Manual Seed (Optional but Recommendable):
- Pick a manual ssed for randomization --> So we get the approximate same start
    ```bash
        torch.manual_seed(41)
        model = Model()
    ```
### Step 4 : Import file 
- Pandas is good option to import csv data and deal with dataframe.
    ```bash
        import pandas as pd
        my_df = pd.read_csv('iris.csv') # Change iris.csv with your file path
        my_df.head() # This optional because it give first 5 entries of the dataframe
    ```
### Step 5 : Change Categorical data to Numerical Data
Since, Computer do not understand names but species contain the names we have to replace each species with numbers. 
```bash
    my_df['species'] = my_df['species'].replace('setosa', 0.0)
    my_df['species'] = my_df['species'].replace('versicolor', 1.0)
    my_df['species'] = my_df['species'].replace('virginica', 2.0)
```
### Step 6 : Train Test Split! X, y
- Data contain **features** (sepal_length, sepal_width, petal_length, petal_width ) and **labels** ( species ). 
- First we have to divide the data into features and label.
    ```bash
        X = my_df.drop('species', axis=1)
        y = my_df['species']
    ```
    - **.drop()** we drop the species column from the dataframe, Remaining data store in X matrix
    - **dataframe['Column_name']** this will store species column in y matrix

- We will use sklearn train_test_split model
    - Make sure we have data in numpy array because **sklearn work train_test_split work with numpy array only**
    In case data not in numpy array 
    ```bash
        X = X.values
        y = y.values
    ```
    - We will divide the data intor 80% for training and 20% for testing.
    ```bash
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=41)
    ```    
    - Convert data again to float tensor and Long Tensor  because **pytorch work with tensor only**
     ```bash
        X_train = torch.FloatTensor(X_train)
        X_test = torch.FloatTensor(X_test)

        y_train = torch.LongTensor(y_train)
        y_test = torch.LongTensor(y_test)
    ```   
### Step 7 : Set the loss and optimizer 
- Loss measure the error how far off the prediction are from original 
- Optimizer algorithm responsible for adjusting the model's parameters (weights and biases) to minimize the **loss function**.  
    - Think of it as the engine that drives the learning process.In machine learning, an optimizer is a crucial component of the training process. It's the **algorithm responsible for adjusting the model's parameters (weights and biases) to minimize the loss function.**  Think of it as the engine that drives the learning process.
    ```bash
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    ```
### Step 8 : Training 
Here we set the traning for 300 epoch we can modify as per our need. 
```bash
    epoch = 300
    losses = []
    for i in range(epoch):
        y_pred = model.forward(X_train) 
        loss = criterion(y_pred, y_train) # predicted values vs the y_train

        # Keep Trach of our losses
        losses.append(loss.detach().numpy())

        #print every 10 epoch
        if i % 10 == 0:
            print(f"Epoch : {i} loss : {loss}")

        # Do some back propagation : Take the error rate of forward propatation and feed it back 
        # thru the network to fine tune the weights

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

```
### Step 8 : Ploting ( Optional)
I like graph then number because those are easy to understand so we can draw the graph of our trained machine model.
Total epoch vs loss, with each epoch loss is reducing that is good.
```bash
    plt.plot(range(epoch), losses)
    plt.ylabel('loss/error')
    plt.xlabel('Epoch')
```
![](images/Epoch_Vs_Loss.jpg)

### Step 9 : Evaluation 
- Evaluate MOdel on test data set (validate model on test set)
- Check with our model with test data how much loss we have reduce or not 
```bash
    with torch.no_grad():                  
    y_eval = model.forward(X_test)     
    loss = criterion(y_eval, y_test)
```
### Step 10 : Checker ( Optional )
- IN this block we are checking how many prediction are correct.
- We change species coloumn back to the name 
- If test and prediction value machine it increase the correct by 1 
```bash
    correct = 0
    with torch.no_grad():
        for i, data in enumerate(X_test):
            y_val = model.forward(data)

            if y_test[i] == 0:
                x = "setosa"
            elif y_test[i] == 1:
                x = "Versicolor"
            elif y_test[i] == 2:
                x= "virginica"
            
        # Will tell us what type of flower class our network think it is
            print(f"{i+1} - {str(y_val)} \t {y_test[i]} \t{y_val.argmax().item()}")

            # Correct or not
            if y_val.argmax().item() == y_test[i]:
                correct +=1

    print(f"we got {correct} correct")

```
- We can give the random data and predict the result.
    - In the prediction high value will be our predicted result
```bash
    with torch.no_grad():
        print(model(new_iris))
```

### Step 11 : Save Our NN model 
```bash
    torch.save(model.state_dict(), 'iris_dataset_train_model.pt')
```

### Step 12 : Load the saved model
- Load the model
```bash
    new_model = Model()
    new_model.load_state_dict(torch.load('iris_dataset_train_model.pt'))
```
- Make sure you loaded your model correctly, Evaluate the model for all the hidden layer
```bash
    new_model.eval()
```

### Step 13 : Testing with already trained model 
```bash
    rand_iris_set = torch.tensor([5.9, 3.1, 5.2, 1.8])
    with torch.no_grad():
    print(new_model(rand_iris_set))
```

---
### Author : shergillkuldeep@outlook.com
**Note :** 
- Feel free to contact if you have any suggestion.
- I want to give a **big thanks** to [Codemy](https://codemy.com)! 
    - Their explanations really helped me grasp PyTorch neural networks.
    - I'd recommend checking them out – you can find their Neural Network materials at [Deep Learning Playlist](https://www.youtube.com/playlist?list=PLCC34OHNcOtpcgR9LEYSdi9r7XIbpkpK1).