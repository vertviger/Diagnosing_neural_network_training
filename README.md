# Neural Network Training Diagnosis

## Specs:
    Task: Classification
    Dataset: Iris, split size 90/30/30, sampled (4 features, 3 classes)
    Hyperparameters: Number of Neurons on hidden layer, Number of Epochs, Learning rate, Batch size, Weight Decay, Dropout Probability, Optimizer.
    Final Model Settings: 
    One Hidden Layer
    16 neurons on hidden layer
    200 epochs
    Lr = 0.05 
    Batch size = 30 
    Weight Decay = 1e-3 
    Dropout Probability = 0.10
    Optimizer = AdamW (Compatible with weight decay)
    Activation functions: Linear on outer layer, RELU on hidden layer

<img width="1792" height="1792" alt="image" src="https://github.com/user-attachments/assets/43bed6c9-d529-4e32-91c5-8ef8597e0086" />

## Final model configuration and results
Such final configuration was chosen because it is a middle ground between training stability and compatibility for any hardware. 
The batch size is set accordingly, and a balanced learning rate prevents overfitting and underfitting.
Dropout probability is set to 10 percent to not cause too much training noise.
As optimizer function I chose AdamW, as Adam uses a great concept of momentum + W part makes it very well compatible with Weight Decay.
Such high number of epochs was chosen to better analyze concepts of underfitting and overfitting.
As measurements I used standard Loss and Accuracy, showcased on the graphs. These metrics fit well for classification tasks. 
As a result model achieved 0.015 Training and 0.057 Validation Loss with 100 percent Test set accuracy.
Possible improvement is adding learning rate decay, as current setup achieves great accuracy and loss fast and then varies about these values for next 100 epochs. 

### Problems and solutions
**Unstable training**:  With small batch sizes the training samples vary drastically, producing big Loss changes, leading to an unstable training process. 
_Measures for batch size of 3_: <img width="1779" height="892" alt="image" src="https://github.com/user-attachments/assets/80304b06-1407-4c34-b98b-26541d3c2ca7" />
The solution to such problem is increasing batch size to reasonable sizes. 
Another solution could be decreasing learning rate. 
Starting with higher values to explore the dataset faster and then lowering it to find minimum with better approximation and less noise.
_Measures for batch size of 48_: <img width="1777" height="892" alt="image" src="https://github.com/user-attachments/assets/f3558c4f-ed8d-4854-9860-3cec6910a553" />
**Underfitting**: Learning rate too small leads to slow training process.
_Lr = 0.0005_: <img width="1773" height="889" alt="image" src="https://github.com/user-attachments/assets/c29836bf-dab8-4a6e-ab2a-d1d50d290036" />
Possible solutions may be: Increasing Lr or Increasing number of epochs
_Lr = 0.05_					           _Lr = 0.0005, N_Epochs = 2000_ 
<img width="1699" height="847" alt="image" src="https://github.com/user-attachments/assets/6cbcc670-faf5-474a-a972-76d472eb917b" />

### Experiment
_Observing impact of batch size on training process_
**Very small batch size**: Training_Set_Size / 18 ( equals to 5 in current dataset)
<img width="1783" height="892" alt="image" src="https://github.com/user-attachments/assets/b7de2051-b80c-4fe9-889a-d9a211fddc79" />
We can observe huge variations of loss function on the whole training process, as small batch sizes produce great instability because of how much data varies from batch to batch.
**Medium batch size**: Training_Set_Size / 3 ( equals to 30 in current dataset)
<img width="1684" height="843" alt="image" src="https://github.com/user-attachments/assets/5d90cda3-5297-4427-a722-91d6e2a0ce3a" />
We can see that the starting accuracy was lower compared to the small batch size, but after approximately 15 epochs it got to the 92+ percent values and was more stable in that range. Also loss function has lower values and less noise.  
**Big batch size**: Training_Set_Size ( equals to 90 in current dataset)
<img width="1678" height="838" alt="image" src="https://github.com/user-attachments/assets/a9265679-b04c-46f1-a6a4-017067616eef" />
Loss and Accuracy curves are much more stable, because the model operates on the whole dataset at once, therefore computing gradients on entire dataset.
However this requires much greater memory capacity and in greater applications, like Image processing, is not possible.
