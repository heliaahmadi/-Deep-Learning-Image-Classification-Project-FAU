import torch as t
from data import ChallengeDataset
from trainer import Trainer
from matplotlib import pyplot as plt
import numpy as np
import model
import pandas as pd
from sklearn.model_selection import train_test_split
import time

_batch_size = 6
_shuffle = True
_epochs = 120
_learning_rate = 0.00001
_early_stopping_patience = 50
optimiser_params = {'lr': _learning_rate}
t.cuda.empty_cache()
# load the data from the csv file and perform a train-test-split
# this can be accomplished using the already imported pandas and sklearn.model_selection modules
df = pd.read_csv('data.csv', sep=';')
df.to_csv('out.csv', index=False)
train, test = train_test_split(df, test_size=0.25, random_state=4)

# set up data loading for the training and validation set each using t.utils.data.DataLoader and ChallengeDataset objects
train_dataloader = t.utils.data.DataLoader(
    ChallengeDataset(train, mode="train", target_height=1024),
    batch_size=_batch_size, shuffle=_shuffle)

test_dataloader = t.utils.data.DataLoader(
    ChallengeDataset(test, mode="val", target_height=1024),
    batch_size=_batch_size)
print("Number of training samples: ", len(train_dataloader.dataset))
print("Number of validation samples: ", len(test_dataloader.dataset))
print("Number of training batches: ", len(train_dataloader))
print("Number of validation batches: ", len(test_dataloader))

# create an instance of our ResNet model
MODEL = model.ResNet()
print("MODEL Loaded")
# set up a suitable loss criterion (you can find a pre-implemented loss functions in t.nn)
# set up the optimizer (see t.optim)
# create an object of type Trainer and set its early stopping criterion
LOSS_FUNCTION = t.nn.BCELoss
OPTIMIZER = t.optim.Adam(MODEL.parameters(), **optimiser_params, weight_decay=0.08)
#OPTIMIZER = t.optim.Adam(MODEL.parameters(), lr=0.001, weight_decay=0.0004)
#OPTIMIZER = t.optim.SGD(MODEL.parameters(), **optimiser_params, momentum=0.9, weight_decay=0.04)
print(optimiser_params)
TRAINER = Trainer(model=MODEL, crit=LOSS_FUNCTION, optim=OPTIMIZER, train_dl=train_dataloader,
                  val_test_dl=test_dataloader, early_stopping_patience=_early_stopping_patience)
# print(LOSS_FUNCTION)
# print(OPTIMIZER)
# print(TRAINER)
# go, go, go... call fit on trainer
start = time.time()
res = TRAINER.fit(epochs =_epochs) 
end = time.time()
print(f"Runtime of the program is {(end - start)/60} minutes")
plt.plot(np.arange(len(res[0])), res[0], label='train loss')
plt.plot(np.arange(len(res[1])), res[1], label='val loss')
plt.yscale('log')
plt.legend()
plt.savefig('losses.png')
