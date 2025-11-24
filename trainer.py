import torch
import torch as t
from sklearn.metrics import f1_score, accuracy_score
from tqdm.autonotebook import tqdm
import pandas as pd

class Trainer:

    def __init__(self,
                 model,  # Model to be trained.
                 crit,  # Loss function
                 optim=None,  # Optimizer
                 train_dl=None,  # Training data set
                 val_test_dl=None,  # Validation (or test) data set
                 cuda=True,  # Whether to use the GPU
                 early_stopping_patience=-1):  # The patience for early stopping
        self._model = model
        self._crit = crit
        self._optim = optim
        self._train_dl = train_dl
        self._val_test_dl = val_test_dl
        self._cuda = cuda

        self._early_stopping_patience = early_stopping_patience
        self._labels =torch.empty(0, 2)
        self._pred = torch.empty(0, 2)

        if cuda:
            self._model = model.cuda()

    def save_checkpoint(self, epoch):
        t.save({'state_dict': self._model.state_dict()}, 'checkpoints/checkpoint_{:03d}.ckp'.format(epoch))
        print("checkpoint saved")

    def restore_checkpoint(self, epoch_n):
        ckp = t.load('checkpoints/checkpoint_{:03d}.ckp'.format(epoch_n), 'cuda' if self._cuda else None)
        self._model.load_state_dict(ckp['state_dict'])

    def save_onnx(self, fn):
        m = self._model.cpu()
        m.eval()
        x = t.randn(1, 3, 300, 300, requires_grad=True)
        y = self._model(x)
        t.onnx.export(m,  # model being run
                      x,  # model input (or a tuple for multiple inputs)
                      fn,  # where to save the model (can be a file or file-like object)
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=10,  # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=['input'],  # the model's input names
                      output_names=['output'],  # the model's output names
                      dynamic_axes={'input': {0: 'batch_size'},  # variable lenght axes
                                    'output': {0: 'batch_size'}})

    def train_step(self, x, y):
        self._optim.zero_grad()  # -reset the gradients. By default, PyTorch accumulates (sums up) gradients when backward() is called. This behavior is not required here, so you need to ensure that all the gradients are zero before calling the backward.
        output = self._model(x)  # -propagate through the network
        loss = self._crit().cuda()
        loss_val = loss(output,y.float())  # -calculate the loss
        loss_val.backward()  # -compute gradient by backward propagation
        self._optim.step()  # -update weights
        return loss_val  # -return the loss

    def val_test_step(self, x, y):
        output = self._model(x)  # propagate through the network and calculate the loss and predictions
        loss = self._crit().cuda()
        loss_val = loss(output, y.float())
        return loss_val, output  # return the loss and the predictions

    def train_epoch(self):
        self._model.train()  # set training mode
        epoch_loss = 0
        for local_batch, local_labels in self._train_dl:  # iterate through the training set
            local_batch, local_labels = local_batch.to("cuda"), local_labels.to("cuda")  # transfer the batch to "cuda()" -> the gpu if a gpu is given
            local_loss = self.train_step(local_batch, local_labels)  # perform a training step
            epoch_loss += local_loss

        avg_loss = epoch_loss / len(self._train_dl)  # calculate the average loss for the epoch and return it
        return avg_loss

    def val_test(self):
        epoch_loss = 0
        results = []
        self._model.eval()  # set eval mode. Some layers have different behaviors during training and testing (for example: Dropout, BatchNorm, etc.). To handle those properly, you'd want to call model.eval()
        # self._model.no_grad() # disable gradient computation. Since you don't need to update the weights during testing, gradients aren't required anymore.
        with t.no_grad():
            for local_batch, local_labels in self._val_test_dl:  # iterate through the validation set
                self._labels = torch.cat((self._labels, local_labels), 0)
                local_batch, local_labels = local_batch.to("cuda"), local_labels.to("cuda")  # transfer the batch to the gpu if given
                local_loss, predictions = self.val_test_step(local_batch, local_labels) # perform a validation step
                epoch_loss += local_loss
                predictions = (predictions>0.5).float()
                self._pred = torch.cat((self._pred,predictions.cpu()), 0)
                results.append([[local_batch, local_labels, predictions]])  # save the predictions and the labels for each batch
        avg_loss = epoch_loss / len(self._val_test_dl)  # calculate the average loss and average metrics of your choice. You might want to calculate these metrics in designated functions

        return avg_loss  # return the loss and print the calculated metrics

    def fit(self, epochs=-1):
        assert self._early_stopping_patience > 0 or epochs > 0
        # create a list for the train and validation losses, and create a counter for the epoch 
        train_losses = []
        val_losses = []
        counter = 0
        flag = 0
        best_val_loss =0

        while counter <= epochs:
            # stop by epoch number
            avg_train_loss = self.train_epoch().float().cpu().detach().numpy()  # train for a epoch and then calculate the loss and metrics on the validation set
            avg_val_loss = self.val_test().float().cpu().detach().numpy()
            train_losses.append(avg_train_loss)  # append the losses to the respective lists
            val_losses.append(avg_val_loss)

            if counter == 0:
                self.save_checkpoint(counter)
                flag = 0
                best_val_loss = avg_val_loss
            elif avg_val_loss < best_val_loss:
                self.save_checkpoint(counter)
                flag = 0
                best_val_loss = avg_val_loss
            else:
                flag += 1  # use the save_checkpoint function to save the model (can be restricted to epochs with improvement)
            # check whether early stopping should be performed using the early stopping criterion and stop if so
            # return the losses for both training and validation
            accuracy, f1 = self.accuracy_f1()
            print("Epoch: " + str(counter) + "/" + str(epochs)  + " Train Loss: ", str(avg_train_loss)  + " Val Loss: ", str(avg_val_loss) + " Accuracy: " + str(accuracy) + " F1 Score: "+str(f1))
            if flag == self._early_stopping_patience:
                break
            #self.save_checkpoint(counter)
            print("Validation Loss not Improving:" + str(flag))
            print("BEST " + str(best_val_loss))
            counter += 1

        return train_losses, val_losses

    def accuracy_f1(self):
        labels = self._labels.detach().numpy()
        predictions = self._pred.detach().numpy()

        accuracy_crack = accuracy_score(labels[:,0], predictions[:, 0])
        accuracy_inactive = accuracy_score(labels[:, 1], predictions[:, 1])
        accuracy = (accuracy_crack+accuracy_inactive)/2
        f1_crack = f1_score(labels[:,0], predictions[:, 0], average='micro')
        f1_inactive = f1_score(labels[:, 1], predictions[:, 1], average='micro')
        f1 = (f1_crack+f1_inactive)/2

        return accuracy, f1
