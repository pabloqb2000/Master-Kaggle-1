import numpy as np
from IPython import display
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim

class Trainer:
    def __init__(self, model, epochs, criterion, optim, lr, stopping_epochs,
                 batches_per_epoch = None, use_cuda = True):
        self.model = model
        self.epochs = epochs
        self.criterion = criterion
        self.optim = optim(model.parameters(), lr)
        self.lr = lr
        self.stopping_epochs = stopping_epochs
        self.training_loss = []
        self.validation_loss = []
        self.record_loss = 9999
        self.epochs_since_loss_record = 0
        self.batches_per_epoch = batches_per_epoch
        self.model_saved_batch = 0

        if not torch.cuda.is_available():
            print("Cuda unavailable")

        self.device = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")
        self.model.to(self.device)


    def train(self, trainloader, validloader):
        if self.batches_per_epoch == None:
            self.batches_per_epoch = len(trainloader)

        for epoch in range(self.epochs):
            trainiter = iter(trainloader)
            for i in range(len(trainloader) // self.batches_per_epoch):
                self.model.train()
                epoch_train_loss = self.train_epoch(trainiter)
                self.training_loss.append(epoch_train_loss)

                self.model.eval()
                epoch_valid_loss = self.evaluate_model(validloader)
                self.validation_loss.append(epoch_valid_loss)

                n = epoch*self.batches_per_epoch + i
                stop, saved = self.early_stopping(epoch_valid_loss, n)

                if n > 0:
                    self.plot_losses(epoch, n) # This line clears the display
                print(f"Epoch train loss: {epoch_train_loss}")
                print(f"Epoch valid loss: {epoch_valid_loss}")

                if saved:
                    print("New valid loss record, model saved!")
                if stop:
                    print("Early stopped at epoch", epoch)
                    break
            if stop:
                break

        # Retrieve best model weights
        state_dict = torch.load('best.pth')
        self.model.load_state_dict(state_dict)

    def plot_losses(self, epoch, n):
        x = np.arange(n+1) / self.batches_per_epoch
        plt.clf()
        plt.title(f"Loss up to epoch: {epoch+1}/{self.epochs}")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.plot(x, self.training_loss, label='Training Loss')
        plt.plot(x, self.validation_loss, label='Validation Loss')
        plt.axvline(self.model_saved_batch / self.batches_per_epoch,
                    ls=':', lw=0.5, color='grey', label = 'Saved model')
        plt.legend()
        display.clear_output(wait=True)
        display.display(plt.gcf())

    def train_epoch(self, trainiter):
        running_loss = 0.
        for _ in range(self.batches_per_epoch):
            x, y = next(trainiter)
            x, y = x.to(self.device), y.to(self.device)
            self.optim.zero_grad()
            out = self.model.forward(x.view(x.shape[0], -1))
            loss = self.criterion(out, y)
            running_loss += loss.item()
            loss.backward()
            self.optim.step()
        return running_loss / self.batches_per_epoch

    def evaluate_model(self, validloader):
        epoch_valid_loss = 0.

        with torch.no_grad():
            for x, y in validloader:
                x, y = x.to(self.device), y.to(self.device)
                out = self.model.forward(x.view(x.shape[0], -1))
                loss = self.criterion(out, y)
                epoch_valid_loss += loss.item()
        return epoch_valid_loss / len(validloader)

    def early_stopping(self, valid_loss, n):
          saved = False
          if valid_loss < self.record_loss:
              self.record_loss = valid_loss
              torch.save(self.model.state_dict(), 'best.pth')
              self.model_saved_batch = n
              self.epochs_since_loss_record = 0
              saved = True
          else:
              self.epochs_since_loss_record += 1

          if self.epochs_since_loss_record > self.stopping_epochs:
              return True, saved
          return False, saved