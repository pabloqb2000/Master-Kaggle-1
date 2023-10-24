import numpy as np
from IPython import display
import matplotlib.pyplot as plt
import torch
from time import time

class Trainer:
    def __init__(self, model, epochs, criterion, optim, lr, stopping_batches,
                 batch_logging_freq = None, use_cuda = True, metrics = {}):
        self.model = model
        self.epochs = epochs
        self.criterion = criterion
        self.optim = optim(model.parameters(), lr)
        self.lr = lr
        self.stopping_batches = stopping_batches 
        self.training_loss = []
        self.validation_loss = []
        self.metrics = metrics
        self.metric_values = {m: [] for m in metrics}
        self.record_loss = 9999
        self.epochs_since_loss_record = 0
        self.batch_logging_freq = batch_logging_freq
        self.model_saved_batch = 0

        if not torch.cuda.is_available():
            print("Cuda unavailable")

        self.device = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")
        self.model.to(self.device)


    def train(self, trainloader, validloader):
        if self.batch_logging_freq == None:
            self.batch_logging_freq = len(trainloader)

        t0 = t = time()
        try:
            for epoch in range(self.epochs): # Epoch loop
                trainiter = iter(trainloader)
                k = len(trainloader) // self.batch_logging_freq
                for i in range(k): # Logging loop
                    self.model.train()
                    epoch_train_loss = self.train_loop(trainiter)
                    self.training_loss.append(epoch_train_loss)

                    self.model.eval()
                    epoch_valid_loss = self.evaluate_model(validloader)
                    self.validation_loss.append(epoch_valid_loss)

                    n = epoch*k + i
                    stop, saved = self.early_stopping(epoch_valid_loss, n)

                    if n > 0:
                        # This line clears the display, all printing should go after this
                        self.plot_losses(epoch, n, k)
                    print(f"Epoch train loss: {epoch_train_loss}")
                    print(f"Epoch valid loss: {epoch_valid_loss}")
                    print(f"{self.batch_logging_freq} batches train time: {time() - t} s")
                    t = time()

                    if saved:
                        print("New valid loss record, model saved!")
                    if stop:
                        display.clear_output(wait=True)
                        print("Early stopped at epoch,", epoch)
                        break
                if stop:
                    break
        except KeyboardInterrupt:
            display.clear_output(wait=True)
            print("Training interrupted at epoch:", epoch)
            stop = True
        finally:
            if not stop:
                display.clear_output(wait=True)

            # Retrieve best model weights
            state_dict = torch.load('best.pth')
            self.model.load_state_dict(state_dict)

            plt.show()
            print("Loaded best model weights!")
            print(f"Total training time: {time() - t0} s")
            print(f"Final train loss: {epoch_train_loss}")
            print(f"Final valid loss: {epoch_valid_loss}")

    def plot_losses(self, epoch, n, k):
        t = np.arange(n+1) / k
        plt.close('all')
        fig, (ax0, *axs) = plt.subplots(1, len(self.metrics)+1, figsize = (18, 6))
        ax0.set_title(f"Loss up to epoch: {epoch+1}/{self.epochs}")
        ax0.set_xlabel("Epoch")
        ax0.set_ylabel("Loss")
        ax0.plot(t, self.training_loss, label='Training Loss')
        ax0.plot(t, self.validation_loss, label='Validation Loss')
        ax0.axvline(self.model_saved_batch / k,
                    ls=':', lw=0.5, color='grey', label='Saved model')
        ax0.legend()

        for ax, m in zip(axs, self.metrics):
            ax.set_title(m)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.plot(t, self.metric_values[m], label='m')
            ax.axvline(self.model_saved_batch / k,
                    ls=':', lw=0.5, color='grey', label='Saved model')

        display.clear_output(wait=True)
        display.display(plt.gcf())

    def train_loop(self, trainiter):
        running_loss = 0.
        for _ in range(self.batch_logging_freq):
            x, y = next(trainiter)
            x, y = x.to(self.device), y.to(self.device)
            self.optim.zero_grad()
            out = self.model.forward(x.view(x.shape[0], -1))
            loss = self.criterion(out, y.view(y.shape[0], -1))
            running_loss += loss.item()
            loss.backward()
            self.optim.step()
        return running_loss / self.batch_logging_freq

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

          if self.epochs_since_loss_record > self.stopping_batches:
              return True, saved
          return False, saved

    def evaluate_model(self, validloader):
        epoch_valid_loss = 0.

        with torch.no_grad():
            for x, y in validloader:
                x, y = x.to(self.device), y.to(self.device)
                out = self.model.forward(x.view(x.shape[0], -1))
                loss = self.criterion(out, y.view(y.shape[0], -1))
                epoch_valid_loss += loss.item()
                for metric in self.metrics:
                    self.metric_values[metric].append(self.metrics[metric](y, out))
        return epoch_valid_loss / len(validloader)
