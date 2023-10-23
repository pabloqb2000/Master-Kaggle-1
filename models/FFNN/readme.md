FFNN over the TFIDF

Parameters:

```python
n_epochs = 500
batch_logging_freq = 1

model = Model([x_dim, 64, n_classes], dropout=0.5)

trainer = train.Trainer(model, epochs=n_epochs, criterion=nn.CrossEntropyLoss(),
                  optim=torch.optim.Adam, lr=1e-3, stopping_batches=10,
                  batch_logging_freq = batch_logging_freq, use_cuda=False)
trainer.train(train_loader, valid_loader)
```