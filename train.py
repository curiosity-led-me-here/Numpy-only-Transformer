import numpy as np
import torch
from torch_model import torch_GPT

# Hyperparams
vocab_size = 10
dim = 64
head_size = 4
learning_rate = 0.01
epochs = 250

# model init
model = torch_GPT(
        dim=dim, 
        head_size=head_size, 
        vocab_size=vocab_size, 
        X=X_data, 
        Y_in=Y_in_data, 
        targets=targets_data, 
        lr=learning_rate
    )

#training loop
def train(model, epochs):
    for epoch in range(epochs):
        predictions = model.full_forward()
        loss = model.backward()
        model.learn()
        if epoch % 20 == 0:
            print(f"Epoch {epoch:03d} | Loss: {loss:.4f}")
    final_predictions = torch.argmax(predictions, dim=-1)
    print(f"Targets:     {targets_data}")
    print(f"Predictions: {final_predictions.tolist()}")

train(model=model, epochs=100)