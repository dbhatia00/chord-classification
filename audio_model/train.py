import typer
from typing import Optional
from preprocess import get_data
from audio_model import AudioModel
import torch
import torch.nn as nn
import torch.optim as optim
import torcheval.metrics as metrics
from tqdm import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def main(
    batch_size: Optional[int] = typer.Option(16),
    epochs: Optional[int] = typer.Option(100),
    lr: Optional[float] = typer.Option(4e-4)):

    print("getting data...")
    # Load data
    train_loader, test_loader = get_data(batch_size, 0.1)

    audio_model = AudioModel().to(device)
    total_params = sum(p.numel() for p in audio_model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total_params}")
    print("Training...")

    loss_func = nn.BCELoss()
    acc = metrics.MultilabelAccuracy()
    # opt = optim.Adam(audio_model.parameters(), lr=lr)
    opt = optim.SGD(audio_model.parameters(), lr=lr, momentum=0.9)

    running_loss = 0.0
    x_total = 0
    for e in range(epochs):
      print(f"Epoch {e}:")
      audio_model.train()
      for i, data in enumerate(tqdm(train_loader)):
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)

        opt.zero_grad()

        outputs = audio_model(inputs)
        loss = loss_func(outputs, labels)
        loss.backward()
        opt.step()

        running_loss += loss.item() * inputs.shape[0]
        x_total += inputs.shape[0]
        acc.update(outputs, labels)

      print(f"Train loss: {running_loss / x_total}")
      print(f"Train accuracy: {acc.compute()}")
      running_loss = 0.0
      x_total = 0
      acc.reset()
      
      audio_model.eval()
      for i, data in enumerate(test_loader):
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = audio_model(inputs)
        loss = loss_func(outputs, labels)
        acc.update(outputs, labels)

      print(f"Test accuracy: {acc.compute()}")
      acc.reset()

if __name__ == '__main__':
  typer.run(main)
