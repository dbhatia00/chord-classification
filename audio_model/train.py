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
    epochs: Optional[int] = typer.Option(10),
    lr: Optional[float] = typer.Option(1e-6)):

    print("getting data...")
    # Load data
    train_loader, test_loader = get_data(batch_size, 0.1)

    audio_model = AudioModel().to(device)

    loss_func = nn.CrossEntropyLoss()
    acc = metrics.BinaryAccuracy()
    opt = optim.Adam(audio_model.parameters(), lr=lr)


    running_loss = 0.0
    for e in range(epochs):
      for i, data in enumerate(tqdm(train_loader)):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)

        # zero the parameter gradients
        opt.zero_grad()

        # forward + backward + optimize
        outputs = audio_model(inputs)
        loss = loss_func(outputs, labels)
        loss.backward()
        opt.step()

        # print statistics
        running_loss += loss.item()
        acc.update(outputs, labels)
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print(f'[{e + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0

    # Compile model
    # model = tf.keras.Sequential([
    #   tf.keras.layers.Conv1D(50, 11, input_shape=(5513, 1)),
    #   tf.keras.layers.MaxPool1D(2),
    #   tf.keras.layers.Conv1D(50, 11),
    #   tf.keras.layers.MaxPool1D(2),
    #   tf.keras.layers.Conv1D(50, 11),
    #   tf.keras.layers.MaxPool1D(2),
    #   tf.keras.layers.Flatten(),
    #   tf.keras.layers.Dense(1000, activation='relu'),
    #   tf.keras.layers.Dense(500, activation='relu'),
    #   tf.keras.layers.Dense(49, activation='softmax'),
    # ])

    # model.compile(
    #    optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
    #    loss=tf.keras.losses.BinaryCrossentropy(),
    #    metrics=[
    #       tf.keras.metrics.BinaryAccuracy()
    #    ])

    # # Train
    # model.fit(dataset, epochs=epochs, shuffle=True)

if __name__ == '__main__':
  typer.run(main)
