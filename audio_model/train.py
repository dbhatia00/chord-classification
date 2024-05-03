import tensorflow as tf
import typer
from typing import Optional
from preprocess import get_data

def main(
    batch_size: Optional[int] = typer.Option(16),
    epochs: Optional[int] = typer.Option(10),
    lr: Optional[float] = typer.Option(2e-4)):

    print("getting data...")
    # Load data
    dataset = get_data(batch_size)

    print("Compiling model...")

    # Compile model
    model = tf.keras.Sequential([
      tf.keras.layers.Conv1D(50, 11, input_shape=(5513, 1)),
      tf.keras.layers.MaxPool1D(2),
      tf.keras.layers.Conv1D(50, 11),
      tf.keras.layers.MaxPool1D(2),
      tf.keras.layers.Conv1D(50, 11),
      tf.keras.layers.MaxPool1D(2),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(1000, activation='relu'),
      tf.keras.layers.Dense(500, activation='relu'),
      tf.keras.layers.Dense(49, activation='softmax'),
    ])

    model.compile(
       optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
       loss=tf.keras.losses.BinaryCrossentropy(),
       metrics=[
          tf.keras.metrics.BinaryAccuracy()
       ])

    # Train
    model.fit(dataset, epochs=epochs, shuffle=True)

if __name__ == '__main__':
  print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
  typer.run(main)
