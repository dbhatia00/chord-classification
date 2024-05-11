import typer
import torch
from preprocess import load_samples_from_file
from audio_model import AudioModel
from tqdm import tqdm
from typing import Optional
import numpy as np
import os
from util import SAMPLE_FREQ, WINDOW_SIZE

def main(model: Optional[str] = typer.Option('model.pt'), 
         filepath: Optional[str] = typer.Option('file.wav'), 
         raw_out: Optional[str] = typer.Option('results/output.txt'),
         notes_out: Optional[str] = typer.Option('results/notes.txt')):
  audio_model = AudioModel()
  audio_model.load_state_dict(torch.load(model))
  audio_model.eval()

  samples = load_samples_from_file(filepath)
  samples = torch.tensor(samples).type(torch.float32)
  preds = []
  for i in tqdm(range(samples.shape[0])):
    pred = audio_model(samples[i])
    preds.append(pred.detach().numpy())
  preds = np.vstack(preds)

  notes = []
  for p in preds:
    note = np.where(p >= 0.3)
    notes.append(note[0].tolist())

  os.makedirs(os.path.dirname(raw_out), exist_ok=True)
  open(raw_out, 'w').close()
  with open(raw_out, 'a') as file:
    for i, pred in enumerate(preds):
      time = (i + WINDOW_SIZE//2) / SAMPLE_FREQ
      file.write(f'{time}: {pred.tolist()}\n')

  os.makedirs(os.path.dirname(notes_out), exist_ok=True)
  open(notes_out, 'w').close()
  with open(notes_out, 'a') as file:
    for i, note in enumerate(notes):
      time = (i + WINDOW_SIZE//2) / SAMPLE_FREQ
      file.write(f'{time}: {note}\n')


if __name__ == '__main__':
  typer.run(main)
