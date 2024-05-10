import typer
import torch
from preprocess import load_samples_from_file
from audio_model import AudioModel
from tqdm import tqdm
from typing import Optional
import numpy as np

def main(model: Optional[str] = typer.Option('model.pt'), 
         filepath: Optional[str] = typer.Option('file.wav'), 
         out: Optional[str] = typer.Option('output.txt')):
  audio_model = AudioModel()
  audio_model.load_state_dict(torch.load(model))
  audio_model.eval()

  samples = load_samples_from_file(filepath)
  preds = []
  for i in tqdm(range(samples.shape[0])):
    tensor = torch.tensor(samples).type(torch.float32)
    pred = audio_model(tensor)
    preds.append(pred.detach().numpy())
  preds = np.vstack(preds)

  # with open(out, 'w') as file:
  #   file.write(str(preds))
  np.savetxt(out, preds)


if __name__ == '__main__':
  typer.run(main)
