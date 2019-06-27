"""Example for loading data
"""
import os, io
import gzip
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor, Resize

import collections
Scene = collections.namedtuple('Scene', ['frames', 'cameras'])


class ShepardMetzler(Dataset):

  def __init__(self, root_dir,
               train=True,
               transform=None, target_transform=None):
    prefix = "train" if train else "test"
    self.root_dir = os.path.join(root_dir, prefix)

    if not os.path.exists(self.root_dir):
      print(f'{self.root_dir} does not exist!')
      return

    self.records = sorted([p for p in os.listdir(self.root_dir) if "pt" in p])
    self.transform = transform
    self.target_transform = target_transform


  def __len__(self):
    return len(self.records)


  def __getitem__(self, idx):
    scene_path = os.path.join(self.root_dir, self.records[idx])
    with gzip.open(scene_path, "r") as f:
      data = torch.load(f)

      byte_to_tensor = lambda x: ToTensor()(Resize(64)((Image.open(io.BytesIO(x)))))
      images = torch.stack([byte_to_tensor(frame) for frame in data.frames])

      viewpoints = torch.from_numpy(data.cameras)
      viewpoints = viewpoints.view(-1, 5)

      if self.transform:
          images = self.transform(images)

      if self.target_transform:
          viewpoints = self.target_transform(viewpoints)

      return images, viewpoints


"""Provide a default target_transform
"""
def transform_viewpoint(v):
  w, z = torch.split(v, 3, dim=-1)
  y, p = torch.split(z, 1, dim=-1)

  # position, [yaw, pitch]
  view_vector = [w, torch.cos(y), torch.sin(y), torch.cos(p), torch.sin(p)]
  v_hat = torch.cat(view_vector, dim=-1)

  return v_hat

