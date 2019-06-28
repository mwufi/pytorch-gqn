import torch
from torch.utils.data import DataLoader
from torch.distributions import Normal

from building_blocks.gqn import GenerativeQueryNetwork
from data.datasets import ShepardMetzler, Scene
from building_blocks.training import partition

"""
--------------------------- Data Pipeline Test ----------------------------

Summary:
- Load the model, dummy dataset, optimizers, and compute all the terms

"""

# Get GPU
cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if cuda else "cpu")


dataset = ShepardMetzler('data/dummy')
model = GenerativeQueryNetwork(x_dim=3, v_dim=7, r_dim=256, h_dim=128, z_dim=64, L=8, share=True).to(device)
train_loader = DataLoader(dataset, batch_size = 2, shuffle=True)

optimizer = torch.optim.Adam(model.parameters(), lr=5 * 10 ** (-5))
model.train()

def step(batch):
  x,v = batch
  x,v = x.to(device), v.to(device)
  x,v, x_q, v_q = partition(x,v, log=True)

  print('Forward pass...')
  x_mu, r, kl = model(x, v, x_q, v_q)

  print("x_mu\t", x_mu.size())
  print("r\t", r.size())
  print("KL\t", kl.size())

  # Log likelihood
  sigma = 0.01
  ll = Normal(x_mu, sigma).log_prob(x_q)

  likelihood = torch.mean(torch.sum(ll, dim=[1, 2, 3]))
  kl_divergence = torch.mean(torch.sum(kl, dim=[1, 2, 3]))

  # Evidence lower bound
  elbo = likelihood - kl_divergence
  loss = -elbo
  print(f"loss {loss}")

  print("Backward pass...")
  loss.backward()

  optimizer.step()
  optimizer.zero_grad()


train_iter = iter(train_loader)
for i in range(3):
  try:
    batch = next(train_iter)
  except StopIteration:
    train_iter = iter(train_loader)
    batch = next(train_iter)
  step(batch)
print('Test passed!')
