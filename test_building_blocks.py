import numpy as np
from building_blocks.representation import TowerRepresentation
from building_blocks.conv_lstm import Conv2dLSTMCell
from building_blocks.gqn import GenerativeQueryNetwork

a = TowerRepresentation(128, v_dim=7, r_dim=256)
b = Conv2dLSTMCell(256,256)
model = GenerativeQueryNetwork(x_dim=3, v_dim=7, r_dim=256, h_dim=128, z_dim=64, L=8)
model2 = GenerativeQueryNetwork(x_dim=3, v_dim=7, r_dim=256, h_dim=128, z_dim=64, L=12)

unshare = GenerativeQueryNetwork(x_dim=3, v_dim=7, r_dim=256, h_dim=128, z_dim=64, L=8, share=False)
unshare2 = GenerativeQueryNetwork(x_dim=3, v_dim=7, r_dim=256, h_dim=128, z_dim=64, L=12, share=False)

def numberParameters(model):
  model_parameters = filter(lambda p: p.requires_grad, model.parameters())
  params = sum([np.prod(p.size()) for p in model_parameters])
  return params

print(numberParameters(a))
print(numberParameters(b))

# these have the same number of parameters because share=True by default
print(numberParameters(model))
print(numberParameters(model2))

# these are like... 10x as big!
print(numberParameters(unshare))
print(numberParameters(unshare2))

print(model2)
print('Success!')