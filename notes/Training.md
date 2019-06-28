
# How to best utilize GPU?


**1 GPU, 2 CPU**
- —batch_size=16 —num_workers=2 —data_parallel=True
- Epoch [1/200]: [174/50569]   0%| , elbo=-1.99e+04, kl=4.42e+01, mu=5.00e-04, sigma=2.00e+00 [06:44<34:09:58]

**4 GPU, 4 CPU**

- —batch_size=64 —num_workers=4 —data_parallel=True
- Epoch [1/200]: [147/12643]   1%| , elbo=-2.00e+04, kl=6.63e+01, mu=5.00e-04, sigma=2.00e+00 [06:09<8:36:21]

- —batch_size=192 —num_workers=16 —data_parallel=True
- Epoch [1/200]: [45/4215]   1%| , elbo=-2.05e+04, kl=3.83e+02, mu=5.00e-04, sigma=2.00e+00 [05:25<8:30:32]


# Run 1

VM:
- 4 K80 GPUs
- 4 CPUs
- 15GB memory

```bash
python run-gqn.py --data_parallel=True --batch_size=192 --workers=16
```

This gives the following:
![CPU/memory utilization](https://raw.githubusercontent.com/mwufi/pytorch-gqn/master/notes/run1/Screen%20Shot%202019-06-28%20at%201.58.52%20PM.png)
![GPU utilization](https://raw.githubusercontent.com/mwufi/pytorch-gqn/master/notes/run1/Screen%20Shot%202019-06-28%20at%202.10.28%20PM.png)
![Training](https://raw.githubusercontent.com/mwufi/pytorch-gqn/master/notes/run1/Screen%20Shot%202019-06-28%20at%201.58.58%20PM.png)

I'm going to let it train for a while and watch on Tensorboard!
