
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
- 4 K80 GPUs (1.87 TFLOPS)
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

**8 hours later**

It's just training, not too much is happening! I think our batch size was too large (the authors use a batch size of 36?) It could be that each GPU can run a gradient step independently, and then all-reduce the gradients? In that case, we should set the batch size to be 36*6, or 144.

I want to make more use of the memory on the GPU. Also I want to make more use of the CPUs. Also, download more RAM next time ;)

Notes on DataParallel:
- OK, so is it 36/4, 36, or 36*4? [same question as mine](https://discuss.pytorch.org/t/dataparallel-results-in-a-different-network-compared-to-a-single-gpu-run/28635?u=ptrblck)
- Our DataParallel algorithm: [comments from Pytorch dev, Facebook AI Research](https://discuss.pytorch.org/t/debugging-dataparallel-no-speedup-and-uneven-memory-allocation/1100/13)
- Following the second comment, he gives a way to use DataParallel on a *portion* of your network (ie, avoid the large linear blocks at the end)[here](https://discuss.pytorch.org/t/are-there-reasons-why-dataparallel-was-used-differently-on-alexnet-and-vgg-in-the-imagenet-example/19844)

# Run 2

Data: same as before, since we changed the logging/eval to iterations!

VM:
- 4 K80 GPUs (1.87 TFLOPS)
- 4 CPUs
- 20GB memory

```bash
python run-gqn.py --data_parallel=True --batch_size=144 --workers=2
```


