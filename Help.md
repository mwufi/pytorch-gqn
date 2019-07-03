
## Help

**During training**

If your computer breaks, you can pass it a `--resume` argument next time:
```
python run-gqn.py --data_parallel=True --batch_size=144 --workers=2 --resume=343000 (or whatever the last checkpoint number was)
```

You might be asking: What do the graphs mean? We only log ELBO and KL at the moment, but we should probably log more metrics! to get a better picture of how our neural net is doing
