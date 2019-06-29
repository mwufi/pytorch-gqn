"""
run-gqn.py

Script to train the a GQN on the Shepard-Metzler dataset
in accordance to the hyperparameter settings described in
the supplementary materials of the paper.
"""
import random
import math
from argparse import ArgumentParser

# Torch
import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

# TensorboardX
from tensorboardX import SummaryWriter

# Ignite
from ignite.contrib.handlers import ProgressBar
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint, Timer
from ignite.metrics import RunningAverage

from building_blocks.gqn import GenerativeQueryNetwork
from building_blocks.annealer import Annealer
from building_blocks.training import partition
from data.datasets import ShepardMetzler, Scene

cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if cuda else "cpu")

# Random seeding
random.seed(99)
torch.manual_seed(99)
if cuda: torch.cuda.manual_seed(99)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if __name__ == '__main__':
    parser = ArgumentParser(description='Generative Query Network on Shepard Metzler Example')
    parser.add_argument('--n_epochs', type=int, default=200, help='number of epochs run (default: 200)')
    parser.add_argument('--batch_size', type=int, default=36, help='multiple of batch size (default: 1)')
    parser.add_argument('--data_dir', type=str, help='location of data', default="data/shepard_metzler_5_parts-torch")
    parser.add_argument('--log_dir', type=str, help='location of logging', default="log")
    parser.add_argument('--checkpoint_dir', type=str, help='location of checkpoints', default="checkpoints")
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
    parser.add_argument('--data_parallel', type=bool, help='whether to parallelise based on data (default: False)', default=False)
    parser.add_argument('--max_n', type=int, help="maximum number of examples in dataset (default: -1, ie all)", default=-1)
    parser.add_argument('--eval_n', type=int, help="evaluate every n iterations", default=1000)
    args = parser.parse_args()

    # Create model and optimizer
    model = GenerativeQueryNetwork(x_dim=3, v_dim=7, r_dim=256, h_dim=128, z_dim=64, L=8).to(device)
    model = nn.DataParallel(model) if args.data_parallel else model
    # TODO: log the throughput improvements if we use --data_parallel=True

    optimizer = torch.optim.Adam(model.parameters(), lr=5 * 10 ** (-5))

    # Rate annealing schemes
    sigma_scheme = Annealer(2.0, 0.7, 2e5)
    mu_scheme = Annealer(5 * 10 ** (-4), 5 * 10 ** (-5), 1.6e6)

    # Load the dataset
    kwargs = {'max_n': args.max_n}
    train_dataset = ShepardMetzler(root_dir=args.data_dir, **kwargs)
    valid_dataset = ShepardMetzler(root_dir=args.data_dir, train=False, **kwargs)

    kwargs = {'num_workers': args.workers, 'pin_memory': True} if cuda else {}
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)

    def step(engine, batch):
        model.train()

        x, v = batch
        x, v = x.to(device), v.to(device)
        x, v, x_q, v_q = partition(x, v)

        # Reconstruction, representation and divergence
        x_mu, _, kl = model(x, v, x_q, v_q)

        # Log likelihood
        sigma = next(sigma_scheme)
        ll = Normal(x_mu, sigma).log_prob(x_q)

        likelihood     = torch.mean(torch.sum(ll, dim=[1, 2, 3]))
        kl_divergence  = torch.mean(torch.sum(kl, dim=[1, 2, 3]))

        # Evidence lower bound
        elbo = likelihood - kl_divergence
        loss = -elbo
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        with torch.no_grad():
            # Anneal learning rate
            mu = next(mu_scheme)
            i = engine.state.iteration
            for group in optimizer.param_groups:
                group["lr"] = mu * math.sqrt(1 - 0.999 ** i) / (1 - 0.9 ** i)

        return {"elbo": elbo.item(), "kl": kl_divergence.item(), "sigma": sigma, "mu": mu }

    # Trainer and metrics
    trainer = Engine(step)
    metric_names = ["elbo", "kl", "sigma", "mu"]
    RunningAverage(output_transform=lambda x: x["elbo"]).attach(trainer, "elbo")
    RunningAverage(output_transform=lambda x: x["kl"]).attach(trainer, "kl")
    RunningAverage(output_transform=lambda x: x["sigma"]).attach(trainer, "sigma")
    RunningAverage(output_transform=lambda x: x["mu"]).attach(trainer, "mu")
    ProgressBar().attach(trainer, metric_names=metric_names)

    # Model checkpointing
    checkpoint_handler = ModelCheckpoint(args.checkpoint_dir, "checkpoint", save_interval=5000, n_saved=2, require_empty=False)
    trainer.add_event_handler(event_name=Events.ITERATION_COMPLETED, handler=checkpoint_handler,
                              to_save={'model': model, 'optimizer': optimizer})

    timer = Timer(average=True).attach(trainer, start=Events.EPOCH_STARTED, resume=Events.ITERATION_STARTED,
                 pause=Events.ITERATION_COMPLETED, step=Events.ITERATION_COMPLETED)

    # Tensorbard writer
    writer = SummaryWriter(logdir=args.log_dir)


    @trainer.on(Events.ITERATION_COMPLETED)
    def log_metrics(engine):
        for key, value in engine.state.metrics.items():
            writer.add_scalar("training/{}".format(key), value, engine.state.iteration)

    @trainer.on(Events.ITERATION_COMPLETED)
    def save_images(engine):
        if engine.state.iteration % args.eval_n == 0:
            with torch.no_grad():
                x, v = engine.state.batch
                x, v = x.to(device), v.to(device)
                x, v, x_q, v_q = partition(x, v)

                x_mu, r, _ = model(x, v, x_q, v_q)

                r = r.view(-1, 1, 16, 16)

                # Send to CPU
                x_mu = x_mu.detach().cpu().float()
                r = r.detach().cpu().float()

                writer.add_image("representation", make_grid(r), engine.state.iteration)
                writer.add_image("reconstruction", make_grid(x_mu), engine.state.iteration)

    @trainer.on(Events.ITERATION_COMPLETED)
    def validate(engine):
        if engine.state.iteration % args.eval_n == 0:
            model.eval()
            with torch.no_grad():
                x, v = next(iter(valid_loader))
                x, v = x.to(device), v.to(device)
                x, v, x_q, v_q = partition(x, v)

                # Reconstruction, representation and divergence
                x_mu, _, kl = model(x, v, x_q, v_q)

                # Validate at last sigma
                ll = Normal(x_mu, sigma_scheme.recent).log_prob(x_q)

                likelihood = torch.mean(torch.sum(ll, dim=[1, 2, 3]))
                kl_divergence = torch.mean(torch.sum(kl, dim=[1, 2, 3]))

                # Evidence lower bound
                elbo = likelihood - kl_divergence

                writer.add_scalar("validation/elbo", elbo.item(), engine.state.iteration)
                writer.add_scalar("validation/kl", kl_divergence.item(), engine.state.iteration)

    @trainer.on(Events.EXCEPTION_RAISED)
    def handle_exception(engine, e):
        writer.close()
        engine.terminate()
        if isinstance(e, KeyboardInterrupt) and (engine.state.iteration > 1):
            import warnings
            warnings.warn('KeyboardInterrupt caught. Exiting gracefully.')
            checkpoint_handler(engine, { 'model_exception': model })
        else: raise e

    trainer.run(train_loader, args.n_epochs)
    writer.close()
