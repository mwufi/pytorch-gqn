# 🌈 pytorch-gqn

Do you want to learn unsupervised representations of 3D scenes? Boy do I have the kick for you 🕍

<img src="https://cdn.arstechnica.net/wp-content/uploads/2018/06/Screen-Shot-2018-06-28-at-4.18.25-PM-800x534.png">

Watch a video: https://vimeo.com/345774866

This is an implementation of [Neural Scene Representation and Rendering (Eslami et al 2018)](https://deepmind.com/blog/neural-scene-representation-and-rendering/). 

The authors are all pretty cool: S. M. Ali Eslami, Danilo Jimenez Rezende, Frederic Besse, Fabio Viola, Ari S. Morcos, Marta Garnelo

Summary:
If you haven't read the blog post already, you should! It's super cool, and will help you get a head start on what's here.

**And** before you commit to reading the rest of this document, please know that there exist other implementations of GQN out there! You've got choices!
- https://github.com/wohlert/generative-query-network-pytorch
- https://github.com/iShohei220/torch-gqn
- https://github.com/ogroth/tf-gqn (if you like Tensorflow)

# Faq: Or, A non-expert's guide to HAVING FUN with this repo ;)

This repository might look a bit intimidating, but I'm always here, and I'll try to make this README informative! 

Steps:
1. Getting the training data (to replicate the networks in the original paper)
2. Training your ~dragon~ *neural nets*
3. How to tell that it works (and exploring what it can do)

Steps 1 and 2 are optional (and more straightforward? Since the authors did all the hard work for us already by telling us how the net is built and so forth). I've trained a GQN for ~40k gradient steps (see `/notes` for more details), so there's checkpoints that you can use out of the box.

However, if you want to do it (just for fun):

**Step 1) Getting training data**

You should probably start by deciding what dataset to use. I used `shepard_metzler_5_parts`, but there are others: [here](https://github.com/deepmind/gqn-datasets). Personally I'd like to see one trained on `rooms_ring_camera_with_object_rotations`, but that's just me 😊

The scripts:
- `download_data.sh` can help you download the data with `gsutil`, and put the data in a directory
- `data/tf2torch.py` will convert the `.tfrecord` files to a _LOT_ of `.pt.gz` files, each one being one particular example. Other implementations (see [wohlert](https://github.com/wohlert/generative-query-network-pytorch/tree/master/scripts)) will do this differently (e.g. by making one big file instead of a lot of little ones)

Great! Now we have training data. When I was writing the various bits, I wrote the tests
- `test_datasets.py` will see if you have the ShepardMetzler dataset ready-to-go
- `test_building_blocks.py` will see if your neural net is OK
- `test_gqn.py` will run one step of training (load the data, load the model, etc). If this passes, you should be ready to scale to full training!

**Step 2) Running the training/evaluation loop**

```
python run-gqn.py --data_parallel=True --batch_size=144 --workers=2
```

This will save backups every 1000 steps, and log changes to [Tensorboard](https://github.com/mwufi/pytorch-gqn/blob/master/notes/Screen%20Shot%202019-07-01%20at%204.57.33%20PM.png).

Note to self (and everyone): Add some tips about logging/keeping track of stuff during training!

_fast forward a few hours, or days_: Awesome! we have a checkpoint, and the images on Tensorboard look pretty good. What now?

**Step 3) Analysing your results**

**What can a trained model do?**

Almost anything, you just have to be patient. Just kidding. You can:
- delete half the weights?
- watch the activations as an image passes through it
- compare the representations for different scenes
- make it rotate!!

Check out the `/fun` folder for my first attempts. Lots more to be done here, so please submit (many) a pull request with suggestions!

## Yes?

**If I _don't_ want to train my own model, can I grab the checkpoints that you have?**

Sure, right [here](https://console.cloud.google.com/storage/browser/transformer-results-bucket/pytorch-gqn/).

**Why do the images in `fun/Interactive.ipynb` suck?**

I only trained it for about 40k gradient steps (each one from a mini-batch of 36 scenes). I think the paper had like... 2 million gradient steps?

## Help

**During training**

If your computer breaks, you can pass it a `--resume` argument next time:
```
python run-gqn.py --data_parallel=True --batch_size=144 --workers=2 --resume=343000 (or whatever the last checkpoint number was)
```

You might be asking: What do the graphs mean? We only log ELBO and KL at the moment, but we should probably log more metrics! to get a better picture of how our neural net is doing
