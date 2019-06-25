
## Getting the dataset

Me: Let's get the simple shepard_metzler dataset here
```
mkdir data
./download-data.sh data
```

## Installing stuff

Computer: Tensorflow is not installed!

Me: OK, let's do it
```
pip install --user tensorflow
```

Computer:
```
The scripts freeze_graph, saved_model_cli, tensorboard, tf_upgrade_v2, tflite_convert, toco and toco_from_protos are installed in '/home/zentang/.local/bin' which is not on PATH.
  Consider adding this directory to PATH or, if you prefer to suppress this warning, use --no-warn-script-location.
```

Me: OK, no problem
```
echo "export PATH=$PATH:/home/zentang/.local/bin" >> .bashrc
source .bashrc
```

## Reading the data

1. Run `tf2torch.py`


