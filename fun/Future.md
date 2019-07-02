
# Things to try

- Looked into exporting to tensorflow.js (through pytorch-to-keras, which just uses ONNX. But ONNX doesn't support LSTMs and custom ops, both of which we have here!! No hope, unless we're willing to do some engineering)

- What happens when you feed it an upside-down image?

- Replicate some of the information gain experiments!

- Add utilities to make it simpler to 1) run the network, 2) visualize it, 3) mess it up

- Add a notebook/movie showing rotation and navigation in the space