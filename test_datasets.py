
import matplotlib.pyplot as plt
from data.datasets import ShepardMetzler, Scene

"""If you run this script (and have downloaded all the files):

```
$$ python test_datasets.py
You have 15 images and 15 viewpoints
images     torch.Size([15, 3, 64, 64])
viewpoints         torch.Size([15, 5])
```
"""
def test_batch(dataset):
  images, viewpoints = dataset[0]
  print(f"You have {len(images)} images and {len(viewpoints)} viewpoints")
  print(f'images\t   {images.size()}')
  print(f'viewpoints\t   {viewpoints.size()}')

  plt.imshow(images[0].transpose(0, 2))
  plt.show()


if __name__ == '__main__':
  dataset = ShepardMetzler('./data/dummy', train=False)
  test_batch(dataset)

