
import random

def shapes(*args):
  for t in args:
    print(t.size())


def partition(images, viewpoints, log=False):
    """
    Partition batch into context and query sets.

    :param images:     torch.Size([B, M, 3, 64, 64])
    :param viewpoints: torch.Size([B, M, 5])

    :return: context images, context viewpoint, query image, query viewpoint
    """
    # Maximum number of context points to use
    b, m, *x_dims = images.shape
    b, m, *v_dims = viewpoints.shape

    # "Squeeze" the batch dimension
    images = images.view((-1, m, *x_dims))
    viewpoints = viewpoints.view((-1, m, *v_dims))

    # Sample random number of views
    n_context = random.randint(2, m - 1)

    indices = random.sample([i for i in range(m)], n_context)

    # Partition into context and query sets
    context_idx, query_idx = indices[:-1], indices[-1]

    x, v = images[:, context_idx], viewpoints[:, context_idx]
    x_q, v_q = images[:, query_idx], viewpoints[:, query_idx]

    if log:
      print(f'# of context views: {n_context - 1}')
      shapes(x,v,x_q,v_q)

    return x, v, x_q, v_q