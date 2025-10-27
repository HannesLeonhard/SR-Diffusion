import torch
import torchvision
import numpy as np
import functools

from globals import device, sigma


def marginal_prob_std(t, sigma):
    """Compute the mean and standard deviation of $p_{0t}(x(t) | x(0))$.

    Args:
        t: A vector of time steps.
        sigma: The $\sigma$ in our SDE.

    Returns:
        The standard deviation.
    """
    global device
    t = t.clone().detach()
    return torch.sqrt((sigma ** (2 * t) - 1) / 2.0 / np.log(sigma))


def diffusion_coeff(t, sigma):
    """Compute the diffusion coefficient of our SDE.

    Args:
      t: A vector of time steps.
      sigma: The $\sigma$ in our SDE.

    Returns:
      The vector of diffusion coefficients.
    """
    global device
    return torch.tensor(sigma**t, device=device, dtype=torch.float32)


marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=sigma)
diffusion_coeff_fn = functools.partial(diffusion_coeff, sigma=sigma)


def loss_fn(model, x, marginal_prob_std, eps=1e-5, debug=False):
    """The loss function for training score-based generative models.

    Args:
      model: A PyTorch model instance that represents a
        time-dependent score-based model.
      x: A mini-batch of training data.
      marginal_prob_std: A function that gives the standard deviation of
        the perturbation kernel.
      eps: A tolerance value for numerical stability.
      debug: Provides extra debug context if set to True
    """
    global device
    rand_t = torch.rand(x.shape[0], device=device) * (1.0 - eps) + eps
    z = torch.randn_like(x)
    std = marginal_prob_std(rand_t)
    perturbed_x = x + z * std[:, None, None, None]
    score = model(perturbed_x, rand_t)
    loss = torch.mean(
        torch.sum((score * std[:, None, None, None] + z) ** 2, dim=(1, 2, 3))
    )
    if debug:
        return loss, rand_t, z, std, perturbed_x, score
    return loss


def loss_fn_sr(model, x, marginal_prob_std, eps=1e-5, debug=False):
    """The loss function for training score-based generative models.

    Args:
      model: A PyTorch model instance that represents a
        time-dependent score-based model.
      x: A mini-batch of training data.
      marginal_prob_std: A function that gives the standard deviation of
        the perturbation kernel.
      eps: A tolerance value for numerical stability.
      debug: Provides extra debug context if set to True
    """
    global device
    rand_t = torch.rand(x.shape[0], device=device) * (1.0 - eps) + eps
    z = torch.randn_like(x)
    std = marginal_prob_std(rand_t)
    perturbed_x = x + z * std[:, None, None, None]
    downsampled_x = torchvision.transforms.functional.adjust_sharpness(x, 0.01)
    conditioned_x = torch.cat([perturbed_x, downsampled_x], dim=1)
    score = model(conditioned_x, rand_t)
    loss = torch.mean(
        torch.sum((score * std[:, None, None, None] + z) ** 2, dim=(1, 2, 3))
    )
    if debug:
        return loss, rand_t, z, std, conditioned_x, score
    return loss
