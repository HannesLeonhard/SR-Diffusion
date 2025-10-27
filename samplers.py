from scipy import integrate
import torch
import numpy as np

from globals import device


def ode_sampler(
    score_model,
    marginal_prob_std,
    diffusion_coeff,
    batch_size=64,
    atol=1e-5,
    rtol=1e-5,
    z=None,
    conditioning_x=None,
    eps=1e-3,
):
    """Generate samples from score-based models with black-box ODE solvers.

    Args:
      score_model: A PyTorch model that represents the time-dependent score-based model.
      marginal_prob_std: A function that returns the standard deviation
        of the perturbation kernel.
      diffusion_coeff: A function that returns the diffusion coefficient of the SDE.
      batch_size: The number of samplers to generate by calling this function once.
      atol: Tolerance of absolute errors.
      rtol: Tolerance of relative errors.
      device: 'cuda' for running on GPUs, and 'cpu' for running on CPUs.
      z: The latent code that governs the final sample. If None, we start from p_1;
        otherwise, we start from the given z.
      eps: The smallest time step for numerical stability.
    """
    global device
    t = torch.ones(batch_size, device=device)
    # Create the latent code
    if z is None:
        init_x = (
            torch.randn(batch_size, 1, 28, 28, device=device)
            * marginal_prob_std(t)[:, None, None, None]
        )
    else:
        init_x = z

# Keep conditioning separate from the ODE state. The ODE evolves only the
# latent image tensor (e.g. shape (B,1,28,28)). The conditioning (e.g. a low-
# resolution or guide image) is passed to the score model during evaluation
# but should NOT be part of the ODE state vector.
    if conditioning_x is not None:
        # Ensure conditioning is on the same device; keep a reference named cond_x
        cond_x = conditioning_x.to(device=device)
    else:
        cond_x = None

    shape = init_x.shape

    def score_eval_wrapper(sample, time_steps):
        """A wrapper of the score-based model for use by the ODE solver."""
        sample = torch.tensor(sample, device=device, dtype=torch.float32).reshape(shape)
        time_steps = torch.tensor(
            time_steps, device=device, dtype=torch.float32
        ).reshape((sample.shape[0],))
        # If there is conditioning, concatenate it to the sample here for the
        # model input, but do NOT change the ODE state shape itself.
        with torch.no_grad():
            if cond_x is not None:
                # cond_x should already have shape (B, C_cond, H, W) and be on device
                model_input = torch.cat([sample, cond_x], dim=1)
            else:
                model_input = sample
            score = score_model(model_input, time_steps)
        # score is expected to have the same shape as `sample` (gradients w.r.t.
        # the latent image only). Flatten and return as numpy array for solver.
        return score.cpu().numpy().reshape((-1,)).astype(np.float32)

    def ode_func(t, x):
        """The ODE function for use by the ODE solver."""
        time_steps = np.ones((shape[0],)) * t
        g = diffusion_coeff(torch.tensor(t, dtype=torch.float32)).cpu().numpy()
        return -0.5 * (g**2) * score_eval_wrapper(x, time_steps)

    # Run the black-box ODE solver.
    # Sanity check: ensure the model returns a flattened vector the same size as the
    # flattened ODE state. This gives a clear error when shapes mismatch instead
    # of the lower-level broadcasting error inside SciPy.
    try:
        f0 = score_eval_wrapper(init_x.reshape(-1).cpu().numpy(), np.ones((shape[0],)) * 1.0)
        y0_flat = init_x.reshape(-1).cpu().numpy()
        if f0.shape != y0_flat.shape:
            raise ValueError(
                f"Score model output flattened shape {f0.shape} does not match ODE state flattened shape {y0_flat.shape}. "
                "If you pass conditioning via `conditioning_x`, it should not be concatenated into the ODE state. "
                "Pass it using the `conditioning_x` argument so the sampler can concatenate it only for model evaluation."
            )
    except Exception:
        # Re-raise to surface helpful diagnostic to the user.
        raise
    res = integrate.solve_ivp(
        ode_func,
        (1.0, eps),
        init_x.reshape(-1).cpu().numpy(),
        rtol=rtol,
        atol=atol,
        method="RK45",
    )
    print(f"Number of function evaluations: {res.nfev}")
    x = torch.tensor(res.y[:, -1], device=device, dtype=torch.float32).reshape(shape)

    return x
