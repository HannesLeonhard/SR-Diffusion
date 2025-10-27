import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Dense(nn.Module):
    """
    Fully connected layer that reshapes outputs to feature maps
    """

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.layer = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.layer(x)[..., None, None]


class GaussianFourierProjection(nn.Module):
    """
    We can incorporate the time information via [Gaussian random features](https://arxiv.org/abs/2006.10739).
    Specifically, we first sample $\omega \sim \mathcal{N}(\mathbf{0}, s^2\mathbf{I})$ which is subsequently fixed
    for the model (i.e., not learnable). For a time step $t$, the corresponding Gaussian random feature is defined as
    \begin{align}
      [\sin(2\pi \omega t) ; \cos(2\pi \omega t)],
    \end{align}
    where $[\vec{a} ; \vec{b}]$ denotes the concatenation of vector $\vec{a}$ and $\vec{b}$.
    This Gaussian random feature can be used as an encoding for time step $t$ so that the score network can
    condition on $t$ by incorporating this encoding. We will see this further in the code.
    """

    def __init__(self, dim, s_squared=30.0):
        super().__init__()
        self.w = nn.Parameter(torch.randn(dim // 2) * s_squared, requires_grad=False)

    def forward(self, t):
        param = 2 * np.pi * t[:, None] * self.w[None, :]
        return torch.cat([torch.sin(param), torch.cos(param)], dim=-1)


class UNetDownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, embed_dim, num_groups, stride):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, bias=False
        )
        self.dense = Dense(embed_dim, out_channels)
        self.gnorm = nn.GroupNorm(num_groups, num_channels=out_channels)

    def forward(self, h, embed):
        h = self.conv(h)
        h += self.dense(embed)
        return self.gnorm(h)


class UNetUpBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels, embed_dim, num_groups, stride, out_padding=0
    ):
        super().__init__()
        self.tconv = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            bias=False,
            output_padding=out_padding,
        )
        self.dense = Dense(embed_dim, out_channels)
        self.gnorm = nn.GroupNorm(num_groups, num_channels=out_channels)

    def forward(self, h, skip, embed):
        if skip != None:
            h = torch.cat([h, skip], dim=1)
        h = self.tconv(h)
        h += self.dense(embed)
        return self.gnorm(h)


# TODO: Use [exponential moving average](https://discuss.pytorch.org/t/how-to-apply-exponential-moving-average-decay-for-variables/10856/3) (EMA) of weights when sampling.
class ScoreNet(nn.Module):
    """
    A time-dependent score-based model build upon the U-Net architecture
    """

    def __init__(
        self,
        marginal_prob_std,
        channels=[32, 64, 128, 256],
        embed_dim=256,
        in_channels=1,
    ):
        """
        Initialize a time-dependent score-based network.

        Args:
          marginal_prob_std: A function that takes time t and gives the standard
            deviation of the perturbation kernel p_{0t}(x(t) | x(0)).
          channels: The number of channels for feature maps of each resolution.
          embed_dim: The dimensionality of Gaussian random feature embeddings.
        """
        super().__init__()
        # Gaussian random feature embedding layer for time
        self.embed = nn.Sequential(
            GaussianFourierProjection(dim=embed_dim), nn.Linear(embed_dim, embed_dim)
        )
        # Encoding layers -> Resolution decreases, channels increas
        self.down_block1 = UNetDownBlock(
            in_channels, channels[0], embed_dim, num_groups=4, stride=1
        )
        self.down_block2 = UNetDownBlock(
            channels[0], channels[1], embed_dim, num_groups=32, stride=2
        )
        self.down_block3 = UNetDownBlock(
            channels[1], channels[2], embed_dim, num_groups=32, stride=2
        )
        self.down_block4 = UNetDownBlock(
            channels[2], channels[3], embed_dim, num_groups=32, stride=2
        )
        # Decoding layers
        self.up_block1 = UNetUpBlock(
            channels[3], channels[2], embed_dim, stride=2, num_groups=32
        )
        self.up_block2 = UNetUpBlock(
            channels[2] * 2,
            channels[1],
            embed_dim,
            stride=2,
            num_groups=32,
            out_padding=1,
        )
        self.up_block3 = UNetUpBlock(
            channels[1] * 2,
            channels[0],
            embed_dim,
            stride=2,
            num_groups=32,
            out_padding=1,
        )
        self.up_tconv4 = nn.ConvTranspose2d(channels[0] * 2, 1, kernel_size=3, stride=1)
        self.act = lambda x: x * torch.sigmoid(x)
        self.marginal_prob_std = marginal_prob_std

    def forward(self, x, t):
        emb = self.act(self.embed(t))
        h1 = self.act(self.down_block1(x, emb))
        h2 = self.act(self.down_block2(h1, emb))
        h3 = self.act(self.down_block3(h2, emb))
        h4 = self.act(self.down_block4(h3, emb))
        h = self.act(self.up_block1(h4, None, emb))
        h = self.act(self.up_block2(h, h3, emb))
        h = self.act(self.up_block3(h, h2, emb))
        h = self.up_tconv4(torch.cat([h, h1], dim=1))
        h = h / self.marginal_prob_std(t)[:, None, None, None]
        return h


class ExponentialMovingAverage:
    """
    Maintains an exponential moving average (EMA) of a model's parameters.

    The implementation keeps a shadow copy of all trainable parameters and
    supports applying the shadow weights to the model and restoring the
    original weights afterwards.
    """

    def __init__(self, model, decay=0.9999, device=None):
        self.decay = decay
        self.device = device
        # shadow holds the EMA parameters
        self.shadow = {}
        # backup is used when apply_shadow is called to store originals
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = (
                    param.data.detach().clone().to(device)
                    if device
                    else param.data.detach().clone()
                )

    def update(self, model):
        """Update the shadow parameters using current model params.

        Should be called after each optimizer step (or at desired frequency).
        """
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_val = (
                    param.data.detach().clone().to(self.device)
                    if self.device
                    else param.data.detach().clone()
                )
                self.shadow[name].mul_(self.decay).add_(
                    new_val, alpha=(1.0 - self.decay)
                )

    def apply_shadow(self, model):
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                # backup on CPU to reduce GPU mem (optional)
                self.backup[name] = param.detach().cpu().clone()
        # copy shadow -> model under no_grad
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad:
                    param.copy_(self.shadow[name].to(param.device))

    def restore(self, model):
        if not self.backup:
            raise RuntimeError("No backup found: call apply_shadow() before restore().")
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad:
                    param.copy_(self.backup[name].to(param.device))
        self.backup = {}

    def copy_to(self, model):
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad:
                    param.copy_(self.shadow[name].to(param.device))

    def state_dict(self):
        # Return a CPU copy of the shadow dict (good for saving)
        return {k: v.cpu() for k, v in self.shadow.items()}

    def load_state_dict(self, state):
        # Load shadow values (state is a dict of name -> tensor)
        for k, v in state.items():
            if k in self.shadow:
                self.shadow[k] = v.to(self.device) if self.device else v.clone()
            else:
                raise KeyError(f"EMA: unexpected key {k} in state_dict")
