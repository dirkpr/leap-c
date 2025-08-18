"""Provides a simple Gaussian layer that allows policies to respect action bounds."""

from gymnasium import spaces
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Beta


class BoundedTransform(nn.Module):
    """A bounded transform.

    The output is squashed with a tanh function and then scaled and shifted to match the space.
    """

    scale: torch.tensor
    loc: torch.tensor

    def __init__(
        self,
        space: spaces.Box,
    ):
        """Initializes the Bounded Transform module.

        Args:
            space: The space that the transform is bounded to.
            padding: The amount of padding to subtract to the bounds.
        """
        super().__init__()
        loc = (space.high + space.low) / 2.0
        scale = (space.high - space.low) / 2.0

        loc = torch.tensor(loc, dtype=torch.float32)
        scale = torch.tensor(scale, dtype=torch.float32)

        self.register_buffer("loc", loc)
        self.register_buffer("scale", scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies the squashing function to the input tensor.

        Args:
            x: The input tensor.

        Returns:
            The squashed tensor, scaled and shifted to match the action space.
        """
        x = torch.tanh(x)
        return x * self.scale[None, :] + self.loc[None, :]

    def inverse(self, x: torch.Tensor, padding: float = 0.001) -> tuple[torch.Tensor]:
        """Applies the inverse squashing function to the input tensor.

        Args:
            x: The input tensor.
            padding: The amount of padding to distance the action of the bounds.

        Returns:
            The inverse squashed tensor, scaled and shifted to match the action space.
        """
        abs_padding = self.scale[None, :] * padding
        x = (x - self.loc[None, :]) / (self.scale[None, :] + 2 * abs_padding)
        return torch.arctanh(x)


class SquashedGaussian(nn.Module):
    """A squashed Gaussian.

    The output is sampled from this distribution and then squashed with a tanh function.
    Finally, the output of the tanh function is scaled and shifted to match the space.
    """

    scale: torch.Tensor
    loc: torch.Tensor

    def __init__(
        self,
        space: spaces.Box,
        log_std_min: float = -4,
        log_std_max: float = 2.0,
    ):
        """Initializes the SquashedGaussian module.

        Args:
            space: The action space of the environment. Used for constraints.
            log_std_min: The minimum value for the logarithm of the standard deviation.
            log_std_max: The maximum value for the logarithm of the standard deviation.
        """
        super().__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        loc = (space.high + space.low) / 2.0
        scale = (space.high - space.low) / 2.0

        loc = torch.tensor(loc, dtype=torch.float32)
        scale = torch.tensor(scale, dtype=torch.float32)

        self.register_buffer("loc", loc)
        self.register_buffer("scale", scale)

    def forward(
        self, mean: torch.Tensor, log_std: torch.Tensor, deterministic: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
        """
        Args:
            mean: The mean of the distribution.
            log_std: The logarithm of the standard deviation of the distribution.
            deterministic: If True, the output will just be tanh(mean), no sampling is taking place.

        Returns:
            An output sampled from the TanhNormal, the log probability of this output
            and a statistics dict containing the standard deviation.
        """
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)

        if deterministic:
            y = mean
        else:
            # reparameterization trick
            y = mean + std * torch.randn_like(mean)

        log_prob = (
            -0.5 * ((y - mean) / std).pow(2) - log_std - np.log(np.sqrt(2) * np.pi)
        )

        y = torch.tanh(y)

        log_prob -= torch.log(self.scale[None, :] * (1 - y.pow(2)) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        y_scaled = y * self.scale[None, :] + self.loc[None, :]

        stats = {"gaussian_unsquashed_std": std.prod(dim=-1).mean().item()}

        return y_scaled, log_prob, stats


class BoundedBeta(nn.Module):
    """Beta distribution rescaled to arbitrary bounds"""

    def __init__(
        self,
        mode=None,
        alpha=None,
        beta=None,
        lower_bound=0.0,
        upper_bound=1.0,
        c: float = 1.0,
        log_c_min: float = -4.0,
        log_c_max: float = 2.0,
    ):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.log_c_min = log_c_min
        self.log_c_max = log_c_max

    def forward(
        self,
        mode: torch.Tensor,
        log_c: torch.Tensor,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
        """Sample from bounded Beta"""
        # Sample from Beta(0, 1)

        assert type(mode) is torch.Tensor, "Mode must be a torch.Tensor"

        log_c = torch.clamp(log_c, self.log_c_min, self.log_c_max)
        c = torch.exp(log_c)

        lower_bound = self.lower_bound
        upper_bound = self.upper_bound

        alpha = torch.zeros_like(mode)
        beta = torch.zeros_like(mode)

        for i, (m, c_val) in enumerate(
            zip(mode[0], c[0])
        ):  # Use [0] to get the actual row
            # Convert mode from bounded space to [0,1] space
            if m < lower_bound[i] or m > upper_bound[i]:
                raise ValueError(
                    f"Mode {m} must be within bounds [{lower_bound[i]}, {upper_bound[i]}]"
                )
            normalized_mode = (m - lower_bound[i]) / (upper_bound[i] - lower_bound[i])

            if normalized_mode == 0.0:
                alpha[0, i], beta[0, i] = (
                    1.0,
                    c_val + 1.0,
                )  # Use [0, i] for (1, n) tensor
            elif normalized_mode == 1.0:
                alpha[0, i], beta[0, i] = c_val + 1.0, 1.0
            else:
                total_concentration = c_val + 2
                alpha_val = normalized_mode * (total_concentration - 2) + 1
                beta_val = total_concentration - alpha_val
                alpha[0, i] = max(0.1, alpha_val)
                beta[0, i] = max(0.1, beta_val)

        self.beta_dist = Beta(alpha, beta)

        beta_samples = self.beta_dist.sample()

        # Rescale to [lower_bound, upper_bound]
        scaled = beta_samples * (self.upper_bound - self.lower_bound) + self.lower_bound

        return scaled

    def log_prob(self, x):
        """Compute log probability"""
        # Transform to [0, 1] space
        normalized = (x - self.lower_bound) / (self.upper_bound - self.lower_bound)

        # Beta log prob + scaling factor
        log_prob_beta = self.beta_dist.log_prob(normalized)
        scaling_factor = torch.log(
            torch.tensor(1.0 / (self.upper_bound - self.lower_bound))
        )

        return log_prob_beta + scaling_factor
