# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from torch import nn
from torch.nn import functional as F

from nemo.collections.asr.data.audio_to_text import AudioToCharWithDursF0Dataset
from nemo.collections.tts.modules.fastspeech2_submodules import LengthRegulator


class GaussianEmbedding(nn.Module):
    """Gaussian embedding layer.."""

    EPS = 1e-6

    def __init__(
        self, vocab, d_emb, sigma_c=2.0, merge_blanks=False,
    ):
        super().__init__()

        # Keep this the same so that existing models still work.
        self.embed = nn.Embedding(len(vocab.labels), d_emb)
        self.pad = vocab.pad
        self.sigma_c = sigma_c
        self.merge_blanks = merge_blanks

        self.length_regulator = LengthRegulator()

        # The idea is to smooth the upsampled embeddings with a Gaussian kernel
        # to approximate the behavior of the regular GaussianEmbedding.
        # The stddev is 0.5 since that's what the regular GaussianEmbedding
        # uses for a token with a duration of 1 (as 1/sigma_c = 0.5), and
        # non-blank tokens usually have a duration of 1.
        # The kernel size is 5 since going larger would have almost no effect.
        # (e.g. expanding to a size of 7 means that the value on the edge of
        # the kernel would be 1.2e-8, which is probably too small to make a
        # difference)
        normal = torch.distributions.normal.Normal(0, 0.5)
        x = torch.linspace(-2, 2, 5)
        kernel = normal.log_prob(x).exp()
        # Normalize so the overall embeddings aren't scaled up or down by a
        # constant.
        kernel = kernel / kernel.sum()
        kernel = kernel.reshape(1, 1, 5)
        # persistent=False ensures that the buffer isn't added to state_dict.
        # This allows for loading of checkpoints made without this code change.
        self.register_buffer("gaussian_kernel", kernel, persistent=False)

    def forward(self, text, durs):
        # Remove <blank> tokens. We keep the first <blank> so that the model
        # knows if there's silence at the beginning of the clip.
        text = torch.cat(
            (
                text[:, 0].unsqueeze(1),
                text[:, 1::2],
            ),
            1
        )

        # Add the duration of each <blank> token to the preceeding token
        # (again, except for the first <blank>).
        durs = torch.cat(
            (
                durs[:, 0].unsqueeze(1),
                durs[:, 1::2] + durs[:, 2::2],
            ),
            1
        )

        # Embed and repeat tokens
        x = self.length_regulator(self.embed(text), durs)

        # Smooth every channel with the Gaussian kernel
        b, c, t = x.shape
        x = x.view(b * c, 1, t)
        x = F.conv1d(
            # Reflect padding ensures that the edges don't change
            F.pad(x, [2, 2], mode="reflect"),
            self.gaussian_kernel,
        )
        x = x.view(b, c, t)

        return x


class MaskedInstanceNorm1d(nn.Module):
    """Instance norm + masking."""

    MAX_CNT = 1e5

    def __init__(self, d_channel: int, unbiased: bool = True, affine: bool = False):
        super().__init__()

        self.d_channel = d_channel
        self.unbiased = unbiased

        self.affine = affine
        if self.affine:
            gamma = torch.ones(d_channel, dtype=torch.float)
            beta = torch.zeros_like(gamma)
            self.register_parameter('gamma', nn.Parameter(gamma))
            self.register_parameter('beta', nn.Parameter(beta))

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor) -> torch.Tensor:  # noqa
        """`x`: [B,C,T], `x_mask`: [B,T] => [B,C,T]."""
        x_mask = x_mask.unsqueeze(1).type_as(x)  # [B,1,T]
        cnt = x_mask.sum(dim=-1, keepdim=True)  # [B,1,1]

        # Mean: [B,C,1]
        cnt_for_mu = cnt.clamp(1.0, self.MAX_CNT)
        mu = (x * x_mask).sum(dim=-1, keepdim=True) / cnt_for_mu

        # Variance: [B,C,1]
        sigma = (x - mu) ** 2
        cnt_fot_sigma = (cnt - int(self.unbiased)).clamp(1.0, self.MAX_CNT)
        sigma = (sigma * x_mask).sum(dim=-1, keepdim=True) / cnt_fot_sigma
        sigma = (sigma + 1e-8).sqrt()

        y = (x - mu) / sigma

        if self.affine:
            gamma = self.gamma.unsqueeze(0).unsqueeze(-1)
            beta = self.beta.unsqueeze(0).unsqueeze(-1)
            y = y * gamma + beta

        return y


class StyleResidual(nn.Module):
    """Styling."""

    def __init__(self, d_channel: int, d_style: int, kernel_size: int = 1):
        super().__init__()

        self.rs = nn.Conv1d(
            in_channels=d_style, out_channels=d_channel, kernel_size=kernel_size, stride=1, padding=kernel_size // 2,
        )

    def forward(self, x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """`x`: [B,C,T], `s`: [B,S,T] => [B,C,T]."""
        return x + self.rs(s)
