import os
os.environ["ACCELERATE_DISABLE_RICH"] = "1"

import sys
from pathlib import Path

exercises_dir = Path("/home/zmd/Code/ARENA_3.0/chapter1_transformer_interp/exercises/")
section_dir = exercises_dir / "part6_othellogpt"
othello_dir = section_dir / "othello_world" / "mechanistic_interpretability"
if str(exercises_dir) not in sys.path:
    sys.path.append(str(exercises_dir))
    sys.path.append(str(othello_dir))

import torch
import torch.nn as nn
import torch.nn.functional
import torch.optim
from torch import Tensor
from torch.utils.data import DataLoader

import numpy as np
import einops
import wandb
from ipywidgets import interact
import plotly.express as px
from ipywidgets import interact
from pathlib import Path
import itertools
import random
from IPython.display import display
import wandb
from jaxtyping import Float, Int, Bool, Shaped, jaxtyped
from typing import List, Union, Optional, Tuple, Callable, Dict
import typeguard
from functools import partial
import copy
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
import dataclasses
import datasets
from IPython.display import HTML
import transformer_lens
import transformer_lens.utils as utils
from transformer_lens.hook_points import HookedRootModule, HookPoint
from transformer_lens import HookedTransformer, HookedTransformerConfig, FactoredMatrix, ActivationCache
from tqdm.notebook import tqdm
from dataclasses import dataclass
from rich import print as rprint
import pandas as pd

from plotly_utils import imshow
from neel_plotly import scatter, line

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Othello!!

# I'm confused why the vocabulary size is 61. Shouldn't it be 60?! (64 − 4)
#
# It turns out that 0 is for "pass"

cfg = HookedTransformerConfig(
    n_layers = 8,
    d_model = 512,
    d_head = 64,
    n_heads = 8,
    d_mlp = 2048,
    d_vocab = 61,
    n_ctx = 59,
    act_fn="gelu",
    normalization_type="LNPre",
    device=device,
)
model = HookedTransformer(cfg)

import part6_othellogpt.tests as tests
import part6_othellogpt.solutions as solutions

# There's a lot of show-and-tell setup before we're asked to do some exercises ourselves.

def cosine_similarities(full_linear_probe):
    cos_sim = nn.CosineSimilarity(dim=0)
    cos_sim(full_linear_probe[1, ..., 1], full_linear_probe[2, ..., 1])

# bleh ...
