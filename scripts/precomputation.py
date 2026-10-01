'''
Script used for precomputing data such as onering coords, some gradients etc., so that autograd does less work during the optimisation loop.
Called by notebooks/surface_fitting.ipynb.
'''

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # make `bcs` importable

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
from torch import autograd as Grad

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.cm as cm
import matplotlib.tri as mtri
import matplotlib
from IPython.display import display, clear_output

import ipywidgets as widgets
import trimesh
import random
import math
import time
import os
import json
import copy

import open3d as o3d
print(o3d.__path__)
print("Using open3d version", o3d.__version__)
o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

from bcs.visuals import *
from bcs.utils import *
from bcs.mesh_processing import *

from bcs import differential
import importlib
importlib.reload(differential)
from bcs.differential import *

two_pi = 2 * torch.pi
diffmod = DifferentialModule()

# -------------------------------
# Argument parsing
# -------------------------------
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--config-filepath", type=str, required=True)
parser.add_argument("--output-filepath", type=str, required=True)

args = parser.parse_args()

config_filepath = args.config_filepath
with open(config_filepath, "r") as f:
    config_dict = json.load(f)
    surface_config = config_dict['surface-config']
    training_config = config_dict['training-config']




output_filepath = args.output_filepath


# -------------------------------
# Device
# -------------------------------
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")
# -------------------------------


# -------------------------------
# BCS setup
# -------------------------------
from bcs import surface
importlib.reload(surface)
from bcs.surface import BCS_fast

bcs = BCS_fast(surface_config, device=device)

mobius_example = False
if 'mobius_example' in surface_config.keys() and surface_config['mobius_example']==True:
    mobius_example=True

if mobius_example==True:
    bcs.do_mobius_edits()





if not 'num_bdry_samples_per_edge' in training_config.keys():
    training_config['num_bdry_samples_per_edge'] = 0
# -------------------------------
# Sample generation
# -------------------------------
training_samples = bcs.compute_samples(num_samples=training_config['num_samples_per_face'], num_bdry_samples= training_config['num_bdry_samples_per_edge']*3)


x = training_samples["uv"]



precomputed_training_data = bcs.precompute_data_from_samples(
    x, detached=True, mobius_example=mobius_example
)

# -------------------------------
# Save
# -------------------------------
os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
torch.save(precomputed_training_data, output_filepath)

print(f"Saved precomputed samples to {output_filepath}")
