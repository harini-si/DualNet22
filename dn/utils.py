# necessary imports
import pickle
import random

import numpy as np
import torch


def deterministic(args):
  # set seed for reproducibility
  random.seed(args.seed)
  np.random.seed(args.seed)

  torch.manual_seed(args.seed)
  torch.cuda.manual_seed_all(args.seed)

  torch.backends.cudnn.benchmark = False
  torch.backends.cudnn.deterministic = True


def load_image_data_pickle(path):
	# load image data from pickle file given path
  with open(path, "rb") as f:
    data = pickle.load(f)
  return data
