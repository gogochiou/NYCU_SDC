import torch
from nuscenes import NuScenes
from nuscenes.prediction import Model

from json import JSONEncoder
import json

model = torch.load("nuScenes_3Dtracking.pth", map_location=torch.device('cpu'))
weights = []
biases = []

for name, param in model.named_parameters():
    if "bias" in name:
        biases.append(param.detach().numpy())
    else:
        weights.append(param.detach().numpy())

model_json = Model.from_weights_biases(weights, biases).to_json()

with open("nuScenes_3Dtracking.json", "w") as f:
    json.dump(model_json, f)