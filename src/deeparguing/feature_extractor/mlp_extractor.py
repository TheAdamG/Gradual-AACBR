import torch
import matplotlib.pyplot as plt
from deeparguing.feature_extractor.feature_extractor import FeatureExtractor

class MLPExtractor(FeatureExtractor):


    def __init__(self, input_size, hidden_sizes, output_size, output_activation = None):
        super(MLPExtractor, self).__init__(output_size)

        layer_sizes = [input_size] + hidden_sizes + [output_size]
        
        self.layers = torch.nn.ModuleList()
        for i in range(len(layer_sizes) - 1):
            self.layers.append(torch.nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            if i < len(layer_sizes) - 2:
                self.layers.append(torch.nn.ReLU())

        if output_activation:
            self.layers.append(output_activation)
        

    def forward(self, case: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            case = layer(case)

        case = case.squeeze()
        
        return case


    def get_output_features(self) -> int:
        return self.no_features

    def plot_parameters(self):
        print("Not plotting NN params")
