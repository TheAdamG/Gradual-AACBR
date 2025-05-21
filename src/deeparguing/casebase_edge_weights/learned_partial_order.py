import torch
import torch.nn.functional as F
from deeparguing.casebase_edge_weights.compute_partial_order import ComputePartialOrder, CompareCases
import matplotlib.pyplot as plt
from deeparguing.feature_extractor.feature_extractor import FeatureExtractor
from typing import List


class LearnedPartialOrder(ComputePartialOrder):

    def __init__(self,  feature_extractors: List[FeatureExtractor], comparison_func: CompareCases, batch_size = None):
        super(LearnedPartialOrder, self).__init__()

        self.feature_extractors = torch.nn.ModuleList(feature_extractors)
        self.comparison_func = comparison_func
        self.batch_size = batch_size


    def forward(self, attacker: torch.Tensor, target: torch.Tensor) -> torch.Tensor:

        # attacker (n, no_features)
        # target (n, no_features)
        if target.ndim == 1:
            target = target.unsqueeze(0)

        if attacker.ndim == 1:
            attacker = attacker.unsqueeze(0)
        
        # TODO: Dimension reshaping to support handling different size attackers/targets
        # When dealing with newcases attacking the casebase: Need to handle this differently
        reshape = False

        if attacker.shape[1] == 1 and target.shape[0] == 1:
            attacker = attacker.squeeze()
            target = target.squeeze()
            reshape = True
        
        for feature_extractor in self.feature_extractors:

            if self.batch_size == None:
                attacker = feature_extractor(attacker)
                target = feature_extractor(target)
            else:
                attacker = self.split_apply_fe(attacker, feature_extractor, self.batch_size)
                target = self.split_apply_fe(target, feature_extractor, self.batch_size)


        if reshape:
            attacker = attacker.unsqueeze(1)
            target = target.unsqueeze(0)


        result = self.comparison_func(attacker, target)
        return result


    def plot_parameters(self):
        for feature_extractor in self.feature_extractors:
            feature_extractor.plot_parameters()


    def split_apply_fe(self, cases, feature_extractor, batch_size):
        split_cases = torch.split(cases, batch_size)
        result = [feature_extractor(cases_i) for cases_i in split_cases]
        result = torch.cat(result, dim=0)
        return result