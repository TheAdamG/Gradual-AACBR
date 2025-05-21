from deeparguing.semantics.gradual_semantics import GradualSemantics
import torch


class SigmoidSemantics(GradualSemantics):

    def __init__(self, max_iters, epsilon) -> None:
        super().__init__(max_iters, epsilon)    

    def aggregation_func(self, A, strengths):
        return torch.matmul(torch.transpose(A, -2, -1), strengths)

    def influence_func(self, base_scores, aggregations):
        return torch.sigmoid(
            torch.log(base_scores/(1 - base_scores)) + aggregations
        )
