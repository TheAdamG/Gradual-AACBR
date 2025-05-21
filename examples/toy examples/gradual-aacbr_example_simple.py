import torch
import deeparguing
import deeparguing.semantics.relu_semantics as rs
import deeparguing.base_scores.constant_base_score as cbs
import deeparguing.irrelevance_edge_weights.feature_weighted_irrelevance as fwi


""""

0: (10, 1)     1: (5: 1)
                    |
                    |
                    v
            2: (default, 0)  

"""


X_train = torch.tensor([
    [10], # (0, 1)
    [5],  # (1, 1)
], dtype=torch.float32)

y_train = torch.tensor([
    [1, 0], 
    [1, 0], 
], dtype=torch.float32)

X_default = torch.tensor([
        [0] 
], dtype=torch.float32) # 4, 5
y_default = torch.tensor([[0, 1]], dtype=torch.float32)

torch.manual_seed(1)

no_features = X_train.shape[-1]
# semantics = ms.MLPBasedSemantics(max_iters=5, epsilon=0)
semantics = rs.ReluSemantics(max_iters=5, epsilon=0)

edge_weights_test = lambda a, t: torch.where(a >= t, 1.0, 0.0) 


model = deeparguing.GradualAACBR(semantics, 
                                   cbs.ConstantBaseScore(1), 
                                   fwi.FeatureWeightedIrrelevance(no_features), 
                                   edge_weights_test
                                   )

model.fit(X_train, y_train, X_default, y_default, use_symmetric_attacks=True)

new_case = torch.tensor([
    [2],
], dtype=torch.float32)

strengths = model(new_case)
print(strengths)

model.show_matrix()
model.show_graph_with_labels()
