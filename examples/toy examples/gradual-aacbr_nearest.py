import torch
import deeparguing
import deeparguing.semantics.relu_semantics as rs
import deeparguing.semantics.sigmoid_semantics as ss
import deeparguing.base_scores.constant_base_score as cbs
import deeparguing.irrelevance_edge_weights.feature_weighted_irrelevance as fwi

import time


"""
        N

d               c
| ↘           ↙ |
|   a       b   | 
  ↘   ↘   ↙   ↙ 
    ↘   δ+  ↙    
      ↘   ↙     
        δ- 

"""


X_train = torch.tensor([
    [0], # (a,
    [1], # (b
    [2], # (c
    [3], # (c
], dtype=torch.float32)

y_train = torch.tensor([
    [0], 
    [0], 
    [1], 
    [1], 
], dtype=torch.float32)

X_default = torch.tensor([
        [4], # (δ+
        [5], # (δ-
], dtype=torch.float32)
y_default = torch.tensor([[1], [0]], dtype=torch.float32)


semantics = rs.ReluSemantics(max_iters=10, epsilon=0)
# semantics = ss.SigmoidSemantics(max_iters=10, epsilon=0)


edge_weights = torch.tensor([
   # a  b  c  d  δ+ δ-
    [0, 0, 0, 0, 1, 1], # a
    [0, 0, 0, 0, 1, 1], # b
    [0, 1, 0, 1, 1, 1], # c
    [1, 0, 0, 0, 1, 1], # d
    [0, 0, 0, 0, 0, 0], # δ+
    [0, 0, 0, 0, 0, 0], # δ-
]) 



def edge_weights_test(attacker, target):
    attacker = attacker.to(dtype = torch.int)
    target = target.to(dtype = torch.int)
    return edge_weights[attacker, target]

def irrelevance(attacker, target):

    condition_a = (attacker == -2) 
    condition_b = torch.logical_or(target == 2,  target == 3)
    mask = condition_a * condition_b.T
    # print(mask)
    return torch.where(mask, 1., 0.) 


no_features = X_train.shape[-1]
model = deeparguing.GradualAACBR(semantics, 
                                   cbs.ConstantBaseScore(1), 
                                   irrelevance,
                                   edge_weights_test
                                   )

print("MODEL FIT")
model.fit(X_train, y_train, X_default, y_default, use_symmetric_attacks=False, use_supports = True)
# model.A = model.A * 10
model.show_matrix()

dummy_n = torch.tensor([
    [-1], # Does not attack anything 
    [-2] # Attacks c and d
  ])

result = model(dummy_n, return_all_strengths = False) # Expected [[1, 0], [0, 1]]
print(result)

