import torch
import deeparguing
import deeparguing.semantics.relu_semantics as rs
import deeparguing.base_scores.constant_base_score as cbs
import deeparguing.irrelevance_edge_weights.feature_weighted_irrelevance as fwi

import time


X_train = torch.tensor([
    [0], # (a,
    [1], # (b
    [2], # (c
], dtype=torch.float32)

y_train = torch.tensor([
    [0], 
    [1], 
    [1], 
], dtype=torch.float32)

X_default = torch.tensor([
        [3]
], dtype=torch.float32)
y_default = torch.tensor([[0]], dtype=torch.float32)


semantics = rs.ReluSemantics(max_iters=5, epsilon=0)


edge_weights = torch.tensor([
    [0, 0.3, 0.4, 0.6],
    [0, 0, 0.2, 0.5], 
    [0, 0, 0, 0.3], 
    [0, 0, 0, 0]
])



def edge_weights_test(attacker, target):
    attacker = attacker.to(dtype = torch.int)
    target = target.to(dtype = torch.int)
    return edge_weights[attacker, target]


no_features = X_train.shape[-1]
model = deeparguing.GradualAACBR(semantics, 
                                   cbs.ConstantBaseScore(1), 
                                   fwi.FeatureWeightedIrrelevance(no_features), 
                                   edge_weights_test
                                   )

print("MODEL FIT")
start_time = time.time()
model.fit(X_train, y_train, X_default, y_default, use_symmetric_attacks=False)
new_fit = model.A
print("--- %s seconds ---" % (time.time() - start_time))
model.show_matrix()


print("MODEL SLOW_FIT")
start_time = time.time()
model.slow_fit(X_train, y_train, X_default, y_default, use_symmetric_attacks=False)
slow_fit = model.A
print("--- %s seconds ---" % (time.time() - start_time))
model.show_matrix()

print(new_fit == slow_fit)

print(torch.all(new_fit == slow_fit))

assert(torch.all(new_fit == slow_fit))
