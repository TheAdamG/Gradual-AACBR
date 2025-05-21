import sys
import os

# Get the parent directory
parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))

# Add parent directory to sys.path
sys.path.append(parent_dir)

print(parent_dir)


import torch
import torch.optim as optim
import numpy as np
from matplotlib import pyplot as plt

import deeparguing.feature_extractor.feature_weighted_extractor as fwe
import deeparguing.feature_extractor.mlp_extractor as mlpe

from helper import load_mushroom
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

from tqdm import tqdm





device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# device=None



# SEED = 42
SEED = 0


# ## Data Set


X, y = load_mushroom()

X = X.astype(np.float32)
np.random.seed(SEED)
indices = np.random.permutation(len(X))

X = X[indices]
y = y[indices]

all_y = np.unique(y, axis=0)

# full_train_size = 625
full_train_size = 250
train_size = int(full_train_size * 0.8)

# train_full, train, val, test = split_data(X, y, SEED)

# print(f"Test Size:  {len(test['X'])}")
# print(f"Train Size:  {len(train['X'])}")
# print(f"Validation Size:  {len(val['X'])}")

# train_full = {"X": X[:full_train_size], "y": y[:full_train_size]}
train = {"X": X[:train_size], "y": y[:train_size]}
val = {"X": X[train_size:full_train_size], "y": y[train_size:full_train_size]}
test = {"X": X[full_train_size:], "y": y[full_train_size:]}




# ### Convert to Torch


# X_train_full, y_train_full = torch.tensor(train_full["X"], device=device),      torch.tensor(train_full["y"], dtype=torch.float32, device=device)
X_train, y_train           = torch.tensor(train["X"]     , device=device),      torch.tensor(train["y"],      dtype=torch.float32, device=device)
X_val, y_val               = torch.tensor(val["X"]       , device=device),      torch.tensor(val["y"],        dtype=torch.float32, device=device)
X_test, y_test             = torch.tensor(test["X"]      , device=device),      torch.tensor(test["y"],       dtype=torch.float32, device=device)



# EPOCHS = 3000
EPOCHS = 500
LR = 2e-2


torch_seed = 0


torch.manual_seed(torch_seed) # TRY DIFFERENT INITIAL WEIGHTS 

no_features = X_train.shape[-1]

# model = mlpe.MLPExtractor(no_features, [50], 10).to(device)
model = mlpe.MLPExtractor(no_features, [], 2).to(device)


criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=LR)




def eval_model(model, X, y):
    # Assess the model pre-training
    with torch.no_grad():
        y_preds = torch.argmax(model(X), dim=1).cpu().detach().numpy()
        y_actual = torch.argmax(y, dim=1).cpu().numpy()

        # print("predictions", y_preds)
        # print("actuals", y_actual)

        results = ( 
            accuracy_score(y_actual, y_preds),
            precision_score(y_actual, y_preds, average='macro', zero_division=0),
            recall_score(y_actual, y_preds, average='macro', zero_division=0),
            f1_score(y_actual, y_preds, average='macro', zero_division=0),
            confusion_matrix(y_actual, y_preds)
        )


        print("Accuracy, precision, recall, f1")
        print(results[:-1])
        print("confusion matrix:")
        print(results[-1])


# print("TRAINING SET")
# eval_model(model, X_train, y_train)

losses = np.zeros((EPOCHS))
for epoch in tqdm(range(EPOCHS)):

    optimizer.zero_grad()
    y_preds = model(X_train)
    loss = criterion(y_preds, y_train)
    loss.backward()
    losses[epoch] = loss.item()
    optimizer.step()




print("TRAINING SET")
eval_model(model, X_train, y_train)

print("VALIDATION SET")
eval_model(model, X_val, y_val)

# print("Loss", losses[np.arange(0, EPOCHS, int(EPOCHS/100))])
print("TEST SET")
eval_model(model, X_test, y_test)

plt.plot(losses)
plt.show()



