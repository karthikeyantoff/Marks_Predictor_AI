import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import optuna
import os
import gc

data = {
    'Hours': [1, 2, 3, 4, 5],
    'Marks': [30, 50, 65, 85, 90],
    'Grade': [1, 2, 3, 4, 5]
}
df = pd.DataFrame(data)
x = torch.tensor(df['Hours'].values, dtype=torch.float32).reshape(-1, 1)
y = torch.tensor(df[['Marks', 'Grade']].values, dtype=torch.float32)

def objective(trial):
    n1 = trial.suggest_int("n1", 16, 34)
    n2 = trial.suggest_int("n2", 34, 64)
    n3 = trial.suggest_int("n3", 64, 124)
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)

    model = nn.Sequential(
        nn.Linear(1, n1),
        nn.ReLU(),
        nn.Linear(n1, n2),
        nn.ReLU(),
        nn.Linear(n2, n3),
        nn.ReLU(),
        nn.Linear(n3, 2)
    )

    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(300):
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    trial.set_user_attr("model_state_dict", model.state_dict())
    return loss.item()

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=30)

n1, n2, n3 = study.best_params["n1"], study.best_params["n2"], study.best_params["n3"]
best_model = nn.Sequential(
    nn.Linear(1, n1),
    nn.ReLU(),
    nn.Linear(n1, n2),
    nn.ReLU(),
    nn.Linear(n2, n3),
    nn.ReLU(),
    nn.Linear(n3, 2)
)
best_model.load_state_dict(study.best_trial.user_attrs["model_state_dict"])
os.makedirs("pytorch_data", exist_ok=True)
torch.save(best_model.state_dict(), "pytorch_data/pytorch.pth")
with open("pytorch_data/best_params.txt", "w") as f:
    f.write(f"{n1},{n2},{n3}")
    
del best_model
gc.collect()
torch.cuda.empty_cache()

print("Model trained and saved!")



                               