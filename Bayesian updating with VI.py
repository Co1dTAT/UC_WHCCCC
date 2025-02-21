import torch
import numpy as np
import matplotlib.pyplot as plt
import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule, PyroSample
import torch.nn as nn
from pyro.infer import SVI, Trace_ELBO, Predictive
from pyro.infer.autoguide import AutoDiagonalNormal
from tqdm.auto import trange

# ------------------------------
# 1. Simulate Old Observations (D)
# ------------------------------
np.random.seed(42)
x_obs = np.hstack([np.linspace(-0.2, 0.2, 500), np.linspace(0.6, 1, 500)])
noise = 0.02 * np.random.randn(x_obs.shape[0])
y_obs = x_obs + 0.3 * np.sin(2 * np.pi * (x_obs + noise)) + 0.3 * np.sin(4 * np.pi * (x_obs + noise)) + noise

x_train = torch.tensor(x_obs.reshape(-1, 1), dtype=torch.float32)
y_train = torch.tensor(y_obs.reshape(-1, 1), dtype=torch.float32)

x_true = np.linspace(-0.5, 1.5, 1000)
y_true = x_true + 0.3 * np.sin(2 * np.pi * x_true) + 0.3 * np.sin(4 * np.pi * x_true)

# ------------------------------
# 2. Define Bayesian Neural Network (BNN)
# ------------------------------
class BNN(PyroModule):
    def __init__(self, in_dim=1, out_dim=1, hid_dim=10, n_hid_layers=5, prior_scale=5.):
        super().__init__()
        self.activation = nn.Tanh()
        self.layers = nn.ModuleList()

        prev_dim = in_dim
        for layer_idx in range(n_hid_layers):
            layer = PyroModule[nn.Linear](prev_dim, hid_dim)
            setattr(layer, f"weight_{layer_idx}", PyroSample(
                dist.Normal(0, prior_scale).expand(layer.weight.shape).to_event(2)
            ))
            setattr(layer, f"bias_{layer_idx}", PyroSample(
                dist.Normal(0, prior_scale).expand(layer.bias.shape).to_event(1)
            ))
            self.layers.append(layer)
            prev_dim = hid_dim

        self.out_layer = PyroModule[nn.Linear](prev_dim, out_dim)
        setattr(self.out_layer, "out_weight", PyroSample(
            dist.Normal(0, prior_scale).expand(self.out_layer.weight.shape).to_event(2)
        ))
        setattr(self.out_layer, "out_bias", PyroSample(
            dist.Normal(0, prior_scale).expand(self.out_layer.bias.shape).to_event(1)
        ))

    def forward(self, x, y=None):
        for layer in self.layers:
            x = self.activation(layer(x))
        mu = self.out_layer(x).squeeze(-1)

        sigma = pyro.sample("sigma", dist.Gamma(0.5, 1.0))
        y_obs = y.squeeze(-1) if y is not None else None
        with pyro.plate("data", x.shape[0]):
            y_pred = pyro.sample("y_pred", dist.Normal(mu, sigma).to_event(1), obs=y_obs)

        return y_pred

# ------------------------------
# 3. Train Model on Old Observations (D)
# ------------------------------
pyro.clear_param_store()
model = BNN(hid_dim=10, n_hid_layers=5, prior_scale=5.)
guide = AutoDiagonalNormal(model)
optimizer = pyro.optim.Adam({"lr": 0.01})
svi = SVI(model, guide, optimizer, loss=Trace_ELBO())

num_epochs = 25000
progress_bar = trange(num_epochs)
for epoch in progress_bar:
    loss = svi.step(x_train, y_train)
    progress_bar.set_postfix(loss=f"{loss / x_train.shape[0]:.3f}")

# ------------------------------
# 4. Generate New Observations (D')
# ------------------------------
x_new = np.linspace(0.2, 0.6, 100)
noise_new = 0.02 * np.random.randn(x_new.shape[0])
y_new = x_new + 0.3 * np.sin(2 * np.pi * (x_new + noise_new)) + 0.3 * np.sin(4 * np.pi * (x_new + noise_new)) + noise_new

x_new_train = torch.tensor(x_new.reshape(-1, 1), dtype=torch.float32)
y_new_train = torch.tensor(y_new.reshape(-1, 1), dtype=torch.float32)

# ------------------------------
# 5. Bayesian Update: Use q_ϕ(θ) as New Prior
# ------------------------------
mu = guide.get_posterior().mean.detach()
stddev = guide.get_posterior().stddev.detach()

new_guide = AutoDiagonalNormal(model)
pyro.clear_param_store()
svi_new = SVI(model, new_guide, optimizer, loss=Trace_ELBO())

num_epochs_new = 10000
progress_bar_new = trange(num_epochs_new)
for epoch in progress_bar_new:
    loss = svi_new.step(x_new_train, y_new_train)
    progress_bar_new.set_postfix(loss=f"{loss / x_new_train.shape[0]:.3f}")

# ------------------------------
# 6. Compute Predictive Distribution
# ------------------------------
predictive = Predictive(model, guide=new_guide, num_samples=500)
x_test = torch.linspace(-0.5, 1.5, 300).reshape(-1, 1)
preds = predictive(x_test)

# ------------------------------
# 7. Visualization
# ------------------------------
def plot_predictions(preds):

    y_pred = preds["y_pred"].mean(dim=0).detach().numpy().squeeze()
    y_std = preds["y_pred"].std(dim=0).detach().numpy().squeeze()
    
    y_pred = np.mean(y_pred, axis=0)  
    y_std = np.mean(y_std, axis=0)  

    x_test_np = x_test.squeeze().numpy()

    xlims = [-0.5, 1.5]
    ylims = [-1.5, 2.5]

    fig, ax = plt.subplots(figsize=(10, 5))
    plt.xlim(xlims)
    plt.ylim(ylims)
    plt.xlabel("X", fontsize=30)
    plt.ylabel("Y", fontsize=30)

    ax.plot(x_true, y_true, 'b-', linewidth=3, label="True function")
    ax.plot(x_obs, y_obs, 'ko', markersize=4, label="Old Observations")
    ax.plot(x_new, y_new, 'ro', markersize=4, label="New Observations")
    ax.plot(x_test_np, y_pred, '-', linewidth=3, color="#408765", label="Predictive Mean")

    y1 = y_pred - 2 * y_std
    y2 = y_pred + 2 * y_std

    print(f"x_test shape: {x_test_np.shape}")
    print(f"y_pred shape: {y_pred.shape}")
    print(f"y_std shape: {y_std.shape}")
    print(f"y1 shape: {y1.shape}, y2 shape: {y2.shape}")

    ax.fill_between(x_test_np, y1, y2, alpha=0.6, color='#86cfac', zorder=5) # fill between y1 and y2

    plt.legend(loc=4, fontsize=15, frameon=False)
    plt.show()
plot_predictions(preds)



