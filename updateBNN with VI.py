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
# 1. Load Previous Variational Parameters
# ------------------------------
np.random.seed(42)
torch.manual_seed(42)

# Simulated new observations (D')
x_new = np.linspace(0.2, 0.6, 100)
noise_new = 0.02 * np.random.randn(x_new.shape[0])
y_new = x_new + 0.3 * np.sin(2 * np.pi * (x_new + noise_new)) + 0.3 * np.sin(4 * np.pi * (x_new + noise_new)) + noise_new

x_new_train = torch.tensor(x_new.reshape(-1, 1), dtype=torch.float32)
y_new_train = torch.tensor(y_new.reshape(-1, 1), dtype=torch.float32)

# Extract the variational parameters from the previous guide
mu = guide.get_posterior().mean.detach()
stddev = guide.get_posterior().stddev.detach()

# ------------------------------
# 2. Define Updated Bayesian Neural Network (UpdatedBNN)
# ------------------------------
class UpdatedBNN(PyroModule):
    def __init__(self, mu, stddev, in_dim=1, out_dim=1, hid_dim=10):
        super().__init__()
        self.activation = nn.Tanh()
        self.layer1 = PyroModule[nn.Linear](in_dim, hid_dim)
        self.layer2 = PyroModule[nn.Linear](hid_dim, out_dim)

        # Initialize priors using learned variational parameters
        self.layer1.weight = PyroSample(dist.Normal(mu[:hid_dim], stddev[:hid_dim]).to_event(2))
        self.layer1.bias = PyroSample(dist.Normal(mu[hid_dim:hid_dim+1], stddev[hid_dim:hid_dim+1]).to_event(1))
        self.layer2.weight = PyroSample(dist.Normal(mu[hid_dim+1:hid_dim+1+hid_dim], stddev[hid_dim+1:hid_dim+1+hid_dim]).to_event(2))
        self.layer2.bias = PyroSample(dist.Normal(mu[hid_dim+1+hid_dim:], stddev[hid_dim+1+hid_dim:]).to_event(1))

    def forward(self, x, y=None):
        x = self.activation(self.layer1(x))
        mu = self.layer2(x).squeeze(-1)
        sigma = pyro.sample("sigma", dist.Gamma(0.5, 1.0))
        with pyro.plate("data", x.shape[0]):
            y_pred = pyro.sample("y_pred", dist.Normal(mu, sigma).to_event(1), obs=y)
        return y_pred

# ------------------------------
# 3. Train Updated Model on New Observations (D')
# ------------------------------
pyro.clear_param_store()
new_model = UpdatedBNN(mu, stddev)
new_guide = AutoDiagonalNormal(new_model)
optimizer = pyro.optim.Adam({"lr": 0.01})
svi_new = SVI(new_model, new_guide, optimizer, loss=Trace_ELBO())

num_epochs_new = 5000  # Reduce epochs to save memory
progress_bar_new = trange(num_epochs_new)
for epoch in progress_bar_new:
    loss = svi_new.step(x_new_train, y_new_train)
    progress_bar_new.set_postfix(loss=f"{loss / x_new_train.shape[0]:.3f}")

# ------------------------------
# 4. Compute Predictive Distribution
# ------------------------------
predictive = Predictive(new_model, guide=new_guide, num_samples=100)  # Reduce samples to save memory
x_test = torch.linspace(-0.5, 1.5, 300).reshape(-1, 1)
preds = predictive(x_test)

# ------------------------------
# 5. Visualization
# ------------------------------
def plot_predictions(preds):
    y_pred = preds["y_pred"].mean(dim=0).detach().numpy().squeeze()
    y_std = preds["y_pred"].std(dim=0).detach().numpy().squeeze()
    x_test_np = x_test.squeeze().numpy()

    plt.figure(figsize=(10, 5))
    plt.xlim([-0.5, 1.5])
    plt.ylim([-1.5, 2.5])
    plt.xlabel("X", fontsize=20)
    plt.ylabel("Y", fontsize=20)
    plt.plot(x_new, y_new, 'ro', markersize=4, label="New Observations")
    plt.plot(x_test_np, y_pred, '-', linewidth=3, color="#408765", label="Predictive Mean")
    plt.fill_between(x_test_np, y_pred - 2 * y_std, y_pred + 2 * y_std, alpha=0.6, color='#86cfac', zorder=5)
    plt.legend(loc=4, fontsize=15, frameon=False)
    plt.show()

plot_predictions(preds)
