import torch
import numpy as np
import matplotlib.pyplot as plt
import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule, PyroSample
import torch.nn as nn
from pyro.infer import MCMC, NUTS
from pyro.infer import Predictive


# 1. Simulate Data
# ------------------------------
np.random.seed(42)

# Generate x values (two separate segments)
x_obs = np.hstack([np.linspace(-0.2, 0.2, 500), np.linspace(0.6, 1, 500)])
noise = 0.02 * np.random.randn(x_obs.shape[0])

# Generate y values with sinusoidal components and noise
y_obs = x_obs + 0.3 * np.sin(2 * np.pi * (x_obs + noise)) + 0.3 * np.sin(4 * np.pi * (x_obs + noise)) + noise

# Convert to PyTorch tensors
x_train = torch.tensor(x_obs.reshape(-1, 1), dtype=torch.float32)
y_train = torch.tensor(y_obs.reshape(-1, 1), dtype=torch.float32)

# Define true function for visualization
x_true = np.linspace(-0.5, 1.5, 1000)
y_true = x_true + 0.3 * np.sin(2 * np.pi * x_true) + 0.3 * np.sin(4 * np.pi * x_true)

# 2. Define Deep Bayesian Neural Network
# ------------------------------
class DeepBNN(PyroModule):
    def __init__(self, in_dim=1, out_dim=1, hid_dim=10, n_hid_layers=5, prior_scale=5.):
        super().__init__()
        self.activation = nn.Tanh()  # Activation function
        self.layers = nn.ModuleList()

        # Define hidden layers
        prev_dim = in_dim
        for layer_idx in range(n_hid_layers):
            layer = PyroModule[nn.Linear](prev_dim, hid_dim)
            
            # Use dictionary assignment to ensure unique sample names
            layer.__setattr__(f"weight_{layer_idx}", PyroSample(
                dist.Normal(0, prior_scale).expand(layer.weight.shape).to_event(2)
            ))

            layer.__setattr__(f"bias_{layer_idx}", PyroSample(
                dist.Normal(0, prior_scale).expand(layer.bias.shape).to_event(1)
            ))

            self.layers.append(layer)
            prev_dim = hid_dim  # Update input dimension for next layer

        # Define output layer
        self.out_layer = PyroModule[nn.Linear](prev_dim, out_dim)
        self.out_layer.__setattr__("out_weight", PyroSample(
            dist.Normal(0, prior_scale).expand(self.out_layer.weight.shape).to_event(2)
        ))
        
        self.out_layer.__setattr__("out_bias", PyroSample(
            dist.Normal(0, prior_scale).expand(self.out_layer.bias.shape).to_event(1)
        ))

    def forward(self, x, y=None):
        # Compute the mean of the output distribution
        for layer in self.layers:
            x = self.activation(layer(x))
        mu = self.out_layer(x)

        # Sample the response noise
        sigma = pyro.sample("sigma", dist.Gamma(0.5, 1.0))

        # Sample the response
        y_pred = pyro.sample("y_pred", dist.Normal(mu, sigma), obs=y)
        return y_pred

# 3.Define model
# ------------------------------
model = DeepBNN(hid_dim=10, n_hid_layers=5, prior_scale=5.)

# Define MCMC sampler
nuts_kernel = NUTS(model, jit_compile=False)
mcmc = MCMC(nuts_kernel, num_samples=50)
mcmc.run(x_train, y_train)

# 4. Compute Predictive Distribution
# ------------------------------
predictive = Predictive(model=model, posterior_samples=mcmc.get_samples())
x_test = torch.linspace(-0.5, 1.5, 300).reshape(-1, 1)
preds = predictive(x_test)

# 5. Visualization
# ------------------------------
def plot_predictions(preds):

    y_pred = preds["y_pred"].mean(dim=0).detach().numpy().squeeze()
    y_std = preds["y_pred"].std(dim=0).detach().numpy().squeeze()

    x_test_np = x_test.squeeze().numpy()

    xlims = [-0.5, 1.5]
    ylims = [-1.5, 2.5]

    fig, ax = plt.subplots(figsize=(10, 5))
    plt.xlim(xlims)
    plt.ylim(ylims)
    plt.xlabel("X", fontsize=30)
    plt.ylabel("Y", fontsize=30)

    ax.plot(x_true, y_true, 'b-', linewidth=3, label="True function")
    ax.plot(x_obs, y_obs, 'ko', markersize=4, label="Observations")
    ax.plot(x_test_np, y_pred, '-', linewidth=3, color="#408765", label="Predictive mean")

    ax.fill_between(x_test_np, y_pred - 2 * y_std, y_pred + 2 * y_std, alpha=0.6, color='#86cfac', zorder=5)

    plt.legend(loc=4, fontsize=15, frameon=False)
    plt.show()

plot_predictions(preds)
