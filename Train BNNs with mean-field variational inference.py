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

# ------------------------------
# 2. Define Deep Bayesian Neural Network
# ------------------------------
class DeepBNN(PyroModule):
    def __init__(self, in_dim=1, out_dim=1, hid_dim=10, n_hid_layers=5, prior_scale=5.):
        super().__init__()
        self.activation = nn.Tanh()
        self.layers = nn.ModuleList()

        # Define hidden layers
        prev_dim = in_dim
        for layer_idx in range(n_hid_layers):
            layer = PyroModule[nn.Linear](prev_dim, hid_dim)
            
            # Assign PyroSample to ensure unique sample names
            setattr(layer, f"weight_{layer_idx}", PyroSample(
                dist.Normal(0, prior_scale).expand(layer.weight.shape).to_event(2)
            ))
            setattr(layer, f"bias_{layer_idx}", PyroSample(
                dist.Normal(0, prior_scale).expand(layer.bias.shape).to_event(1)
            ))

            self.layers.append(layer)
            prev_dim = hid_dim  # Update input dimension for next layer

        # Define output layer
        self.out_layer = PyroModule[nn.Linear](prev_dim, out_dim)
        setattr(self.out_layer, "out_weight", PyroSample(
            dist.Normal(0, prior_scale).expand(self.out_layer.weight.shape).to_event(2)
        ))
        setattr(self.out_layer, "out_bias", PyroSample(
            dist.Normal(0, prior_scale).expand(self.out_layer.bias.shape).to_event(1)
        ))

    def forward(self, x, y=None):
        # Compute the mean of the output distribution
        for layer in self.layers:
            x = self.activation(layer(x))
        mu = self.out_layer(x).squeeze(-1)  # (batch_size,)

        # Sample the response noise
        sigma = pyro.sample("sigma", dist.Gamma(0.5, 1.0))

        # Ensure `y` is not None before calling `.squeeze(-1)`
        y_obs = y.squeeze(-1) if y is not None else None

        # Use pyro.plate to ensure batch processing
        with pyro.plate("data", x.shape[0]):
            y_pred = pyro.sample("y_pred", dist.Normal(mu, sigma).to_event(1), obs=y_obs)
        
        return y_pred

# ------------------------------
# 3. Train BNN using Variational Inference (VI)
# ------------------------------
pyro.clear_param_store()

# Define model
model = DeepBNN(hid_dim=10, n_hid_layers=5, prior_scale=5.)
mean_field_guide = AutoDiagonalNormal(model)  # Variational distribution
optimizer = pyro.optim.Adam({"lr": 0.01})
svi = SVI(model, mean_field_guide, optimizer, loss=Trace_ELBO())

# Train model
num_epochs = 25000
progress_bar = trange(num_epochs)

for epoch in progress_bar:
    loss = svi.step(x_train, y_train)
    progress_bar.set_postfix(loss=f"{loss / x_train.shape[0]:.3f}")

# ------------------------------
# 4. Compute Predictive Distribution
# ------------------------------
predictive = Predictive(model=mean_field_guide, num_samples=500)
x_test = torch.linspace(-0.5, 1.5, 300).reshape(-1, 1)
preds = predictive(x_test)

# ------------------------------
# 5. Visualization
# ------------------------------
def plot_predictions(model, guide, x_test):

    predictive = Predictive(model, guide=guide, num_samples=500)
    samples = predictive(x_test)

    y_pred = model(x_test).mean.detach().numpy().squeeze()
    y_std = model(x_test).stddev.detach().numpy().squeeze()

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
    
plot_predictions(model, mean_field_guide, x_test)

