import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

import seaborn as sn
from bar_plot_alpaca_eval import configure_plotting_sn_params


# plt.rcParams["font.family"] = "Times New Roman"
# SCALE = 13
SCALE = 20
# SCALE = 20
# SCALE = 8
# HEIGHT_SCALE =0.8
# HEIGHT_SCALE =0.5
HEIGHT_SCALE = 1
LEGEND_Y_CORD = -0.75 * (HEIGHT_SCALE / 2.0)
SUBPLOT_ADJUST = 1 / HEIGHT_SCALE  # -(0.05 + LEGEND_Y_CORD)
LEGEND_X_CORD = 0.45
PLOT_FROM_CACHE = False
PLOT_SAFTEY_MARGIN = 1.25
MODEL_NAME_MAP = {}

sn = configure_plotting_sn_params(sn, SCALE, HEIGHT_SCALE)

# import matplotlib.pyplot as plt
# import numpy as np

# Data for plotting
# model_names = ['AlphaCode', 'Incoder', 'CodeGeex', 'CodeGeex-Mono', 'PaLM Coder',
#             'Codex',
# human_eval_scores = [17.1, 15.2, 17.6, 26.9, 32.9, 38.6, 47.0, 67.7, 65.8, 87.7]

import seaborn as sns


# Set the seaborn style
sns.set_style("whitegrid")


# Define colors for each loss
num_dpo_hinge = 2
num_other = 7  # Number of other loss types to plot

dpo_hinge_colors = plt.cm.Blues(np.linspace(0.5, 0.9, num_dpo_hinge))  # Shades of blue for dpo and hinge
other_colors = plt.cm.Reds(np.linspace(0.2, 1, num_other))  # Shades of red for other losses

loss_colors = {
    "dpo": dpo_hinge_colors[0],
    "hinge": dpo_hinge_colors[1],
    "dbaql": other_colors[0],
    "aql": other_colors[1],
    "padll": other_colors[4],
    "aqfl": other_colors[2],
    "cell": other_colors[4],
    "lrml": other_colors[6],
    "pfl": other_colors[5],
}

loss_names = {
    "dpo": "DPO",
    "hinge": "SLiC",
    "dbaql": "DBAQL",
    "aql": "AQL",
    "padll": "PADLL",
    "aqfl": "AQFL",
    "cell": "CELL",
    "lrml": "LRML",
    "pfl": "PFL",
}


def log_ratio_modulated_loss(logits_in: torch.Tensor, beta, tau=0.05) -> torch.FloatTensor:
    logits = beta * logits_in
    # Modulate the mixing coefficient based on the log ratio magnitudes
    log_ratio_modulation = torch.sigmoid(logits / tau)
    logistic_component = -F.logsigmoid(logits)
    exp_component = torch.exp(-logits)
    # Blend between logistic and exponential component based on log ratio modulation
    losses = logistic_component * (1 - log_ratio_modulation) + exp_component * log_ratio_modulation
    return losses


def logistic_log_loss(logits_in, beta) -> torch.FloatTensor:
    logits = beta * logits_in
    losses = -F.logsigmoid(logits)
    return losses


# Generate logit values from -10 to 10
logits = torch.arange(-20, 40, 0.01, dtype=torch.float)

# Calculate losses for different beta values
betas = [0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5]
losses = [log_ratio_modulated_loss(logits, beta) for beta in betas]
losses_dpo = [logistic_log_loss(logits, beta) for beta in betas]


# Create a 3x3 subfigure plot
fig, axs = plt.subplots(3, 3, figsize=(SCALE, int(HEIGHT_SCALE * SCALE)))
fig.suptitle(r"Comparison of DPO vs LRML for different $\boldsymbol{\beta}$ Values")

# Plot each loss in the grid
for i, (ax, beta, loss, loss_dpo) in enumerate(zip(axs.flat, betas, losses, losses_dpo)):
    ax.plot(logits, loss_dpo, label="DPO", color=loss_colors["dpo"])
    ax.plot(logits, loss, label="LRML", color=loss_colors["lrml"])
    ax.set_xlabel(r"Logits $\boldsymbol{\rho}$")
    ax.set_ylabel(r"Loss $f(\boldsymbol{\rho})$")
    ax.set_title(r"$\boldsymbol{\beta} =$" + f"{beta}")
    ax.legend()

# Add labels and title
plt.savefig("./plots/losses_sweep.pdf", bbox_inches="tight")
plt.savefig("./plots/losses_sweep.png", bbox_inches="tight")
print("./plots/losses_sweep.png")


print("Done")
plt.clf()


##################### Gradients

logits = torch.arange(-20, 40, 0.01, dtype=torch.float, requires_grad=True)

# Calculate losses for different beta values
betas = [0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5]

losses_gradients = []
losses_dpo_gradients = []

for beta in betas:
    logits.grad = None  # Reset gradients
    loss = log_ratio_modulated_loss(logits, beta).sum()
    loss.backward()
    losses_gradients.append(logits.grad.clone().detach().numpy())

    logits.grad = None  # Reset gradients
    loss_dpo = logistic_log_loss(logits, beta).sum()
    loss_dpo.backward()
    losses_dpo_gradients.append(logits.grad.clone().detach().numpy())

fig, axs = plt.subplots(3, 3, figsize=(SCALE, int(HEIGHT_SCALE * SCALE)))
fig.suptitle(r"Comparison of DPO vsLRML Gradients for Different $\boldsymbol{\beta}$ Values")

for i, (ax, beta, loss_grad, loss_dpo_grad) in enumerate(zip(axs.flat, betas, losses_gradients, losses_dpo_gradients)):
    ax.plot(logits.detach().numpy(), loss_dpo_grad, label="DPO", color=loss_colors["dpo"])
    ax.plot(logits.detach().numpy(), loss_grad, label="LRML", color=loss_colors["lrml"])
    ax.set_xlabel(r"Logits $\boldsymbol{\rho}$")
    ax.set_ylabel(r"Loss Gradient $\boldsymbol{\nabla}_{\rho}f(\boldsymbol{\rho})$")
    ax.set_title(r"$\boldsymbol{\beta} =$" + f"{beta}")
    ax.legend()

plt.tight_layout(rect=[0, 0, 1, 0.96])

plt.savefig("./plots/gradients_sweep.pdf", bbox_inches="tight")
plt.savefig("./plots/gradients_sweep.png", bbox_inches="tight")
print("./plots/gradients_sweep.png")

print("Done")
plt.clf()
