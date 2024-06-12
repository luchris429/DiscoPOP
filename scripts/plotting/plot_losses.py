import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

import seaborn as sn
from bar_plot_alpaca_eval import configure_plotting_sn_params


# plt.rcParams["font.family"] = "Times New Roman"
# SCALE = 13
SCALE = 11
# SCALE = 20
# SCALE = 8
# HEIGHT_SCALE =0.8
# HEIGHT_SCALE =0.5
HEIGHT_SCALE = 0.8
LEGEND_Y_CORD = -1.2 * (HEIGHT_SCALE / 2.0)
SUBPLOT_ADJUST = 1 / HEIGHT_SCALE  # -(0.05 + LEGEND_Y_CORD)
LEGEND_X_CORD = 0.45
PLOT_FROM_CACHE = False
PLOT_SAFTEY_MARGIN = 1.25
MODEL_NAME_MAP = {}

dpo_hinge_colors = plt.cm.Blues(np.linspace(0.5, 0.9, 2))  # Shades of blue for dpo and hinge
other_colors = plt.cm.Reds(np.linspace(0.2, 1, 7))  # Shades of red for other losses

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

sn = configure_plotting_sn_params(sn, SCALE, HEIGHT_SCALE)
plt.gcf().subplots_adjust(bottom=0.40, left=0.2, top=0.95)

# import matplotlib.pyplot as plt
# import numpy as np

# Data for plotting
# model_names = ['AlphaCode', 'Incoder', 'CodeGeex', 'CodeGeex-Mono', 'PaLM Coder',
#             'Codex',
# human_eval_scores = [17.1, 15.2, 17.6, 26.9, 32.9, 38.6, 47.0, 67.7, 65.8, 87.7]

import seaborn as sns


# Set the seaborn style
sns.set_style("whitegrid")


beta = 0.05


def performance_adaptive_decay_logistic_loss(logits: torch.Tensor) -> torch.FloatTensor:
    base_decay = 0.9
    mismatch_penalty = 0.5  # Penalty decay for mismatched choices
    mismatches = (logits < 0).float()  # Identify mismatches

    adaptive_decay = base_decay * (1 - mismatches * mismatch_penalty)
    weighted_losses = adaptive_decay * -F.logsigmoid(beta * logits)
    return weighted_losses


def logistic_log_loss(logits) -> torch.FloatTensor:
    losses = -F.logsigmoid(beta * logits)
    return losses


def ipo_loss(logits) -> torch.FloatTensor:
    losses = (logits - 1 / (2 * 0.1)) ** 2
    return losses


def hinge_loss(logits) -> torch.FloatTensor:
    losses = torch.relu(1 - beta * logits)
    return losses


def adaptive_quantile_loss(logits: torch.Tensor) -> torch.FloatTensor:
    percentile = 0.5  # Start with the median quantile
    moving_quantile_weight = 0.01  # Weight for updating the moving quantile

    moving_quantile = percentile + moving_quantile_weight * (torch.sigmoid(logits.mean()) - percentile)

    quantile_weights = torch.sigmoid(-beta * (logits - moving_quantile))

    logistic_losses = -F.logsigmoid(beta * logits)
    hinge_losses = torch.relu(1 - beta * logits)

    # Blend the logistic and hinge losses based on the dynamic quantile weight
    losses = quantile_weights * logistic_losses + (1 - quantile_weights) * hinge_losses
    return losses


def combined_exp_logistic_loss(logits: torch.Tensor) -> torch.FloatTensor:
    exp_losses = torch.exp(-beta * logits)
    log_losses = -F.logsigmoid(beta * logits)
    # Combine the losses with a tunable mixing coefficient
    alpha = 0.5
    losses = alpha * exp_losses + (1 - alpha) * log_losses
    return losses


def log_ratio_modulated_loss(logits: torch.Tensor) -> torch.FloatTensor:
    # Modulate the mixing coefficient based on the log ratio magnitudes
    log_ratio_modulation = torch.sigmoid(logits)
    logistic_component = -F.logsigmoid(beta * logits)
    exp_component = torch.exp(-beta * logits)
    # Blend between logistic and exponential component based on log ratio modulation
    losses = logistic_component * (1 - log_ratio_modulation) + exp_component * log_ratio_modulation
    return losses


def policy_focused_loss(logits: torch.Tensor) -> torch.FloatTensor:
    focus_scale = 2.0  # Scale to emphasize or de-emphasize based on the correctness of predictions
    is_correct = logits > 0

    logistic_losses = -F.logsigmoid(logits)
    hinge_losses = torch.relu(1 - logits)

    focused_loss = torch.where(
        is_correct,
        logistic_losses / focus_scale,  # De-emphasize correct predictions
        hinge_losses * focus_scale,  # Emphasize incorrect predictions
    )
    return focused_loss


def dynamic_blended_adaptive_quantile_loss(logits) -> torch.FloatTensor:
    import torch.nn.functional as F

    # Constants for the loss function
    starting_quantile = 0.5
    quantile_adapt_rate = 0.01
    temperature = 0.9
    dynamic_blend_rate = 1.0

    logits_variability = logits.var()

    # Calculate an adaptive quantile based on a moving target
    starting_quantile + quantile_adapt_rate * (torch.sigmoid(logits.mean()) - starting_quantile)

    # Calculate dynamic blending coefficient based on logits variability
    dynamic_blend_coeff = torch.sigmoid(logits_variability) * dynamic_blend_rate

    # Prepare components of the blended loss
    logistic_loss = -F.logsigmoid(beta * logits / temperature)
    exp_loss = torch.exp(beta * logits * temperature)

    # Blend the losses dynamically
    losses = dynamic_blend_coeff * logistic_loss + (1 - dynamic_blend_coeff) * exp_loss
    return losses


def adaptive_quantile_feedback_loss(logits) -> torch.FloatTensor:
    import torch.nn.functional as F

    quantile_update_rate = 0.05
    distance_scale = 0.1

    logits_std = logits.std()

    adaptive_quantile = logits_std * torch.sigmoid(-logits).mean()
    adaptive_quantile += quantile_update_rate * (torch.sigmoid(logits.mean()) - adaptive_quantile)

    distance_from_quantile = (logits - adaptive_quantile).abs()
    blend_rate = torch.sigmoid(distance_scale * distance_from_quantile)

    logistic_losses = -F.logsigmoid(beta * logits)
    hinge_losses = torch.relu(1 - beta * logits)

    losses = blend_rate * logistic_losses + (1 - blend_rate) * hinge_losses
    return losses


# Generate logit values from -10 to 10
logits = torch.arange(-10, 40, 0.01, dtype=torch.float)

# Calculate losses for both functions
loss1 = logistic_log_loss(logits)
loss2 = ipo_loss(logits)
loss3 = hinge_loss(logits)
loss4 = performance_adaptive_decay_logistic_loss(logits)
loss5 = adaptive_quantile_loss(logits)
loss6 = combined_exp_logistic_loss(logits)
loss7 = log_ratio_modulated_loss(logits)
loss8 = policy_focused_loss(logits)
loss9 = dynamic_blended_adaptive_quantile_loss(logits)
loss10 = adaptive_quantile_feedback_loss(logits)

# Define the colors for each loss function

# dpo_hinge_colors = plt.cm.Blues(np.linspace(0.5, 0.9, 2))  # Shades of blue for dpo and hinge
# other_colors = plt.cm.Reds(np.linspace(0.5, 0.9, 3))  # Shades of red for other losses

# loss_colors = {
#     "dpo": dpo_hinge_colors[0],
#     "hinge": dpo_hinge_colors[1],
#     "aql": other_colors[0],
#     "padll": other_colors[1],
#     "cell": "gray",  # Not plotted
#     "lrml": other_colors[2],
#     "pfl": "gray"    # Not plotted
# }


plt.figure(figsize=(SCALE, int(HEIGHT_SCALE * SCALE)))

# Plot the results
# plt.figure()
# plt.plot(logits, loss1, label='DPO', color= dpo_hinge_colors[0])
# #plt.plot(logits, loss2, label='IPO')
# plt.plot(logits, loss3, label='SLiC', color= dpo_hinge_colors[1])
# plt.plot(logits, loss5, label='Adaptive Quantile Loss', color= other_colors[0],)
# plt.plot(logits, loss4, label='Performance Adaptive Decay Log Loss', color= other_colors[1],)
# plt.plot(logits, loss7, label='Log Ratio Modulated Loss', color= other_colors[2],)
# #plt.plot(logits, loss9, label='New', color= other_colors[2],)
# #plt.plot(logits, loss8, label='Policy Focused Loss')

# plt.plot(logits, loss5, label='AQL')
plt.plot(logits, loss10, label="AQFL", color=loss_colors["aqfl"])
plt.plot(logits, loss4, label="PADLL", color=loss_colors["padll"])
plt.plot(logits, loss7, label="LRML", color=loss_colors["lrml"])
plt.plot(logits, loss1, label="DPO", color=loss_colors["dpo"])
# plt.plot(logits, loss2, label='IPO')
plt.plot(logits, loss3, label="SLiC", color=loss_colors["hinge"])
# plt.plot(logits, loss9, label='New', color= other_colors[2],)
# plt.plot(logits, loss8, label='Policy Focused Loss')
plt.xlabel(r"Logits $\rho$")
plt.ylabel(r"Loss $f(\rho)$")
plt.title("Discovered Objective Functions")
plt.legend(
    loc="lower center",
    bbox_to_anchor=(LEGEND_X_CORD, LEGEND_Y_CORD),
    ncol=2,
    fancybox=True,
    shadow=True,
)
plt.tight_layout()
plt.savefig("./plots/losses_function.png", bbox_inches="tight")
plt.savefig("./plots/losses_function.pdf", bbox_inches="tight")
print("./plots/losses_function.png")
# plt.grid(True)
# plt.savefig("loss_new.pdf",bbox_inches='tight')
# plt.show()
print("Done")
plt.clf()


# Generate logit values from -10 to 10
logits = torch.arange(-10, 40, 0.01, dtype=torch.float, requires_grad=True)

# Calculate losses for each function
loss_functions = [
    ("aqfl", adaptive_quantile_feedback_loss),
    ("padll", performance_adaptive_decay_logistic_loss),
    ("lrml", log_ratio_modulated_loss),
    ("dpo", logistic_log_loss),
    ("hinge", hinge_loss),
]

# Colors for the gradients
dpo_hinge_colors = plt.cm.Blues(np.linspace(0.5, 0.9, 2))  # Shades of blue for DPO and SLiC
other_colors = plt.cm.Reds(np.linspace(0.5, 0.9, 3))  # Shades of red for other losses

# Plot the gradients
plt.figure(figsize=(SCALE, int(HEIGHT_SCALE * SCALE)))

# Define color mapping for each loss function
# color_mapping = {
#     'DPO': dpo_hinge_colors[0],
#     'SLiC': dpo_hinge_colors[1],
#     #'Adaptive Quantile Loss': other_colors[0],
#     'PADLL': other_colors[0],
#     'LRML': other_colors[2]
# }

for name, loss_fn in loss_functions:
    logits.grad = None  # Clear any existing gradients
    loss = loss_fn(logits)
    loss.backward(torch.ones_like(logits))  # Compute the gradients
    grad = logits.grad.clone().detach().numpy()  # Extract the gradients
    # plt.plot(logits.detach().numpy(), grad, label=name, color=color_mapping[name])
    plt.plot(logits.detach().numpy(), grad, label=loss_names[name], color=loss_colors[name])

plt.xlabel(r"Logits $\rho$")
plt.ylabel(r"Gradient $f'(\rho)$")
plt.title(r"Gradient of Objective Functions")
plt.legend(
    loc="lower center",
    bbox_to_anchor=(LEGEND_X_CORD, LEGEND_Y_CORD),
    ncol=2,
    fancybox=True,
    shadow=True,
)
plt.tight_layout()
plt.savefig("./plots/losses_gradient.png", bbox_inches="tight")
plt.savefig("./plots/losses_gradient.pdf", bbox_inches="tight")
print("./plots/losses_gradient.png")
# plt.grid(True)
# plt.savefig("loss_new.pdf",bbox_inches='tight')
# plt.show()
print("Done")
plt.clf()
