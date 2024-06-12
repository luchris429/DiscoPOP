import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import seaborn as sn
from bar_plot_alpaca_eval import configure_plotting_sn_params


# plt.rcParams["font.family"] = "Times New Roman"
# SCALE = 13
SCALE = 10.5
# SCALE = 20
# SCALE = 8
# HEIGHT_SCALE =0.8
# HEIGHT_SCALE =0.5
HEIGHT_SCALE = 0.8
LEGEND_Y_CORD = -0.75 * (HEIGHT_SCALE / 2.0)
SUBPLOT_ADJUST = 1 / HEIGHT_SCALE  # -(0.05 + LEGEND_Y_CORD)
LEGEND_X_CORD = 0.45
PLOT_FROM_CACHE = False
PLOT_SAFTEY_MARGIN = 1.25
MODEL_NAME_MAP = {}

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


plt.figure(figsize=(SCALE, int(HEIGHT_SCALE * SCALE)))


# Read DataFrame
df = pd.read_csv("./scripts/plotting/imdb_results_2.csv")
defined_betas = [0.025, 0.05, 0.1, 0.25, 0.5, 1.0]

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
    "padll": other_colors[2],
    "aqfl": other_colors[4],
    "cell": other_colors[4],
    "lrml": other_colors[6],
    "pfl": other_colors[6],
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

# Filter the DataFrame to include only rows with beta values in defined_betas
df_filtered = df[df["beta"].isin(defined_betas)]

# Group by 'loss' and 'beta' and calculate mean and std of 'kl_divergence' and 'reward'
mean_df = (
    df_filtered.groupby(["loss", "beta"])
    .agg({"kl_divergence": ["mean", "std"], "reward": ["mean", "std"]})
    .reset_index()
)

# Flatten the MultiIndex columns
mean_df.columns = [
    "loss",
    "beta",
    "kl_divergence_mean",
    "kl_divergence_std",
    "reward_mean",
    "reward_std",
]

for loss in ["dpo", "lrml"]:
    loss_df = mean_df[mean_df["loss"] == loss]
    plt.errorbar(
        loss_df["kl_divergence_mean"],
        loss_df["reward_mean"],
        # xerr=row['kl_divergence_std'],
        # yerr=row['reward_std'],
        label=loss_names[loss],
        color=loss_colors[loss],
        linestyle=":",
        marker="o",
        ms=9,
        lw=2.5,
    )

    # Annotate each point with the corresponding beta value
    # Annotate each point with the corresponding beta value
    for i, row in loss_df.iterrows():
        if loss == "lrml":
            offset_x, offset_y = -0.001, 0.001  # Adjust these values as needed
            ha, va = "right", "bottom"
        else:
            offset_x, offset_y = 0.005, -0.001  # Adjust these values as needed
            ha, va = "left", "top"

        plt.text(
            row["kl_divergence_mean"] + offset_x,
            row["reward_mean"] + offset_y,
            r"$\boldsymbol{\beta} =$" + f"{row['beta']}",
            fontsize=18,
            ha=ha,
            va=va,
            color=loss_colors[loss],
        )

# Add labels and title
plt.xlabel("KL Divergence")
plt.ylabel("Reward")
plt.title("IMDb Positive Text Generation: DPO vs LRML")
plt.legend(
    loc="lower center",
    bbox_to_anchor=(LEGEND_X_CORD, LEGEND_Y_CORD),
    ncol=2,
    fancybox=True,
    shadow=True,
)
plt.xlim(0, 1.5)
plt.savefig("./plots/imdb_dpo_lrml.pdf", bbox_inches="tight")
plt.savefig("./plots/imdb_dpo_lrml.png", bbox_inches="tight")
print("./plots/imdb_dpo_lrml.png")


plt.figure(figsize=(SCALE, int(HEIGHT_SCALE * SCALE)))

mean_df.columns = [
    "loss",
    "beta",
    "kl_divergence_mean",
    "kl_divergence_std",
    "reward_mean",
    "reward_std",
]

for loss in ["hinge", "lrml"]:
    loss_df = mean_df[mean_df["loss"] == loss]
    plt.errorbar(
        loss_df["kl_divergence_mean"],
        loss_df["reward_mean"],
        # xerr=row['kl_divergence_std'],
        # yerr=row['reward_std'],
        label=loss_names[loss],
        color=loss_colors[loss],
        linestyle=":",
        marker="o",
        ms=9,
        lw=2.5,
    )

    # Annotate each point with the corresponding beta value
    # Annotate each point with the corresponding beta value
    for i, row in loss_df.iterrows():
        if loss == "lrml":
            offset_x, offset_y = -0.001, 0.001  # Adjust these values as needed
            ha, va = "right", "bottom"
        else:
            offset_x, offset_y = 0.005, -0.001  # Adjust these values as needed
            ha, va = "left", "top"

        plt.text(
            row["kl_divergence_mean"] + offset_x,
            row["reward_mean"] + offset_y,
            r"$\boldsymbol{\beta} =$" + f"{row['beta']}",
            fontsize=18,
            ha=ha,
            va=va,
            color=loss_colors[loss],
        )

# Add labels and title
plt.xlabel("KL Divergence")
plt.ylabel("Reward")
plt.title("IMDb Positive Text Generation: SLiC vs LRML")
plt.legend(
    loc="lower center",
    bbox_to_anchor=(LEGEND_X_CORD, LEGEND_Y_CORD),
    ncol=2,
    fancybox=True,
    shadow=True,
)
plt.xlim(0, 1.5)
plt.savefig("./plots/imdb_slic_lrml.pdf", bbox_inches="tight")
plt.savefig("./plots/imdb_slic_lrml.png", bbox_inches="tight")
print("./plots/imdb_slic_lrml.png")


plt.figure(figsize=(SCALE, int(HEIGHT_SCALE * SCALE)))

for loss in ["dpo", "padll"]:
    loss_df = mean_df[mean_df["loss"] == loss]
    plt.errorbar(
        loss_df["kl_divergence_mean"],
        loss_df["reward_mean"],
        # xerr=row['kl_divergence_std'],
        # yerr=row['reward_std'],
        label=loss_names[loss],
        color=loss_colors[loss],
        linestyle=":",
        marker="o",
        ms=9,
        lw=2.5,
    )

    # Annotate each point with the corresponding beta value
    # Annotate each point with the corresponding beta value
    for i, row in loss_df.iterrows():
        if loss == "padll":
            offset_x, offset_y = -0.001, 0.001  # Adjust these values as needed
            ha, va = "right", "bottom"
        else:
            offset_x, offset_y = 0.005, -0.001  # Adjust these values as needed
            ha, va = "left", "top"

        plt.text(
            row["kl_divergence_mean"] + offset_x,
            row["reward_mean"] + offset_y,
            r"$\boldsymbol{\beta} =$" + f"{row['beta']}",
            fontsize=18,
            ha=ha,
            va=va,
            color=loss_colors[loss],
        )

# Add labels and title
plt.xlabel("KL Divergence")
plt.ylabel("Reward")
plt.title("IMDb Positive Text Generation: DPO vs PADLL")
plt.legend(
    loc="lower center",
    bbox_to_anchor=(LEGEND_X_CORD, LEGEND_Y_CORD),
    ncol=2,
    fancybox=True,
    shadow=True,
)
plt.xlim(0, 1.5)
plt.savefig("./plots/imdb_dpo_padll.pdf", bbox_inches="tight")
plt.savefig("./plots/imdb_dpo_padll.png", bbox_inches="tight")
print("./plots/imdb_dpo_padll.png")


plt.figure(figsize=(SCALE, int(HEIGHT_SCALE * SCALE)))

for loss in ["hinge", "padll"]:
    loss_df = mean_df[mean_df["loss"] == loss]
    plt.errorbar(
        loss_df["kl_divergence_mean"],
        loss_df["reward_mean"],
        # xerr=row['kl_divergence_std'],
        # yerr=row['reward_std'],
        label=loss_names[loss],
        color=loss_colors[loss],
        linestyle=":",
        marker="o",
        ms=9,
        lw=2.5,
    )

    # Annotate each point with the corresponding beta value
    # Annotate each point with the corresponding beta value
    for i, row in loss_df.iterrows():
        if loss == "padll":
            offset_x, offset_y = -0.001, 0.001  # Adjust these values as needed
            ha, va = "right", "bottom"
        else:
            offset_x, offset_y = 0.005, -0.001  # Adjust these values as needed
            ha, va = "left", "top"

        plt.text(
            row["kl_divergence_mean"] + offset_x,
            row["reward_mean"] + offset_y,
            r"$\boldsymbol{\beta} =$" + f"{row['beta']}",
            fontsize=18,
            ha=ha,
            va=va,
            color=loss_colors[loss],
        )

# Add labels and title
plt.xlabel("KL Divergence")
plt.ylabel("Reward")
plt.title("IMDb Positive Text Generation: SLiC vs PADLL")
plt.legend(
    loc="lower center",
    bbox_to_anchor=(LEGEND_X_CORD, LEGEND_Y_CORD),
    ncol=2,
    fancybox=True,
    shadow=True,
)
plt.xlim(0, 1.5)
plt.savefig("./plots/imdb_slic_padll.pdf", bbox_inches="tight")
plt.savefig("./plots/imdb_slic_padll.png", bbox_inches="tight")
print("./plots/imdb_slic_padll.png")


plt.figure(figsize=(SCALE, int(HEIGHT_SCALE * SCALE)))

for loss in ["dpo", "aqfl"]:
    loss_df = mean_df[mean_df["loss"] == loss]
    plt.errorbar(
        loss_df["kl_divergence_mean"],
        loss_df["reward_mean"],
        # xerr=row['kl_divergence_std'],
        # yerr=row['reward_std'],
        label=loss_names[loss],
        color=loss_colors[loss],
        linestyle=":",
        marker="o",
        ms=9,
        lw=2.5,
    )

    # Annotate each point with the corresponding beta value
    # Annotate each point with the corresponding beta value
    for i, row in loss_df.iterrows():
        if loss == "aqfl":
            offset_x, offset_y = -0.001, 0.001  # Adjust these values as needed
            ha, va = "right", "bottom"
        else:
            offset_x, offset_y = 0.005, -0.001  # Adjust these values as needed
            ha, va = "left", "top"

        plt.text(
            row["kl_divergence_mean"] + offset_x,
            row["reward_mean"] + offset_y,
            r"$\boldsymbol{\beta} =$" + f"{row['beta']}",
            fontsize=18,
            ha=ha,
            va=va,
            color=loss_colors[loss],
        )

# Add labels and title
plt.xlabel("KL Divergence")
plt.ylabel("Reward")
plt.title("IMDb Positive Text Generation: DPO vs AQFL")
plt.legend(
    loc="lower center",
    bbox_to_anchor=(LEGEND_X_CORD, LEGEND_Y_CORD),
    ncol=2,
    fancybox=True,
    shadow=True,
)
plt.xlim(0, 1.5)
plt.savefig("./plots/imdb_dpo_aqfl.pdf", bbox_inches="tight")
plt.savefig("./plots/imdb_dpo_aqfl.png", bbox_inches="tight")
print("./plots/imdb_dpo_aqfl.png")


plt.figure(figsize=(SCALE, int(HEIGHT_SCALE * SCALE)))

for loss in ["hinge", "aqfl"]:
    loss_df = mean_df[mean_df["loss"] == loss]
    plt.errorbar(
        loss_df["kl_divergence_mean"],
        loss_df["reward_mean"],
        # xerr=row['kl_divergence_std'],
        # yerr=row['reward_std'],
        label=loss_names[loss],
        color=loss_colors[loss],
        linestyle=":",
        marker="o",
        ms=9,
        lw=2.5,
    )

    # Annotate each point with the corresponding beta value
    # Annotate each point with the corresponding beta value
    for i, row in loss_df.iterrows():
        if loss == "aqfl":
            offset_x, offset_y = -0.001, 0.001  # Adjust these values as needed
            ha, va = "right", "bottom"
        else:
            offset_x, offset_y = 0.005, -0.001  # Adjust these values as needed
            ha, va = "left", "top"

        plt.text(
            row["kl_divergence_mean"] + offset_x,
            row["reward_mean"] + offset_y,
            r"$\boldsymbol{\beta} =$" + f"{row['beta']}",
            fontsize=18,
            ha=ha,
            va=va,
            color=loss_colors[loss],
        )

# Add labels and title
plt.xlabel("KL Divergence")
plt.ylabel("Reward")
plt.title("IMDb Positive Text Generation: SLiC vs AQFL")
plt.legend(
    loc="lower center",
    bbox_to_anchor=(LEGEND_X_CORD, LEGEND_Y_CORD),
    ncol=2,
    fancybox=True,
    shadow=True,
)
plt.xlim(0, 1.5)
plt.savefig("./plots/imdb_slic_aqfl.pdf", bbox_inches="tight")
plt.savefig("./plots/imdb_slic_aqfl.png", bbox_inches="tight")
print("./plots/imdb_slic_aqfl.png")


# plt.rcParams["font.family"] = "Times New Roman"
# SCALE = 13
SCALE = 10.5
# SCALE = 20
# SCALE = 8
# HEIGHT_SCALE =0.8
# HEIGHT_SCALE =0.5
HEIGHT_SCALE = 0.8
LEGEND_Y_CORD = -(HEIGHT_SCALE / 2.0)
SUBPLOT_ADJUST = 1 / HEIGHT_SCALE  # -(0.05 + LEGEND_Y_CORD)
LEGEND_X_CORD = 0.45
PLOT_FROM_CACHE = False
PLOT_SAFTEY_MARGIN = 1.25
MODEL_NAME_MAP = {}


plt.figure(figsize=(SCALE, int(HEIGHT_SCALE * SCALE)))

for loss in [
    "dpo",
    "hinge",
    "aqfl",
    "padll",
    "lrml",
]:
    loss_df = mean_df[mean_df["loss"] == loss]
    plt.errorbar(
        loss_df["kl_divergence_mean"],
        loss_df["reward_mean"],
        # xerr=row['kl_divergence_std'],
        # yerr=row['reward_std'],
        label=loss_names[loss],
        color=loss_colors[loss],
        linestyle=":",
        marker="o",
        ms=9,
        lw=2.5,
    )


# Add labels and title
plt.xlabel("KL Divergence")
plt.ylabel("Reward")
plt.title("IMDb Positive Text Generation")
plt.legend(
    loc="lower center",
    bbox_to_anchor=(LEGEND_X_CORD, LEGEND_Y_CORD),
    ncol=4,
    fancybox=True,
    shadow=True,
)
plt.xlim(0, 1.5)
plt.savefig("./plots/imdb_all.pdf", bbox_inches="tight")
plt.savefig("./plots/imdb_all.png", bbox_inches="tight")
print("./plots/imdb_all.png")


# # Define a list of markers to use
# # SCALE = 13
# SCALE = 10.5
# # SCALE = 20
# # SCALE = 8
# # HEIGHT_SCALE =0.8
# # HEIGHT_SCALE =0.5
# HEIGHT_SCALE =0.8
# LEGEND_Y_CORD = - 0.75* (HEIGHT_SCALE / 2.0)
# SUBPLOT_ADJUST = 1 / HEIGHT_SCALE  # -(0.05 + LEGEND_Y_CORD)
# LEGEND_X_CORD = 0.45
# PLOT_FROM_CACHE = False
# PLOT_SAFTEY_MARGIN = 1.25
# MODEL_NAME_MAP = {}

# sn = configure_plotting_sn_params(sn, SCALE, HEIGHT_SCALE)
# plt.gcf().subplots_adjust(bottom=0.1, left=0.2, top=0.95)

# # import matplotlib.pyplot as plt
# # import numpy as np

# # Data for plotting
# # model_names = ['AlphaCode', 'Incoder', 'CodeGeex', 'CodeGeex-Mono', 'PaLM Coder',
# #             'Codex',
# # human_eval_scores = [17.1, 15.2, 17.6, 26.9, 32.9, 38.6, 47.0, 67.7, 65.8, 87.7]
# from matplotlib import cm
# import seaborn as sns


# # Set the seaborn style
# sns.set_style("whitegrid")


# markers = ['o', 's', '^', 'D', '*', 'p', 'h', 'p', '*', 'h']

# plt.figure(figsize=(SCALE, int(HEIGHT_SCALE * SCALE)))

# # Plot lines for each loss method
# for loss in ["aqfl", "padll", "lrml", "dpo", "hinge"]:
#     loss_df = mean_df[mean_df['loss'] == loss]
#     plt.plot(loss_df['kl_divergence_mean'],
#              loss_df['reward_mean'],
#              label=loss_names[loss],
#              color=loss_colors[loss],
#              linestyle=":",
#              lw=2.5)

# # Plot markers for each beta
# used_markers = {}
# for loss in ["aqfl", "padll", "lrml", "dpo", "hinge"]:
#     loss_df = mean_df[mean_df['loss'] == loss]
#     for i, (index, row) in enumerate(loss_df.iterrows()):
#         marker = markers[i % len(markers)]
#         plt.scatter(row['kl_divergence_mean'],
#                     row['reward_mean'],
#                     color=loss_colors[loss],
#                     marker=marker, s=100)
#         # Add to the used_markers dict for legend
#         if row['beta'] not in used_markers:
#             used_markers[row['beta']] = marker

# # Add labels and title
# plt.xlabel('KL Divergence')
# plt.ylabel('Reward')
# plt.title('IMDb Positive Text Generation: All')

# # Create legends
# # Legend for colors (methods)
# first_legend = plt.legend(loc="lower center", bbox_to_anchor=(
#             LEGEND_X_CORD, LEGEND_Y_CORD), ncol=3, fancybox=True, shadow=True)

# # Legend for markers (betas)
# marker_handles = [plt.Line2D([0], [0], marker=m, color='w', label=r'$\boldsymbol{\beta}=$' + f'{b}',
#                              markerfacecolor='black', markersize=9)
#                   for b, m in used_markers.items()]
# plt.legend(handles=marker_handles, title=r'$\boldsymbol\beta$ Values', loc='lower right', fancybox=True, shadow=True)

# # Add the first legend back
# plt.gca().add_artist(first_legend)

# plt.xlim(0, 1.5)
# plt.savefig(f'./plots/imdb_all.pdf')
# plt.savefig(f'./plots/imdb_all.png')
# print(f'./plots/imdb_all.png')


# print('Done')
# plt.clf()
