import pandas as pd


def configure_plotting_sn_params(sn, SCALE, HEIGHT_SCALE, use_autolayout=True):
    pd.set_option("mode.chained_assignment", None)
    sn.set(
        rc={
            "figure.figsize": (SCALE, int(HEIGHT_SCALE * SCALE)),
            "figure.autolayout": use_autolayout,
            "text.usetex": True,
            "text.latex.preamble": "\n".join(
                [
                    r"\usepackage{siunitx}",  # i need upright \micro symbols, but you need...
                    r"\sisetup{detect-all}",  # ...this to force siunitx to actually use your fonts
                    r"\usepackage{helvet}",  # set the normal font here
                    r"\usepackage{sansmath}",  # load up the sansmath so that math -> helvet
                    r"\usepackage{amsmath}",
                    r"\sansmath",  # <- tricky! -- gotta actually tell tex to use!
                ]
            ),
        }
    )
    sn.set(font_scale=2.0)
    sn.set_style(
        "white",
        {
            "font.family": "serif",
            "font.serif": "Times New Roman",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 14,
        },
    )
    sn.color_palette("colorblind")
    return sn


def plot_alaca_eval_bar_chart():
    import matplotlib.pyplot as plt

    import seaborn as sn

    # plt.rcParams["font.family"] = "Times New Roman"
    # SCALE = 13
    SCALE = 11
    # SCALE = 8
    # HEIGHT_SCALE =0.8
    # HEIGHT_SCALE =0.5
    HEIGHT_SCALE = 1.0
    1 / HEIGHT_SCALE  # -(0.05 + LEGEND_Y_CORD)

    sn = configure_plotting_sn_params(sn, SCALE, HEIGHT_SCALE)
    # plt.gcf().subplots_adjust(bottom=0.40, left=0.2, top=0.95)

    # import matplotlib.pyplot as plt
    # import numpy as np

    # Data for plotting
    # model_names = ['AlphaCode', 'Incoder', 'CodeGeex', 'CodeGeex-Mono', 'PaLM Coder',
    #             'Codex',
    # human_eval_scores = [17.1, 15.2, 17.6, 26.9, 32.9, 38.6, 47.0, 67.7, 65.8, 87.7]

    import seaborn as sns

    # Set the seaborn style
    sns.set_style("whitegrid")

    import numpy as np
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.patches import Patch

    import seaborn as sns

    # Define the evaluation scores
    scores = {
        "DPO": 65.29,
        "SLiC": 62.09,
        "DBQL": 61.56,
        "AQL": 63.57,
        "PADLL": 67.20,
        "AQFL": 63.38,
        "CELL": 63.96,
        "LRML - DiscoPOP": 70.83,
        "PFL": 63.00,
    }

    # Sort dictionary by values
    scores = dict(sorted(scores.items(), key=lambda item: item[1], reverse=False))

    model_names, eval_scores = [], []
    for model_name, score in scores.items():
        model_names.append(model_name)
        eval_scores.append(score)

    # Choose a color palette
    blue_palette = sns.light_palette("skyblue", reverse=True, n_colors=len(scores))

    # Convert Seaborn color palette to a Matplotlib colormap
    cmap = LinearSegmentedColormap.from_list("custom_cmap", blue_palette)

    # Set the seaborn style
    sns.set_style("whitegrid")

    # Define the figure and axis
    fig, ax = plt.subplots()

    # Identify baseline models
    baseline_models = ["DPO", "SLiC"]

    # Create the horizontal bar chart
    bars = ax.barh(model_names, eval_scores, color=blue_palette, edgecolor="grey")

    # Custom function to add gradient to the bars
    def gradient_bars(bars, cmap, vmin, vmax):
        grad = np.atleast_2d(np.linspace(0, 1, 256))
        ax = bars[0].axes
        lim = ax.get_xlim() + ax.get_ylim()
        for bar in bars:
            bar.set_zorder(1)
            bar.set_facecolor("none")
            x, y = bar.get_xy()
            w, h = bar.get_width(), bar.get_height()
            ax.imshow(
                grad,
                extent=[x, x + w, y, y + h],
                aspect="auto",
                zorder=0,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
            )
        ax.axis(lim)

    gradient_bars(bars, cmap, vmin=61, vmax=72)

    # Highlight baseline models with a different style
    for bar, model_name in zip(bars, model_names):
        if model_name in baseline_models:
            bar.set_edgecolor("black")
            bar.set_hatch("//")

    # Remove the spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    # Add horizontal grid lines only
    ax.xaxis.grid(True, linestyle="--", which="major", color="grey", alpha=0.25)
    ax.yaxis.grid(False)
    ax.set(xlim=(61, 72))

    # Add value labels to the bars
    def add_labels(bars):
        for bar in bars:
            width = bar.get_width()
            ax.annotate(
                f"{width:.2f}%",
                xy=(width, bar.get_y() + bar.get_height() / 2),
                xytext=(3, 0),  # 3 points horizontal offset
                textcoords="offset points",
                ha="left",
                va="center",
            )

    add_labels(bars)

    # Set labels and title
    plt.xlabel("Win Rate - LC (\%)")
    plt.title("Held Out Alpaca Eval Performance")

    # Create legend
    handles = [
        Patch(color="skyblue", label="Discovered"),
        Patch(facecolor="skyblue", edgecolor="black", hatch="//", label="Baselines"),
    ]
    ax.legend(handles=handles, title="Model Type", loc="lower right")

    # Set background color
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")

    # Adjust layout
    plt.tight_layout()
    plt.savefig("./plots/alpaca_eval_bar_horizontal.png")
    plt.savefig("./plots/alpaca_eval_bar_horizontal.pdf")
    print("./plots/alpaca_eval_bar_horizontal.png")
    plt.clf()


if __name__ == "__main__":
    plot_alaca_eval_bar_chart()
