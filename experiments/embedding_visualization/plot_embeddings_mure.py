"""
Plot a simple visualization of the embeddings
"""

import pandas as pd
from pathlib import Path
import numpy as np

import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

import tempfile
import os
import joblib

exp_dir = Path(__file__).absolute().parents[1]
n_sampled_points = 9
sampled_points = None

mem = joblib.Memory(location=os.path.join(tempfile.gettempdir(), "embedding_joblib"))


def compute_two_dim(ken=True):
    """Compute a 2D representation of the embeddings, for attributes in
    "all" or "categorical"
    """
    # Load the embeddings
    if ken:
        fname = exp_dir / "embedding_visualization/mure_ken_embeddings_yago3.parquet"
    else:
        fname = exp_dir / "embedding_visualization/mure_embeddings_yago3.parquet"
    df = pd.read_parquet(fname)
    # Avoid redundant embeddings
    df = df.groupby("County").first().reset_index()
    values = df["County"]
    embeddings = df.iloc[:, 1:].to_numpy()

    # l1 normalization of the embeddings
    embeddings = embeddings / np.abs(embeddings).sum(axis=0)

    # a low-dimensional representation to get an intuition
    pca = PCA(n_components=2)
    two_dim_data = pca.fit_transform(embeddings)
    values = values.str[1:-1].str.replace("_", " ").str.replace(" County", "")
    # In the numerical attributes we don't have the county names and
    # states with spaces. Filter these out
    mask = values.str.replace(", ", "aa").str.contains(" ")
    values = values[~mask].reset_index(drop=True)
    two_dim_data = two_dim_data[~mask]

    return values, two_dim_data


two_dim_embeddings = {
    True: mem.cache(compute_two_dim)(True),
    False: mem.cache(compute_two_dim)(False),
}


for ken in (True, False):
    ken_name = ""
    if ken:
        ken_name = "_ken"
    values, two_dim_data = two_dim_embeddings[ken]
    num_attributes = pd.read_parquet(
        exp_dir / "embedding_visualization/num_attributes_county.parquet"
    )
    # Make sure that we have the same values
    num_attributes = pd.merge(
        values.rename("County"), num_attributes, on="County", how="left"
    )

    state = values.str.split(", ").str[-1]
    num_attributes["state"] = state
    state_num = np.unique(state, return_inverse=True)[1]

    # Select a few data points: rejection sampling: choose them not to
    # close to one another
    np.random.seed(3)
    np.random.seed(5)
    if sampled_points is None:
        # First the most populated counties
        # (complicated dance because of NA)
        sampled_points = num_attributes["Pop"].fillna(0).argsort()[-4:]

        # The least populated county
        candidate = num_attributes["Pop"].fillna(1e6).argsort()[:1]
        sampled_points = np.append(sampled_points, candidate)
        min_accepted_distance = 1
        while len(sampled_points) < n_sampled_points:
            candidate = np.random.choice(len(values), 1)
            if candidate[0] in sampled_points:
                continue
            # Only keep if in already sampled states
            sampled_states = values[sampled_points].str.split(", ").str[-1]
            this_state = values[candidate[0]].split(", ")[-1]
            if not this_state in set(sampled_states):
                continue
            # No more than 3 in the same state
            if (sampled_states == this_state).sum() >= 3:
                continue
            # We want a weakly-populated county
            if not (
                num_attributes.iloc[candidate[0]]["Pop"]
                < num_attributes["Pop"].quantile(0.1)
            ):
                continue
            # Screen for distances on both figures
            distance_to_points = (
                two_dim_embeddings[True][1][candidate]
                - two_dim_embeddings[True][1][sampled_points]
            )
            if np.min(np.abs(distance_to_points)) < min_accepted_distance:
                min_accepted_distance *= 0.75
                continue
            distance_to_points = (
                two_dim_embeddings[False][1][candidate]
                - two_dim_embeddings[False][1][sampled_points]
            )
            if np.min(np.abs(distance_to_points)) < min_accepted_distance:
                min_accepted_distance *= 0.75
                continue
            sampled_points = np.append(sampled_points, candidate)

    # Then we plot it, adding the categories in the scatter plot:
    log_pop = np.log(num_attributes["Pop"])
    symbols_state = {v: s for v, s in zip(np.unique(sampled_states), "<>^vP")}

    f = plt.figure(figsize=(4, 3))
    ax = plt.axes([0.001, 0.001, 0.998, 0.998])

    # Plot data points not in selected states
    state_mask = ~num_attributes["state"].isin(sampled_states)
    ax.scatter(
        x=two_dim_data[state_mask, 0],
        y=two_dim_data[state_mask, 1],
        c=log_pop[state_mask],
        vmin=log_pop.min(),
        vmax=log_pop.max(),
        marker="o",
        alpha=0.15,
        s=10,
    )

    # Plot data points in selected states
    for this_state, symbol in symbols_state.items():
        state_mask = num_attributes["state"] == this_state
        ax.scatter(
            x=two_dim_data[state_mask, 0],
            y=two_dim_data[state_mask, 1],
            c=log_pop[state_mask],
            vmin=log_pop.min(),
            vmax=log_pop.max(),
            marker=symbol,
            alpha=0.35,
            s=15,
        )

    # Plot selected data points
    for i, idx in enumerate(sampled_points):
        this_x = two_dim_data[idx, 0]
        this_y = two_dim_data[idx, 1]

        ax.scatter(
            x=this_x,
            y=this_y,
            marker=symbols_state[state[idx]],
            c=log_pop[idx],
            vmin=log_pop.min(),
            vmax=log_pop.max(),
            edgecolor="k",
            s=15,
        )

        # The name of the county
        name = values[idx]
        # Break the line if too close to the border
        if this_x > np.percentile(two_dim_data[:, 0], 95) or this_x < np.percentile(
            two_dim_data[:, 0], 10
        ):
            name = ",\n".join([s.strip() for s in name.split(",")])

        # Clever positioning of the label
        ha = "center"
        if this_x < np.percentile(two_dim_data[:, 0], 15):
            ha = "right"
        if this_x > np.percentile(two_dim_data[:, 0], 85):
            ha = "left"

        va = "center"
        if ha == "center":
            va = "bottom"
        if this_y < np.percentile(two_dim_data[sampled_points, 1], 35):
            va = "top"
        if this_y > np.percentile(two_dim_data[sampled_points, 1], 66):
            va = "bottom"

        ax.text(
            x=this_x,
            y=this_y,
            s=name,
            fontsize=10,
            ha=ha,
            va=va,
        )

    plt.axis("off")

    # Set tight boundaries
    if ken:
        plt.xlim(-0.0043, 0.0073)
        plt.ylim(-0.0048, 0.0025)
    else:
        plt.xlim(-0.005, 0.0094)
        plt.ylim(-0.0028, 0.0027)

    ax.text(
        0.99,
        0.99,
        (
            "County embeddings\nusing MuRE + KEN"
            if ken
            else "County embeddings\nusing MuRE"
        ),
        ha="right",
        va="top",
        fontsize=11,
        transform=ax.transAxes,
    )

    cax = plt.axes([0.93, 0.2, 0.02, 0.4])
    colorbar = f.colorbar(plt.cm.ScalarMappable(cmap="viridis"), cax=cax)
    colorbar.set_ticks([])
    cax.set_ylabel("County population")

    f.savefig(
        exp_dir / f"embedding_visualization/embedding_{ken_name}.pdf",
        transparent=True,
    )
    f.savefig(exp_dir / f"embedding_visualization/embedding_{ken_name}.png", dpi=200)


# Save a table of the populations selected counties
tab = num_attributes.iloc[sampled_points].sort_values("Pop")[["County", "Pop"]]
open(exp_dir / "embedding_visualization/embedding_selected_counties.tex", "w").write(
    tab.to_latex()
)
