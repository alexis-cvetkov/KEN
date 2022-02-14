# -*- coding: utf-8 -*-

"""A function to convert a set of tables to triples."""

import os
import numpy as np
import pandas as pd
import pickle


def dataframes_to_triples(
    df_dict, df_params, save_dir, add_prefix=False, add_suffix=False
):

    ### Turn dataframes to triplets and concatenate them
    print("Turn dataframes to triplets and concatenate them")
    triplets = []
    for df_name, df in df_dict.items():
        params = df_params[df_name]
        head_col = params["head_col"]
        # Add column name as prefix for the head column
        if add_prefix:
            df[head_col] = head_col + df[head_col].astype(str)
        if "rel_col" not in params.keys():  # Melt dataframe and rename cols
            # Prefix column names with head_col to avoid collisions
            df.columns = [
                head_col + " -> " + col if col != head_col else col
                for col in df.columns
            ]
            params["cat_attr"] = [
                head_col + " -> " + attr for attr in params["cat_attr"]
            ]
            df = df.melt(id_vars=[head_col], var_name="rel", value_name="tail")
            df.rename(columns={head_col: "head"}, inplace=True)
        else:  # Just rename columns
            df.rename(
                columns={
                    params["head_col"]: "head",
                    params["rel_col"]: "rel",
                    params["tail_col"]: "tail",
                },
                inplace=True,
            )
        triplets.append(df)
    triplets = pd.concat(triplets)
    ### Tokenize triplets
    print("Tokenize triplets")
    # Get list of categorical attributes
    cat_attr = list(
        np.unique(np.concatenate([params["cat_attr"] for params in df_params.values()]))
    )
    mask = triplets["rel"].isin(cat_attr)
    # Add suffixes for categorical attributes
    if add_suffix:
        triplets["tail"][mask] = triplets["rel"][mask] + triplets["tail"][mask].astype(
            str
        )
    # Replace missing tails by distinct 'NaN' entities for each relation
    na_mask = triplets["tail"][mask].isna()
    triplets["tail"][mask][na_mask] = "NaN_" + triplets["rel"][mask][na_mask]
    # Build entity_to_idx
    entities = pd.unique(np.concatenate([triplets["head"], triplets["tail"][mask]]))
    n_entities = len(entities)
    entity_to_idx = dict(zip(entities, np.arange(n_entities)))
    # Replace entities by their idx in triplets
    triplets["head"] = triplets["head"].map(entity_to_idx)
    triplets.loc[mask, "tail"] = triplets.loc[mask, "tail"].map(entity_to_idx)
    triplets["tail"] = triplets["tail"].astype(np.float64)
    # Tokenize relations
    relations = triplets["rel"].unique()
    n_relations = len(relations)
    num_attr = [attr for attr in relations if attr not in cat_attr]
    rel_to_idx = dict(zip(num_attr + cat_attr, range(n_relations)))
    triplets["rel"] = triplets["rel"].map(rel_to_idx)
    # Shuffle triplets
    triplets = triplets.sample(frac=1)

    ### Save triplets as record array and metadata
    print("Save triplets and metadata")
    # Create dir
    os.makedirs(save_dir, exist_ok=True)
    # Save triplets
    dtypes = {"head": np.uint32, "rel": np.uint16, "tail": np.float64}
    triplets = triplets.to_records(index=False, column_dtypes=dtypes)
    np.save(f"{save_dir}/triplets.npy", triplets)
    # Save metadata
    metadata = {
        "entity_to_idx": entity_to_idx,
        "rel_to_idx": rel_to_idx,
        "n_entities": n_entities,
        "n_relations": n_relations,
        "cat_attr": cat_attr,
        "n_cat_attr": len(cat_attr),
        "num_attr": num_attr,
        "n_num_attr": len(num_attr),
    }
    with open(f"{save_dir}/metadata.pkl", "wb") as f:
        pickle.dump(metadata, f)
    return
