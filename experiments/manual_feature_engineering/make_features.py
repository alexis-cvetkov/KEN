import pandas as pd
from pathlib import Path

from KEN.dataloader import DataLoader

exp_dir = Path(__file__).absolute().parents[1]
data_dir = exp_dir / "datasets"


def kdd14():
    df = pd.read_parquet(data_dir / "kdd14/target.parquet")
    df.columns = ["projectid", "target"]
    # Add MEAN(donation_amount) and MEAN(LENGTH(donation_message))
    use_cols = ["projectid", "donation_to_project", "donation_message"]
    donations = pd.read_csv(data_dir / "kdd14/raw/donations.csv", usecols=use_cols)
    donations["donation_message"].fillna("", inplace=True)
    donations["LENGTH(donation_message)"] = donations["donation_message"].apply(len)
    del donations["donation_message"]
    mean_stats = donations.groupby("projectid").mean().reset_index()
    mean_stats.columns = [
        "projectid",
        "MEAN(donation_to_project)",
        "MEAN(LENGTH(donation_message))",
    ]
    df = df.merge(mean_stats, on="projectid", how="left")
    # Add COUNT(donations)
    counts = donations["projectid"].value_counts().reset_index()
    counts.columns = ["projectid", "COUNT(donations)"]
    df = df.merge(counts, on="projectid", how="left")
    df["COUNT(donations)"].fillna(0, inplace=True)
    ### Add static features
    use_cols = [
        "projectid",
        "primary_focus_subject",
        "primary_focus_area",
        "resource_type",
        "poverty_level",
        "grade_level",
        "eligible_double_your_impact_match",
        "eligible_almost_home_match",
        "total_price_excluding_optional_support",
        "total_price_including_optional_support",
        "students_reached",
    ]
    projects = pd.read_csv(data_dir / "kdd14/raw/projects.csv", usecols=use_cols)
    # Map boolean features to 0/1
    for col in ["eligible_almost_home_match", "eligible_double_your_impact_match"]:
        projects[col] = (projects[col] == "t").astype(int)
    # Replace mssing values in categorical features with a "Missing" token
    na_cat_cols = [
        "primary_focus_subject",
        "primary_focus_area",
        "resource_type",
        "grade_level",
    ]
    projects[na_cat_cols] = projects[na_cat_cols].fillna("Missing", axis=0)
    # Merge features
    df = df.merge(projects, on="projectid", how="left")
    # Encode categorical features
    oce_columns = [
        "primary_focus_subject",
        "primary_focus_area",
        "resource_type",
        "poverty_level",
        "grade_level",
    ]
    for col in oce_columns:
        enc_col = pd.get_dummies(df[col], prefix=col, prefix_sep=" == ")
        df = pd.concat([df, enc_col], axis=1)
        del df[col]
    # Save feature and target to parquet file
    del df["projectid"]
    df.to_parquet(exp_dir / "manual_feature_engineering/kdd14.parquet", index=False)
    return


def kdd15():
    # Load target data
    df = pd.read_parquet(data_dir / "kdd15/target.parquet")
    df.columns = ["enrollment_id", "target"]
    # Add COUNT(logs)
    logs = pd.read_csv(data_dir / "kdd15/raw/log_train.csv")
    log_counts = logs["enrollment_id"].value_counts()
    log_counts = log_counts.rename_axis("enrollment_id").reset_index(name="COUNT(logs)")
    df = pd.merge(df, log_counts, on="enrollment_id", how="left")
    ### Add MEAN(relative_log_time)
    course_dates = pd.read_csv(data_dir / "kdd15/raw/date.csv")
    enrollments = pd.read_csv(data_dir / "kdd15/raw/enrollment_train.csv")
    logs = pd.merge(logs, enrollments, on="enrollment_id")
    logs = pd.merge(logs, course_dates, on="course_id")
    # Compute log_time relative to course start/end dates
    for col in ["time", "from", "to"]:
        logs[col] = pd.to_datetime(logs[col])
    logs["rel_time"] = (logs["time"] - logs["from"]) / (logs["to"] - logs["from"])
    mean_log_times = logs.groupby("enrollment_id").mean("rel_time")
    mean_log_times.columns = ["MEAN(relative_log_time)"]
    df = pd.merge(df, mean_log_times, on="enrollment_id", how="left")
    ### Add course start date (number of days relative to the first course)
    # Add course id
    df = pd.merge(df, enrollments[["enrollment_id", "course_id"]], on="enrollment_id")
    # Compute number of days after the first course started
    course_dates[["from", "to"]] = course_dates[["from", "to"]].apply(pd.to_datetime)
    course_dates["course_start_date"] = (
        course_dates["from"] - course_dates["from"].min()
    ).dt.days
    df = pd.merge(df, course_dates[["course_id", "course_start_date"]], on="course_id")
    # Add number of modules per course
    modules = pd.read_csv(data_dir / "kdd15/raw/object.csv")
    module_counts = modules["course_id"].value_counts()
    module_counts = module_counts.rename_axis("course_id").reset_index(
        name="COUNT(course.modules)"
    )
    df = pd.merge(df, module_counts, on="course_id")
    # Encode course_id
    enc_col = pd.get_dummies(df["course_id"], prefix=col, prefix_sep=" == ")
    df = pd.concat([df, enc_col], axis=1)
    del df["course_id"]
    # Save feature and target to parquet file
    del df["enrollment_id"]
    df.to_parquet(exp_dir / "manual_feature_engineering/kdd15.parquet", index=False)
    return


def us_elections():
    # Load triples
    dl = DataLoader(data_dir / "yago3/triplets/full", use_features="all")
    triples = dl.get_dataframe()
    ### Load target dataframe
    df = pd.read_parquet(data_dir / "us_elections/target_log.parquet")
    df.columns = ["entity", "party", "target"]
    # Remove entities that are not in triples
    mask = df["entity"].isin(list(dl.entity_to_idx.keys()))
    df = df[mask]
    # Map entities to their index
    df["entity"] = df["entity"].map(dl.entity_to_idx).astype(int)
    ### Merge latitude, longitude, population, area, density
    for rel in [
        "<hasLatitude>",
        "<hasLongitude>",
        "<hasNumberOfPeople>",
        "<hasArea>",
        "<hasPopulationDensity>",
    ]:
        mask = triples["rel"] == dl.rel_to_idx[rel]
        df_attr = triples[["head", "tail"]][mask]
        df_attr.columns = ["entity", rel]
        df_attr = df_attr.groupby(by="entity").mean()
        df = pd.merge(df, df_attr, on="entity", how="left")
    # Encode parties
    enc_col = pd.get_dummies(df["party"], prefix="party", prefix_sep=" == ")
    df = pd.concat([df, enc_col], axis=1)
    del df["party"]
    # Save feature and target to parquet file
    del df["entity"]
    df.to_parquet(
        exp_dir / "manual_feature_engineering/us_elections.parquet", index=False
    )
    return


def housing_prices():
    # Load triples
    dl = DataLoader(data_dir / "yago3/triplets/full", use_features="all")
    triples = dl.get_dataframe()
    ### Load target dataframe
    df = pd.read_parquet(data_dir / "housing_prices/target_log.parquet")
    df.columns = ["entity", "target"]
    # Remove entities that are not in triples
    mask = df["entity"].isin(list(dl.entity_to_idx.keys()))
    df = df[mask]
    # Map entities to their index
    df["entity"] = df["entity"].map(dl.entity_to_idx).astype(int)
    ### Merge latitude, longitude, population, area, density
    for rel in [
        "<hasLatitude>",
        "<hasLongitude>",
        "<hasNumberOfPeople>",
        "<hasArea>",
        "<hasPopulationDensity>",
    ]:
        mask = triples["rel"] == dl.rel_to_idx[rel]
        df_attr = triples[["head", "tail"]][mask]
        df_attr.columns = ["entity", rel]
        df_attr = df_attr.groupby(by="entity").mean()
        df = pd.merge(df, df_attr, on="entity", how="left")
    # Save feature and target to parquet file
    del df["entity"]
    df.to_parquet(
        exp_dir / "manual_feature_engineering/housing_prices.parquet", index=False
    )
    return

def us_accidents():
    # Load triples
    dl = DataLoader(data_dir / "yago3/triplets/full", use_features="all")
    triples = dl.get_dataframe()
    ### Load target dataframe
    df = pd.read_parquet(data_dir / "us_accidents/counts.parquet")
    df.columns = ["entity", "target"]
    # Remove entities that are not in triples
    mask = df["entity"].isin(list(dl.entity_to_idx.keys()))
    df = df[mask]
    # Map entities to their index
    df["entity"] = df["entity"].map(dl.entity_to_idx).astype(int)
    ### Merge latitude, longitude, population, area, density
    for rel in [
        "<hasLatitude>",
        "<hasLongitude>",
        "<hasNumberOfPeople>",
        "<hasArea>",
        "<hasPopulationDensity>",
    ]:
        mask = triples["rel"] == dl.rel_to_idx[rel]
        df_attr = triples[["head", "tail"]][mask]
        df_attr.columns = ["entity", rel]
        df_attr = df_attr.groupby(by="entity").mean()
        df = pd.merge(df, df_attr, on="entity", how="left")
    # Save feature and target to parquet file
    del df["entity"]
    df.to_parquet(
        exp_dir / "manual_feature_engineering/us_accidents.parquet", index=False
    )
    return

def movie_revenues():
    # Load triples
    dl = DataLoader(data_dir / "yago3/triplets/full", use_features="all")
    triples = dl.get_dataframe()
    ### Load target dataframe
    df = pd.read_parquet(data_dir / "movie_revenues/target.parquet")
    df.columns = ["entity", "target"]
    # Remove entities that are not in triples
    mask = df["entity"].isin(list(dl.entity_to_idx.keys()))
    df = df[mask]
    # Map entities to their index
    df["entity"] = df["entity"].map(dl.entity_to_idx).astype(int)
    ### Merge movie attributes (we identified attributes present
    ### in at least 5% of movies beforehand)
    # Add movie duration
    rel = "<hasDuration>"
    mask = triples["rel"] == dl.rel_to_idx[rel]
    df_attr = triples[["head", "tail"]][mask]
    df_attr.columns = ["entity", rel]
    df_attr = df_attr.groupby(by="entity").mean()
    df = pd.merge(df, df_attr, on="entity", how="left")
    # Add the country in which is produced the movie (if several, keep only one)
    rel = "<isLocatedIn>"
    mask = triples["rel"] == dl.rel_to_idx[rel]
    df_attr = triples[["head", "tail"]][mask]
    df_attr.columns = ["entity", rel]
    df_attr = df_attr.groupby(by="entity").first()
    df = pd.merge(df, df_attr, on="entity", how="left")
    # One-hot encoding
    most_common = df[rel].value_counts().index[:5]
    mask = ~df[rel].isin(most_common)
    mask2 = ~df[rel].isna()
    df.loc[mask * mask2, rel] = "Rare country"
    enc_col = pd.get_dummies(df[rel], prefix="<isLocatedIn>", prefix_sep=" == ", dummy_na=True)
    df = pd.concat([df, enc_col], axis=1)
    del df[rel]
    #
    for rel in ['<actedIn>', '<created>', '<directed>', '<edited>', '<wroteMusicFor>']:
        mask = triples["rel"] == dl.rel_to_idx[rel]
        df_attr = triples[["head", "tail"]][mask]
        df_attr.columns = [rel, "entity"]
        df_attr = df_attr.groupby(by="entity").count()
        df_attr.columns = [f"COUNT({rel})"]
        df = pd.merge(df, df_attr, on="entity", how="left")
    # Save feature and target to parquet file
    del df["entity"]
    df.to_parquet(
        exp_dir / "manual_feature_engineering/movie_revenues.parquet", index=False
    )
    return

def company_employees():
    # Load triples
    dl = DataLoader(data_dir / "yago3/triplets/full", use_features="all")
    triples = dl.get_dataframe()
    ### Load target dataframe
    df = pd.read_parquet(data_dir / "company_employees/target.parquet")
    df.columns = ["entity", "target"]
    # Remove entities that are not in triples
    mask = df["entity"].isin(list(dl.entity_to_idx.keys()))
    df = df[mask]
    # Map entities to their index
    df["entity"] = df["entity"].map(dl.entity_to_idx).astype(int)
    ### Merge movie attributes present in at least 5% of companies beforehand
    # Forward relations
    for rel, rel_idx in dl.rel_to_idx.items():
        mask = triples["rel"] == rel_idx
        head = triples["head"][mask]
        if df["entity"].isin(head).sum() >= 0.05 * len(df):
            df_attr = triples[["head", "tail"]][mask]
            df_attr.columns = ["entity", rel]
            if rel_idx < dl.n_num_attr:
                df_attr = df_attr.groupby(by="entity").mean()
            else:
                df_attr = df_attr.groupby(by="entity").count()
                df_attr.columns = [f"COUNT({rel})"]
            df = pd.merge(df, df_attr, on="entity", how="left")
    # Inverse relations
    for rel, rel_idx in dl.rel_to_idx.items():
        mask = triples["rel"] == rel_idx
        tail = triples["tail"][mask]
        if rel_idx > dl.n_num_attr and df["entity"].isin(tail).sum() >= 0.05 * len(df):
            df_attr = triples[["head", "tail"]][mask]
            df_attr.columns = [rel, "entity"]
            df_attr = df_attr.groupby(by="entity").count()
            df_attr.columns = [f"COUNT(inv_{rel})"]
            df = pd.merge(df, df_attr, on="entity", how="left")
    # Save feature and target to parquet file
    del df["entity"]
    df.to_parquet(
        exp_dir / "manual_feature_engineering/company_employees.parquet", index=False
    )
    return