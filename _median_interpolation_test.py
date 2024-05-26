import pandas as pd

data_df = pd.DataFrame(
    {
        "birth_year": [1990, 1985, 1991],
        "param1": [1995, None, 1996],
        "param2": [1987, 1983, None],
    }
)
new_data_df = pd.DataFrame(
    {
        "param1": [1994, None, 1997],
        "param2": [None, 1982, 1990],
    }
)

medians = (
    data_df.drop("birth_year", axis=1)
    .apply(lambda x: x - data_df["birth_year"])
    .median()
)

print(medians.head())


def infer_birth_year(row, medians):
    values = []
    for param in medians.index:
        if pd.notna(row[param]):
            inferred_year = row[param] - medians[param]
            values.append(inferred_year)
    if values:
        return sum(values) / len(values)
    return None


new_data_df["inferred_birth_year"] = new_data_df.apply(
    infer_birth_year, axis=1, args=(medians,)
)


def fill_na_values(row, medians):
    for param in medians.index:
        if pd.isna(row[param]) and pd.notna(row["inferred_birth_year"]):
            row[param] = row["inferred_birth_year"] + medians[param]
    return row


new_data_df = new_data_df.apply(fill_na_values, axis=1, args=(medians,))

print(data_df)

print(new_data_df)
