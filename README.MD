## Notes

Please download the latest version of the file named 'latest.7z' from https://huggingface.co/datasets/cbdb/cbdb-sqlite/blob/main/latest.7z. After downloading, decompress the file, and move the 'latest.db' file to the root directory of this repository.

Still working...

## TODO list

- Using CBDB Access to compare the loss between model and rules: 99_compare_model_and_rules

## Done

- Create the fake birth year by median
- If there are multiple time information to calculate the inferred birth year, then just calculate the medium(as a test, maybe find a weight by using recursive_feature_elimination.py in the future)
- Use the inferred birth year to calculate all the year information for NA
- Save the data_interpolated.csv
- Train the model by 999_gredient_boosting.ipynb. Skip validation_set.csv personids in advance
- Get the loss
