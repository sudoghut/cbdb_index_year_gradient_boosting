## Notes

Please download the latest version of the file named 'latest.7z' from https://huggingface.co/datasets/cbdb/cbdb-sqlite/blob/main/latest.7z. After downloading, decompress the file, and move the 'latest.db' file to the root directory of this repository.

Still working...

## Score

20240602 validation set

### xgboost:
MAE: 10.033610447844351
RMSE: 22.9241680850365
R2: 0.9962479865532473
SD: 374.3629629607035
Percent within 10 years: 0.7209443099273608

### Rule based:
MAE: 31.145703611457037
RMSE: 159.95510733839154
R2: 0.852460556943477
STD: 416.56165855544583
percent within 10 years: 0.651307596513076

## TODO list

- Trying k-fold
- Ensemble learning

## Done

- Create the fake birth year by median
- If there are multiple time information to calculate the inferred birth year, then just calculate the medium(as a test, maybe find a weight by using recursive_feature_elimination.py in the future)
- Use the inferred birth year to calculate all the year information for NA
- Save the data_interpolated.csv
- Train the model by 999_gredient_boosting.ipynb. Skip validation_set.csv personids in advance
- Get the loss
- Using CBDB Access to compare the loss between model and rules: 99_compare_model_and_rules
