# Welcome to parks_pkg_dsci310_08
**DSCI 310 Group 08**\
**Authors**: Shivani Aggarwal, Luna Gulec, Jingyuan Liu, Sarenna Ulman

[![Run Tests](https://github.com/UBC-DSCI-310-2025W2/parks_pkg_dsci310_08/actions/workflows/test.yml/badge.svg)](https://github.com/UBC-DSCI-310-2025W2/parks_pkg_dsci310_08/actions/workflows/test.yml)
[![Documentation Status](https://github.com/UBC-DSCI-310-2025W2/parks_pkg_dsci310_08/actions/workflows/docs.yml/badge.svg)](https://github.com/UBC-DSCI-310-2025W2/parks_pkg_dsci310_08/actions/workflows/docs.yml)

**`parks_pkg_dsci310_08` is a Python package that contains several utility functions for managing local file directories, data pre-processing, and model coefficient extraction, specifically for [U.S. Parks Access Ranking](https://github.com/rfordatascience/tidytuesday/blob/main/data/2021/2021-06-22/readme.md) analysis, for example, in [this project](https://github.com/UBC-DSCI-310-2025W2/dsci-310-group-08) we wrote.**

This package sits within multiple layers of Python data science ecosystems, such as Data Engineering and Pre-processing and File Management. 

While our package relies on several foundational libraries like `pandas`, `numpy`, `pathlib`, and `scikit-learn`, it is customized to simplify steps into a cleaner, and more project-specific format, saving several steps of manual wrapping.

We have published our package to [TestPyPI](https://test.pypi.org/). You can view this [package](https://test.pypi.org/project/parks_pkg_dsci310_08/) on TestPyPI.

## Get started

You can install this package into your preferred Python environment using pip:

```bash
$ pip install -i https://test.pypi.org/simple/ parks_pkg_dsci310_08
```

or directly from GitHub using:

```bash
$ pip install git+https://github.com/ljy0401/parks_pkg_dsci310_08.git
```

To use parks_pkg_dsci310_08 in your code:

```python
>>> from parks_pkg_dsci310_08.parks import create_directory, find_measurement_last_time, get_model_coefficients, split_dataset
>>> import pandas as pd

>>> folder_path = "data/processed/predictions"
>>> created_path = create_directory(folder_path)

>>> data_raw = pd.read_csv('example_data.csv') 
>>> data_processed = find_measurement_last_time(data_raw, 'year', 'city', 'rank')
>>> print(data_processed)

>>> # Given a fitted sklearn Pipeline with a ColumnTransformer and Ridge step
>>> coef_df = get_model_coefficients(final_pipe)
>>> coef_df.head()

>>> data_processed = pd.read_csv(data_path)
>>> X_train, X_test, y_train, y_test = split_dataset(
    data_processed, target_col="rank", test_size=0.2, random_state=73
    )
```

## Copyright

- Copyright © 2026 Shivani Aggarwal, Luna Gulec, Jingyuan Liu, Sarenna Ulman.
- Free software distributed under the [MIT License](./LICENSE).
