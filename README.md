Predict CO2 emission according to vehicle characteristics
==============================

This is a project by Polina, Vincent and Denis.
Data from the EEA is used in order to build prediction models for CO2 emissions based on vehicle chracteristics.

Steps done:
* Data cleaned (errors, outliers, duplicates, missing data)
* Data enriched (dichotomization, interpretation of innovative technology codes)
* Evaluation of regression models (select models, select hyperparameters, vizualise with mlflow, arbitrate on robustness and compare scoring)
* Evaluation of categorization models
* Use model to make prediction based on new vehicle characteristics

Results
-------
* Regression scoring of investigated models and corresponding hyperparameters
![results_regression01](https://github.com/user-attachments/assets/5f128c32-77d3-43db-aea1-3f6b02ac539d)

* Classification scoring of investigated models and corresponding hyperparameters
![classif_results](https://github.com/user-attachments/assets/b49c4906-9e05-4bc5-af07-26c2643e96cc)


Project Organization
------------

    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    ├── data               <- Should be in your computer but not on Github (only in .gitignore)
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's name, and a short `-` delimited description, e.g.
    │                         `1.0-alban-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, links, and all other explanatory materials.
    │
    ├── reports            <- The reports that you'll make during this project as PDF
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   ├── visualization  <- Scripts to create exploratory and results oriented visualizations
    │   │   └── visualize.py

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
