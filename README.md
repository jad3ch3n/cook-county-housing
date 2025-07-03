# Cook County Housing Analysis

This project analyzes housing sale data from Cook County to evaluate property valuation models and assess their fairness across income and geographic groups. It combines predictive modeling with ethical reflection to explore real-world consequences of statistical decisions.

## Project Parts
- **Part 1:** Data Exploration & Predictive Modeling
  - Cleaned and explored a large property sales dataset
  - Built log-linear regression models
  - Investigated variable importance and model fit

- **Part 2:** Human Context & Fairness
  - Explored residual plots to assess regressive patterns
  - Interpreted model fairness in the context of Cook County's tax history
  - Reflected on equity implications of property assessments

## Ethical Motivation
Historically, property tax assessments in Cook County have been shown to overestimate lower-value homes and underestimate higher-value ones, disproportionately impacting low-income and non-white homeowners. This project aims to build models that are not only accurate (low RMSE) but also fair in how errors are distributed.

## 📂 Data Access
To reproduce this project, download the dataset from the following Box folder:
👉 [Cook County Data (Box)](https://berkeley.box.com/s/6l0lr3qfuz3i67act3kphquf21c2zm0m)

You will need the following files:
- `cook_county_train.csv`
- `cook_county_test.csv`
- `codebook.txt`

Place them in a local `data/` folder.

Alternatively, you can run the provided script:
```bash
python download_data.py
```

## 📦 Setup
Recommended environment:
- Python 3.8+
- pandas
- matplotlib
- seaborn
- numpy
- scikit-learn

You can install the packages via:
```bash
pip install -r requirements.txt
```

## 📓 Notebooks
- `CookCountyAnalysis.ipynb`: Full analysis combining Parts 1 & 2

---
Feel free to fork, cite, or adapt this for your own fairness-centered data science work.
