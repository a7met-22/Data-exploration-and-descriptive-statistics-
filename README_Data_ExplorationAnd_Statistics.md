# E-Commerce Customer Churn: Data Exploration and Descriptive Statistics

An educational data-analysis notebook that performs the first-pass
exploration of an **e-commerce customer churn dataset**: it inspects the
dataset's dimensions and types, quantifies missing values, generates and
interprets descriptive statistics for all numerical columns, and
visualizes the frequency distributions of the categorical columns.

The notebook follows a classic EDA workflow narrated in Arabic: inspect
first, describe second, visualize third, and only then formulate
hypotheses about what drives churn.

> **Project:** Data Exploration and Statistics on Customer Churn\
> **Notebook:** `Data_ExplorationAnd_Statistics.ipynb`\
> **Domain:** Data Analysis · Exploratory Data Analysis · Descriptive
> Statistics · Pandas · Seaborn / Matplotlib\
> **Environment:** Google Colab

------------------------------------------------------------------------

## 1. Project Overview

The goal is to understand the structure and quality of a synthetic
e-commerce customer dataset (`ecommerce_customer_churn_dataset.csv`)
before any predictive modeling, and to surface the anomalies that will
matter later for churn prediction.

The notebook executes the following steps in order:

1.  Upload the CSV into Colab and load it into a pandas DataFrame with a
    head preview.
2.  Run an initial inspection: DataFrame dimensions, column dtypes via
    `df.info()`, and missing-value counts per column.
3.  Generate comprehensive descriptive statistics with `df.describe()`
    and interpret them in a long Arabic markdown block (central
    tendency, dispersion, and range per column).
4.  Compute and print value counts for the categorical columns
    (`Gender`, `Country`, `City`, `Signup_Quarter`).
5.  Draw count plots for all four categorical columns in a 2×2 grid to
    reveal dominant categories visually.

The central analytical principle demonstrated is:

> **Never model before you inspect: knowing the shape, types, missing
> values, and impossible values of a dataset determines what cleaning
> and which hypotheses are even possible.**

------------------------------------------------------------------------

## 2. Key Features

### Three-stage inspection workflow

-   **Structure:** `df.shape`, `df.info()` — 50,000 rows × 25 columns;
    20 float64, 1 int64 (`Churned`), 4 object columns.
-   **Completeness:** `df.isnull().sum()` — every column's missing count
    is listed, exposing that only 10 of 25 columns are fully populated.
-   **Distribution:** `df.describe()` plus a detailed written
    interpretation.

### Written interpretation, not just numbers

-   A dedicated Arabic markdown cell walks through every numerical
    column, calling out means, standard deviations, quartiles, and
    minimum/maximum values, and flags anomalies explicitly (Age up to
    200, negative `Total_Purchases`, `Cart_Abandonment_Rate` above
    100%).

### Categorical composition analysis

-   Value counts for all four categorical columns, showing a slight
    female majority, eight countries led by the USA, forty cities led by
    Houston, and an almost perfectly uniform signup-quarter
    distribution.
-   Seaborn count plots in a 2×2 `plt.subplot` grid for visual
    comparison.

### Teaching-oriented narration

-   The task and subtask blocks (in Arabic) define exactly what each
    stage must produce, and short "Reasoning" markdown notes explain the
    intent of the next code cell before it runs.

------------------------------------------------------------------------

## 3. Analysis Structure

| Stage | Code element | Responsibility |
|-------|--------------|----------------|
| Load data | `files.upload()` + `pd.read_csv` | Brings the CSV into Colab and into a DataFrame. |
| Structure check | `df.shape`, `df.info()` | Reports rows/columns and per-column dtypes. |
| Missing values | `df.isnull().sum()` | Quantifies incompleteness per column. |
| Numerical summary | `df.describe()` | Central tendency, dispersion, range for 21 numeric columns. |
| Categorical counts | `df[col].value_counts()` | Frequency tables for `Gender`, `Country`, `City`, `Signup_Quarter`. |
| Categorical plots | `sns.countplot` in a 2×2 grid | Visual frequency distributions. |

### Dataset configuration

| Setting | Value |
|---------|-------|
| Dataset | `ecommerce_customer_churn_dataset.csv` (synthetic) |
| Shape | 50,000 rows × 25 columns |
| Numeric columns | 21 (20 float64 + 1 int64 target `Churned`) |
| Categorical columns | 4 (`Gender`, `Country`, `City`, `Signup_Quarter`) |
| Churn rate | 28.9% (`Churned` mean = 0.289) |
| Fully populated columns | 10 of 25 |

### High-level workflow

``` text
        ecommerce_customer_churn_dataset.csv
                      |
                      v
        Upload + load into DataFrame, preview head
                      |
                      v
        Initial inspection (shape, dtypes, missing values)
                      |
                      v
        df.describe() + written interpretation
        (central tendency, dispersion, anomalies)
                      |
                      v
        Categorical value_counts + count plots
        (Gender / Country / City / Signup_Quarter)
                      |
                      v
        Key insights and hypotheses for churn analysis
```

------------------------------------------------------------------------

## 4. Technical Highlights

### Quantified missingness

The missing-value audit is column-exact and shows the pattern clearly:
`Social_Media_Engagement_Score` (6,000), `Credit_Balance` (5,500),
`Mobile_App_Usage` (5,000), `Returns_Rate` (4,491), and
`Wishlist_Items` (4,000) are the least complete, while identity,
behavioral-purchase, and target columns (`Gender`, `Country`,
`Membership_Years`, `Total_Purchases`, `Churned`, …) have none.

### Anomaly detection from descriptive statistics

The interpretation block extracts concrete data-quality issues directly
from `describe()` output:

-   `Age`: min 5, max 200 → impossible values requiring clipping or
    removal.
-   `Total_Purchases`: minimum of **−13** → clearly erroneous entries.
-   `Cart_Abandonment_Rate`: mean 57.08% but maximum 143.74% → values
    above 100% are suspicious and need redefinition or cleaning.
-   `Average_Order_Value`: mean $123 vs maximum $9,666 → heavy right
    tail.
-   `Churned`: mean 0.289 establishes the baseline churn rate used as a
    reference for all later hypotheses.

### Balanced vs. skewed categorical structure

`Signup_Quarter` is nearly uniform (Q1 12,453 → Q3 12,558), while
`Country` and `City` are heavily skewed (USA 17,384 ≈ 35% of customers;
top five US cities each ≈ 3,400–3,550). This contrast matters: quarter
is unlikely to be predictive, whereas geography may proxy market
differences.

### Recorded results

| Quantity | Value |
|----------|-------|
| DataFrame dimensions | 50,000 rows, 25 columns |
| `Age` missing / `Returns_Rate` missing | 2,495 / 4,491 |
| `Age` mean / max | 37.80 / 200 |
| `Total_Purchases` min | −13 |
| `Cart_Abandonment_Rate` mean / max | 57.08% / 143.74% |
| `Average_Order_Value` mean / max | $123.12 / $9,666.38 |
| `Churned` mean (churn rate) | 0.289 |
| Gender split (F / M / Other) | 25,116 / 23,947 / 937 |
| Top country / count | USA / 17,384 |
| Signup_Quarter range | 12,453–12,558 (near-uniform) |

------------------------------------------------------------------------

## 5. Technology Stack

| Technology | Role |
|------------|------|
| Python | Core implementation language. |
| Google Colab | Notebook execution environment and file-upload widget. |
| pandas | DataFrame loading, `info`, `isnull`, `describe`, `value_counts`. |
| Matplotlib | Figure and subplot management for the count-plot grid. |
| Seaborn | `countplot` visualizations of categorical distributions. |

------------------------------------------------------------------------

## 6. Running the Notebook

### Requirements

The notebook is designed for Google Colab with the interactive upload
widget. No GPU is required.

Before running:

-   Run the first cell, select `ecommerce_customer_churn_dataset.csv`
    in the upload widget, and let the cell finish loading the file.
-   Re-run the upload cell if the widget says it is only available in
    the current browser session.

### Execution order

1.  Upload and load the CSV; preview the head.
2.  Display `df.describe()` (an early, unannotated pass).
3.  Read the task/subtask markdown blocks (Arabic).
4.  Run the initial inspection cell (shape, dtypes, missing values).
5.  Run the descriptive-statistics cell and read the interpretation
    markdown.
6.  Run the categorical value-counts cell.
7.  Run the count-plot cell to produce the 2×2 figure.

### Example output

The inspection cell prints the 50,000 × 25 dimensions, the full
`df.info()` listing, and a 25-row missing-value table; the frequency
cell prints value counts for all four categorical columns; the plotting
cell renders four count plots (with a seaborn palette deprecation
warning on current seaborn versions).

------------------------------------------------------------------------

## 7. Configuration

Settings are defined inline so they can be reviewed and adjusted.

Review these settings:

-   File name: `ecommerce_customer_churn_dataset.csv` (uploaded at
    runtime).
-   Categorical columns analyzed:
    `['Gender', 'Country', 'City', 'Signup_Quarter']`.
-   Plot layout: 2×2 subplot grid, figure size 18×12, `viridis`
    palette, 45-degree rotated x-tick labels.

No credentials are required.

------------------------------------------------------------------------

## 8. Limitations and Considerations

This notebook is an exploration stage, not a completed analysis.

| Area | Current approach | Important limitation |
|------|------------------|----------------------|
| Notebook scope | Stops after categorical frequency plots. | The declared task also requires correlation matrices, scatter plots, box plots, and crosstabulations (e.g., churn vs. features) — none are implemented yet, so no relationship between variables is actually tested. |
| Missing values | Counted and reported. | No strategy is chosen (imputation, flagging, or dropping), and missingness is not analyzed against the `Churned` target (MNAR check). |
| Outliers | Flagged in markdown only. | Impossible values (Age 5–200, negative purchases, abandonment > 100%) are documented but not cleaned, capped, or investigated row-by-row. |
| Categorical plots | Single `sns.countplot` grid. | A `FutureWarning` appears because `palette` is passed without `hue`; on seaborn ≥ 0.14 the call will need `hue=col, legend=False`. |
| Duplicates | Not examined. | No `duplicated()` check on customers or rows, and no unique-ID column is verified. |
| Churn analysis | Baseline rate computed. | No churn rate breakdown by category (e.g., churn by country or gender), so the stated hypotheses remain unverified. |
| Reproducibility | Static CSV upload. | No seed-dependent operations yet, but no versioning of the dataset either. |

### Recommendations

-   Complete the declared task: add a correlation heatmap for numerical
    columns, scatter plots and box plots of key behavioral features
    against `Churned`, and crosstabs of churn by `Gender`, `Country`,
    and `Signup_Quarter`.
-   Fix the seaborn call (`hue=col, legend=False`) to silence the
    deprecation warning.
-   Clean anomalies: clip `Age` to a valid range, floor `Total_Purchases`
    at zero, and decide how `Cart_Abandonment_Rate` above 100% should be
    interpreted or corrected.
-   Quantify missingness per column as a percentage and test whether
    missingness correlates with churn before imputing.
-   Check for duplicated rows and verify whether one row equals one
    customer.
-   Turn the written interpretation into a reusable data-quality
    report function.

------------------------------------------------------------------------

## 9. Repository Structure

``` text
.
├── Data_ExplorationAnd_Statistics.ipynb
└── ecommerce_customer_churn_dataset.csv   (uploaded at runtime)
```

The dataset is not stored in the repository; it is uploaded through the
Colab file widget when the first cell runs.

------------------------------------------------------------------------

## 10. Possible Future Improvements

-   Add bivariate analysis: correlation matrix heatmap, churn-stratified
    box plots, and scatter plots for engagement vs. purchase features.
-   Compute churn rate per category and per numeric decile to rank
    candidate predictors.
-   Build a cleaning pipeline (outlier clipping, missing-value strategy)
    and validate it with before/after distribution plots.
-   Proceed to feature engineering and a churn classifier (logistic
    regression baseline, then gradient boosting), using this EDA as the
    contract for data quality.
-   Extend the categorical analysis with normalized (percentage)
    frequency tables alongside raw counts.

------------------------------------------------------------------------

## 11. Key Takeaway

This project demonstrates a disciplined first contact with a messy
dataset: exact dimensions and dtypes, a per-column missing-value audit,
descriptive statistics read column by column, and categorical
distributions — all before any hypothesis is tested.

The most important design lesson is that **descriptive statistics are a
data-quality tool, not just a summary: the same `describe()` output that
reports a 28.9% churn baseline also reveals impossible ages, negative
purchase counts, and abandonment rates above 100%, and those anomalies —
plus the planned-but-unbuilt correlation, box-plot, and crosstab stages —
must be resolved before any churn model can be trusted.**