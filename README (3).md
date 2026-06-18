# Customer Prediction & Analytics System

A data science project that walks through the full pipeline — from messy raw data to a working machine learning model — using a real-world e-commerce dataset as the foundation.

---

## Why This Project Exists

Every time Amazon says *"customers who bought this also bought..."* or your bank flags an unusual transaction — that's this exact pipeline running under the hood. This project is my attempt to understand and build that from scratch, step by step.

I started with one question: **what does it actually take to go from a CSV file full of messy sales records to a model that predicts customer behavior?**

Turns out, the answer is mostly just patience, careful data cleaning, and a lot of `df.isnull().sum()`.

---

## What's Inside

```
customer-analytics/
│
├── data/
│   ├── raw/                    ← The original, untouched dataset
│   └── cleaned/                ← After all the cleaning work
│
├── notebooks/
│   ├── 01_data_cleaning.ipynb  ← Start here. EDA + cleaning.
│   └── 02_ml_model.ipynb       ← Model training and evaluation
│
├── src/
│   ├── clean.py                ← All cleaning logic, reusable
│   ├── visualize.py            ← Plot functions
│   └── model.py                ← ML pipeline
│
├── outputs/
│   └── plots/                  ← Every chart gets saved here
│
└── README.md
```

---

## Tech Stack

| Tool | What I Used It For |
|------|--------------------|
| Python 3.10 | Everything |
| Pandas | Loading, cleaning, reshaping data |
| NumPy | Numerical operations under the hood |
| Scikit-Learn | Building and evaluating the ML model |
| Matplotlib | Visualizing distributions, trends, and results |

---

## The Pipeline

### Step 1 — Load and take a first look

Before touching anything, just look at the data. Run these three lines and read what they tell you.

```python
import pandas as pd

df = pd.read_csv('data/raw/ecommerce_sales.csv')

print(df.head())       # What do the first few rows look like?
print(df.info())       # What types are the columns? Any nulls?
print(df.describe())   # What's the range? Are there suspicious values?
```

This step alone will usually tell you 80% of what needs fixing.

---

### Step 2 — Handle missing values

Missing data is the most common problem in real datasets. There's no single right answer — you have to think about what makes sense for each column.

```python
# First, see exactly what's missing
print(df.isnull().sum())

# For numeric columns, fill with the median
# (median is safer than mean when there are outliers)
df['revenue'].fillna(df['revenue'].median(), inplace=True)
df['session_time'].fillna(df['session_time'].mean(), inplace=True)

# For rows where the customer ID itself is missing — just drop them.
# There's nothing useful we can do with an anonymous record.
df.dropna(subset=['customer_id'], inplace=True)

# Confirm everything looks clean
print(df.isnull().sum())
```

---

### Step 3 — Remove duplicates

Simple but easy to forget.

```python
print(f"Duplicates found: {df.duplicated().sum()}")
df.drop_duplicates(inplace=True)
print(f"Dataset shape after cleaning: {df.shape}")
```

---

### Step 4 — Deal with outliers

Outliers can quietly ruin a model. A single data entry error — like revenue recorded as 999999 instead of 999 — will throw off everything downstream.

I use the IQR method because it's robust and doesn't assume a normal distribution.

```python
def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    before = len(df)
    df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    print(f"{column}: removed {before - len(df)} outliers")
    return df

df = remove_outliers(df, 'revenue')
df = remove_outliers(df, 'session_time')
```

---

### Step 5 — Explore the data (EDA)

Now that the data is clean, actually look at it. Ask questions. See what stands out.

```python
# Which product categories are most popular?
print(df['product_category'].value_counts())

# Who are the top spenders?
print(df.groupby('customer_id')['revenue'].sum()
        .sort_values(ascending=False)
        .head(10))

# Do any features correlate with whether someone actually buys?
print(df[['revenue', 'session_time', 'cart_items', 'pages_visited']].corr())

# What percentage of sessions end in a purchase?
print(f"Conversion rate: {df['purchased'].mean() * 100:.1f}%")
```

---

### Step 6 — Visualize

Numbers in a table are easy to miss. A chart makes patterns obvious immediately.

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('E-Commerce Sales — Overview', fontsize=15, fontweight='bold', y=1.01)

# Revenue distribution — is it skewed? Are there peaks?
axes[0, 0].hist(df['revenue'], bins=30, color='#4F46E5', edgecolor='white', alpha=0.85)
axes[0, 0].set_title('Revenue Distribution')
axes[0, 0].set_xlabel('Revenue (₹)')
axes[0, 0].set_ylabel('Count')

# Monthly trend — is there seasonality?
monthly = df.groupby('month')['revenue'].sum()
axes[0, 1].plot(monthly.index, monthly.values, marker='o', color='#0EA5E9', linewidth=2.5)
axes[0, 1].set_title('Revenue by Month')
axes[0, 1].set_xlabel('Month')

# Which categories drive the most revenue?
cat_rev = df.groupby('product_category')['revenue'].sum().sort_values()
axes[1, 0].barh(cat_rev.index, cat_rev.values, color='#10B981')
axes[1, 0].set_title('Revenue by Category')

# How many sessions actually convert?
counts = df['purchased'].value_counts()
axes[1, 1].pie(counts, labels=['Browsed only', 'Purchased'],
               autopct='%1.1f%%', colors=['#E2E8F0', '#6366F1'],
               startangle=90)
axes[1, 1].set_title('Conversion Rate')

plt.tight_layout()
plt.savefig('outputs/plots/eda_overview.png', dpi=150, bbox_inches='tight')
plt.show()
```

---

### Step 7 — Train the model

With clean data and a clear understanding of the features, training a model is actually the easy part.

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# What we're using to predict
X = df[['age', 'session_time', 'cart_items', 'pages_visited', 'revenue']]

# What we're trying to predict: did this customer buy something?
y = df['purchased']

# Keep 20% of data aside — the model never sees this until evaluation
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale so no single feature dominates just because of its units
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print(classification_report(y_test, y_pred))
```

---

## Getting Started

**Requirements:** Python 3.8 or higher

```bash
# Install dependencies
pip install pandas numpy scikit-learn matplotlib jupyter

# Launch the notebook
jupyter notebook
```

Open `notebooks/01_data_cleaning.ipynb` and run the cells top to bottom.

---

## Checklist

**Data Cleaning & EDA**
- [ ] Raw data loaded and inspected
- [ ] All null values handled
- [ ] Duplicates removed
- [ ] Outliers treated with IQR method
- [ ] EDA completed — distributions, correlations, value counts
- [ ] 4-panel visualization saved to `outputs/plots/`
- [ ] Cleaned CSV exported to `data/cleaned/`

**ML Model**
- [ ] Features selected and documented
- [ ] 80/20 train-test split applied
- [ ] RandomForestClassifier trained
- [ ] Accuracy and classification report recorded
- [ ] Feature importances visualized

---

## Things I Learned

- Cleaning data takes longer than you expect. Always.
- EDA is not optional — it changes which features you trust and which you question.
- The IQR method is simple, but it works. Don't overcomplicate outlier detection.
- Median is almost always safer than mean for filling missing values in skewed columns.
- A model trained on dirty data will give you confident wrong answers. Clean first.

---

## Real-World Applications

This same pipeline, at larger scale, powers:

- **Amazon** — "Customers who bought this also bought..."
- **Netflix** — content recommendations based on viewing patterns
- **Banking** — real-time fraud scoring on every transaction

The fundamentals don't change. Only the scale does.

---

*Built with Python · Pandas · NumPy · Scikit-Learn · Matplotlib*
