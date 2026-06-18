print("Program started")
import pandas as pd
import matplotlib.pyplot as plt

# ==============================
# LOAD DATASET
# ==============================
file_path = "Amazon Sale Report.csv"

df = pd.read_csv(file_path)

print("="*50)
print("DATASET SHAPE")
print(df.shape)

print("\n" + "="*50)
print("FIRST 5 ROWS")
print(df.head())

print("\n" + "="*50)
print("COLUMN NAMES")
print(df.columns)

print("\n" + "="*50)
print("DATA INFO")
print(df.info())

# ==============================
# MISSING VALUES
# ==============================
print("\n" + "="*50)
print("MISSING VALUES")

missing = df.isnull().sum()
print(missing[missing > 0])

# Fill numeric missing values
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns

for col in numeric_cols:
    df[col] = df[col].fillna(df[col].median())

# Fill categorical missing values
categorical_cols = df.select_dtypes(include=['object']).columns

for col in categorical_cols:
    if df[col].isnull().sum() > 0:
        df[col] = df[col].fillna("Unknown")

# ==============================
# DUPLICATES
# ==============================
print("\n" + "="*50)
print("DUPLICATE ROWS")

duplicates = df.duplicated().sum()
print("Duplicates Found:", duplicates)

df = df.drop_duplicates()

print("Shape after removing duplicates:", df.shape)

# ==============================
# BASIC STATISTICS
# ==============================
print("\n" + "="*50)
print("STATISTICAL SUMMARY")

print(df.describe())

# ==============================
# TOTAL REVENUE
# ==============================
if 'Amount' in df.columns:

    total_revenue = df['Amount'].sum()

    print("\n" + "="*50)
    print("TOTAL REVENUE")
    print(f"₹ {total_revenue:,.2f}")

# ==============================
# TOP CATEGORIES
# ==============================
if 'Category' in df.columns:

    print("\n" + "="*50)
    print("TOP CATEGORIES")

    print(df['Category'].value_counts().head(10))

# ==============================
# TOP STATES
# ==============================
if 'ship-state' in df.columns:

    print("\n" + "="*50)
    print("TOP STATES")

    print(df['ship-state'].value_counts().head(10))

# ==============================
# ORDER STATUS
# ==============================
if 'Status' in df.columns:

    print("\n" + "="*50)
    print("ORDER STATUS")

    print(df['Status'].value_counts())

# ==============================
# SALES BY CATEGORY
# ==============================
if 'Category' in df.columns and 'Amount' in df.columns:

    sales_by_category = (
        df.groupby('Category')['Amount']
        .sum()
        .sort_values(ascending=False)
    )

    plt.figure(figsize=(10,5))
    sales_by_category.plot(kind='bar')
    plt.title("Sales by Category")
    plt.xlabel("Category")
    plt.ylabel("Revenue")
    plt.tight_layout()
    plt.show()

# ==============================
# TOP 10 STATES BY SALES
# ==============================
if 'ship-state' in df.columns and 'Amount' in df.columns:

    state_sales = (
        df.groupby('ship-state')['Amount']
        .sum()
        .sort_values(ascending=False)
        .head(10)
    )

    plt.figure(figsize=(10,5))
    state_sales.plot(kind='bar')
    plt.title("Top 10 States by Revenue")
    plt.xlabel("State")
    plt.ylabel("Revenue")
    plt.tight_layout()
    plt.show()

# ==============================
# MONTHLY SALES TREND
# ==============================
if 'Date' in df.columns and 'Amount' in df.columns:

    try:
        df['Date'] = pd.to_datetime(df['Date'])

        monthly_sales = (
            df.groupby(df['Date'].dt.to_period('M'))['Amount']
            .sum()
        )

        monthly_sales.index = monthly_sales.index.astype(str)

        plt.figure(figsize=(10,5))
        monthly_sales.plot(marker='o')
        plt.title("Monthly Sales Trend")
        plt.xlabel("Month")
        plt.ylabel("Revenue")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    except:
        print("Date column format could not be processed.")

# ==============================
# EXPORT CLEANED DATA
# ==============================
df.to_csv("cleaned_amazon_sales.csv", index=False)

print("\n" + "="*50)
print("Analysis Complete!")
print("Cleaned file saved as: cleaned_amazon_sales.csv")