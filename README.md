
# Data Cleaning with AI Support

## Student Information
- **Name:** Lavigne Kaye S. Sistona  
- **Course Year:** BSCS 4  
- **Date:** 2025-09-28  

## Dataset
- **Source:** Kaggle - [Titanic: Machine Learning from Disaster](https://www.kaggle.com/c/titanic/data)  
- **Name:** Titanic  

## Issues Found
- **Missing values:**  
  - `Age` column had several missing entries.  [77 missing values (19.87%)]
  - `Cabin` had many missing values.  [687 missing values (77.1%)]
  - `Embarked` had a few missing entries.  [2 missing values (0.22%)]
- **Duplicates:**  
  - No exact duplicate rows, but checked to ensure uniqueness.  
- **Inconsistencies:**  
  - Mixed categorical values (e.g., "male"/"female" vs capitalized forms if any).  
  - Titles are not standardized.

## Cleaning Steps
1. **Missing values:**  
   - Filled missing `Age` values with the median.  
   - Dropped the `Cabin` column due to excessive missing data and created new 'Has_Cabin' binary feature
   - Filled missing `Embarked` values with the mode.  
2. **Duplicates:**  
   - Checked for duplicate `PassengerId` entries, confirmed none existed.  
3. **Inconsistencies:**  
   - Standardized categorical columns (`Sex`, `Embarked`).  
   - Converted string-based numeric columns to integer/float where applicable.  
   - Standardized the title and the name separating then to different columns.
4. **Outliers:**  
   - Detected extreme `Fare` values and capped them at the 99th percentile.  

## AI Prompts Used
- **Prompt 1:** "Generate pandas code to handle missing values in the provided dataset, it should fill all the missing values."  
- **Generated Code:**  
  ```python
    print("Before handling missing values:")
    print(f"Dataset shape: {df.shape}")
    print("Missing values per column:")
    print(df.isnull().sum())

    age_median_by_group = df.groupby(['Pclass', 'Sex'])['Age'].median()
    df['Age'] = df.apply(lambda row: age_median_by_group[row['Pclass'], row['Sex']] 
                        if pd.isnull(row['Age']) else row['Age'], axis=1)
    print(f"Filled Age missing values with median by Pclass & Sex")

    embarked_mode = df['Embarked'].mode()[0]
    df['Embarked'] = df['Embarked'].fillna(embarked_mode)
    print(f"Filled Embarked missing values with mode: {embarked_mode}")

    df['Has_Cabin'] = df['Cabin'].notna().astype(int)
    df = df.drop('Cabin', axis=1)
    print("Created 'Has_Cabin' feature and dropped 'Cabin' column")

    print("\nAfter handling missing values:")
    print(f"Dataset shape: {df.shape}")
    print("Remaining missing values:")
    print(df.isnull().sum())
  ```

- **Prompt 2:** "Create a python function that detects and treats outliers in a DataFrame using IQR and visualization"  
- **Generated Code:**  
  ```python
    print("=== OUTLIER DETECTION ===")

    numerical_cols = ['Age', 'SibSp', 'Parch', 'Fare']

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.ravel()

    for i, col in enumerate(numerical_cols):
        df[col].hist(bins=30, ax=axes[i])
        axes[i].set_title(f'Distribution of {col}')
        axes[i].set_xlabel(col)
        axes[i].set_ylabel('Frequency')

    plt.tight_layout()
    plt.show()

    def detect_outliers_iqr(df, column):
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
        return outliers, lower_bound, upper_bound

    print("Outlier analysis:")
    for col in numerical_cols:
        outliers, lower, upper = detect_outliers_iqr(df, col)
        print(f"{col}: {len(outliers)} outliers (bounds: [{lower:.2f}, {upper:.2f}])")

    fare_q95 = df['Fare'].quantile(0.95)
    df['Fare'] = np.where(df['Fare'] > fare_q95, fare_q95, df['Fare'])
    print(f"\nFare values capped at 95th percentile: {fare_q95:.2f}")

    df['AgeGroup'] = pd.cut(df['Age'], 
                        bins=[0, 12, 18, 35, 60, 100], 
                        labels=['Child', 'Teen', 'Adult', 'Middle', 'Senior'])

    print("\nAfter outlier treatment - Summary:")
    display(df[numerical_cols].describe())
  ```


## Results  

- **Rows before cleaning:** 891  
- **Rows after cleaning:** 891 (no rows dropped, only cleaned)  

### Shape Comparison  
- **Before:** (891, 12) → 891 rows, 12 columns  
- **After:** (891, 11) → 891 rows, 14 columns (2 columns added)  

### Missing Values  
- **Before:**  
  - Age: 177 missing  
  - Cabin: 687 missing  
  - Embarked: 2 missing  
- **After:**  
  - Age: 0 missing (filled with median)  
  - Cabin: removed  and change to has_Cabin
  - Embarked: 0 missing (filled with mode)  

### Descriptive Statistics Comparison  

**Age**  
- Before: count = 714, mean ≈ 29.7, std ≈ 14.5, min = 0.42, max = 80  
- After: count = 891, mean ≈ 29.4, std ≈ 13.3, min = 0.42, max = 80  

**Fare**  
- Before: count = 891, mean ≈ 32.2, std ≈ 49.7, min = 0, max ≈ 512  
- After: count = 891, mean ≈ 27.1, std ≈ 37.4, min = 0, max ≈ 250 (capped at 99th percentile)  

**Embarked**  
- Before: 889 non-null values, 2 missing  
- After: 891 non-null values, no missing, categories standardized `{S, C, Q}`  

## Generated outputs
- Cleaned dataset saved to `data/cleaned_dataset.csv`.


**Video:** [Youtube Link:  ](https://youtu.be/Om5p8PGYWEo)
