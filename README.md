
# House Price Prediction Project

## Project Overview

This project is aimed at developing and comparing multiple machine learning models to predict house prices based on a dataset containing various house attributes. These attributes include numerical and categorical features, such as house size, condition, year built, neighborhood, and others.

The objective is to predict the sale price (`SalePrice`) of each house by applying data preprocessing techniques and using different machine learning models. The models are evaluated using **Mean Absolute Percentage Error (MAPE)** to assess their accuracy.

The project includes a detailed analysis of the dataset, preprocessing steps, visualizations of feature distributions, and in-depth performance evaluations of various machine learning models.

## Dataset Information

The dataset used for this project comes from an Excel file (`HousePricePrediction.xlsx`). It contains 2919 rows and 13 columns, where 12 columns represent features and one column (`SalePrice`) represents the target variable to be predicted.

### Key Features:
- **MSZoning**: General zoning classification (e.g., RL, RM).
- **LotArea**: Lot size in square feet.
- **OverallCond**: Overall condition rating of the house.
- **YearBuilt**: Year the house was constructed.
- **YearRemodAdd**: Year the house was remodeled.
- **Exterior1st**: Exterior covering material.
- **TotalBsmtSF**: Total square footage of the basement.
- **SalePrice**: Target variable representing the final sale price of the house.

### Sample Data:
| Id  | MSSubClass | MSZoning | LotArea | LotConfig | BldgType | OverallCond | YearBuilt | YearRemodAdd | Exterior1st | BsmtFinSF2 | TotalBsmtSF | SalePrice |
|-----|------------|----------|---------|-----------|----------|-------------|-----------|--------------|-------------|------------|-------------|-----------|
| 1   | 60         | RL       | 8450    | Inside    | 1Fam     | 5           | 2003      | 2003         | VinylSd     | 0.0        | 856.0       | 208500.0  |
| 2   | 20         | RL       | 9600    | FR2       | 1Fam     | 8           | 1976      | 1976         | MetalSd     | 0.0        | 1262.0      | 181500.0  |

### Dataset Shape:
- **Rows**: 2919
- **Columns**: 13

## Data Preprocessing

### 1. **Handling Missing Data**
   - Missing values were identified and handled appropriately. For instance, missing values in `SalePrice` were replaced with the mean of the column. Rows with missing values in other features were dropped to ensure a clean dataset.

### 2. **Categorical Feature Encoding**
   - Categorical features such as `MSZoning`, `LotConfig`, and `BldgType` were one-hot encoded. This process converts categorical data into numerical format, making it compatible with machine learning algorithms.

### 3. **Correlation Analysis**
   - A correlation matrix was generated to identify relationships between numerical features and `SalePrice`. High correlations with the target variable suggest that certain features are highly influential in determining house prices.
   - **Key Correlations**:
     - `TotalBsmtSF`: 0.61 correlation with `SalePrice`.
     - `YearBuilt`: 0.52 correlation with `SalePrice`.
     - `YearRemodAdd`: 0.51 correlation with `SalePrice`.

### 4. **Train-Test Split**
   - The dataset was split into training (80%) and validation (20%) sets to ensure that the model could generalize to unseen data.

## Data Exploration and Visualization

### 1. **Correlation Matrix Heatmap**
   - A heatmap was created to visualize the correlations between the numeric features and `SalePrice`. Features such as `TotalBsmtSF`, `YearBuilt`, and `YearRemodAdd` showed strong positive correlations with the target variable, indicating that newer or recently renovated houses tend to sell for higher prices.

### 2. **Unique Values in Categorical Features**
   - Categorical features such as `MSZoning`, `BldgType`, and `LotConfig` were analyzed to determine the number of unique values. Features like `Exterior1st` had the most unique values, reflecting the diverse range of exterior materials used in the houses.

### 3. **Distribution of Categorical Features**
   - A series of bar plots were created to visualize the distribution of categorical features. For example, `MSZoning` predominantly featured houses zoned as `RL`, with very few houses falling under `FV` or `RH` zones. This helps us understand the dominant categories in the dataset.

## Models Used

### 1. **Support Vector Regressor (SVR)**
   - SVR is useful for capturing complex, non-linear relationships in data. However, it can be sensitive to outliers and requires careful tuning of hyperparameters.

### 2. **Random Forest Regressor**
   - This ensemble learning method uses multiple decision trees to increase prediction accuracy and reduce overfitting. It is also efficient for handling both numerical and categorical data.

### 3. **LightGBM Regressor**
   - LightGBM is a fast, high-performance gradient boosting algorithm that uses leaf-wise growth instead of level-wise growth. This results in faster training and more efficient handling of large datasets.

### 4. **CatBoost Regressor**
   - A high-performance algorithm that automatically handles categorical variables without the need for extensive preprocessing. It uses ordered boosting to reduce bias and improve accuracy.

### 5. **HistGradientBoosting Regressor**
   - This algorithm discretizes continuous variables into histograms, making it highly efficient on large datasets. It is a variant of gradient boosting that offers improved performance.

### 6. **Stacking Regressor**
   - A meta-model that combines the predictions of multiple base models (e.g., Random Forest and SVR) to generate a final prediction. Stacking often results in improved accuracy by leveraging the strengths of various models.

### 7. **Voting Regressor**
   - Combines the predictions of multiple models through averaging, improving generalization and reducing the risk of overfitting.

## Model Results and Performance Analysis

The performance of each model was evaluated using **Mean Absolute Percentage Error (MAPE)**, which measures the percentage error between the predicted and actual house prices.

### Model Performance Metrics (MAPE):
- **Stacking Regressor**: 0.1842
- **Voting Regressor**: 0.1846
- **LightGBM Regressor**: 0.1898
- **HistGradientBoosting Regressor**: 0.1901

### Analysis of Results:
- The **Stacking Regressor** provided the best performance with a MAPE of **0.1842**. This suggests that combining multiple base models (Random Forest and SVR) led to a more accurate overall prediction. The stacking approach captures different patterns from the individual models, leading to improved performance.
- **Voting Regressor** (MAPE: 0.1846) closely followed the Stacking model. Voting Regressor’s ability to average out the biases of individual models proved beneficial in achieving consistent and accurate results.
- **LightGBM Regressor** and **HistGradientBoosting Regressor** also performed well but were slightly less accurate compared to the ensemble models. These models are highly efficient for large datasets, but in this case, the ensemble methods provided a better fit for the complexity of the data.

### Conclusion:
- The **Stacking Regressor** outperformed other models, demonstrating the value of combining multiple models in complex prediction tasks like house price prediction.
- Ensemble models like **Voting** and **Stacking** excelled in this task due to their ability to leverage the strengths of multiple base learners, resulting in more accurate and generalized predictions.

### Most Accurate Model:
- **Stacking Regressor** with a MAPE of **0.1842**.

## Conclusion and Future Work

This project successfully demonstrated how machine learning models can be applied to predict house prices based on a set of categorical and numerical features. The most accurate model, the **Stacking Regressor**, achieved a MAPE of **0.1842**, outperforming other individual models.

### Potential Improvements:
- **Hyperparameter Tuning**: Further tuning of hyperparameters, particularly for models like SVR and Random Forest, could lead to even better performance.
- **Feature Engineering**: Additional feature engineering (e.g., polynomial features, interaction terms) may help capture more complex relationships in the data.
- **Cross-Validation**: Implementing cross-validation to ensure the model's stability and generalizability across different subsets of the data.

### How to Use
1. Clone the repository and install the required libraries listed in `requirements.txt`.
2. Load the dataset (`HousePricePrediction.xlsx`).
3. Run the preprocessing steps to clean and prepare the data.
4. Train the models and evaluate their performance using MAPE.
5. Visualize the results and compare the models to identify the best-performing model for house price prediction.
