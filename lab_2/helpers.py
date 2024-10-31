import pandas as pd
from sklearn.preprocessing import LabelEncoder


def data_normalizer():
    # Розділимо ознаки і цільову змінну
    df = pd.read_csv('./adult.csv')
    # Checking and handling missing values
    df.dropna(inplace=True)
    # Randomly sample 20% of the DataFrame because got a 'vegetable' instead of pc with me
    df = df.sample(frac=0.2, random_state=42)  # random_state for reproducibility

    df = df.rename(columns={'39': 'income'})

    column_names = ['age', 'workclass', 'year_income', 'education', 'education_years', 'marital_status',
                    'occupation', 'relationship', 'race', 'sex', 'capital_gain', 'capital_loss',
                    'hours_per_week', 'native_country', 'income']
    df.columns = column_names
    # Encoding categorical variables
    df = pd.get_dummies(df, columns=['workclass', 'education', 'native_country', 'relationship',
                                     'marital_status', 'occupation', 'sex', 'race'])

    # Label encoding для цільової змінної income
    le = LabelEncoder()
    df['income'] = le.fit_transform(df['income'])
    # Splitting into features and target after encoding
    X = df.drop(columns=['income'])
    y = df['income']
    result = {'X': X, 'y': y}
    return result