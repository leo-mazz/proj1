import pandas as pd
from pandas.api.types import CategoricalDtype
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import FeatureUnion
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

from dataset import adult
import experiment


# For numerical columns: want to be able to select them and normalize them
class ColumnsSelector(BaseEstimator, TransformerMixin):
    def __init__(self, type):
        self.type = type

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.select_dtypes(include=[self.type])


num_pipeline = Pipeline(steps=[
    ("num_attr_selector", ColumnsSelector(type='int')),
    ("scaler", StandardScaler())
])


# Fill missing categorical values with most present in the column
class CategoricalImputer(BaseEstimator, TransformerMixin):

    def __init__(self, columns=None, strategy='most_frequent'):
        self.columns = columns
        self.strategy = strategy


    def fit(self, X, y=None):
        if self.columns is None:
            self.columns = X.columns

        if self.strategy is 'most_frequent':
            self.fill = {column: X[column].value_counts().index[0] for column in self.columns}
        else:
            self.fill ={column: '0' for column in self.columns}
            
        return self
        
    def transform(self,X):
        X_copy = X.copy()
        for column in self.columns:
            X_copy[column] = X_copy[column].fillna(self.fill[column])
        return X_copy

# One-hot encoding for categories
class CategoricalEncoder(BaseEstimator, TransformerMixin):
  
    def __init__(self, dropFirst=True, train_data=None, test_data=None):
        self.categories = dict()
        self.dropFirst = dropFirst
        self.train_data = train_data
        self.test_data = test_data

    def fit(self, X, y=None, train_data=None, test_data=None):
        join_df = pd.concat([self.train_data, self.test_data])
        join_df = join_df.select_dtypes(include=['object'])
        for column in join_df.columns:
            self.categories[column] = join_df[column].value_counts().index.tolist()
        return self

    def transform(self, X):
        X_copy = X.copy()
        X_copy = X_copy.select_dtypes(include=['object'])
        for column in X_copy.columns:
            X_copy[column] = X_copy[column].astype({column:
                    CategoricalDtype(self.categories[column])})
        return pd.get_dummies(X_copy, drop_first=self.dropFirst)


@experiment.step('train_adult')
def train_and_get_accuracy(train_data, test_data):
    cat_pipeline = Pipeline(steps=[
        ("cat_attr_selector", ColumnsSelector(type='object')),
        ("cat_imputer", CategoricalImputer(columns=
            ['workClass','occupation', 'native-country'])),
        ("encoder", CategoricalEncoder(dropFirst=True, train_data=train_data, test_data=test_data))
    ])

    full_pipeline = FeatureUnion([("num_pipe", num_pipeline), ("cat_pipeline", cat_pipeline)])


    train_data.drop(['fnlwgt', 'education-num'], axis=1, inplace=True)
    test_data.drop(['fnlwgt', 'education-num'], axis=1, inplace=True)


    train_copy = train_data.copy()
    train_copy["income"] = train_copy["income"].apply(lambda x:0 if x=='<=50K' else 1)

    X_train = train_copy.drop('income', axis =1)
    Y_train = train_copy['income']


    X_train_processed=full_pipeline.fit_transform(X_train)

    model = LogisticRegression(random_state=0)
    model.fit(X_train_processed, Y_train)


    test_copy = test_data.copy()
    test_copy["income"] = test_copy["income"].apply(lambda x:0 if 
                        x=='<=50K.' else 1)

    X_test = test_copy.drop('income', axis =1)
    Y_test = test_copy['income']


    X_test_processed = full_pipeline.fit_transform(X_test)

    predicted_classes = model.predict(X_test_processed)
    acc = accuracy_score(predicted_classes, Y_test.values)

    return acc, {'results': {'adult_test_accuracy': round(acc, 4)}}


if __name__ == "__main__":
    print(train_and_get_accuracy(*adult.get_frames()))
