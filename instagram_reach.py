import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Load the dataset
# Replace 'your_dataset.csv' with the actual file name
data = pd.read_csv('instagram_reach.csv')

# Split the data into features and target variables
X = data[['Username', 'Caption', 'Hashtag', 'Followers', 'Time_Since_Posted']]
y_likes = data['Likes']
y_time_since_posted = data['Time_Since_Posted']

# Split the data into training and testing sets
X_train, X_test, y_likes_train, y_likes_test, y_time_train, y_time_test = train_test_split(
    X, y_likes, y_time_since_posted, test_size=0.2, random_state=42
)

# Preprocessing for numerical features
numeric_features = ['Followers']
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

# Preprocessing for text features
text_features = ['Username', 'Caption', 'Hashtag']
text_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='')),
    ('vectorizer', CountVectorizer())
])

# Bundle preprocessing for numerical and text features
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('text', text_transformer, text_features)
    ]
)

# Define the model for predicting the number of likes
model_likes = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# Define the model for predicting time since posted
model_time_since_posted = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# Train the models
model_likes.fit(X_train, y_likes_train)
model_time_since_posted.fit(X_train, y_time_train)

# Make predictions
likes_predictions = model_likes.predict(X_test)
time_predictions = model_time_since_posted.predict(X_test)

# Evaluate the models
likes_rmse = mean_squared_error(y_likes_test, likes_predictions, squared=False)
time_rmse = mean_squared_error(y_time_test, time_predictions, squared=False)

print(f'Root Mean Squared Error (Likes): {likes_rmse}')
print(f'Root Mean Squared Error (Time Since Posted): {time_rmse}')
