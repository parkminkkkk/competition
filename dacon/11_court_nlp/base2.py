#TfidfVectorizer + LogisticRegression

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

path = 'd:/study/_data/court/'
train = pd.read_csv(path + 'train.csv')
test = pd.read_csv(path + 'test.csv')

import re

def extract_numerical_feature(text):
    # 예시: 텍스트에서 숫자 개수 추출
    numerical_count = len(re.findall(r'\d+', text))
    return numerical_count

# Extract numerical features from text
train['numerical_feature'] = train['facts'].apply(extract_numerical_feature)
test['numerical_feature'] = test['facts'].apply(extract_numerical_feature)

# Standardize numerical features
scaler = StandardScaler()
train['numerical_feature'] = scaler.fit_transform(train['numerical_feature'].values.reshape(-1, 1))
test['numerical_feature'] = scaler.transform(test['numerical_feature'].values.reshape(-1, 1))

# Define additional feature extraction function
def extract_additional_features(df):
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    X_facts = vectorizer.fit_transform(df['facts'])
    X_party1 = vectorizer.transform(df['first_party'])
    X_party2 = vectorizer.transform(df['second_party'])
    return np.concatenate([X_party1.todense(), X_party2.todense(), X_facts.todense(), df['numerical_feature'].values.reshape(-1, 1)], axis=1)



# Get additional features for training and testing data
X_train_additional = extract_additional_features(train)
X_test_additional = extract_additional_features(test)

# Combine TF-IDF features and additional features
X_train_combined = np.concatenate([train, X_train_additional], axis=1)
X_test_combined = np.concatenate([test, X_test_additional], axis=1)

# Define Model & Train
model = LogisticRegression(random_state=337)
model.fit(X_train_combined, train)

# Inference & Submission
submit = pd.read_csv(path + 'sample_submission2.csv')
pred = model.predict(X_test_combined)
submit['first_party_winner'] = pred
submit.to_csv('./_save/court/baseline_submit.csv', index=False)
print('Done')