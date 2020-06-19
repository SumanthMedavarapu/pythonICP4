import pandas as pd

train_df = pd.read_csv('train.csv')
X_train= train_df.drop("Survived",axis=1)                                       #dropping Survived column and axis = 1 means column
Y_train= train_df["Survived"]                                                   # storing Survived column into y_train
train_df['Sex'] = train_df['Sex'].map({'female': 1, 'male': 0}).astype(int)     # keeping female = 1 and male = 0 in sex Column
print(train_df['Survived'].corr(train_df['Sex']))                               # finding correlation and printing it