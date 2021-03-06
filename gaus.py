import pandas as pd
glass_data = pd.read_csv('glass.csv')
x=glass_data.drop('Type',axis=1)
y=glass_data['Type']
from sklearn import model_selection
X_train, X_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.2,random_state=0) # 70% training and 30% test
#from sklearn.naive_bayes import GaussianNB
#model = GaussianNB()
#model.fit(X_train, y_train)
#y_pred = model.predict(X_test)
#from sklearn import metrics
#print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
#print("classification_report\n",metrics.classification_report(y_test,y_pred))
from sklearn.naive_bayes import BernoulliNB
from sklearn import metrics
from sklearn.metrics import accuracy_score
BernNB = BernoulliNB(binarize= 0)
BernNB.fit(X_train, y_train)
print(BernNB)
y_expect = y_test
y_pred = BernNB.predict(X_test)
print(accuracy_score(y_expect, y_pred))
