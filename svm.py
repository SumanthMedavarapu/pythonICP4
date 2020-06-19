import pandas as pd                                 #importing pandas
from sklearn import model_selection                 # sklearn model_selection Splits arrays or matrices into random train and test subsets
from sklearn import svm                             # importing svm
from sklearn import metrics                         # Importing metrics module to measure classification performance.
glass_data = pd.read_csv('glass.csv')               #reading glass.csv file
x=glass_data.drop('Type',axis=1)                    # dropping Type column and axis = 1 means column
y=glass_data['Type']                                # storing Type column into y
X_train, X_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.2,random_state=0)  #splitting data into two subsets for training and testing. 80% training and 20% test
lc = svm.SVC(kernel="linear")                       # Training a SVM with a Linear Kernel
lc.fit(X_train, y_train)                            # training X_train and y_train
y_pred = lc.predict(X_test)                         # making a prediction
print("Accuracy:",metrics.accuracy_score(y_test, y_pred)) #finding accuracy and printing it
print("classification_report\n",metrics.classification_report(y_test,y_pred,zero_division=1))  #finding classification report and printing it
#print("confusion matrix\n",metrics.confusion_matrix(y_test,y_pred))  #finding confusion martix and printing it