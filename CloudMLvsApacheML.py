
# In[1]:


import numpy as np
import datetime
from sklearn.svm import LinearSVC
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import os
def svm(data,labels):
    clf = LinearSVC()

    clf.fit(data,labels)
    return clf
def svm_test(data,labels,clf):
    predicted = clf.predict(data)
    count=0
    for i in range(36224):
      if(labels[i]==predicted[i]):
        count=count+1
    print('Accuracy is'+' '+str(count/36224))
train=np.load('casia_train.npy')
train_labels=np.load('casia_train_labels.npy')
test=np.load('casia_test.npy')
test_labels=np.load('casia_test_labels.npy')
print(datetime.datetime.now())
clf=svm(train,train_labels)
count_correct_labels=svm_test(test,test_labels,clf)
count_correct_labels
print(datetime.datetime.now())


# In[2]:


import numpy as np
import datetime
from sklearn.svm import LinearSVC
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import os
def svm(data,labels):
    clf = LogisticRegression(random_state=0, multi_class='ovr')

    clf.fit(data,labels)
    return clf
def svm_test(data,labels,clf):
    predicted = clf.predict(data)
    count=0
    for i in range(36224):
      if(labels[i]==predicted[i]):
        count=count+1
    print('Accuracy is'+' '+str(count/36224))
train=np.load('casia_train.npy')
train_labels=np.load('casia_train_labels.npy')
test=np.load('casia_test.npy')
test_labels=np.load('casia_test_labels.npy')
print(datetime.datetime.now())
clf=svm(train,train_labels)
count_correct_labels=svm_test(test,test_labels,clf)
count_correct_labels
print(datetime.datetime.now())


# In[ ]:


from pyspark.ml.classification import LogisticRegression
from pyspark.ml.classification import LinearSVC
from pyspark.ml.classification import RandomForestClassifier
import datetime
from pyspark.sql import SparkSession
from pyspark import SparkContext
sc = SparkContext('local', 'my app')
from pyspark.ml.linalg import Vectors
from pyspark import SQLContext
sqlContext = SQLContext(sc)
import numpy as np
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName('abc').getOrCreate()



x_train=np.load('casia_train.npy')
x_train=x_train.astype(int)
y_train=list(np.load('casia_train_labels.npy'))

test=np.load('casia_test.npy')
test=x_train.astype(int)
test_labels=list(np.load('casia_test_labels.npy'))       
df_list = []
i = 0
df_list_test = []


for element in x_train:  # row
            tup = (int(y_train[i]), Vectors.dense(element))
            i = i + 1
            df_list.append(tup)


Train_sparkframe = spark.createDataFrame(df_list, schema=['label', 'features'])

from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.classification import OneVsRest
from pyspark.ml.classification import LinearSVC

lr = LogisticRegression()
ovr = OneVsRest(classifier=lr)
print(datetime.datetime.now())
# Fit the model
mlrModel = ovr.fit(Train_sparkframe)
print(datetime.datetime.now())


# In[ ]:




