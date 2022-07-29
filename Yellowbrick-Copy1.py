#!/usr/bin/env python
# coding: utf-8

# In[257]:


get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from yellowbrick.style.palettes import PALETTES, SEQUENCES, color_palette

YBfruits = pd.read_table('fruit_data_with_colors.txt')
YBfruits.head(1)


# In[245]:


print(YBfruits['fruit_name'].unique())
print(YBfruits['fruit_label'].unique())


# In[237]:


print(YBfruits.groupby('fruit_name').size().to_string().title())


# In[258]:


features = ['mass', 'width', 'height', 'color_score']
X = YBfruits[features]
y = YBfruits['fruit_label']


# In[247]:


print(X.head(1).to_string(index = False).title(), "   ", "Category = ", y.head(1).to_string(index = False).title())


# In[199]:


from yellowbrick.target import ClassBalance

visualizerA = ClassBalance(labels = [i for i in YBfruits['fruit_name'].unique()])
visualizerA.fit(y)
visualizerA.show();


# In[159]:


from yellowbrick.features import Rank1D

visualizer2 = Rank1D(algorithm = 'shapiro')
visualizer2.fit(X,y)
visualizer2.transform(X)
visualizer2.show();


# In[160]:


from yellowbrick.features import Rank2D
import matplotlib.pyplot as plt
visualizer1 = Rank2D(algorithm = 'pearson', size = (596,396), title= 'Correlation Between Independent Variables')
visualizer1.fit_transform(X)
visualizer1.show();


# In[161]:


from yellowbrick.model_selection import FeatureImportances
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(multi_class = 'auto', solver = 'liblinear')
visualizer3 = FeatureImportances(model, 
                                 stack = True, 
                                 relative = False, 
                                 xlabel = '\n 1 = Apple | 2 = Mandarin | 3 = Orange | 4 = Lemon')
visualizer3.fit(X,y)
visualizer3.show();


# In[165]:


modelb = LogisticRegression(multi_class = 'auto', solver = 'liblinear')
visualizer3b = FeatureImportances(modelb, 
                                 stack = True, 
                                 relative = False, 
                                  topn = 2)
visualizer3b.fit(X,y)
visualizer3b.show();


# In[170]:


from sklearn import datasets
toydf = datasets.load_diabetes()
x2 = toydf.data
y2 = toydf.target

from yellowbrick.regressor import ResidualsPlot
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(x2, y2, test_size = 0.25, random_state = 0)

visualizer = ResidualsPlot(LinearRegression())
visualizer.fit(X_train, y_train)
visualizer.score(X_test, y_test)
visualizer.show();


# In[356]:


import warnings
warnings.filterwarnings('ignore')


# In[363]:


from yellowbrick.features.pca import pca_decomposition
import matplotlib.pyplot as plt


X = YBfruits[features]
y = YBfruits['fruit_label'].astype('category').cat.codes

visualizer = PCA(projection = 3, 
                 scale = True, 
                 size = (996,696), 
                 classes = ['apple', 'lemon','mandarin', 'orange'],
                 proj_features=True,
                 features = ['height', 'width','mass','color_score'])

visualizer.fit_transform(X,y)
visualizer.show();


# In[395]:


from yellowbrick.regressor import ResidualsPlot
from sklearn.neighbors import KNeighborsClassifier
from yellowbrick.contrib.classifier import DecisionViz
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split as tts

subfeatures = ['width', 'height']
y = YBfruits['fruit_label'].astype('category').cat.codes.tolist()
X = YBfruits[subfeatures]
X = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = tts(X, y, test_size = 0.25, random_state = 0)

viz5 = DecisionViz(KNeighborsClassifier(), 
                   features = ['width', 'height'], 
                   classes = ['apple', 'orange','mandarin','lemon'], 
                   alpha =1)

viz5.fit(X_train, y_train)
viz5.draw(X_test, y_test)
viz5.show();

