#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
import seaborn as sns 


# In[11]:


review=pd.read_csv(r'D:\Users\wuxiao\Desktop\数据分析\数据分析案例\google-play-store-apps\googleplaystore_user_reviews.csv')
app=pd.read_csv(r'D:\Users\wuxiao\Desktop\数据分析\数据分析案例\google-play-store-apps\googleplaystore.csv')
review.info()
print('\n')
app.info()


# In[24]:


app.duplicated().value_counts()


# In[31]:


app.drop_duplicates(subset='App',inplace=True)
app=app.fillna(0)
app.info()


# In[32]:


app.App.value_counts()


# In[34]:


app.head()


# In[35]:


app.info()


# In[56]:


app.Size.strip('M')


# In[58]:


app.Size.str.strip('M')


# In[159]:


#Size转换
def size_transform(x):
    if 'Varies with device' in x:
        return float(str(x).replace('Varies with device', 'NaN'))
    elif 'M' in x:
        return float(str(x).replace('M',''))
    elif 'k' in x:
        return float(str(x).replace('k',''))/1000
    
app.Size=app.Size.transform(size_transform)
#Installs转换
def installs_transform(x):
    if '+' in x:
        x=x.replace('+','')
    if ',' in x:
        x=x.replace(',','')
    if 'Free' in x:
        x=0
    return int(x)
app.Installs=app.Installs.transform(installs_transform)
#Price转换
def price_transform(x):
    if '$' in x:
        x=x.replace('$','')
    if 'Everyone' in x:
        x=0
    return float(x)
app.Price=app.Price.transform(price_transform)
#Reviews转换为数值型
def reviews_transform(x):
    if 'M' in x:
        x=x.replace('M','')
        x=float(x)*1000000
    return int(x)
app.Reviews=app.Reviews.transform(reviews_transform)


# In[216]:


def reviews_transform(x):
    if 'M' in x:
        x=x.replace('M','')
        x=float(x)*1000000
    return int(x)
app.Reviews=app.Reviews.transform(reviews_transform)


# In[217]:


app.head()


# In[192]:





# In[169]:


app.Price.value_counts()


# In[162]:


app.head()


# In[154]:


def installs_transform(x):
    if '+' in x:
        x=x.replace('+','')
    if ',' in x:
        x=x.replace(',','')
    if 'Free' in x:
        x=0
    return int(x)


# In[155]:


b=installs_transform('100,100+')


# In[156]:


b


# In[157]:


a=app.Installs


# In[127]:


a


# In[151]:


a.transform(installs_transform).value_counts()


# In[298]:


x=app.Rating
y=app.Size
z=np.log(app.Installs[app.Installs!=0])
p=np.log10(app.Reviews[app.Reviews!=0])
t=app.Type[(app.Type=='Free') | (app.Type=='Paid')]
price=app.Price

aa=pd.DataFrame(list(zip(x,y,z,p,t,price)),columns=['Rating','Size','Installs','Reviews','Type','Price'])
pp=sns.pairplot(aa,hue='Type')


# In[325]:


app['Category'].value_counts()#自动降序排列


# In[323]:


number=app['Category'].value_counts()#自动降序排列
plt.rcParams['figure.figsize'] = (20.0, 10.0)
plt.pie(number.values,labels=number.index,autopct='%.2f%%')


# In[305]:


number.values


# In[328]:





# In[344]:


sns.distplot(app.Rating,bins=10)
print('app平均得分为:%.5f'%(np.average(app.Rating)))


# In[365]:


plt.rcParams['figure.figsize'] = (8, 5)
axes=plt.axes()
axes.set_xlim([0,5])
plt.hist(app.Rating,bins=50)


# In[ ]:




