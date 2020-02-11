# 项目描述
### 基于kaggle公开数据集，对谷歌应用市场的APP情况进行数据探索和分析。


from kaggle:
https://www.kaggle.com/lava18/google-play-store-apps

#分析思路：
> 0、数据准备
>1、数据概览
>2、种类对Rating的影响
>3、定价策略
>4、因素相关性分析
>5、用户评价
>6、总结



#0、数据准备
####（1）模块及数据导入
导入所需数据模块：
```
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
%matplotlib inline
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
import seaborn as sns 
```
导入数据，并检查数据的完整性：
```
review=pd.read_csv(r'D:\Users\wuxiao\Desktop\数据分析\数据分析案例\google-play-store-apps\googleplaystore_user_reviews.csv')
app=pd.read_csv(r'D:\Users\wuxiao\Desktop\数据分析\数据分析案例\google-play-store-apps\googleplaystore.csv')
review.info()
print('\n')
app.info()
```
![image.png](https://upload-images.jianshu.io/upload_images/18032205-e1957aa7247751a4.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
包含两个数据表：app和review，其中app是主要的分析来源。
####（2）对app表进行处理：
app中共13个字段，每个字段10841行，分别解释为：
- App:app名称
- Category：app所属类别
- Rating：评分
- Reviews:评论数
- Size：app内存大小
- Installs：安装次数
- Type：免费或收费
- Price：价格
- Content Rating：内容等级
- Genres： 体裁类型（对Category的二级分类）
- Last Updated：最近更新日期
- Current Ver：版本号
- Andriod Ver：安卓版本号

数据清洗：
A、去重和补数据
检查数据重复性：
```
app.duplicated().value_counts()
```
![image.png](https://upload-images.jianshu.io/upload_images/18032205-712f776879334315.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
对数据进行去重（因为app是不允许重名的，而在数据里面有app重名，但个别类目有差异的情况，可以认为数据错误）、缺失值不处理（因为如果补0的话，会对数据有很大的影响，比如评分补0）：
```
app.drop_duplicates(subset='App',inplace=True)
app.info()
```
B、数据转换
- Size全部转化为以MB为单位的float
- Installs转化为int
- Price转化为float
- Reviews转化为int
```
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
```
C、不良数据处理
查看处理后的数据整体情况：
![image.png](https://upload-images.jianshu.io/upload_images/18032205-52d6ac52d27f4aa1.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
发现Rating中的存在19的值，将其删除：
```
def Rating_transform(x):
    if x>5:
        return float('NaN')
    else:return x
app.Rating=app.Rating.transform(Rating_transform)
```
数据处理完后变为：
![image.png](https://upload-images.jianshu.io/upload_images/18032205-67e0bde92a2dd263.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
####(3)对review表进行处理：
review表共包含5个字段，每个字段64295行，分别解释为：

- App:app名称
- Translated_Review：用户评价
- Sentiment：态度
- Sentiment_Polarity：情绪极性
-Sentiment_Subjectivity：情绪主观性（下面的分析不会用到这一点）

数据清洗：
A、去无效数据和去重
由于存在大量的无评论数据，认为是无效数据；同时存在大量的重复数据：
```
review=review.dropna()
review.drop_duplicates()
```
处理后数据变为：
![image.png](https://upload-images.jianshu.io/upload_images/18032205-bc7afa2b887158f3.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


#1、数据概览
####（1）数据相关性（如果之前对Rating缺失值补0，会很大程度影响实际评价）
```
x=app.Rating
y=app.Size
z=np.log(app.Installs[app.Installs!=0])
p=np.log10(app.Reviews[app.Reviews!=0])
t=app.Type[(app.Type=='Free') | (app.Type=='Paid')]
price=app.Price

aa=pd.DataFrame(list(zip(x,y,z,p,t,price)),columns=['Rating','Size','Installs','Reviews','Type','Price'])
pp=sns.pairplot(aa,hue='Type')
```
![](https://upload-images.jianshu.io/upload_images/18032205-2a979a3b2e705cc0.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
####（2）不同种类app的个数占比
```
number=app['Category'].value_counts()#自动降序排列
plt.rcParams['figure.figsize'] = (20.0, 10.0)
plt.pie(number.values,labels=number.index,autopct='%.2f%%')
```
![image.png](https://upload-images.jianshu.io/upload_images/18032205-48f0de03d84ca12e.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
- 从图中可以看出，排名前五的为FAMILY、GAME、TOOLS 、BUSINESS 和MEDICAL类。
(由于种类数太多，lable发生重叠，主要有两种调整方案：
- 针对text依次调整：[https://www.jianshu.com/p/0a76c94e9db7](https://www.jianshu.com/p/0a76c94e9db7)
- 或者采用plotly在线绘图api，绘图优化更好，好像是可以实现lable单列的)

#2、种类对Rating的影响
####（1）Rating整体情况
```
plt.figure(figsize=(20,10))
plt.hist(app.Rating,bins=50)
print('app平均得分为%.3f'%(np.mean(app.Rating)))#np.average不会忽略nan值
```
![image.png](https://upload-images.jianshu.io/upload_images/18032205-56944faba4908099.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
可知：平均得分4.173分；得分在4.0~4.5分比较集中。

####（2）种类对Rating的影响
A、不同种类Rating的差异性分析
```
import scipy.stats as stats
#挑选占比前十的种类进行单因素方差分析
#f_oneway函数不接受nan参数
f=stats.f_oneway(app.loc[app.Category=='FAMILY','Rating'].dropna(),app.loc[app.Category=='GAME','Rating'].dropna(),app.loc[app.Category=='TOOLS','Rating'].dropna(),app.loc[app.Category=='BUSINESS','Rating'].dropna(),app.loc[app.Category=='MEDICAL','Rating'].dropna(),app.loc[app.Category=='PERSONALIZATION','Rating'].dropna(),app.loc[app.Category=='PRODUCTIVITY','Rating'].dropna(),app.loc[app.Category=='LIFESTYLE','Rating'].dropna(),app.loc[app.Category=='FINANCE','Rating'].dropna(),app.loc[app.Category=='SPORTS','Rating'].dropna())
f
```
显示：
![image.png](https://upload-images.jianshu.io/upload_images/18032205-c6541c5cd08b2650.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
由于p值是远小于给定显著性水平（好像是0.05），因此认为不同Category具有明显不同的Rating情况。
```
groups=app.groupby('Category').filter(lambda x:len(x)>=288).reset_index()
groups.Rating.hist(by=groups.Category,figsize=(20,20))
#只能用’方法‘来画图，因为plt.hist的函数没有by这个选项）
```
![image.png](https://upload-images.jianshu.io/upload_images/18032205-73b44cfee5752eca.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
可以看出不同种类的Rating，有的比较集中，有的分散，总体来说，具有明显差异性。
B、表现最好的种类
```
fig,ax=plt.subplots(figsize=(10,10))
ax.set_title('App ratings across major categories')
plt.xticks(rotation=90)
sns.boxplot(x=groups.Category,y=groups.Rating)
```
![image.png](https://upload-images.jianshu.io/upload_images/18032205-a39a19986f736ca5.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

（seaborn常见绘图总结：[https://blog.csdn.net/qq_40195360/article/details/86605860](https://blog.csdn.net/qq_40195360/article/details/86605860)）
从箱图中可以看出：
- 表现最好的种类是Books and Reference以及Health and Fitness，两者有50%以上的评价在4.5及以上；
- Business、Travel and local和Tools的表现不够好，普遍评分不高，且存在个别极端低分。

#3、app的Size大小对Rating的影响
```
#要保证x和y的个数一致
x=app[['Size','Rating']].dropna().Size
y=app[['Size','Rating']].dropna().Rating
sns.jointplot(x,y)
```
![image.png](https://upload-images.jianshu.io/upload_images/18032205-9dcc234c5dd46506.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
图示结果表明，Size越小，得高分的app越多，当Size增大时，高分app变少。但这可能是由于Size较大的样本量变少导致的，需要进一步分析：
```
x=list(app[['Size','Rating']].dropna().Size)
y=list(app[['Size','Rating']].dropna().Rating)

a=np.linspace(0,100,21)

#建造arrayy记录对应size段的个数和高分Rating的个数
arrayy=np.zeros((21,2))
for i in range(len(x)):
    arrayy[int(np.floor(x[i]/5)),0]+=1
    if y[i]>=4.173:  #高于平均分的认为是高分app
        arrayy[int(np.floor(x[i]/5)),1]+=1
#b来记录每个size分段的高分率
b=arrayy[:,1]/arrayy[:,0]

plt.bar(a,b,width=3)
plt.xlabel('Size/Mb')
plt.ylabel('HighRating_ratio')
```
![image.png](https://upload-images.jianshu.io/upload_images/18032205-1ab5228cee5fae1d.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
从图中可以看出，Size越大，似乎高分率更高，需要对此进行进一步挖掘。
对各个种类的Size做差异性分析：
```
f=stats.f_oneway(app.loc[app.Category=='FAMILY','Size'].dropna(),app.loc[app.Category=='GAME','Size'].dropna(),app.loc[app.Category=='TOOLS','Size'].dropna(),app.loc[app.Category=='BUSINESS','Size'].dropna(),app.loc[app.Category=='MEDICAL','Size'].dropna(),app.loc[app.Category=='PERSONALIZATION','Size'].dropna(),app.loc[app.Category=='PRODUCTIVITY','Size'].dropna(),app.loc[app.Category=='LIFESTYLE','Size'].dropna(),app.loc[app.Category=='FINANCE','Size'].dropna(),app.loc[app.Category=='SPORTS','Size'].dropna())
```
![image.png](https://upload-images.jianshu.io/upload_images/18032205-0e99d2f72e45f62c.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
落入拒绝域，说明各个种类的Size有明显区别：
```
fig,ax=plt.subplots(figsize=(10,10))
ax.set_title('App Size across major categories')
plt.xticks(rotation=90)
plt.ylim((0,100))
sns.boxplot(x=groups.Category,y=groups.Size)
```
![image.png](https://upload-images.jianshu.io/upload_images/18032205-bddb6e14033d4a45.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
可以看出，Game和Family类别的Size明显比其他种类大，说明该类应用因功能所需，通常需要更大的Size。
对表现最好的Books and Reference和Health and Fitness进行Size-Rating关联性分析：
```
aaa=app.loc[(app.Category=='BOOKS_AND_REFERENCE')|(app.Category=='HEALTH_AND_FITNESS'),['Size','Rating','Category']]
sns.scatterplot(y=aaa.Size,x=aaa.Rating,hue=aaa.Category)
```
![image.png](https://upload-images.jianshu.io/upload_images/18032205-3b6d465f06818018.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
- 从图中可以看出，Size-Rating的关联性不够强，可以一定程度认为，HEALTH类的超低分段主要集中在50M以下，BOOKS的中低分段主要集中在60M以下，因此这两类app的Size不宜做得过小，但实际上当Size较小时仍有很多高得分，因此Size-Rating关联性缺乏数据支撑。

#3、定价策略
####（1）总体价格策略
![image.png](https://upload-images.jianshu.io/upload_images/18032205-97b7fb5c65997212.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
- 收费软件的价格一般处在0~50之间，但极少数app收费超过了300；
- 高分收费软件的价格一般处在0~30之间。
```
groups=app.groupby('Category').filter(lambda x:len(x)>=330).reset_index()
plt.figure(figsize=(10,10))
sns.set(style='darkgrid')
plt.xticks(rotation=45)
sns.stripplot(x='Category',y='Price',data=groups)
```
![image.png](https://upload-images.jianshu.io/upload_images/18032205-4f974e515fe92cc2.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
- 收费较高的种类有:FAMILY、MEDICAL
- 个别app收费不合理（Price>200），他们是：
![image.png](https://upload-images.jianshu.io/upload_images/18032205-29610d588e8a0c9f.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
####（2）不同种类的定价策略
针对五大类FAMILY、GAME、TOOLS 、BUSINESS 和MEDICAL的收费app占比：
```
#确定双饼图所需的数据
aaa=app.loc[(app.Category=='FAMILY')|(app.Category=='GAME')|(app.Category=='TOOLS')|(app.Category=='BUSINESS')|(app.Category=='MEDICAL')]
bbb=aaa[['Type','Category']]
bbb['num']=0  #新建一列用于count，因为groupby的count函数只针对数值栏有效？？？
ccc=dict(bbb.groupby(['Category','Type']).count().num)

x1=[ccc[('FAMILY', 'Free')],ccc[('FAMILY', 'Paid')],ccc[('GAME', 'Free')],ccc[('GAME', 'Paid')],ccc[('TOOLS', 'Free')],ccc[('TOOLS', 'Paid')],ccc[('BUSINESS', 'Free')],ccc[('BUSINESS', 'Paid')],ccc[('MEDICAL', 'Free')],ccc[('MEDICAL', 'Paid')]]
x2=list(dict(aaa.Category.value_counts()).values())
#画图
label1=['Free','Paid']*5
label2=['FAMILY','GAME','TOOLS','BUSINESS','MEDICAL']

color=['aqua','linen','lightcoral','olive','gold']
plt.figure(figsize=(10,10))
plt.pie(x1,labels=label1,labeldistance=0.5,radius=0.8,wedgeprops=dict(linewidth=2))
plt.pie(x2,labels=label2,autopct='%3.1f%%',colors=color,labeldistance=1.1,radius=1,pctdistance=0.9,wedgeprops=dict(linewidth=3,width=0.18))
```
（双饼图制作参考https://www.jianshu.com/p/428daad5b3c6）
![image.png](https://upload-images.jianshu.io/upload_images/18032205-b0af59a090516e1e.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
- 可以看出，绝大部分的软件仍然是免费形式。
免费软件和收费软件的下载量差异和评分差异如何？
```
temp=app[(app.Type=='Paid')|(app.Type=='Free')].groupby('Type').agg({'Rating':'mean','Reviews':'sum','Size':'mean','Installs':'sum','Price':'mean'})
temp.columns=['Rating_mean','Review_sum','Size_mean','Install_sum','Price_mean']
temp
```
![image.png](https://upload-images.jianshu.io/upload_images/18032205-fe2054aaf9646919.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
- 收费软件整体评价比免费软件更高，说明收费软件在质量上确实更有保证；
- 收费软件和免费软件的Size大小相差不大，甚至更小，说明收费软件更加注重“质”而不是“量”；
```
temp=app[(app.Type=='Paid')|(app.Type=='Free')][['Type','Category','Reviews','Installs']]
temp1=temp.groupby(['Type','Category']).Reviews.sum()/temp.groupby(['Type','Category']).Installs.sum()
temp1=temp1.to_frame()
#要对表进行透视
temp2=temp1.pivot_table(index='Category',columns='Type') 
temp2.columns=['Free', 'Paid']
#temp2=pd.DataFrame({'Free':temp_free,'Paid':temp_paid}) #boxplot只能对dataframe画图
fig,ax=plt.subplots(1,1)
ax.set_title('review_ratio',fontsize=15)
sns.boxplot(data=temp2)
```
![image.png](https://upload-images.jianshu.io/upload_images/18032205-feb4a12cdfea69ab.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
上图是免费/收费软件评论量和下载量的比率，收费软件的review_ratio明显高于免费软件，说明：
- 愿意选择收费软件的用户也有更强烈的意愿对软件提出意见和建议。

#4、因素相关性分析
```
corrmat=app.corr()
plt.figure(figsize=(15,10))
#用热力学图来表示相关性
sns.heatmap(corrmat,annot=True,cmap=sns.diverging_palette(220, 20, as_cmap=True))
```
![image.png](https://upload-images.jianshu.io/upload_images/18032205-2519c7b283213e74.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
其中评论量和安装量相关系数达到0.63，可以认为两者是强相关性关系，进一步分析：
```
temp=app
temp=temp[temp.Reviews>0]
temp=temp[temp.Installs>0]
temp['Reviews(Log Scaled)']=np.log(temp.Reviews)
temp['Installs(Log Scaled)']=np.log(temp.Installs)
sns.lmplot(x='Reviews(Log Scaled)',y='Installs(Log Scaled)',data=temp)
#没有阴影显示置信区间是因为画不出来？？
```
![image.png](https://upload-images.jianshu.io/upload_images/18032205-78cdee4e97ceda56.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
可以看出，评论数和下载安装量呈现一定的线性正相关关系，这表明:
- 一方面，可能是用户趋于下载评论量更多的app并反过来促进app下载量增加，另一方面也有可能是用户保持一定的评论活跃度，因此随着app下载增加，评论量也会不断增加；
- 考虑到以上因素，商家为了促进app的下载和推广，可以考虑增加评论的方式来吸引用户下载。

#5、用户评价
####（1）用户对不同app的态度
```
app_merge=pd.merge(app,review,on='App')

mergetemp=app_merge.groupby('Category').agg({'Sentiment':'value_counts'})
#数据转换
mergetemp=mergetemp/mergetemp.groupby('Category').sum()
#表透视
mergetemp=mergetemp.unstack()

mergetemp.plot(kind='bar',figsize=(10,6),stacked=True)
```
![image.png](https://upload-images.jianshu.io/upload_images/18032205-12b0b87b2d543a9e.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
- Health and Fitness 类软件表现良好，获得了超过85%的积极评价；
- Game and Social 类软件表现较差，负面评价较多。
```
sns.boxplot(x='Type',y='Sentiment_Polarity',data=app_merge)
```
![image.png](https://upload-images.jianshu.io/upload_images/18032205-1ec2435d15a5ed8a.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
- 免费软件的整体评价较收费软件更差，并且收获了一些极端的负值情绪极性，代表用户对某些免费软件极端不满意；
- 收费软件收获的情绪极性好得多，表明这些用户对收费软件更加“容忍和宽容”，这与前文中分析得到的“收费软件整体评分比免费软件更高”是一致相合的。

####（2）评价关键词
（采用jieba分词+wc做词云图参见：
http://www.mzh.ren/python-jieba-and-wordcloud.html，
本文中是英文评论，所以没有用到jieba分词）
```
from wordcloud import WordCloud
wc=WordCloud(background_color="white", max_words=200, colormap="Set2")

#略过了创建停用词库进行数据清洗的环节
free = app_merge.loc[app_merge.Type=='Free']['Translated_Review_new'].apply(lambda x: '' if x=='nan' else x)
wc.generate(''.join(str(free)))
plt.figure(figsize=(10, 10))
plt.imshow(wc, interpolation='bilinear')
plt.axis("off")
plt.show()
```
![image.png](https://upload-images.jianshu.io/upload_images/18032205-141ab6a947d1c561.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
- 免费软件的高频关键词：
正面的有: good, love, best, great；
负面的有：ads, bad, hate。
```
from wordcloud import WordCloud
wc=WordCloud(background_color="white", max_words=200, colormap="Set2")

#略过了创建停用词库进行数据清洗的环节
Paid = app_merge.loc[app_merge.Type=='Paid']['Translated_Review_new'].apply(lambda x: '' if x=='nan' else x)
wc.generate(''.join(str(Paid)))
plt.figure(figsize=(10, 10))
plt.imshow(wc, interpolation='bilinear')
plt.axis("off")
plt.show()
```
![image.png](https://upload-images.jianshu.io/upload_images/18032205-5756b808572776f4.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
- 收费软件的高频关键词：
正面的有: great, love, easy；
负面的有：malware, problem。

- 综上，从免费软件和收费软件的评价上来看，显然收费软件获得的评论更加积极，收费软件的用户也趋于对其提出问题建议，而不是一味的表达自己的不满意。

#6、总结
- 软件平均得分4.173分，Books and Reference以及Health and Fitness类软件表现良好，而Business、Travel and local和Tools的评分普遍不高；
- 各类软件因其功能所需，Size大小存在差异，例如Game和Family类别的Size明显比其他种类大，而Size-Rating的关联性不够强，需要更多的数据才能探究两者关系；
- 收费软件的价格一般处在0 ~ 50之间，而定价在0 ~ 30之间更容易获得高评价；
- 收费软件整体评分比免费软件更高，同时用户评论也更加积极，说明收费软件在质量上确实更有保证；
- 用户下载量和用户评论存在一定的正相关关系，为了促进app的下载和推广，可以考虑增加评论的方式来吸引用户下载。


