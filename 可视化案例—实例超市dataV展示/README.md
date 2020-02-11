# 项目描述
### DataV是阿里开发的灵活性极高的可视化软件。本项目利用DataV实现数据可视化

###一、Navicat连接阿里云RDS数据库
[https://www.cnblogs.com/zhangmeihuizi/articles/10392417.html](https://www.cnblogs.com/zhangmeihuizi/articles/10392417.html)
ps：本地ip会变动，所以需要自己更改
###二、下载tableau官方数据《示例超市》数据，导入mysql
workbench导入csv太慢了，用navicat导入数据，导入后再用workbench修改数据类型/列名。
注意，tableau是可以处理货币格式的，但是导入mysql需要转化成decimal，无法附带货币单位。
###三、画图
dataV基本可视化操作：[https://help.aliyun.com/document_detail/74195.html?spm=a2c4g.11186623.6.681.2b284862jzWCHo](https://help.aliyun.com/document_detail/74195.html?spm=a2c4g.11186623.6.681.2b284862jzWCHo)
[https://blog.csdn.net/u010886217/article/details/85224596](https://blog.csdn.net/u010886217/article/details/85224596)
目标图：

![image.png](https://upload-images.jianshu.io/upload_images/18032205-f1c4861aed854d36.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

1、热力图部分
下载geoatlas编码，left join表来求得各个省份的平均利润率
```
select sum(profit)/sum(saleamount) as profitrate_province,province,编码 from market left join geoatlas on market.province=geoatlas.名称 group by province,编码;
```
2、销售额部分
3、顶部概述部分

四、页面分享
https://datav.aliyuncs.com/share/19377ea7f80eb43b2221e9ee98a2d138

![image.png](https://upload-images.jianshu.io/upload_images/18032205-33fc17addcad2f70.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
