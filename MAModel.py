#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2017-05-15 20:00:00
# @Author  : 林子珩 (zhlin@hhu.edu.cn)
# @Link    : https://edlinus.cn
# @Version : 0.1
import os
import numpy as np
import matplotlib.pyplot as plt

'''
迭代计算回归系数
xita:上一次迭代出的系数数组
dx:数据系列的均方差
x:数据系列
'''
def iteration(xita,dx,x,count=0):
    _xita=np.zeros(len(xita))
    rou=np.zeros(len(xita))
    ds=dx/(1+sum(np.power(xita,2)))
    for i in range(0,len(xita)):
        rou[i]=rho(x,i+1)[0,1]
        _xita[i]=cparams(xita,rou[i],ds,dx,i+1)
    dx=ds
    if count!=0:
        cor=np.corrcoef(xita,_xita)[0,1]
    else:
        cor=0
    if cor>=0.9:
        return _xita,ds
    else:
        return iteration(_xita,dx,x,count+1)
'''
计算相关系数
x:系列
ss:下标号
'''
def rho(x,ss):
    xt=x[ss:]
    xl=x[:len(x)-ss]
    cor=np.corrcoef(xt,xl)
    return cor
'''
计算数组和
narr:数组
lens:计算长度，默认为整个数组
'''
def sum(narr,lens=0):
    if lens==0:
        lens=len(narr)
    s=0
    for i in range(0,lens):
        s+=narr[i]
    return s
'''
计算系数
xita:上一次迭代出的系数数组
rou:样本相关系数
ds:随机参数的方差
dx:样本的方差
n:系数序号
'''
def cparams(xita,rou,ds,dx,n):
    lens=len(xita)
    _xita=-rou*dx/ds
    if n==lens:
        return _xita
    else:
        for i in range(0,lens-n,n+1):
            _xita+=xita[i]*xita[i+n]
        return _xita
'''
交叉乘
m1:系列1
m2:系列2
n:终止位置
'''
def cross(m1,m2,n):
    s=0
    for i in range(0,n):
        s+=m1[i]*m2[n-i-1]
    return s
'''
计算随机变量的值
x:数据系列
xita:回归系数数组
'''
def cran(x,xita):
    ran=np.zeros(len(xita)+1)
    ran[0]=x[0];
    for i in range(1,len(xita)+1):
        s=0
        for j in range(0,i):
            s+=cross(xita,ran,j+1)
        ran[i]=s+x[i]
    return ran
'''
计算随机变量的偏态系数
ran:随机变量数组
'''
def ccs(ran):
    std=pow(np.std(ran),3)
    n=len(ran)
    s=0
    for r in ran:
        s+=pow(r,3)
    cs=s/((n-3)*std)
    return cs
'''
计算相关系数数组
x:系列数组
'''
def ccorr(x,q):
    cor=np.zeros(q)
    for i in range(0,q):
        cor[i]=rho(x,i+1)[0,1]
    return cor

'''
计算数值在数组中的位置
num:数字
list:数组
'''
def pos(num,list):
    ub = 0;db=0
    for i in range(0,len(list)-1):
        ubn=list[i];dbn=list[i+1]
        if num>=ubn and num<=dbn :
            ub=i;db=i+1
    return ub,db
'''
生成PIII型分布的随机误差
u:0-1均匀分布的随机数
csr:随机误差的离势系数
plist:φ值表矩阵
'''
def p3num(u,csr,dr,plist):
    u=u*100
    xp=pos(csr,plist[:,0])
    yp=pos(u,plist[0,:])
    if xp[0]==xp[1] or yp[0]==yp[1]:
        return 0
    cst1=plist[xp[0],yp[0]]+plist[xp[1],yp[0]]*(u-plist[0,yp[0]])/(plist[0,yp[1]]-plist[0,yp[0]])
    cst2=plist[xp[0],yp[1]]+plist[xp[1],yp[1]]*(u-plist[0,yp[1]])/(plist[0,yp[1]]-plist[0,yp[0]])
    p3n=cst1+(cst2-cst1)*(csr-plist[xp[0],0])/(plist[xp[1],0]-plist[xp[0],0])
    p3n=dr*p3n
    return p3n
'''
确定滑动阶数q
dk:标准差
cor:相关系数数组
'''
def checkq(dk,cor):
    q=0;count=0
    for i in range(0,len(cor)):
        if dk>cor[i]:
            for j in range(i,len(cor)):
                if dk>cor[j]:
                    count+=1
            if count==len(cor)-i-1:
                q=i+1
                count=0
    return q

filepath=os.path.dirname(os.path.realpath(__file__))
x=np.loadtxt(filepath+"\\x.dat")#数据导入
n=len(x) #数据数目
xm=np.mean(x) #平均值
y=x-xm #中心化系列

dx=np.std(y)#系列标准差
cor=ccorr(x,int(len(x)/2))#相关系数数组
dk=pow(1+2*sum(np.power(cor,2)),0.5)/np.sqrt(60)#误差项
q=checkq(dk,cor)#滑动阶数
xt=np.zeros(q)#初试xita数组
[xita,ds]=iteration(xt,pow(dx,2),y)
ran=cran(y,xita)#随机变量数组
b=ran[len(ran)-1] #回归方程截距
str1=""
for c in range(0,len(xita)):
    b+=-xita[c]*xm
    if xita[c]>=0:
        str1+='+'+str(xita[c]) + "ε|t-" + str(c+1)
    else:
        str1+=str(xita[c]) + "ε|t-" + str(c+1)
str0="回归方程为：y="
str0+=str(b) + str1 + "+rt"
print(str0)

csr=ccs(ran)
dr=np.std(ran)
#fp = input("请输入预报长度:\n") #预报长度
fp=10
xin=ran[len(ran)-q:]
plist=np.loadtxt(filepath+"\\φ.dat")
yf=np.zeros(fp)
for i in range(0,fp):
    u=np.random.rand()
    p3n=p3num(u,csr,dr,plist)
    #print(p3n)
    yf[i]+=b+p3n
    for j in range(0,len(xita)):
        yf[i]+=xin[j]*xita[len(xita)-j-1]
    for k in range(0,len(xita)-1):
        xin[k]=xin[k+1]
    xin[len(xita)-1]=p3n
np.savetxt(filepath+"\\y.dat",yf,fmt='%s')
print("预报结果已保存在"+filepath+"\\y.dat")
plt.figure(figsize=(8,4))
xl=range(1,len(yf)+1)
plt.plot(xl,yf,label="$Forecast Stage(m)$",color="red",linewidth=2)
plt.ylabel("Stage(m)")
plt.xlabel("Time(day)")
plt.legend()
plt.show()