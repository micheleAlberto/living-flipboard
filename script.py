import numpy as np 
import cv2
from collections import Counter
import matplotlib.pyplot as plt
from sklearn import cluster, datasets
from sklearn import svm
from random import shuffle, randint



def mk_locality(n,w,h):
    def locality_pattern(n):
        return ( (i,j) 
            for i in range(-n,n+1) 
            for j in range(-n,n+1) 
            if not (j==0) 
            and not (i==0) ) 
    print list(locality_pattern(n))
    def locality(i,j):
        return (
            (cv2.borderInterpolate(i+a, w, cv2.BORDER_WRAP),
             cv2.borderInterpolate(j+b, h, cv2.BORDER_WRAP)) 
            for a,b in locality_pattern(n))
    return locality
def center(n):
    return 2*n*(n+1)

def query(c,x,n):
    cent=center(n)
    x_up=list(x)
    x_up[cent]=1  
    x_do=list(x)
    x_do[cent]=0
    x_do=tuple(x_do)
    x_up=tuple(x_up)
    if c[x_up]==c[x_do]:
        return np.random.random()
    else:
        if np.random.random()>0.2:
            return c[x_up]>c[x_do]
        else :
            return not c[x_up]>c[x_do]
   
def make_counter(IM,n):
    cv2.borderInterpolate(p, len, borderType)

def feat(i,j,im):
    return tuple( [im[i,j] for i,j in loc(i,j)])

def generalize(c,g):
    selected=sorted(c,key=c.get)[-g:]
    return {s:c[s] for s in selected}

k_means = cluster.KMeans(n_clusters=10)
clf = svm.SVR(kernel='rbf')

w = 180
h = 150
def each_pixel():
    return ( (i,j) 
        for i in range(w)  
        for j in range(h) )

def each_pixel_of(im):
    return ( (i,j) 
        for i in range(im.shape[0])  
        for j in range(im.shape[1]) )


im = np.array([[
        np.random.binomial(1, 0.5) 
    for i in range(h)] 
    for j in range(w)])

im = np.array([[
        np.random.binomial(1, 0.5) 
    for i in range(h)] 
    for j in range(w)])

me=np.mean(im)
print me
"""
for i,j in each_pixel_of(im):
    #print im[i,j]
    im[i,j]=1 if np.mean(im[i,j])<me else 0
"""
#im= im[:,:,0]>me
cv2.imwrite('zero.png',im*255)
for label,n in enumerate(range(3,9)):
    loc=mk_locality(n,w,h)
    def feat(i,j,im):
        return np.array( [im[i,j] for i,j in loc(i,j)])
    print 'eval pixel feats'
    pixels=[(i,j) for i,j in each_pixel() if randint(0,100)>90]
    X=[feat(i,j,im) for i,j in pixels]
    Y=[im[i,j]      for i,j in pixels]
    print 'fiting'
    clf.fit(X,Y)
    #for x,y in zip(X,Y):
    #   print x,y
    for prove in range(n+1):
        print prove
        #print 'shuffling pixels'
        #shuffled_each_pixel=shuffle(each_pixel())
        print 'update feat tensor'
        feat_tensor=np.array(
            [
            feat(i,j,im)
            for j in range(w)
            for i in range(h)]
            )
        print 'prediction'
        im=clf.predict(feat_tensor)
        im.shape=(w,h)
        im=im>np.median(im)
        cv2.imwrite('{}_{}.png'.format(label,prove),im*255)
        print 'update pixel feats'
        pixels=[(i,j) for i,j in each_pixel() if randint(0,100)>90]
        X=[feat(i,j,im) for i,j in pixels]
        Y=[im[i,j]      for i,j in pixels]
        print 're-fit'
        clf.fit(X,Y)
