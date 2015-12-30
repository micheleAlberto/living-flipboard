import numpy as np 
import cv2
from collections import Counter
import matplotlib.pyplot as plt



def mk_locality(n,w,h):
    def locality_pattern(n):
        return ( (i,j) 
            for i in range(-n,n+1) 
            for j in range(-n,n+1) ) 
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

w = 144
h = 120
def each_pixel():
    return ( (i,j) 
        for i in range(w)  
        for j in range(h) )


im = np.array([[
        np.random.binomial(1, 0.9) 
    for i in range(h)] 
    for j in range(w)])
for label,n in enumerate([9,7,5,3,2,9,5,4,3,8,6,4,2]):
    loc=mk_locality(n,w,h)
    def feat(i,j,im):
        return tuple( [im[i,j] for i,j in loc(i,j)])
    for prove in range(3):
        print prove
        c=Counter( [feat(i,j,im) for i,j in each_pixel()])
        for i,j in each_pixel():
            im[i,j]=query(c,feat(i,j,im),n)
        cv2.imwrite('{}.png'.format(label),im*255)


