# To compute the subgrid dissipation we first compute mean(T_{ij}^sgs)

# import modules

import numpy as np
import struct as st
import time
import gc
from eddy import *
import glob

path='/p/work/jortizta/hybrid_spd_frinf_3/run/datafiles5p_complete/'
pathGrid='/p/work/jortizta/hybrid_spd_frinf_3/run/'

Re=100000

i,j,k=481,258,4610

iIni,iEnd=1,400
iRed=iEnd-iIni+1

kmin5p,kmax5p=1,2
skip=0

_, _,nx5pSlice, _, xe5pSlice,xc5pSlice,_, _, _, _=grid5p(pathGrid+'x3_grid.in',kmin5p,kmax5p,skip)

_,_,_,re,rc=readgrid(pathGrid+'x1_grid.in')
J,Jsym,the,thc=azi_grid(j)
_,_,_,ze,zc=readgrid(pathGrid+'x3_grid.in')

reRed=re[iIni-1:iEnd]
rcRed=rc[iIni-1:iEnd]

zeRed=xe5pSlice
zcRed=xc5pSlice

fileHeader='tvp_'
fileRes=glob.glob(path+fileHeader+'*000.5p')
listRes=[ii[-11:-3] for ii in fileRes]
nit=len(listRes)
print('Number of iteration:', nit)



t11=np.load('tauSgs11mean.npy')
t12=np.load('tauSgs12mean.npy')
t13=np.load('tauSgs13mean.npy')
t22=np.load('tauSgs22mean.npy')
t23=np.load('tauSgs23mean.npy')
t33=np.load('tauSgs33mean.npy')

s11=np.load('s11mean.npy')
s12=np.load('s12mean.npy')
s13=np.load('s13mean.npy')
s22=np.load('s22mean.npy')
s23=np.load('s23mean.npy')
s33=np.load('s33mean.npy')

# initialize variables

dSgs=np.zeros([iRed,j,nx5pSlice])


# begin loop

for it, fileNumber in enumerate(listRes):

    start=time.time()
    print('Iteration:',it,'fileNumber',fileNumber)

    fileHeader='up_'
    fileName   =  path + fileHeader + fileNumber + '.5p'
    _,_,_,_,_,_,_,_,_,_,uRtmp=read5pSlice(fileName,kmin5p,kmax5p,skip)

    #uRstag=bc(uRtmp[:iEnd,],'u',J,Jsym)
    uRstag=uRtmp[:iEnd,]
    
    del uRtmp

    fileHeader='vp_'
    fileName   =  path + fileHeader + fileNumber + '.5p'
    _,_,_,_,_,_,_,_,_,_,uTtmp=read5pSlice(fileName,kmin5p,kmax5p,skip)

    #uTstag=bc(uTtmp[:iEnd,],'v',J,Jsym)
    uTstag=uTtmp[:iEnd,]

    del uTtmp

    fileHeader='wp_'
    fileName   =  path + fileHeader + fileNumber + '.5p'
    _,_,_,_,_,_,_,_,_,_,uZtmp=read5pSlice(fileName,kmin5p,kmax5p,skip)

    #uZstag=bc(uZtmp[:iEnd,],'w',J,Jsym)
    uZstag=uZtmp[:iEnd,]

    del uZtmp

    fileHeader='tvp_'
    fileName   =  path + fileHeader + fileNumber + '.5p'
    _,_,_,_,_,_,_,_,_,_,tVtmp=read5pSlice(fileName,kmin5p,kmax5p,skip)

    #tV=bc(tVtmp[:iEnd,],'w',J,Jsym)
    tV=tVtmp[:iEnd,]

    del tVtmp

    uR,uT,uZ=center(uRstag,uTstag,uZstag)
    uX,uY=cyl2car(uR,uT,rcRed,thc)

    del uR,uT,uRstag,uTstag,uZstag

# compute gradients

    duXdZ = np.gradient(uX, zcRed, axis=2)
    duXdR = np.gradient(uX, rcRed, axis=0) 
    duXdT = np.gradient(uX, thc, axis=1)

    del uX

    duXdX, duXdY = diff_cyl2car(duXdR, duXdT, rcRed, thc, iRed, j, nx5pSlice)

    del duXdR,duXdT

 
    duYdZ = np.gradient(uY, zcRed, axis=2) 
    duYdR = np.gradient(uY, rcRed, axis=0)
    duYdT = np.gradient(uY, thc, axis=1)

    del uY

    duYdX, duYdY = diff_cyl2car(duYdR, duYdT, rcRed, thc, iRed, j, nx5pSlice)
 
    del duYdR,duYdT

    duZdZ = np.gradient(uZ, zcRed, axis=2)
    duZdR = np.gradient(uZ, rcRed, axis=0)
    duZdT = np.gradient(uZ, thc, axis=1)
 
    del uZ  
 
    duZdX, duZdY = diff_cyl2car(duZdR, duZdT, rcRed, thc, iRed, j, nx5pSlice)
 
    del duZdR,duZdT


    dSgs+=(2*tV*duXdX-t11)*(duXdX-s11)
    del duXdX    

    dSgs+=2*(tV*(duXdY+duYdX)-t12)*(0.5*(duXdY+duYdX)-s12)
    del duXdY,duYdX

    dSgs+=2*(tV*(duXdZ+duZdX)-t13)*(0.5*(duXdZ+duZdX)-s13)
    del duXdZ,duZdX

    dSgs+=(2*tV*duYdY-t22)*(duYdY-s22)
    del duYdY

    dSgs+=2*(tV*(duYdZ+duZdY)-t23)*(0.5*(duYdZ+duZdY)-s23)
    del duYdZ,duZdY

    dSgs+=(2*tV*duZdZ-t33)*(duZdZ-s33)
    del duZdZ


    print(time.time()-start)


dSgs/=(-1*nit)

np.save('dissipationSgs.npy',dSgs)











