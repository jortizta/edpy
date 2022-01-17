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

# initialize variables

t11=np.zeros([iRed,j,nx5pSlice])
t12=np.zeros([iRed,j,nx5pSlice])
t13=np.zeros([iRed,j,nx5pSlice])
t22=np.zeros([iRed,j,nx5pSlice])
t23=np.zeros([iRed,j,nx5pSlice])
t33=np.zeros([iRed,j,nx5pSlice])

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

    del uRstag,uTstag,uZstag

    uX,uY=cyl2car(uR,uT,rcRed,thc)

    del uR,uT

# compute gradients

    duXdZ = np.gradient(uX, zcRed, axis=2)
    duXdR = np.gradient(uX, rcRed, axis=0) 
    duXdT = np.gradient(uX, thc, axis=1)

    del uX

    duXdX, duXdY = diff_cyl2car(duXdR, duXdT, rcRed, thc, iRed, j, nx5pSlice)

    del duXdR,duXdT

    t11+=2*tV*duXdX
    del duXdX    
 
    duYdZ = np.gradient(uY, zcRed, axis=2) 
    duYdR = np.gradient(uY, rcRed, axis=0)
    duYdT = np.gradient(uY, thc, axis=1)

    del uY

    duYdX, duYdY = diff_cyl2car(duYdR, duYdT, rcRed, thc, iRed, j, nx5pSlice) 
    del duYdR,duYdT

    t12+=tV*(duXdY+duYdX)
    del duXdY,duYdX

    t22+=2*tV*duYdY
    del duYdY

    duZdZ = np.gradient(uZ, zcRed, axis=2)
    duZdR = np.gradient(uZ, rcRed, axis=0)
    duZdT = np.gradient(uZ, thc, axis=1)
 
    del uZ  
 
    duZdX, duZdY = diff_cyl2car(duZdR, duZdT, rcRed, thc, iRed, j, nx5pSlice)
 
    del duZdR,duZdT

    t23+=tV*(duYdZ+duZdY)
    del duYdZ,duZdY

    t13+=tV*(duXdZ+duZdX)
    del duXdZ,duZdX

    t33+=2*tV*duZdZ
    del duZdZ


    print(time.time()-start)

t11/=nit
t12/=nit
t13/=nit
t22/=nit
t23/=nit
t33/=nit

np.save('tauSgs11mean.npy',t11)
np.save('tauSgs12mean.npy',t12)
np.save('tauSgs13mean.npy',t13)
np.save('tauSgs22mean.npy',t22)
np.save('tauSgs23mean.npy',t23)
np.save('tauSgs33mean.npy',t33)











