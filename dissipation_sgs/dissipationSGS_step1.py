# To compute the subgrid dissipation we then compute mean(S_{ij}^sgs)

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

s11=np.zeros([iRed,j,nx5pSlice])
s12=np.zeros([iRed,j,nx5pSlice])
s13=np.zeros([iRed,j,nx5pSlice])
s22=np.zeros([iRed,j,nx5pSlice])
s23=np.zeros([iRed,j,nx5pSlice])
s33=np.zeros([iRed,j,nx5pSlice])

# begin loop

for it, fileNumber in enumerate(listRes):

    start=time.time()
    print('Iteration:',it,'fileNumber',fileNumber)

    fileHeader='up_'
    fileName   =  path + fileHeader + fileNumber + '.5p'
    _,_,_,_,_,_,_,_,_,_,uRtmp=read5pSlice(fileName,kmin5p,kmax5p,skip)

    fileHeader='vp_'
    fileName   =  path + fileHeader + fileNumber + '.5p'
    _,_,_,_,_,_,_,_,_,_,uTtmp=read5pSlice(fileName,kmin5p,kmax5p,skip)

    fileHeader='wp_'
    fileName   =  path + fileHeader + fileNumber + '.5p'
    _,_,_,_,_,_,_,_,_,_,uZtmp=read5pSlice(fileName,kmin5p,kmax5p,skip)


#    uRstag=bc(uRtmp[:iEnd,],'u',J,Jsym)
#    uTstag=bc(uTtmp[:iEnd,],'v',J,Jsym)
#    uZstag=bc(uZtmp[:iEnd,],'w',J,Jsym)

    uRstag=uRtmp[:iEnd,]
    uTstag=uTtmp[:iEnd,]
    uZstag=uZtmp[:iEnd,]

    del uRtmp,uTtmp,uZtmp


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


    s11+=duXdX
    del duXdX    

    s12+=0.5*(duXdY+duYdX)
    del duXdY,duYdX

    s13+=0.5*(duXdZ+duZdX)
    del duXdZ,duZdX

    s22+=duYdY
    del duYdY

    s23+=0.5*(duYdZ+duZdY)
    del duYdZ,duZdY

    s33+=duZdZ
    del duZdZ


    print(time.time()-start)

s11/=nit
s12/=nit
s13/=nit
s22/=nit
s23/=nit
s33/=nit


np.save('s11mean.npy',s11)
np.save('s12mean.npy',s12)
np.save('s13mean.npy',s13)
np.save('s22mean.npy',s22)
np.save('s23mean.npy',s23)
np.save('s33mean.npy',s33)











