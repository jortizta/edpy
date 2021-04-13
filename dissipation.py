# DISSIPATION: postprocessing of Eddy.f
# Updated: March 1 21
# Version: fixed last term in dissipation
# computation line 


import numpy as np
import struct as st
import time
import gc
from eddy import *
import glob

path='/p/work/jortizta/hybrid_spd_frinf_3/run/datafiles5p_complete/'
pathGrid='/p/work/jortizta/hybrid_spd_frinf_3/run/'
pathMean='/p/work/jortizta/hybrid_spd_frinf_3/run/pyPost/production/'

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

fileHeader='wp_'
fileRes=glob.glob(path+fileHeader+'*.5p')
listRes=[ii[-11:-3] for ii in fileRes]
nit=len(listRes)
print('Number of iteration:', nit)

# read mean

uXmean=np.load(pathMean+'uXmean_full5p.npy')
uYmean=np.load(pathMean+'uYmean_full5p.npy')
uZmean=np.load(pathMean+'uZmean_full5p.npy')

# initialize variables

dis=np.zeros([iRed,j,nx5pSlice])

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

    uRstag=uRtmp[:iEnd,]
    uTstag=uTtmp[:iEnd,]
    uZstag=uZtmp[:iEnd,]

#   uRstag=bc(uRtmp[:iEnd,],'u',J,Jsym)
#   uTstag=bc(uTtmp[:iEnd,],'v',J,Jsym)
#   uZstag=bc(uZtmp[:iEnd,],'w',J,Jsym)


    print('read:',time.time()-start)

    del uRtmp,uTtmp,uZtmp


    uR,uT,uZ=center(uRstag,uTstag,uZstag)
    uX,uY=cyl2car(uR,uT,rcRed,thc)

    del uR,uT,uRstag,uTstag,uZstag

# compute fluc

    uf=uX-uXmean
    vf=uY-uYmean
    wf=uZ-uZmean

    del uX,uY,uZ


    print('fluctuations:',time.time()-start)

# compute gradients

    dUfdZ = np.gradient(uf, zcRed, axis=2)
    dUfdR = np.gradient(uf, rcRed, axis=0) 
    dUfdTh = np.gradient(uf, thc, axis=1)

    del uf

    dUfdX, dUfdY = diff_cyl2car(dUfdR, dUfdTh, rcRed, thc, iRed, j, nx5pSlice)

    del dUfdR,dUfdTh

    dVfdZ = np.gradient(vf, zcRed, axis=2) 
    dVfdR = np.gradient(vf, rcRed, axis=0)
    dVfdTh = np.gradient(vf, thc, axis=1)

    del vf

    dVfdX, dVfdY = diff_cyl2car(dVfdR, dVfdTh, rcRed, thc, iRed, j, nx5pSlice)

    del dVfdR,dVfdTh

    dWfdZ = np.gradient(wf, zcRed, axis=2)
    dWfdR = np.gradient(wf, rcRed, axis=0)
    dWfdTh = np.gradient(wf, thc, axis=1)

    del wf  
 
    dWfdX, dWfdY = diff_cyl2car(dWfdR, dWfdTh, rcRed, thc, iRed, j, nx5pSlice)

    del dWfdR,dWfdTh


    print('gradient 3:',time.time()-start)

# square gradients and multiply gradients

    dis += ( 2*dUfdX**2 + 2*dVfdY**2 + 2*dWfdZ**2 + 
             + dUfdY**2 + dVfdX**2 + 2*dUfdY*dVfdX +
             + dUfdZ**2 + dWfdX**2 + 2*dUfdZ*dWfdX +
             + dVfdZ**2 + dWfdY**2 + 2*dVfdZ*dWfdY )


    print('diss:',time.time()-start)

nu=1/Re
dis *= (nu/nit)

np.save('dissipation.npy',dis)












