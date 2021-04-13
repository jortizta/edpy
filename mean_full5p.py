# Dissipation mean: computes the mean of the 5p files retaining the groups of 5 planes.
# Last updated: March 1 2021

import numpy as np
from eddy import *
import time
import glob

path='/p/work3/jortizta/hybrid_spd_fr2_6/run/datafiles5p/'
pathGrid='/p/work3/jortizta/hybrid_spd_fr2_6/run/'


i,j,k=619,258,4610

iIni,iEnd=1,400
iRed=iEnd-iIni+1

kmin5p,kmax5p=1,2
skip=0

_, _,nx5pSlice, _, xe5pSlice,xc5pSlice,_, _, _, _=grid5p(pathGrid+'x3_grid.in',kmin5p,kmax5p,skip)

_,_,_,re,rc=readgrid(pathGrid+'x1_grid.in')
J,Jsym,the,thc=azi_grid(j)

reRed=re[iIni-1:iEnd]
rcRed=rc[iIni-1:iEnd]


fileHeader='up_'
fileRes=glob.glob(path+fileHeader+'*0.5p')
listRes=[i[-11:-3] for i in fileRes]

nit=len(listRes)

uRmean=np.zeros([iRed,j,nx5pSlice])
uTmean=np.zeros([iRed,j,nx5pSlice])
uZmean=np.zeros([iRed,j,nx5pSlice])

for it, fileNumber in enumerate(listRes):

    start=time.time()
    print('Iteration:',it,'of',nit,'fileNumber:',fileNumber)

    fileHeader='up_'
    fileName   =  path + fileHeader + fileNumber + '.5p'
    _,_,_,_,_,_,_,_,_,_,uRtmp=read5pSlice(fileName,kmin5p,kmax5p,skip)

    uRstag=uRtmp[:iEnd,]


#    print('time1:',time.time()-start)   

    del uRtmp

    fileHeader='vp_'
    fileName   =  path + fileHeader + fileNumber + '.5p'
    _,_,_,_,_,_,_,_,_,_,uTtmp=read5pSlice(fileName,kmin5p,kmax5p,skip)

    uTstag=uTtmp[:iEnd,]

    del uTtmp


#    print('time2:',time.time()-start)   

    fileHeader='wp_'
    fileName   =  path + fileHeader + fileNumber + '.5p'
    _,_,_,_,_,_,_,_,_,_,uZtmp=read5pSlice(fileName,kmin5p,kmax5p,skip)

    uZstag=uZtmp[:iEnd,]

    del uZtmp


#    print('time3:',time.time()-start)   

    uR,uT,uZ=center(uRstag,uTstag,uZstag)

    uRmean+=uR
    uTmean+=uT
    uZmean+=uZ 
    
    print('tpi:',time.time()-start)   

uRmean/=nit
uTmean/=nit
uZmean/=nit

uXmean,uYmean=cyl2car(uRmean,uTmean,rcRed,thc)

np.save('uRmean_full5p.npy',uRmean)
np.save('uTmean_full5p.npy',uTmean)
np.save('uXmean_full5p.npy',uXmean)
np.save('uYmean_full5p.npy',uYmean)
np.save('uZmean_full5p.npy',uZmean)

np.save('uRmean.npy',uRmean[:,:,2::5])
np.save('uTmean.npy',uTmean[:,:,2::5])
np.save('uXmean.npy',uXmean[:,:,2::5])
np.save('uYmean.npy',uYmean[:,:,2::5])
np.save('uZmean.npy',uZmean[:,:,2::5])



