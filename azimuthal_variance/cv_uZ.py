import numpy as np
from eddy import *
import time
import glob

path='/p/work/jortizta/hybrid_spd_frinf_3/run/datafiles5p_complete/'
pathGrid='/p/work/jortizta/hybrid_spd_frinf_3/run/'

i,j,k=481,258,4610

iIni,iEnd=1,400
iRed=iEnd-iIni+1

kmin5p,kmax5p=1,2
skip=0

_,kRed,kSlice,_,_,_,_,_,_,_=grid5p(pathGrid+'x3_grid.in',kmin5p,kmax5p,skip)

_,_,_,re,rc=readgrid(pathGrid+'x1_grid.in')
J,Jsym,the,thc=azi_grid(j)

reRed=re[iIni-1:iEnd]
rcRed=rc[iIni-1:iEnd]


fileHeader='wp_'
fileRes=glob.glob(path+fileHeader+'*00.5p')
listRes=[i[-11:-3] for i in fileRes]

nit=len(listRes)

uZmean=np.load('uZmean.npy')

uZmeanPlane=np.mean(uZmean,axis=1)

#uRrms=np.zeros([iRed,j,kRed])
#uTrms=np.zeros([iRed,j,kRed])
cv=np.zeros([iRed,kRed])

uZtmp=np.zeros([iRed,j,kSlice])

for it, fileNumber in enumerate(listRes):

    start=time.time()
    print('Iteration:',it,'of',nit,'fileNumber:',fileNumber)

    fileHeader='wp_'
    fileName   =  path + fileHeader + fileNumber + '.5p'
    _,_,_,_,_,_,_,_,_,_,uZstag=read5pSlice(fileName,kmin5p,kmax5p,skip)

    uZtmp[:,:,:-1]=0.5*(uZstag[:iEnd,:,1:]+uZstag[:iEnd,:,0:-1])
    uZtmp[:,:,-1]=uZstag[:iEnd,:,-1]
    uZ=uZtmp[:,:,2::5]

    tmp1 = np.mean((uZ-uZmeanPlane[:,np.newaxis,:])**2,axis=1)
    
    cv += tmp1**0.5/uZmeanPlane

    print('tpi:',time.time()-start)
    
    
cv/=nit

np.save('cv_uZ.npy',cv)
