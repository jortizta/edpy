import numpy as np
from eddy import *
import time
import glob

path='/p/work/jortizta/hybrid_spd_frinf_2/run/datafiles5p/'
pathGrid='/p/work/jortizta/hybrid_spd_frinf_2/run/'

i,j,k=431,258,2562

iIni,iEnd=1,380
iRed=iEnd-iIni+1

kmin5p,kmax5p=1,2
skip=0

_,kRed,kSlice,_,_,_,_,_,_,_=grid5p(pathGrid+'x3_grid.in',kmin5p,kmax5p,skip)

_,_,_,re,rc=readgrid(pathGrid+'x1_grid.in')
J,Jsym,the,thc=azi_grid(j)

reRed=re[iIni-1:iEnd]
rcRed=rc[iIni-1:iEnd]


fileHeader='up_'
fileRes=glob.glob(path+fileHeader+'*.5p')
listRes=[i[-11:-3] for i in fileRes]

nit=len(listRes)

uRmean=np.zeros([iRed,j,kRed])
uTmean=np.zeros([iRed,j,kRed])
uZmean=np.zeros([iRed,j,kRed])
uZtmp=np.zeros([iRed,j,kSlice])

for it, fileNumber in enumerate(listRes):

    start=time.time()
    print('Iteration:',it,'of',nit,'fileNumber:',fileNumber)

    fileHeader='up_'
    fileName   =  path + fileHeader + fileNumber + '.5p'
    _,_,_,_,_,_,_,_,_,_,uRtmp=read5pSqueeze(fileName,kmin5p,kmax5p,skip)

    uRstag=uRtmp[:iEnd,]

    fileHeader='vp_'
    fileName   =  path + fileHeader + fileNumber + '.5p'
    _,_,_,_,_,_,_,_,_,_,uTtmp=read5pSqueeze(fileName,kmin5p,kmax5p,skip)

    uTstag=uTtmp[:iEnd,]

    fileHeader='wp_'
    fileName   =  path + fileHeader + fileNumber + '.5p'
    _,_,_,_,_,_,_,_,_,_,uZstag=read5pSlice(fileName,kmin5p,kmax5p,skip)

    uZtmp[:,:,:-1]=0.5*(uZstag[:iEnd,:,1:]+uZstag[:iEnd,:,0:-1])
    uZtmp[:,:,-1]=uZstag[:iEnd,:,-1]
    uZ=uZtmp[:,:,2::5]

    uR,uT=centerUV(uRstag,uTstag)

    uRmean+=uR
    uTmean+=uT
    uZmean+=uZ 
    
    print('tpi:',time.time()-start)   

uRmean/=nit
uTmean/=nit
uZmean/=nit

uXmean,uYmean=cyl2car(uRmean,uTmean,rcRed,thc)

np.save('uRmean.npy',uRmean)
np.save('uTmean.npy',uTmean)
np.save('uXmean.npy',uXmean)
np.save('uYmean.npy',uYmean)
np.save('uZmean.npy',uZmean)

uXrms=np.zeros([iRed,j,kRed])
uYrms=np.zeros([iRed,j,kRed])
uZrms=np.zeros([iRed,j,kRed])
uXZ=np.zeros([iRed,j,kRed])
uYZ=np.zeros([iRed,j,kRed])

uRrms=np.zeros([iRed,j,kRed])
uTrms=np.zeros([iRed,j,kRed])
uRZ=np.zeros([iRed,j,kRed])

uZtmp=np.zeros([iRed,j,kSlice])

for it, fileNumber in enumerate(listRes):

    start=time.time()
    print('Iteration:',it,'of',nit,'fileNumber:',fileNumber)

    fileHeader='up_'
    fileName   =  path + fileHeader + fileNumber + '.5p'
    _,_,_,_,_,_,_,_,_,_,uRtmp=read5pSqueeze(fileName,kmin5p,kmax5p,skip)

    uRstag=uRtmp[:iEnd,]

    fileHeader='vp_'
    fileName   =  path + fileHeader + fileNumber + '.5p'
    _,_,_,_,_,_,_,_,_,_,uTtmp=read5pSqueeze(fileName,kmin5p,kmax5p,skip)

    uTstag=uTtmp[:iEnd,]

    fileHeader='wp_'
    fileName   =  path + fileHeader + fileNumber + '.5p'
    _,_,_,_,_,_,_,_,_,_,uZstag=read5pSlice(fileName,kmin5p,kmax5p,skip)

    uZtmp[:,:,:-1]=0.5*(uZstag[:iEnd,:,1:]+uZstag[:iEnd,:,0:-1])
    uZtmp[:,:,-1]=uZstag[:iEnd,:,-1]
    uZ=uZtmp[:,:,2::5]

    uR,uT=centerUV(uRstag,uTstag)

    uX,uY=cyl2car(uR,uT,rcRed,thc)

    uXrms+=(uX-uXmean)**2
    uYrms+=(uY-uYmean)**2
    uZrms+=(uZ-uZmean)**2
    uRrms+=(uR-uRmean)**2
    uTrms+=(uT-uTmean)**2

    uRZ+=(uR-uRmean)*(uZ-uZmean)
    uXZ+=(uX-uXmean)*(uZ-uZmean)
    uYZ+=(uY-uYmean)*(uZ-uZmean)
    
    print('tpi:',time.time()-start)   

uXrms/=nit
uYrms/=nit
uZrms/=nit
uTrms/=nit
uRrms/=nit

uRZ/=nit
uXZ/=nit
uYZ/=nit

tke=0.5*(uXrms+uYrms+uZrms)


np.save('uRZ.npy',uRZ)
np.save('uXZ.npy',uXZ)
np.save('uYZ.npy',uYZ)

np.save('uXrms.npy',uXrms**0.5)
np.save('uYrms.npy',uYrms**0.5)
np.save('uZrms.npy',uZrms**0.5)

np.save('uRrms.npy',uRrms**0.5)
np.save('uTrms.npy',uTrms**0.5)

np.save('tke.npy',tke)

