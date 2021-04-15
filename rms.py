import numpy as np
from eddy import *
import time
import glob

path='/p/work/jortizta/spd_re1e5_fr2_2/run/datafiles5p/'
pathGrid='/p/work/jortizta/spd_re1e5_fr2_2/run/'


i,j,k=848,514,3074

iIni,iEnd=1,728
iRed=iEnd-iIni+1

kmin5p,kmax5p=109,110
skip=60


_,kRed,kSlice,_,_,_,_,_,_,_=grid5p(pathGrid+'x3_grid.in',kmin5p,kmax5p,skip)

_,_,_,re,rc=readgrid(pathGrid+'x1_grid.in')
J,Jsym,the,thc=azi_grid(j)

reRed=re[iIni-1:iEnd]
rcRed=rc[iIni-1:iEnd]


fileHeader='up_'
fileRes=glob.glob(path+fileHeader+'*0.5p')
listRes=[i[-11:-3] for i in fileRes]

nit=len(listRes)

uRmean=np.load('uRmean.npy')
uTmean=np.load('uTmean.npy')
uXmean=np.load('uXmean.npy')
uYmean=np.load('uYmean.npy')
uZmean=np.load('uZmean.npy')

uXX=np.zeros([iRed,j,kRed])
uYY=np.zeros([iRed,j,kRed])
uZZ=np.zeros([iRed,j,kRed])
uXZ=np.zeros([iRed,j,kRed])
uXY=np.zeros([iRed,j,kRed])
uYZ=np.zeros([iRed,j,kRed])

uRR=np.zeros([iRed,j,kRed])
uTT=np.zeros([iRed,j,kRed])
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

    uXX+=(uX-uXmean)**2
    uYY+=(uY-uYmean)**2
    uZZ+=(uZ-uZmean)**2
    uRR+=(uR-uRmean)**2
    uTT+=(uT-uTmean)**2

    uRZ+=(uR-uRmean)*(uZ-uZmean)

    uXY+=(uX-uXmean)*(uY-uYmean)
    uXZ+=(uX-uXmean)*(uZ-uZmean)
    uYZ+=(uY-uYmean)*(uZ-uZmean)
    
    print('tpi:',time.time()-start)   

uXX/=nit
uYY/=nit
uZZ/=nit
uTT/=nit
uRR/=nit

uRZ/=nit

uXY/=nit
uXZ/=nit
uYZ/=nit

tke=0.5*(uXX+uYY+uZZ)


np.save('uRZ.npy',uRZ)
np.save('uXY.npy',uXY)
np.save('uXZ.npy',uXZ)
np.save('uYZ.npy',uYZ)

np.save('uXrms.npy',uXX**0.5)
np.save('uYrms.npy',uYY**0.5)
np.save('uZrms.npy',uZZ**0.5)

np.save('uRrms.npy',uRR**0.5)
np.save('uTrms.npy',uTT**0.5)

np.save('uXX.npy',uXX)
np.save('uYY.npy',uYY)
np.save('uZZ.npy',uZZ)

np.save('tke.npy',tke)

