import numpy as np
#from numpy-1.19.0 import *
from eddy import *
import time
import glob

path='/p/work/jortizta/hybrid_spd_fr2_5/run/datafiles5p/'
pathGrid='/p/work/jortizta/hybrid_spd_fr2_5/run/'

i,j,k=619,258,4610

iIni,iEnd=1,200
iRed=iEnd-iIni+1

kmin5p,kmax5p=1,2
skip=0

_,kRed,kSlice,_,_,_,_,_,_,_=grid5p(pathGrid+'x3_grid.in',kmin5p,kmax5p,skip)

_,_,_,re,rc=readgrid(pathGrid+'x1_grid.in')
J,Jsym,the,thc=azi_grid(j)

reRed=re[iIni-1:iEnd]
rcRed=rc[iIni-1:iEnd]


fileHeader='up_'
fileRes=glob.glob(path+fileHeader+'*0.5p')
listRes=[i[-11:-3] for i in fileRes]

nit=len(listRes)


uXmean=np.load('./uXmean.npy')
uZmean=np.load('./uZmean.npy')

uXmean=uXmean[:iEnd,:,:]
uZmean=uZmean[:iEnd,:,:]

uXrms=np.zeros([iRed,j,kRed])
uZrms=np.zeros([iRed,j,kRed])

uZtmp=np.zeros([iRed,j,kSlice])

duXdYm=np.zeros([iRed,j,kRed])
duZdYm=np.zeros([iRed,j,kRed])
duXdXm=np.zeros([iRed,j,kRed])
duZdXm=np.zeros([iRed,j,kRed])


for it, fileNumber in enumerate(listRes):

    start=time.time()
    print('Iteration:',it,'of',nit,'fileNumber:',fileNumber)

    fileHeader='up_'
    fileName   =  path + fileHeader + fileNumber + '.5p'
    _,_,_,_,_,_,_,_,_,_,uRtmp=read5pSqueeze(fileName,kmin5p,kmax5p,skip)

    uRstag=uRtmp[:iEnd,]
    
    del uRtmp

    fileHeader='vp_'
    fileName   =  path + fileHeader + fileNumber + '.5p'
    _,_,_,_,_,_,_,_,_,_,uTtmp=read5pSqueeze(fileName,kmin5p,kmax5p,skip)

    uTstag=uTtmp[:iEnd,]
  
    del uTtmp

    fileHeader='wp_'
    fileName   =  path + fileHeader + fileNumber + '.5p'
    _,_,_,_,_,_,_,_,_,_,uZstag=read5pSlice(fileName,kmin5p,kmax5p,skip)

    uZtmp[:,:,:-1]=0.5*(uZstag[:iEnd,:,1:]+uZstag[:iEnd,:,0:-1])
    uZtmp[:,:,-1]=uZstag[:iEnd,:,-1]
   
    del uZstag

    uZ=uZtmp[:,:,2::5]

    uR,uT=centerUV(uRstag,uTstag)

    del uRstag,uTstag

    uX,uY=cyl2car(uR,uT,rcRed,thc)

    uXf=uX-uXmean

    del uX
    uZf=uZ-uZmean
    del uZ

    duXfdR=np.gradient(uXf,rcRed,axis=0)
    duXfdT=np.gradient(uXf,thc,axis=1)
    del uXf

    duZfdR=np.gradient(uZf,rcRed,axis=0)
    duZfdT=np.gradient(uZf,thc,axis=1)
    del uZf

#    print('shape:',uXf.shape,kRed)

    duXdX,duXdY=diff_cyl2car(duXfdR,duXfdT,rcRed,thc,iRed,j,kRed)
    del duXfdR,duXfdT

    duZdX,duZdY=diff_cyl2car(duZfdR,duZfdT,rcRed,thc,iRed,j,kRed)
    del duZfdR,duZfdT

    duXdYm+=duXdY**2
    duZdYm+=duZdY**2
    duXdXm+=duXdX**2
    duZdXm+=duZdX**2

    print('max duXdYm',np.max(duXdY))    
    print('max duZdXm',np.max(duZdX))    

    del duXdY,duZdY
    print('tpi:',time.time()-start)   

duXdYm/=nit
duZdYm/=nit
duXdXm/=nit
duZdXm/=nit


np.save('duXdYm.npy',duXdYm)
np.save('duZdYm.npy',duZdYm)
np.save('duXdXm.npy',duXdXm)
np.save('duZdXm.npy',duZdXm)



