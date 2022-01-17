#Updated Mon Apr 19
#The pressure shows a tiny low freq oscillation, a reference pressure is taken 
#to bring it back to zero

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

_,kRed,kSlice,_,_,_,_,_,_,_=grid5p(pathGrid+'x3_grid.in',kmin5p,kmax5p,skip)

_,_,_,re,rc=readgrid(pathGrid+'x1_grid.in')
J,Jsym,the,thc=azi_grid(j)

reRed=re[iIni-1:iEnd]
rcRed=rc[iIni-1:iEnd]


fileHeader='pp_'
fileRes=glob.glob(path+fileHeader+'*0.5p')
listRes=[i[-11:-3] for i in fileRes]

nit=len(listRes)

pmean=np.zeros([iRed,j,kRed])

for it, fileNumber in enumerate(listRes):

    start=time.time()
    print('Iteration:',it,'of',nit,'fileNumber:',fileNumber)

    fileHeader='pp_'
    fileName   =  path + fileHeader + fileNumber + '.5p'
    _,_,_,_,_,_,_,_,_,_,ptmp=read5pSqueeze(fileName,kmin5p,kmax5p,skip)


    p=ptmp[:iEnd,]
    p-=p[-1,0,-1] #substract a reference pressure  

    pmean+=p

    print('tpi:',time.time()-start)   

pmean/=nit

np.save('pmean.npy',pmean)

