# EDDY:library for postprocessing of Eddy.f
# Updated: Mar 24 21
# Version: initial version validated with Fortran version 



import numpy as np
from eddy import *
import time
import glob

# path

start=time.time()

path='/p/work3/jortizta/hybrid_spd_fr2_6/run/datafiles5p/'
pathGrid='/p/work3/jortizta/hybrid_spd_fr2_6/run/'
pathMean='./'

i,j,k=619,258,4610

iIni,iEnd=1,400
iRed=iEnd-iIni+1

kmin5p,kmax5p=1,2
skip=0


_, nx5pSqueeze,nx5pSlice, _, xe5pSlice,xc5pSlice,xe5pSqueeze,xc5pSqueeze, _, _=grid5p(pathGrid+'x3_grid.in',kmin5p,kmax5p,skip)

_,_,_,re,rc=readgrid(pathGrid+'x1_grid.in')
J,Jsym,the,thc=azi_grid(j)

reRed=re[iIni-1:iEnd]
rcRed=rc[iIni-1:iEnd]

zeRed=xe5pSlice
zcRed=xc5pSlice

uXmean=np.load(pathMean+'uXmean_full5p.npy') #size iRed,j,nx5pSlice
uYmean=np.load(pathMean+'uYmean_full5p.npy')
uZmean=np.load(pathMean+'uZmean_full5p.npy')

uXX=np.load('uXX.npy') #size iRed,j,nx5pSqueeze
uYY=np.load('uYY.npy')
uZZ=np.load('uZZ.npy')
uXY=np.load('uXY.npy')
uXZ=np.load('uXZ.npy')
uYZ=np.load('uYZ.npy')


print('data read, computing gradients:',time.time()-start)   

prod=np.zeros([iRed,j,nx5pSqueeze])

duxdrFull=np.gradient(uXmean,rcRed,axis=0)
duxdr=duxdrFull[:,:,2::5]
del duxdrFull
duxdtFull=np.gradient(uXmean,thc,axis=1)
duxdt=duxdtFull[:,:,2::5]
del duxdtFull
duxdzFull=np.gradient(uXmean,zcRed,axis=2)
duxdz=duxdzFull[:,:,2::5]
del duxdzFull

duxdx,duxdy = diff_cyl2car(duxdr,duxdt,rcRed,thc,iRed,j,nx5pSqueeze)
del duxdr,duxdt

duydrFull=np.gradient(uYmean,rcRed,axis=0)
duydr=duydrFull[:,:,2::5]
del duydrFull
duydtFull=np.gradient(uYmean,thc,axis=1)
duydt=duydtFull[:,:,2::5]
del duydtFull
duydzFull=np.gradient(uYmean,zcRed,axis=2)
duydz=duydzFull[:,:,2::5]
del duydzFull

duydx,duydy = diff_cyl2car(duydr,duydt,rcRed,thc,iRed,j,nx5pSqueeze)
del duydr,duydt

duzdrFull=np.gradient(uZmean,rcRed,axis=0)
duzdr=duzdrFull[:,:,2::5]
del duzdrFull
duzdtFull=np.gradient(uZmean,thc,axis=1)
duzdt=duzdtFull[:,:,2::5]
del duzdtFull
duzdzFull=np.gradient(uZmean,zcRed,axis=2)
duzdz=duzdzFull[:,:,2::5]

duzdx,duzdy = diff_cyl2car(duzdr,duzdt,rcRed,thc,iRed,j,nx5pSqueeze)
del duzdr,duzdt

p11=uXX*duxdx
prod+=p11
np.save('p11.npy',p11)
del p11

p12=uXY*duxdy
prod+=p12
np.save('p12.npy',p12)
del p12,duxdy

p13=uXZ*duxdz
prod+=p13
np.save('p13.npy',p13)
del p13

p21=uXY*duydx
prod+=p21
np.save('p21.npy',p21)
del p21

p22=uYY*duydy
prod+=p22
np.save('p22.npy',p22)
del p22

p23=uYZ*duydz
prod+=p23
np.save('p23.npy',p23)
del p23

p31=uXZ*duzdx
prod+=p31
np.save('p31.npy',p31)
del p31

p32=uYZ*duzdy
prod+=p32
np.save('p32.npy',p32)
del p32

p33=uZZ*duzdz
prod+=p33
np.save('p33.npy',p33)
del p33


np.save('production.npy',prod)


print('done',time.time()-start)   

