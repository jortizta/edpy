# Version Mon 26 10:18 AM
# mean wave flux implementation is corrected
# ds is exported
# name of the variables is changed to improve readability  


import numpy as np
from eddy import * 
import glob
import time

def normal(th,r):
    ny=th.shape[0]
    n=np.zeros([ny,3])
    ds=np.zeros([ny,])

    for j in range(1,ny-1):
        
        x1=r[j-1]*np.cos(th[j-1]) #before
        y1=r[j-1]*np.sin(th[j-1])
        x2=r[j+1]*np.cos(th[j+1]) #after
        y2=r[j+1]*np.sin(th[j+1])
        
        vx=x2-x1
        vy=y2-y1
        mag=(vx**2+vy**2)**0.5 #magnitude
        tmp=np.cross([vx,vy,0],[0,0,1]) #normal vector
        tmp/=mag # unitary normal vector
        n[j,:]=tmp
        ds[j]=mag/2 
#the surface differential is approx. half the magnitude of v_tangential 
#since it is a 3 point stencil
    
    n[0,:]=[1,0,0]
    n[-1,:]=[1,0,0]
    #n[loc(th,np.pi),:]=[-1,0,0]
    ds[0]=ds[1]
    ds[-1]=ds[-2]
    
    return n,ds  

def interface(tkePlane,r,xl,lVk,lHk,x,t):
    if t==1:
        ri,th,i=interface_1(tkePlane,r)

    elif t==2:

        ri,th,i=interface_2(tkePlane,r)

    elif t==3:
        ri,th,i=interface_3(tkePlane,r,xl,lVk,lHk,x)

    elif t==4:
        ri,th,i=interface_4(tkePlane,r)

    elif t==5:
        ri,th,i=interface_5(tkePlane,r)

    elif t==6:
        ri,th,i=interface_6(tkePlane,r)

    elif t==7:
        ri,th,i=interface_7(tkePlane,r)

    elif t==8:
        ri,th,i=interface_8(tkePlane,r,xl,lVk,lHk,x)

    return ri,th,i


def interface_1(tkePlane,r):
    ny=tkePlane.shape[1]
    tkePlane/=np.max(tkePlane)
    th=np.linspace(0,2*np.pi,ny)
    ri=np.zeros(ny,)
    i=np.zeros([ny,],dtype=np.int)
    for j in range(ny):
        index=loc(tkePlane[:,j],0.05)
        i[j]=int(index)
        ri[j]=r[index]
    return ri,th,i


def interface_2(tkePlane,r):
    ny=tkePlane.shape[1]
    tkePlane/=np.max(tkePlane)
    th=np.linspace(0,2*np.pi,ny)
    ri=np.zeros(ny,)
    i=np.zeros([ny,],dtype=np.int)
    for j in range(ny):
         d=r[loc(tkePlane[:,j],0.05)]
         index=loc(r,1.5*d) 
         i[j]=int(index)
         ri[j]=r[index]
    return ri,th,i

def interface_2_alternative(tkePlane,r):
# not very smooth surface which leads to wave flux artifacts
    ny=tkePlane.shape[1]
    tkeCl=np.mean(tkePlane[1,:])
    th=np.linspace(0,2*np.pi,ny)
    ri=np.zeros(ny,)   
    i=np.zeros([ny,],dtype=np.int)
    for j in range(ny):
        lk=r[loc(tkePlane[:,j],tkeCl/2)]              
        index=loc(r,1.5*lk)    
        i[j]=int(index)
        ri[j]=r[index]        
    return ri,th,i

def interface_3(tkePlane,r,xl,lVk_total,lHk_total,x):

    ny=tkePlane.shape[1]
    th=np.linspace(0,2*np.pi,ny)
    ri=np.zeros(ny,)   
    i=np.zeros([ny,],dtype=np.int)

    lHk=lHk_total[loc(xl,x)]
    lVk=lVk_total[loc(xl,x)]
    
    a=1.5*lHk
    b=1.5*lVk
    rE=a*b/np.sqrt((b*np.cos(th))**2+(a*np.sin(th))**2)
        
    for j in range(ny):
        index=loc(r,rE[j])   
        i[j]=int(index)
        ri[j]=r[index]        
    return ri,th,i


def interface_4(tkePlane,r):
    ny=tkePlane.shape[1]
    th=np.linspace(0,2*np.pi,ny)
    ri=np.zeros(ny,)
    i=np.zeros([ny,],dtype=np.int)
    for j in range(ny):
        index=loc(r,0.75)
        i[j]=int(index)
        ri[j]=r[index]
    return ri,th,i

def interface_5(tkePlane,r):
    ny=tkePlane.shape[1]
    th=np.linspace(0,2*np.pi,ny)
    ri=np.zeros(ny,)
    i=np.zeros([ny,],dtype=np.int)
    for j in range(ny):
        index=loc(r,2)
        i[j]=int(index)
        ri[j]=r[index]
    return ri,th,i

def interface_6(tkePlane,r):
    ny=tkePlane.shape[1]
    tkePlane/=np.max(tkePlane)
    th=np.linspace(0,2*np.pi,ny)
    ri=np.zeros(ny,)
    i=np.zeros([ny,],dtype=np.int)
    for j in range(ny):
        index=loc(tkePlane[:,j],0.02)
        i[j]=int(index)
        ri[j]=r[index]
    return ri,th,i

def interface_7(tkePlane,r):
    ny=tkePlane.shape[1]
    tkePlane/=np.max(tkePlane)
    th=np.linspace(0,2*np.pi,ny)
    ri=np.zeros(ny,)
    i=np.zeros([ny,],dtype=np.int)
    for j in range(ny):
        index=loc(tkePlane[:,j],0.01)
        i[j]=int(index)
        ri[j]=r[index]
    return ri,th,i

def interface_8(tkePlane,r,xl,lVk_total,lHk_total,x):

    ny=tkePlane.shape[1]
    th=np.linspace(0,2*np.pi,ny)
    ri=np.zeros(ny,)   
    i=np.zeros([ny,],dtype=np.int)

    lHk=lHk_total[loc(xl,x)]
    lVk=lVk_total[loc(xl,x)]
    
    a=2*lHk
    b=2*lVk
    rE=a*b/np.sqrt((b*np.cos(th))**2+(a*np.sin(th))**2)
        
    for j in range(ny):
        index=loc(r,rE[j])   
        i[j]=int(index)
        ri[j]=r[index]        
    return ri,th,i

# Inputs:

# interface_type 1: isoline where tke=5% of the peak value at that streamwise loc
# interface_type 2: 1.5 Lk where Lk is r(theta) location where max_tke/2
# interface_type 3: ellipse defined by 1.5 Lk_horizontal and Lk_vertical
# interface_type 4: circle of 0.75D
# interface_type 5: circle of 2D
# interface_type 6: isoline where tke=2% of the peak value at that streamwise loc
# interface_type 7: isoline where tke=1% of the peak value at that streamwise loc
# interface_type 8: ellipse defined by 2 Lk_horizontal and Lk_vertical
 
interface_type=8

path='/p/work3/jortizta/hybrid_spd_fr2_6/run/datafiles5p/'
pathGrid='/p/work3/jortizta/hybrid_spd_fr2_6/run/'
pathMean='/p/work3/jortizta/hybrid_spd_fr2_6/run/pyPost/rms/'

i,j,k=619,258,4610

iIni,iEnd=1,400
iRed=iEnd-iIni+1

kmin5p,kmax5p=1,2
skip=0


_,kRed,kSlice,_,_,_,_,xc,_,_=grid5p(pathGrid+'x3_grid.in',kmin5p,kmax5p,skip)

_,_,_,re,rc=readgrid(pathGrid+'x1_grid.in')
J,Jsym,the,thc=azi_grid(j)

reRed=re[iIni-1:iEnd]
rcRed=rc[iIni-1:iEnd]


fileHeader='up_'
fileRes=glob.glob(path+fileHeader+'*100000.5p')
listRes=[i[-11:-3] for i in fileRes]

nit=len(listRes)


tke=np.load(pathMean+'tke.npy')
uXmean=np.load(pathMean+'uXmean.npy')
uYmean=np.load(pathMean+'uYmean.npy')
pmean=np.load(pathMean+'pmean.npy')

raw_data=np.load('./lVk_fr2.npy')
xlk=raw_data[:,0]
lVk_total=raw_data[:,1]

raw_data=np.load('./lHk_fr2.npy')
lHk_total=raw_data[:,1]

interface_total=np.zeros([j,kRed],dtype=np.int)
normal_x_total=np.zeros([j,kRed])
normal_y_total=np.zeros([j,kRed])
ds_total=np.zeros([j,kRed])
uNmean_total=np.zeros([j,kRed])
fluxMean=np.zeros([j,kRed])

for k in range(kRed):
    data_slice=tke[:,:,k]    
    rI,thI,irI=interface(data_slice,rcRed,xlk,lVk_total,lHk_total,xc[k],interface_type)
    n,ds=normal(thI,rI)
    interface_total[:,k]=irI
    normal_x_total[:,k]=n[:,0]
    normal_y_total[:,k]=n[:,1]
    ds_total[:,k]=ds
 
    uNmean_total[:,k]=uXmean[irI,J,k]*normal_x_total[:,k] + uYmean[irI,J,k]*normal_y_total[:,k]
    fluxMean[:,k]=pmean[irI,J,k]*uNmean_total[:,k]
 

fluxTurb=np.zeros([j,kRed])

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

    fileHeader='pp_'
    fileName   =  path + fileHeader + fileNumber + '.5p'
    _,_,_,_,_,_,_,_,_,_,ptmp=read5pSqueeze(fileName,kmin5p,kmax5p,skip)

    p=ptmp[:iEnd,]
    p-=p[-1,0,-1]

    uR,uT=centerUV(uRstag,uTstag)

    uX,uY=cyl2car(uR,uT,rcRed,thc)

    del uR, uT

    pf =p-pmean

    uN=np.zeros([j,])
    uNf=np.zeros([j,])
    fluxf=np.zeros([j,kRed])

    for k in range(kRed):
   
        irI=interface_total[:,k]

        uN=uX[irI,J,k]*normal_x_total[:,k] + uY[irI,J,k]*normal_y_total[:,k]
        uNf=uN-uNmean_total[:,k] 
        fluxf[:,k]=uNf*pf[irI,J,k]    

    fluxTurb+=fluxf
    
    print('tpi:',time.time()-start)   


fluxTurb/=nit

np.save('fluxMean_i'+str(interface_type)+'.npy',fluxMean)
#np.save('fluxTurb_i'+str(interface_type)+'.npy',fluxTurb)

intFluxMean=np.zeros([kRed,])
intFluxTurb=np.zeros([kRed,])

for k in range(kRed):

    intFluxMean[k]=np.sum(ds_total[:,k]*fluxMean[:,k])
    intFluxTurb[k]=np.sum(ds_total[:,k]*fluxTurb[:,k])

np.save('intfluxMean_i'+str(interface_type)+'.npy',intFluxMean)
#np.save('intfluxTurb_i'+str(interface_type)+'.npy',intFluxTurb)

np.save('ds_i'+str(interface_type)+'.npy',ds_total)
 
