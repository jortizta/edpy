# EDDY:library for postprocessing of Eddy.f
# Updated: Tue Jan 7 21
# Version: fixed center routine

import numpy as np
import struct as st

# -------------------------------------------------------------

def readgrid(filename):

    f = open(filename)

    nx = int(f.read(4))

    #nx = np.fromfile(f, dtype=int, count=1)

    index, x = np.loadtxt(filename, skiprows=1, unpack=1)

    xe = np.zeros(nx+1)
    xc = np.zeros(nx+1)

    xe[:len(x)] = x[:]
    # only required for consistency of the allocation
    xe[-1] = xe[-2] + (xe[-2]-xe[-3])

    for ii in range(1, nx+1):

        xc[ii] = (xe[ii] + xe[ii-1]) * 0.5

    xc[0] = xe[0] - (xc[1]-xe[0])

    return nx, index, x, xe, xc

# -------------------------------------------------------------


def readpln(filename):

    f = open(filename, 'rb')

    header = np.dtype([('dummy1', 'int32'), ('nstep', 'int32'),
                       ('time', 'float64'), ('dt', 'float64'), ('g', 'float64'),
                       ('dens_0', 'float64'), ('Re', 'float64'), ('Pr', 'float64'),
                       ('dummy2', 'int32'), ('dummy3', 'int32'), ('norm', 'int32'),
                       ('index', 'int32'), ('iu', 'int32'), ('iv',
                                                             'int32'), ('iw', 'int32'),
                       ('dummy4', 'int32'), ('dummy5',
                                             'int32'), ('cloc', 'float64'),
                       ('eloc', 'float64'), ('dummy6',
                                             'int32'), ('dummy7', 'int32'),
                       ('np1', 'int32'), ('np2', 'int32'),
                       ('dummy8', 'int32'), ('dummy9', 'int32')])

    my_header = np.fromfile(f, header)

    np1 = my_header['np1'][0]
    np2 = my_header['np2'][0]

    f.close()

    f = open(filename, 'rb')

    plane = np.dtype([('dummy1', 'int32'), ('nstep', 'int32'),
                      ('time', 'float64'), ('dt', 'float64'), ('g', 'float64'),
                      ('dens_0', 'float64'), ('Re', 'float64'), ('Pr', 'float64'),
                      ('dummy2', 'int32'), ('dummy3', 'int32'), ('norm', 'int32'),
                      ('index', 'int32'), ('iu', 'int32'), ('iv',
                                                            'int32'), ('iw', 'int32'),
                      ('dummy4', 'int32'), ('dummy5',
                                            'int32'), ('cloc', 'float64'),
                      ('eloc', 'float64'), ('dummy6',
                                            'int32'), ('dummy7', 'int32'),
                      ('np1', 'int32'), ('np2', 'int32'),
                      ('dummy8', 'int32'), ('dummy9', 'int32'),
                      ('gc1', 'float64', (np1,)), ('ge1', 'float64', (np1,)),
                      ('dummy10', 'int32'), ('dummy11', 'int32'),
                      ('gc2', 'float64', (np2,)), ('ge2', 'float64', (np2,)),
                      ('dummy12', 'int32'), ('dummy13', 'int32'),
                      ('data', 'float32', (np1*np2,)), ('dummy14', 'int32')])

    my_plane = np.fromfile(f, plane)

    if my_plane['dummy13'][0] == my_plane['dummy14'][0]:
        print('Reading correct')
    else:
        print('Reading error')
        print(my_plane['dummy1'][0], my_plane['dummy2']
              [0], my_plane['dummy3'][0])
        print(my_plane['dummy4'][0], my_plane['dummy5']
              [0], my_plane['dummy6'][0])
        print(my_plane['dummy7'][0], my_plane['dummy8']
              [0], my_plane['dummy9'][0])
        print(my_plane['dummy10'][0], my_plane['dummy11']
              [0], my_plane['dummy12'][0])
        print(my_plane['dummy13'][0], my_plane['dummy14'][0])

    f.close()

    return my_plane

# --------------------------------------------------------

def loc(v, a):
    i = (np.abs(a - v)).argmin()
    return i

# -------------------------------------------------------------

def readres(filename,arrangement='F'):
    # remember: f.seek(),f.tell()

    f = open(filename, 'rb')

    _, i, j, k, jp, _ = st.unpack('i'*6, f.read(24))

    data = np.zeros([k, j, i], dtype=float)

    for kk in range(0, k):

        dummy1 = st.unpack('i', f.read(4))
        data[kk,:, :] = np.reshape(
            st.unpack('d'*i*j, f.read(8*i*j)), (j, i), order=arrangement)
        dummy2 = st.unpack('i', f.read(4))

        if dummy1 != dummy2:
            print('Error reading', kk)
            break

    _, it, _, _, time, _, _, dt, grav, _ = st.unpack(
        'iiiidiiddi', f.read(4*7+3*8))

    data=np.swapaxes(data,0,2)
    return i, j, k, it, time, dt, grav, data

#------------------------------------------------------------
def readres_old(filename):
    # remember: f.seek(),f.tell()

    f = open(filename, 'rb')

    _, i, j, k, jp, _ = st.unpack('i'*6, f.read(24))

    data = np.zeros([i, j, k], dtype=float)

    for kk in range(0, k):

        dummy1 = st.unpack('i', f.read(4))
        data[:, :, kk] = np.reshape(
            st.unpack('d'*i*j, f.read(8*i*j)), (i, j), order='F')
        dummy2 = st.unpack('i', f.read(4))

        if dummy1 != dummy2:
            print('Error reading', kk)
            break

    _, it, _, _, time, _, _, dt, grav, _ = st.unpack(
        'iiiidiiddi', f.read(4*7+3*8))

    return i, j, k, it, time, dt, grav, data

# -------------------------------------------------------------


def read5p(filename):
    # remember: f.seek(),f.tell()

    f = open(filename, 'rb')

    _, i, j, k, jp, _ = st.unpack('i'*6, f.read(24))

    data = np.zeros([i, j, k], dtype=float)

    for kk in range(0, k):

        dummy1 = st.unpack('i', f.read(4))
        data[:, :, kk] = np.reshape(
            st.unpack('d'*i*j, f.read(8*i*j)), (i, j), order='F')
        dummy2 = st.unpack('i', f.read(4))

        if dummy1 != dummy2:
            print('Error reading', kk)
            break

    _, it, _, _, time, _, _, dt, grav, _, _, kmin5p, kmax5p, nzg = st.unpack(
        'iiiidiiddiiiii', f.read(4*11+3*8))

    return i, j, k, it, time, dt, grav, kmin5p, kmax5p, nzg, data

# ----------------------------------------------------------------------


def azi_grid(ny):
    dth = 2*np.pi/(ny-2)
    xe = dth*np.arange(ny)
    xc = xe-dth/2

    return xe, xc

# ----------------------------------------------------------------------


def symmetry(data, var, nyt, full):
    ny = int((nyt-2)/4)

    if var == 'v':
        tmp1 = 0.5*(data[:, 0:ny+1, :]+data[:, 2*ny:3*ny+1, :])
        tmp2 = 0.5*(data[:, ny:2*ny+1, :]+data[:, 3*ny:-1, :])
        dataAve = 0.5*(tmp1+tmp2[:, ::-1, :])

    else:
        tmp1 = 0.5*(data[:, 1:ny+1, :]+data[:, 2*ny+1:3*ny+1, :])
        tmp2 = 0.5*(data[:, ny+1:2*ny+1, :]+data[:, 3*ny+1:-1, :])
        dataAve = 0.5*(tmp1+tmp2[:, ::-1, :])

    if full:
        if var == 'v':
            tmp3 = np.concatenate(
                (dataAve, np.flip(dataAve[:, :-1, :], axis=1)), axis=1)
            tmp4 = np.concatenate((tmp3, tmp3[:, 1:-1, :]), axis=1)

        else:
            tmp3 = np.concatenate((dataAve, np.flip(dataAve, axis=1)), axis=1)
            tmp4 = np.concatenate((tmp3, tmp3), axis=1)

        dataAve = np.empty(data.shape)
        dataAve[:, 1:-1, :] = tmp4
        dataAve[:, 0, :] = dataAve[:, -2, :]
        dataAve[:, -1, :] = dataAve[:, 1, :]

    return dataAve

# ---------------------------------------------------------------------


def grid5p(filename, kmin5p, kmax5p, skip):

    nx, index, x, xe, xc = readgrid(filename)

    kmin5p=kmin5p-1
    kmax5p=kmax5p-1

    xc5p = xc[kmin5p:kmax5p+1]
    xe5p = xe[kmin5p:kmax5p+1]

    nk = 1
    for ii in range(1, np.int(np.floor((nx-2-kmax5p)/20.0))+1):

        xcc0 = kmax5p+nk*20-2
        xcc1 = kmax5p+nk*20-1
        xcc2 = kmax5p+nk*20
        xcc3 = kmax5p+nk*20+1
        xcc4 = kmax5p+nk*20+2

        xc5p = np.append(xc5p, [xc[xcc0]], axis=0)
        xc5p = np.append(xc5p, [xc[xcc1]], axis=0)
        xc5p = np.append(xc5p, [xc[xcc2]], axis=0)
        xc5p = np.append(xc5p, [xc[xcc3]], axis=0)
        xc5p = np.append(xc5p, [xc[xcc4]], axis=0)

        xe5p = np.append(xe5p, [xe[xcc0]], axis=0)
        xe5p = np.append(xe5p, [xe[xcc1]], axis=0)
        xe5p = np.append(xe5p, [xe[xcc2]], axis=0)
        xe5p = np.append(xe5p, [xe[xcc3]], axis=0)
        xe5p = np.append(xe5p, [xe[xcc4]], axis=0)

        nk = nk + 1

    nx5p = xc5p.shape[0]

    xc5pSqueeze = xc5p[kmax5p-kmin5p+1+skip*5+2::5]
    xe5pSqueeze = xe5p[kmax5p-kmin5p+1+skip*5+2::5]
    nx5pSqueeze = xc5pSqueeze.shape[0]

    xc5pSlice = xc5p[kmax5p-kmin5p+1+skip*5:]
    xe5pSlice = xe5p[kmax5p-kmin5p+1+skip*5:]

    nx5pSlice = xc5pSlice.shape[0]

    return nx, nx5pSqueeze,nx5pSlice, nx5p, xe5pSlice,xc5pSlice,xe5pSqueeze, xc5pSqueeze, xe5p, xc5p


# ---------------------------------------------------------------------------

def read5pSqueeze_old(filename, kmin5p, kmax5p, skip):

    f = open(filename, 'rb')

    _, i, j, k, jp, _ = st.unpack('6i', f.read(24))
    nxny = i*j
    kpoints = (kmax5p-kmin5p+1)
    var = '1i'+str(nxny)+'d1i'
    sizeVar = st.calcsize('='+var)

    # do not read # of "skip" blocks of 5 planes
    f.seek(24+kpoints*sizeVar+skip*5*sizeVar)

    nb = int((k-kpoints-skip*5)/5+1)
    data = np.empty([i, j, nb-1])
    for b in range(0, nb-1):

        f.seek(2*sizeVar, 1)
        dummy1 = st.unpack('i', f.read(4))
        data[:, :, b] = np.reshape(
            st.unpack('d'*i*j, f.read(8*i*j)), (i, j), order='F')
        dummy2 = st.unpack('i', f.read(4))
        f.seek(2*sizeVar, 1)

        if dummy1 != dummy2:
            print('Error reading', kk)
            break

    _, it, _, _, time, _, _, dt, grav, _, _, kmin5p, kmax5p, nzg = st.unpack(
        'iiiidiiddiiiii', f.read(4*11+3*8))

    return i, j, k, it, time, dt, grav, kmin5p, kmax5p, nzg, data


def read5pSqueeze(filename, kmin5p, kmax5p, skip):

    f = open(filename, 'rb')

    _, i, j, k, jp, _ = st.unpack('6i', f.read(24))
    nxny = i*j
    kpoints = (kmax5p-kmin5p+1)
    var = '1i'+str(nxny)+'d1i'
    sizeVar = st.calcsize('='+var)

    # do not read # of "skip" blocks of 5 planes
    f.seek(24+kpoints*sizeVar+skip*5*sizeVar)

    nb = int((k-kpoints-skip*5)/5+1)
    data = np.zeros([nb-1, j, i],dtype=float)
    #print('What')
    for b in range(0, nb-1):

        f.seek(2*sizeVar, 1)
        dummy1 = st.unpack('i', f.read(4))
        data[b,:, :] = np.reshape(
            st.unpack('d'*i*j, f.read(8*i*j)), (j, i), order='C')

       # datatmp= np.reshape(st.unpack('d'*i*j, f.read(8*i*j)),
       # (j, i), order='F')
   
       # data[b,:,:]=datatmp

        dummy2 = st.unpack('i', f.read(4))
        f.seek(2*sizeVar, 1)

        if dummy1 != dummy2:
            print('Error reading', kk)
            break

    _, it, _, _, time, _, _, dt, grav, _, _, kmin5p, kmax5p, nzg = st.unpack(
        'iiiidiiddiiiii', f.read(4*11+3*8))

    data=np.swapaxes(data,0,2)

    #print(data.shape)

    return i, j, k, it, time, dt, grav, kmin5p, kmax5p, nzg, data







# ----------------------------------------------------------------------------------

def read5pSlice_old(filename,kmin5p,kmax5p,skip):
#remember: f.seek(),f.tell()
#skip a chunk of the 5p files but still read the 5 planes

        f = open(filename,'rb')

        _,i,j,k,jp,_ = st.unpack('i'*6,f.read(24))

        nxny=i*j
        kpoints=(kmax5p-kmin5p+1)
        var='1i'+str(nxny)+'d1i'
        sizeVar=st.calcsize('='+var)

        #do not read # of "skip" blocks of 5 planes
        f.seek(24+kpoints*sizeVar+skip*5*sizeVar)

        nb=int(k-kpoints-skip*5+1)

        data=np.zeros([i,j,nb-1],dtype=float)

        for kk in range (0,nb-1):

                dummy1 = st.unpack('i',f.read(4))
                data[:,:,kk]=np.reshape(st.unpack('d'*i*j,f.read(8*i*j)),(i,j),order='F')
                dummy2 = st.unpack('i',f.read(4))

                if dummy1!=dummy2:
                        print('Error reading',kk)
                        break

        _,it,_,_,time,_,_,dt,grav,_,_,kmin5p,kmax5p,nzg=st.unpack('iiiidiiddiiiii',f.read(4*11+3*8))


        return i,j,k,it,time,dt,grav,kmin5p,kmax5p,nzg,data

# ----------------------------------------------------------------------------------

def read5pSlice(filename,kmin5p,kmax5p,skip):
#remember: f.seek(),f.tell()
#skip a chunk of the 5p files but still read the 5 planes

        f = open(filename,'rb')

        _,i,j,k,jp,_ = st.unpack('i'*6,f.read(24))

        nxny=i*j
        kpoints=(kmax5p-kmin5p+1)
        var='1i'+str(nxny)+'d1i'
        sizeVar=st.calcsize('='+var)

        #do not read # of "skip" blocks of 5 planes
        f.seek(24+kpoints*sizeVar+skip*5*sizeVar)

        nb=int(k-kpoints-skip*5+1)

        data=np.zeros([nb-1,j,i],dtype=float)

        for kk in range (0,nb-1):

                dummy1 = st.unpack('i',f.read(4))
                data[kk,:,:]=np.reshape(st.unpack('d'*i*j,f.read(8*i*j)),(j,i),order='C')
                dummy2 = st.unpack('i',f.read(4))

                if dummy1!=dummy2:
                        print('Error reading',kk)
                        break

        _,it,_,_,time,_,_,dt,grav,_,_,kmin5p,kmax5p,nzg=st.unpack('iiiidiiddiiiii',f.read(4*11+3*8))


        data=np.swapaxes(data,0,2)

        return i,j,k,it,time,dt,grav,kmin5p,kmax5p,nzg,data






# ----------------------------------------------------------------------------------


def readresSlice(filename, kmin, kmax):

    f = open(filename, 'rb')

    _, i, j, k, jp, _ = st.unpack('6i', f.read(24))
    nxny = i*j
    kpoints = (kmax-kmin+1)
    var = '1i'+str(nxny)+'d1i'
    varT = kpoints*var
    sizeVar = st.calcsize('='+var)
    sizeVarT = kpoints*sizeVar

    f.seek(24+(kmin-1)*sizeVar)

    data = st.unpack('='+varT, f.read(st.calcsize('='+var)*kpoints))

    f.seek(24+k*sizeVar)

    _, it, _, _, time, _, _, dt, grav, _ = st.unpack(
        'iiiidiiddi', f.read(4*7+3*8))

    data = list(data)
    del data[(nxny+2):-1:(nxny+2)]
    del data[0:-1:(nxny+1)]
    del data[-1]

    data = np.reshape(np.array(data), (i, j, kpoints), order='F')

    return i, j, k, it, time, dt, grav, data


# -----------------------------------------------------------------------------------

def diff_cyl2car(dUdR, dUdTh, rc, thc, i, j, k):


    _,jsym,_,_=azi_grid(j)

    # note that the velocity is centered
    dUdX = np.zeros([i, j, k])
    dUdY = np.zeros([i, j, k])

    cosTh = np.cos(thc)
    sinTh = np.sin(thc)

    sinThDr = sinTh[np.newaxis, :]/rc[:, np.newaxis]
    cosThDr = cosTh[np.newaxis, :]/rc[:, np.newaxis]

    dUdX = dUdR * cosTh[np.newaxis, :, np.newaxis] - \
        sinThDr[:, :, np.newaxis]*dUdTh
    dUdY = dUdR * sinTh[np.newaxis, :, np.newaxis] + \
        cosThDr[:, :, np.newaxis]*dUdTh

    dUdX[0, :, :] = dUdX[1,jsym, :]
    dUdX[:, 0, :] = dUdX[:,-2, :]
#    print('imposed') 

    dUdY[0, :, :] = dUdY[1,jsym, :]
    dUdY[:, 0, :] = dUdY[:,-2, :]
 

    return dUdX, dUdY

# -----------------------------------------------------------------------------------


def cyl2car(Ucyl, Vcyl, rc, thc):

    i,j,k=Ucyl.shape

    _,jsym,_,_=azi_grid(j)

    # note that the velocity is centered
    UcCar = np.zeros([i, j, k])
    VcCar = np.zeros([i, j, k])

    cosTh = np.cos(thc)
    sinTh = np.sin(thc)

    UcCar = Ucyl*cosTh[np.newaxis, :, np.newaxis]-sinTh[np.newaxis, :, np.newaxis]*Vcyl
    VcCar = Ucyl*sinTh[np.newaxis, :, np.newaxis]+cosTh[np.newaxis, :, np.newaxis]*Vcyl

    UcCar[0, :, :] = UcCar[1,jsym, :]
    UcCar[:, 0, :] = UcCar[:,-2, :]
    VcCar[0, :, :] = VcCar[1,jsym, :]
    VcCar[:, 0, :] = VcCar[:,-2, :]
 
    return UcCar, VcCar


# --------------------------------------------------------------------------------
def azi_grid(ny):
    dth=2*np.pi/(ny-2)
    xe=dth*np.arange(ny)
    xc=xe-dth/2

    j=np.arange(ny)
    jsym=j+int((ny-2)/2)
    jsym[jsym>(ny-2)]=jsym[jsym>(ny-2)]-(ny-2)


    return j,jsym,xe,xc

# --------------------------------------------------------------------------------
def bc(data,var,j,jsym):

     data[:,0,:]=data[:,-2,:]
     data[:,-1,:]=data[:,1,:]

#     j=np.arange(ny)
#     jsym=j+int((ny-2)/2)
#     jsym[jsym>(ny-2)]=jsym[jsym>(ny-2)]-(ny-2)

     if var=='v':

         data[0,j,:]=-data[1,jsym,:]

     elif var=='u':

         data[0,j,:]=0.5*(-data[1,jsym,:]+data[1,j,:])

     else:

         data[0,j,:]=data[1,jsym,:]

     return data

# --------------------------------------------------------------------------------

def center(dataU, dataV, dataW):


    i,j,k=dataU.shape

    _,jsym,_,_=azi_grid(j)

    # Center velocity

    dataUc = np.zeros([i, j, k])
    dataUc[1:, ] = 0.5*(dataU[0:-1, ]+dataU[1:, ])

    dataVc = np.zeros([i, j, k])
    dataVc[:, 1:, ] = 0.5*(dataV[:, 1:, ]+dataV[:, 0:-1, ])

    dataWc = np.zeros([i, j, k])
    dataWc[:, :, 1:] = 0.5*(dataW[:, :, 1:]+dataW[:, :, 0:-1])

    dataUc[0, :, :] = dataUc[1,jsym, :]
    dataUc[:, 0, :] = dataUc[:,-2, :]

    dataVc[:, 0, :] = dataVc[:,-2, :] 

    dataWc[:, :, 0] = dataWc[:,:,1]
    dataWc[:, 0, :] = dataWc[:,-2, :]



    return dataUc, dataVc, dataWc


# ----------------------------------------------------------------------------


def centerUV(dataU, dataV):

    i,j,k=dataU.shape

    _,jsym,_,_=azi_grid(j)

    # Center velocity

    dataUc = np.zeros([i, j, k])
    dataUc[1:, ] = 0.5*(dataU[0:-1, ]+dataU[1:, ])

    dataVc = np.zeros([i, j, k])
    dataVc[:, 1:, ] = 0.5*(dataV[:, 1:, ]+dataV[:, 0:-1, ])

    dataUc[0, :, :] = dataUc[1,jsym, :]

    dataUc[:, 0, :] = dataUc[:,-2, :]

    dataVc[:, 0, :] = dataVc[:,-2, :]
   
 
    return dataUc, dataVc

# --------------------------------------------------------------------------





def centerUV_old(dataU, dataV):

    i,j,k=dataU.shape

    # Center velocity

    dataUc = np.zeros([i, j, k])
    dataUc[:-1, ] = 0.5*(dataU[1:, ]+dataU[0:-1, ])

    dataVc = np.zeros([i, j, k])
    dataVc[:, :-1, ] = 0.5*(dataV[:, 1:, ]+dataV[:, 0:-1, ])

    dataUc[-1, :, :] = dataU[-1, :, :]
    dataVc[:, -1, :] = dataV[:, -1, :]

    print('fixed')

    return dataUc, dataVc

#---------------------------------------------------------------------------


