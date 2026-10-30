import math
JRLL=[2,2,2,2,3,3,3,3,4,4,4,4]; JPLL=[1,3,5,7,0,2,4,6,1,3,5,7]
def compress(v):
    r=0;b=0
    while v:
        r|=(v&1)<<b; v>>=2; b+=1
    return r
def gaia_radec(sid):
    order=12; nside=1<<order; npface=nside*nside
    ipix=sid//34359738368
    face=ipix//npface; ipf=ipix%npface
    ix=compress(ipf); iy=compress(ipf>>1)
    jr=JRLL[face]*nside-ix-iy-1
    if jr<nside: nr=jr; z=1-nr*nr/(3*npface); ks=0
    elif jr>3*nside: nr=4*nside-jr; z=nr*nr/(3*npface)-1; ks=0
    else: nr=nside; z=(2*nside-jr)*2/(3*nside); ks=(jr-nside)&1
    jp=(JPLL[face]*nr+ix-iy+1+ks)//2
    if jp>4*nside: jp-=4*nside
    if jp<1: jp+=4*nside
    phi=(jp-(ks+1)*0.5)*(math.pi/2/nr)
    return math.degrees(phi)%360, 90-math.degrees(math.acos(z))
if __name__=='__main__':
    ra,dec=gaia_radec(2855836699866130816); print(ra/15,dec)
