import numpy as np, math
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory
q='20260921_062224'
maps=[];gnss=[]
with open(f'/data/data/reLocationData/zc_data/{q}.mcap','rb') as fh:
    rd=make_reader(fh,decoder_factories=[DecoderFactory()])
    for s,ch,m,dec in rd.iter_decoded_messages(topics=['/map','/ll/position/gps/gnss_fix']):
        if ch.topic=='/map':
            p=dec.pose.pose.position; maps.append((m.log_time,p.x,p.y))
        else: gnss.append((m.log_time,dec.latitude,dec.longitude))
M=np.array(maps);G=np.array(gnss)
lat0,lon0=G[0,1],G[0,2]
e=(G[:,2]-lon0)*math.cos(math.radians(lat0))*111320; n=(G[:,1]-lat0)*110540
j=np.clip(np.searchsorted(M[:,0],G[:,0]),0,len(M)-1)
mx,my=M[j,1],M[j,2]
# fit 2D rigid (yaw+2t) map<-enu via least squares on centered coords
ce,cn=e-e.mean(),n-n.mean(); cm,cy=mx-mx.mean(),my-my.mean()
# complex form: cm+i*cy = z*(ce+i*cn)
E=ce+1j*cn; Mc=cm+1j*cy
z=(E.conj()@Mc)/(E.conj()@E).real
res=np.abs(Mc-z*E)
print(f"rigid fit scale={abs(z):.3f} yaw={math.degrees(np.angle(z)):.1f}deg; residual |map-fit(enu)|: p50={np.median(res):.2f} p95={np.percentile(res,95):.2f} max={res.max():.2f} m")
