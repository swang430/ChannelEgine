"""逐射线天线复电场。接收表使用接收有效长度的相位约定，收端按转置收缩。"""
import numpy as np


def _basis(az, zen):
    sa, ca = np.sin(az), np.cos(az)
    sz, cz = np.sin(zen), np.cos(zen)
    radial = np.column_stack((sz*ca, sz*sa, cz))
    theta = np.column_stack((cz*ca, cz*sa, -sz))
    phi = np.column_stack((-sa, ca, np.zeros_like(sa)))
    return radial, theta, phi


def terminal_field(pattern, azimuth_deg, zenith_deg, carrier_hz):
    """局部等角或非等角网格上双线性插值，方位周期延拓，再转到全局球面基。"""
    if (isinstance(carrier_hz,(bool,np.bool_)) or isinstance(pattern.get("carrier_hz"),(bool,np.bool_))
            or not isinstance(carrier_hz,(int,float,np.integer,np.floating)) or not np.isreal(carrier_hz) or not np.isfinite(carrier_hz) or carrier_hz<=0
            or not isinstance(pattern.get("version"),str) or not pattern["version"].strip()
            or pattern.get("carrier_hz") != carrier_hz or pattern.get("gain_convention") != "absolute_field"):
        raise ValueError("天线方向图版本、绝对场增益约定或频点不符")
    rotation = np.asarray(pattern["local_to_global"],dtype=float)
    if (rotation.shape != (3,3) or not np.allclose(rotation.T@rotation,np.eye(3),atol=1e-8,rtol=0)
            or not np.isclose(np.linalg.det(rotation),1,atol=1e-8,rtol=0)):
        raise ValueError("天线安装姿态须为右手正交旋转矩阵")
    az, ze = (np.asarray(pattern[k],dtype=float) for k in ("azimuth_deg","zenith_deg"))
    raw = np.asarray(pattern["field"])
    if raw.dtype.kind not in "fiu":
        raise ValueError("复场须以实部、虚部两个实数表示")
    table = raw.astype(float)
    if (az.ndim!=1 or ze.ndim!=1 or min(az.size,ze.size)<2 or np.any(np.diff(az)<=0)
            or np.any(np.diff(ze)<=0) or not np.isfinite(az).all() or not np.isfinite(ze).all()
            or az[0]<0 or az[-1]>=360 or ze[0]<0 or ze[-1]>180
            or table.shape!=(len(ze),len(az),2,2) or not np.isfinite(table).all()):
        raise ValueError("复方向图需要递增的方位、天顶角及两极化的实虚部")
    a,z = np.asarray(azimuth_deg,float),np.asarray(zenith_deg,float)
    if a.ndim!=1 or a.shape!=z.shape or not np.isfinite(a).all() or not np.all((z>=0)&(z<=180)):
        raise ValueError("射线角度无效")
    radial,gt,gp = _basis(np.deg2rad(a),np.deg2rad(z))
    local = radial @ rotation
    la = np.arctan2(local[:,1],local[:,0])
    lz = np.arccos(np.clip(local[:,2],-1,1))
    aa=(np.rad2deg(la)-az[0])%360+az[0]
    aa=np.where(np.isclose(aa,az[0]+360,atol=1e-9,rtol=0),az[0],aa)
    zz=np.rad2deg(lz)
    if np.any((zz<ze[0]-1e-9)|(zz>ze[-1]+1e-9)):
        raise ValueError("方向图缺少本次射线所在的天顶角覆盖")
    zz=np.clip(zz,ze[0],ze[-1])
    coverage=pattern.get("azimuth_coverage")
    if coverage not in ("periodic","sector"):
        raise ValueError("请声明完整周期方向图或扇区扫描")
    if coverage=="sector" and np.any(aa>az[-1]+1e-9):
        raise ValueError("射线超出方位测量范围")
    az=np.append(az,az[0]+360)
    grid=table[...,0]+1j*table[...,1]
    grid=np.concatenate((grid,grid[:,:1]),axis=1)
    ai=np.clip(np.searchsorted(az,aa,side="right")-1,0,len(az)-2)
    zi=np.clip(np.searchsorted(ze,zz,side="right")-1,0,len(ze)-2)
    aw=((aa-az[ai])/(az[ai+1]-az[ai]))[:,None]
    zw=((zz-ze[zi])/(ze[zi+1]-ze[zi]))[:,None]
    field=(1-zw)*((1-aw)*grid[zi,ai]+aw*grid[zi,ai+1])+zw*((1-aw)*grid[zi+1,ai]+aw*grid[zi+1,ai+1])
    _,lt,lp=_basis(la,lz)
    xyz=(field[:,:1]*lt+field[:,1:]*lp)@rotation.T
    return np.column_stack(((xyz*gt).sum(axis=1),(xyz*gp).sum(axis=1)))


def project_rays(tx, rx, propagation, complex_path):
    """每条射线只施加一次两端方向响应和一次传播幅相。"""
    tx,rx,propagation,complex_path=(np.asarray(v,dtype=np.complex128) for v in (tx,rx,propagation,complex_path))
    n=len(complex_path)
    if (tx.shape!=(n,2) or rx.shape!=(n,2) or propagation.shape!=(n,2,2)
            or complex_path.shape!=(n,) or not all(np.isfinite(x).all() for x in (tx,rx,propagation,complex_path))):
        raise ValueError("极化传递或天线复数分量缺失，无法核验")
    result = np.einsum("ni,nij,nj->n",rx,propagation,tx)*complex_path
    if not np.isfinite(result).all():
        raise ValueError("极化投影超出数值范围")
    return result
