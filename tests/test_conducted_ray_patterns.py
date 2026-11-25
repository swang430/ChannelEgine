import numpy as np
import pytest
from mimo_ota_simulator.conducted_ray_patterns import terminal_field, project_rays

def pattern():
    return dict(version="fixture",azimuth_coverage="periodic",carrier_hz=1e9,local_to_global=np.eye(3).tolist(),
                zenith_deg=[0,180],azimuth_deg=[0,180],gain_convention="absolute_field",
                field=[[[[1,0],[0,0]]]*2]*2)

def test_rotation_and_polarization():
    spec=pattern()
    v=terminal_field(spec,[0],[90],1e9)
    assert np.allclose(v,[[1,0]])
    spec["local_to_global"]=[[1,0,0],[0,0,-1],[0,1,0]]
    rotated=terminal_field(spec,[0],[90],1e9)
    assert np.allclose(abs(rotated),[[0,1]],atol=1e-12)
    assert np.allclose(project_rays(v,rotated,[np.eye(2)],[1]),0,atol=1e-12)
    assert np.allclose(project_rays(2*v,v,[np.eye(2)],[1j]),[2j])
    with pytest.raises(ValueError):
        terminal_field(spec,[0],[90],2e9)

def test_evidence_and_numeric_boundaries():
    spec=pattern()
    spec.update(azimuth_deg=[30,40],azimuth_coverage="sector")
    assert np.allclose(terminal_field(spec,[30],[90],1e9),[[1,0]])
    with pytest.raises(ValueError):
        terminal_field(spec,[220],[90],1e9)
    with pytest.raises(ValueError):
        terminal_field(spec,[35],[90],None)
    spec["field"]=np.array(spec["field"],complex)*1j
    with pytest.raises(ValueError):
        terminal_field(spec,[35],[90],1e9)
    v=np.array([[20,0]],np.int8)
    assert project_rays(v,v,[np.eye(2,dtype=np.int8)],[1])==400
