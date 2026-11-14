import copy

import pytest


def reader():
    from mimo_ota_simulator import imported_cluster_trajectory
    return imported_cluster_trajectory


def document():
    path = dict(clusterId=0, delayNs=2., hReal=-1., hImag=2.,
                aoaPhi=180., aoaTheta=90., aodPhi=-10., aodTheta=30.,
                channelType=1, dopplerHz=None)
    frames = []
    for i, time in enumerate([0., 1., 1.145913773532114]):
        frames.append(dict(sampleIndex=i, frameId=i, internalLinkId=10+i,
                           timeSeconds=time, inputPathCount=4, status='clustered',
                           clustering=dict(rawPaths=4, filteredPaths=3, nCluster=1,
                                           paths=[copy.deepcopy(path)])))
    return dict(schemaVersion='meta3d-cluster-trajectory/1',
                source=dict(taskId='synthetic-test', mpdbPath='s3://test/R0B0S0D.mpdb',
                            contentVersion=None, softwareVersion=None, additionalLoss=None),
                carrierHz=2.4e9, txAnt=0, rxAnt=0, angleCoordinateSystem='LCS',
                units=dict(time='s', delay='ns', angles='deg', doppler='Hz',
                           carrier='Hz', response='complex_scalar'),
                clusterIdentity='frame-local-only', requestedClusterCount=8,
                timing=dict(source='synthetic-test', timesSeconds=[f['timeSeconds'] for f in frames]),
                frames=frames)


def test_preserves_complete_document_and_unknowns():
    data = document()
    result = reader().validate_trajectory(data)
    assert result == data
    assert result['frames'][-1]['timeSeconds'] == 1.145913773532114
    assert result['frames'][0]['clustering']['paths'][0]['hImag'] == 2.
    assert result['frames'][0]['clustering']['paths'][0]['dopplerHz'] is None
    assert result['source']['contentVersion'] is None


@pytest.mark.parametrize('bad', ['time', 'unit', 'imag', 'nan', 'count', 'budget',
                                'identity', 'source', 'bool', 'doppler'])
def test_rejects_corruption(bad):
    data = document()
    frame = data['frames'][0]
    path = frame['clustering']['paths'][0]
    if bad == 'time': data['frames'][2]['timeSeconds'] = 2.
    if bad == 'unit': data['units']['delay'] = 's'
    if bad == 'imag': del path['hImag']
    if bad == 'nan': path['hReal'] = float('nan')
    if bad == 'count': frame['clustering']['nCluster'] = 2
    if bad == 'budget': frame['clustering']['paths'] *= 9
    if bad == 'identity': data['clusterIdentity'] = 'physical'
    if bad == 'source': del data['source']['mpdbPath']
    if bad == 'bool': frame['timeSeconds'] = False
    if bad == 'doppler': del path['dopplerHz']
    with pytest.raises(ValueError): reader().validate_trajectory(data)


@pytest.mark.parametrize('status,count', [('no_paths', 0), ('no_valid_paths', 4)])
def test_empty_frame_remains_in_timeline(status, count):
    data = document()
    data['frames'][1].update(status=status, inputPathCount=count, clustering=None)
    assert len(reader().validate_trajectory(data)['frames']) == 3


def test_reader_rejects_duplicate_json_keys(tmp_path):
    path = tmp_path / 'bad.json'
    path.write_text('{"schemaVersion": 1, "schemaVersion": 2}')
    with pytest.raises(ValueError, match='重复'): reader().read_trajectory(path)


def test_sampled_response_uses_complex_h_delay_and_actual_time():
    data = document()
    data['frames'][1].update(status='no_paths', inputPathCount=0, clustering=None)
    rows = reader().sampled_responses(data, [0., 125e6])
    assert [r['timeSeconds'] for r in rows] == data['timing']['timesSeconds']
    assert rows[0]['real'] == pytest.approx([-1., 2.])
    assert rows[0]['imag'] == pytest.approx([2., 1.])
    assert rows[1]['status'] == 'unavailable'
    assert rows[1]['real'] is None
    assert rows[1]['reason'] == 'no_paths'


def test_response_grid_must_be_explicit_and_finite():
    for grid in ([], [float('nan')], [True]):
        with pytest.raises(ValueError): reader().sampled_responses(document(), grid)


def test_exchange_version_rejects_old_native_claim():
    data = document()
    data['schemaVersion'] = 'lauraycs-cluster-trajectory/1'
    with pytest.raises(ValueError, match='版本'):
        reader().validate_trajectory(data)


def test_file_round_trip_preserves_full_payload(tmp_path):
    import json
    data = document()
    data['source']['extraMetadata'] = {'unknown': None, 'detail': [1, 'original']}
    data['angleCoordinateSystem'] = None
    before = copy.deepcopy(data)
    path = tmp_path / 'trajectory.json'
    path.write_text(json.dumps(data), encoding='utf-8')
    loaded = reader().read_trajectory(path)
    assert loaded == before
    reader().sampled_responses(loaded, [0.])
    assert loaded == before


def test_reader_enforces_size_limit(tmp_path, monkeypatch):
    module = reader()
    monkeypatch.setattr(module, 'MAX_BYTES', 8)
    path = tmp_path / 'large.json'
    path.write_bytes(b' ' * 9)
    with pytest.raises(ValueError, match='大小限制'):
        module.read_trajectory(path)


def test_response_sums_all_clusters_without_normalization():
    data = document()
    result = data['frames'][0]['clustering']
    extra = copy.deepcopy(result['paths'][0])
    extra.update(clusterId=3, hReal=3., hImag=-4., delayNs=0.)
    result['paths'].append(extra)
    result['nCluster'] = 2
    rows = reader().sampled_responses(data, [0., 125e6])
    assert rows[0]['real'] == pytest.approx([2., 5.])
    assert rows[0]['imag'] == pytest.approx([-2., -3.])
    assert rows[0]['status'] == 'model_reconstruction'


def test_no_valid_paths_response_stays_unknown():
    data = document()
    data['frames'][1].update(status='no_valid_paths', inputPathCount=4, clustering=None)
    row = reader().sampled_responses(data, [0.])[1]
    assert row == dict(sampleIndex=1, timeSeconds=1., status='unavailable',
                       reason='no_valid_paths', real=None, imag=None)
