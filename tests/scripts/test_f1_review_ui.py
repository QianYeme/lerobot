import csv
import http.client
import json
import threading
from http.server import ThreadingHTTPServer

import pytest

from scripts.run_f1_review_ui import ConflictError, FIELDS, ReviewStore, make_handler


@pytest.fixture
def store(tmp_path):
    samples = tmp_path / 'evidence_samples'
    samples.mkdir()
    records = []
    for ep in [0, 2, 4, 6, 7, 9]:
        (samples / f'ep{ep}.mp4').write_bytes(bytes(range(100)) * 10)
        records.append({'episode': ep, 'frames': 479, 'fps': 30, 'split': 'train',
                        'clip': f'evidence_samples/ep{ep}.mp4'})
    (tmp_path / 'label_manifest.json').write_text(json.dumps({'records': records}))
    with (samples / 'review_intervals.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows([{'episode': r['episode']} for r in records])
    return ReviewStore(tmp_path, tmp_path / 'cache')


def row(label='PRECONTACT', start='20', end='40'):
    return {'label': label, 'first_reliable_frame': start, 'last_reliable_frame': end,
            'reviewer': 'human_test', 'reason': 'Visible gap between jaw and cup'}


def test_save_readback_backup_preserves_other_episode_and_restart(store):
    revision = store.revision()
    reply = store.save(0, [row()], revision)
    assert reply['rows'][0]['first_reliable_frame'] == '20'
    assert reply['rows'][0]['evidence'] == 'ep0.mp4#frames=20-40'
    assert reply['revision'] != revision
    store.save(2, [row(start='60', end='80')], reply['revision'])
    restarted = ReviewStore(store.result, store.cache)
    assert [len(x['rows']) for x in restarted.bootstrap()['episodes']] == [1, 1, 0, 0, 0, 0]
    assert len(list((store.csv_path.parent / 'csv_backups').glob('*.csv'))) == 2


def test_stale_page_does_not_overwrite(store):
    revision = store.revision()
    store.save(0, [row()], revision)
    before = store.csv_path.read_bytes()
    with pytest.raises(ConflictError):
        store.save(2, [row()], revision)
    assert store.csv_path.read_bytes() == before


def test_pilot_allowlist_and_contact_only_labels(store):
    manifest = json.loads((store.result / 'label_manifest.json').read_text())
    fit, evaluate = [0, 2, 4, 6, 7, 9], [10, 19, 29, 38, 49, 57]
    for record, ep in zip(manifest['records'], evaluate):
        record['episode'] = ep
    manifest['experiment_id'] = 'F1P_CONTACT_SIGNAL_PILOT'
    (store.result / 'label_manifest.json').write_text(json.dumps(manifest))
    (store.result / 'subset_manifest.json').write_text(json.dumps({'pilot_fit': fit, 'pilot_eval': evaluate}))
    pilot = ReviewStore(store.result, store.cache)
    assert list(pilot.records) == evaluate
    assert pilot.labels == ['PRECONTACT', 'CONTACT_SUPPORTED', 'UNKNOWN', 'OCCLUDED']
    assert pilot.validate(10, [row()])[0]['episode'] == '10'
    with pytest.raises(ValueError):
        pilot.validate(10, [row('CLOSING')])
    with pytest.raises(ValueError):
        pilot.validate(0, [row()])
    manifest['records'][0]['episode'] = 3  # Original Dev is never allowed.
    (store.result / 'label_manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        ReviewStore(store.result, store.cache)


@pytest.mark.parametrize('patch', [
    {'first_reliable_frame': '-1'}, {'last_reliable_frame': '479'},
    {'last_reliable_frame': '19'}, {'first_reliable_frame': '20.5'},
    {'reviewer': ''}, {'reason': ''}, {'label': 'STABLE'},
    {'uncertain_start': '25', 'uncertain_end': '30'},
    {'uncertain_start': '5', 'uncertain_end': ''},
])
def test_invalid_interval_never_touches_csv(store, patch):
    before = store.csv_path.read_bytes()
    with pytest.raises(ValueError):
        store.save(0, [{**row(), **patch}], store.revision())
    assert store.csv_path.read_bytes() == before


def test_unknown_not_reliable_and_axis_overlap_rules(store):
    unknown = {'label': 'OCCLUDED', 'uncertain_start': '100', 'uncertain_end': '120',
               'reason': 'Jaw occludes contact edge', 'reviewer': 'human_test'}
    assert len(store.validate(0, [row(), row('CLOSING'), unknown])) == 3
    with pytest.raises(ValueError):
        store.validate(0, [row(), row('CONTACT_SUPPORTED')])
    with pytest.raises(ValueError):
        store.validate(0, [{**unknown, 'first_reliable_frame': '100', 'last_reliable_frame': '120'}])
    with pytest.raises(ValueError):
        store.validate(3, [row()])


def test_http_save_ranges_blind_routes_and_cross_origin(store):
    store.videos = store.sources
    server = ThreadingHTTPServer(('127.0.0.1', 0), make_handler(store))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    conn = http.client.HTTPConnection('127.0.0.1', server.server_port)
    try:
        conn.request('GET', '/videos/0', headers={'Range': 'bytes=100-199'})
        reply = conn.getresponse()
        assert reply.status == 206 and reply.getheader('Content-Range') == 'bytes 100-199/1000'
        assert len(reply.read()) == 100
        conn.request('HEAD', '/videos/0', headers={'Range': 'bytes=-100'})
        reply = conn.getresponse()
        assert reply.status == 206 and reply.getheader('Content-Length') == '100'
        assert reply.read() == b''
        conn.request('GET', '/videos/0', headers={'Range': 'bytes=1001-'})
        reply = conn.getresponse()
        assert reply.status == 416
        reply.read()
        conn.request('GET', '/signal_integrity.json')
        reply = conn.getresponse()
        assert reply.status == 404
        reply.read()
        body = json.dumps({'episode': 0, 'rows': [row()], 'revision': store.revision()})
        conn.request('POST', '/api/save', body, {'Content-Type': 'application/json', 'Origin': 'http://evil.invalid'})
        reply = conn.getresponse()
        assert reply.status == 403
        reply.read()
        conn.request('POST', '/api/save', body, {'Content-Type': 'application/json'})
        reply = conn.getresponse()
        assert reply.status == 200 and json.loads(reply.read())['rows'][0]['label'] == 'PRECONTACT'
        conn.request('GET', '/api/bootstrap', headers={'Host': 'evil.invalid'})
        reply = conn.getresponse()
        assert reply.status == 403
        reply.read()
    finally:
        conn.close()
        server.shutdown()
        server.server_close()
        thread.join()
