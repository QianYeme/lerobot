"""Real Edge UI smoke using an isolated temporary CSV, never research labels."""
import argparse
import csv
import hashlib
import json
import shutil
import sys
import tempfile
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'outputs/f1_rgb_review_ui/test_dependencies'))
from playwright.sync_api import sync_playwright
from scripts.run_f1_review_ui import DEFAULT_RESULT, FIELDS, ReviewStore, make_handler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--result', type=Path, default=DEFAULT_RESULT)
    args = parser.parse_args()
    source_result = args.result.resolve()
    official = source_result / 'evidence_samples/review_intervals.csv'
    original_hash = hashlib.sha256(official.read_bytes()).hexdigest()
    with tempfile.TemporaryDirectory(prefix='f1_ui_smoke_') as temp:
        result = Path(temp)
        samples = result / 'evidence_samples'
        samples.mkdir()
        manifest = json.loads((source_result / 'label_manifest.json').read_text())
        pilot = manifest.get('experiment_id') == 'F1P_CONTACT_SIGNAL_PILOT'
        first, second = [r['episode'] for r in manifest['records'][:2]]
        shutil.copy2(source_result / 'label_manifest.json', result / 'label_manifest.json')
        if pilot:
            shutil.copy2(source_result / 'subset_manifest.json', result / 'subset_manifest.json')
        for record in manifest['records']:
            source = source_result / record['clip']
            shutil.copy2(source, samples / source.name)
        with (samples / 'review_intervals.csv').open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows([{'episode': r['episode']} for r in manifest['records']])
        store = ReviewStore(result, ROOT / 'outputs/f1_rgb_review_ui/video_cache')
        for ep, path in store.sources.items():
            stamp = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
            store.videos[ep] = store.cache / f'ep{ep:03d}_{stamp}.mp4'
            assert store.videos[ep].is_file()
        server = ThreadingHTTPServer(('127.0.0.1', 0), make_handler(store))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        errors = []
        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(channel='msedge', headless=True)
                page = browser.new_page(viewport={'width': 1600, 'height': 1000})
                page.on('pageerror', lambda error: errors.append(str(error)))
                page.goto(f'http://127.0.0.1:{server.server_port}')
                page.wait_for_selector(f'[data-episode="{first}"]')
                page.wait_for_function("document.querySelector('#video').readyState >= 2")
                page.locator('#play').click()
                page.wait_for_function("document.querySelector('#video').currentTime > 1")
                page.locator('#play').click()
                page.locator('#timeline').evaluate("e => { e.value=250; e.dispatchEvent(new Event('input')); e.dispatchEvent(new Event('change')); }")
                page.wait_for_function("document.querySelector('#frameText').textContent.includes('250 /') && !document.querySelector('#still').hidden && document.querySelector('#still').src.includes('frame=250')")
                page.locator('[data-step="1"]').click()
                page.wait_for_function("document.querySelector('#frameText').textContent.includes('251 /')")
                page.locator('#reviewer').fill('UI_SMOKE_NOT_RESEARCH')
                page.locator('#add').click()
                page.locator('[data-capture="first_reliable_frame"]').click()
                assert page.locator('[data-field="first_reliable_frame"]').input_value() == '251'
                page.locator('[data-field="first_reliable_frame"]').fill('20')
                page.locator('[data-field="last_reliable_frame"]').fill('40')
                page.locator('[data-field="reason"]').fill('Synthetic UI test only; not a visual label')
                if pilot:
                    assert page.locator('[data-field="label"] option').count() == 4
                else:
                    page.locator('#save').click()
                page.wait_for_selector('#saveStatus.saved')
                assert store.bootstrap()['episodes'][0]['rows'][0]['first_reliable_frame'] == '20'
                page.reload()
                page.wait_for_selector('[data-field="first_reliable_frame"]')
                assert page.locator('[data-field="last_reliable_frame"]').input_value() == '40'
                page.locator(f'[data-episode="{second}"]').click()
                page.wait_for_function(f"document.querySelector('#episodeTitle').textContent.includes('{second:03d}')")
                assert page.locator('.interval').count() == 0
                page.locator(f'[data-episode="{first}"]').click()
                page.wait_for_selector('.interval')
                prefix = 'pilot_' if pilot else ''
                page.screenshot(path=str(ROOT / f'outputs/f1_rgb_review_ui/{prefix}desktop_smoke.png'), full_page=True)
                page.set_viewport_size({'width': 800, 'height': 1000})
                page.screenshot(path=str(ROOT / f'outputs/f1_rgb_review_ui/{prefix}narrow_smoke.png'), full_page=True)
                assert page.evaluate('document.documentElement.scrollWidth <= window.innerWidth')
                assert not errors, errors
                browser.close()
        finally:
            server.shutdown()
            server.server_close()
            thread.join()
    assert hashlib.sha256(official.read_bytes()).hexdigest() == original_hash
    print(json.dumps({'browser': 'Microsoft Edge headless', 'pilot': pilot, 'autosave_tested': pilot, 'playback': 'PASS', 'seek_exact_250_step_251': 'PASS',
                      'save_disk_readback_reload': 'PASS', 'episode_switch': 'PASS', 'narrow_no_overflow': 'PASS',
                      'page_errors': errors, 'official_csv_unchanged': True}))


if __name__ == '__main__':
    main()
