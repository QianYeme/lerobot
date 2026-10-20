"""Isolated headless browser checks; synthetic labels are never saved to project CSV."""
import csv
import hashlib
import io
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'outputs/f1_rgb_review_ui/test_dependencies'))
from playwright.sync_api import sync_playwright

OUT = ROOT / 'experiments/force/f1rm_teleop_load_retrospective_match/index1/result'


def main():
    official = OUT / 'contact_annotations.csv'
    before = hashlib.sha256(official.read_bytes()).hexdigest()
    errors = []
    with tempfile.TemporaryDirectory(prefix='f1rm_smoke_') as temp:
        with sync_playwright() as p:
            browser = p.chromium.launch(channel='msedge', headless=True)
            page = browser.new_page(accept_downloads=True)
            page.on('pageerror', lambda error: errors.append(str(error)))
            page.goto((OUT / 'review.html').as_uri())
            page.wait_for_function("document.querySelector('#video').readyState>=2")
            assert page.locator('#trial option').count() == 10
            page.evaluate("document.querySelector('#video').currentTime=2")
            page.wait_for_function("Math.abs(document.querySelector('#video').currentTime-2)<0.01")
            page.locator('#rows button').nth(0).click()
            assert page.locator('#rows input').nth(0).input_value() == '60'
            page.locator('#forward').click()
            page.wait_for_function("Math.abs(document.querySelector('#video').currentTime-61/30)<0.001")
            page.locator('#export').click()
            assert '未完成' in page.locator('#status').inner_text()
            # Ten synthetic UNKNOWN rows exercise export, not a research annotation.
            for i in range(10):
                page.locator('#trial').select_option(index=i)
                page.locator('#reviewed').check()
                page.locator('#reviewer').fill('SYNTHETIC_TEST_ONLY')
                page.locator('#rows select').select_option('UNKNOWN')
                inputs = page.locator('#rows input')
                inputs.nth(0).fill('0')
                inputs.nth(1).fill('149')
                inputs.nth(2).fill('SYNTHETIC_TEST_ONLY')
            with page.expect_download() as download:
                page.locator('#export').click()
            path = Path(temp) / 'synthetic.csv'
            download.value.save_as(path)
            rows = list(csv.DictReader(io.StringIO(path.read_text(encoding='utf-8-sig'))))
            assert len(rows) == 10 and len({r['trial_id'] for r in rows}) == 10
            page.locator('#trial').select_option(index=0)
            assert page.locator('#reviewer').input_value() == 'SYNTHETIC_TEST_ONLY'
            page.locator('#add').click()
            page.locator('#rows input').nth(3).fill('0')
            page.locator('#rows input').nth(4).fill('1')
            page.locator('#rows input').nth(5).fill('SYNTHETIC_OVERLAP')
            page.locator('#export').click()
            assert '重叠' in page.locator('#status').inner_text()
            assert not errors, errors
            browser.close()
    assert hashlib.sha256(official.read_bytes()).hexdigest() == before
    result = dict(status='PASS', checks=['video decode/seek', 'frame capture/step', '10 trial options',
                  'incomplete export rejected', 'synthetic CSV roundtrip', 'trial switch draft retained',
                  'overlap rejected', 'no JS errors', 'official CSV hash unchanged'])
    (OUT / 'browser_smoke.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
