"""Fetch the additional author files used by the expanded fresh-test protocol.

Optional and network-dependent: the package already contains these files.
Every file comes from the pinned commit and must match the Git blob SHA
listed by the GitHub tree API; any mismatch aborts before writing.
Usage: python simulation/code/fetch_expanded_sources.py
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'public_data/fgf2_author_repository'
MANIFEST = ROOT / 'public_data/fgf2_expanded_source_manifest.json'
REPO = 'Mijan/LFNS_MSB'
COMMIT = '5c917abda0618d75c00c9cab45f24ed893dd71f1'
DATA = 'FGF2_models/data/'
B3 = 'Inference_results/Fgf_B3/results/sus_3_20/'
EXPERIMENT_NOTE = 'Author-provided processed experimental measurements; not new measurements'
SIMULATION_ROLE = "Original authors' B3 simulation export; used only to infer the fgf_sp_60 command timing"


def condition_files(folder, tokens, suffixes=('.txt', '_trunc.txt', '_mean_trunc.txt')):
    return [f'{DATA}{folder}/{token}{suffix}' for token in tokens for suffix in suffixes]


FILES = (condition_files('fgf_sus', ('0-25ng', '25ng'))
         + condition_files('fgf_3_20', ('25ng',))
         + condition_files('fgf_sp_5', ('0-25ng', '25ng'))
         + condition_files('fgf_sp_10', ('0-25ng', '2-5ng', '25ng', '250ng'))
         + [f'{DATA}fgf_sp_10/time.txt', f'{DATA}fgf_sp_10/time_trunc.txt']
         + condition_files('fgf_sp_60', ('2-5ng', '25ng', '250ng'))
         + [f'{DATA}fgf_sp_60/time.txt', f'{DATA}fgf_sp_60/time_trunc.txt']
         + condition_files('fgf_mixed', ('0-25ng', '25ng'), ('_trunc.txt', '_mean_trunc.txt', '_unnormalized.txt'))
         + [f'{B3}sim_times.txt']
         + [f'{B3}sim_{p}_{t}_measurements.txt' for p in ('sus', 'sp_10', 'sp_60') for t in ('2-5ng', '25ng', '250ng')])


def fetch(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent': 'fgf2-fetch'}), timeout=60) as response:
        return response.read()


def git_blob_sha(raw):
    return hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest()


def main():
    tree = json.loads(fetch(f'https://api.github.com/repos/{REPO}/git/trees/{COMMIT}?recursive=1'))
    upstream = {item['path']: item['sha'] for item in tree['tree'] if item['type'] == 'blob'}
    contents = {}
    for path in FILES:
        raw = fetch(f'https://raw.githubusercontent.com/{REPO}/{COMMIT}/{path}')
        if git_blob_sha(raw) != upstream[path]:
            raise SystemExit(f'Blob mismatch for {path}; nothing written.')
        contents[path] = raw
    today = datetime.now(timezone.utc).date().isoformat()
    manifest = [{'repository': REPO, 'commit_sha': COMMIT,
                 'scope': 'Additional files for the expanded fresh-test protocol; the pilot manifest is unchanged.'}]
    for path, raw in contents.items():
        target = SOURCE / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
        entry = {'path': path, 'blob_sha': upstream[path], 'source_size_bytes': len(raw),
                 'source_url': f'https://github.com/{REPO}/blob/{COMMIT}/{path}', 'retrieved_utc_date': today}
        entry.update({'data_role': SIMULATION_ROLE} if path.startswith(B3) else {'note': EXPERIMENT_NOTE})
        manifest.append(entry)
    MANIFEST.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8', newline='\n')
    print(f'Fetched and verified {len(contents)} files')


if __name__ == '__main__':
    main()
