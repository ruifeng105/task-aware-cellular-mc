"""Fetch the author B3 posterior and simulation summaries used by b3_model.py.

Optional and network-dependent: the package already contains these files.
Every file comes from the pinned commit and must match its Git blob SHA.
Usage: python simulation/code/fetch_b3_sources.py
"""
from datetime import datetime, timezone
import json
from pathlib import Path

from fetch_expanded_sources import COMMIT, REPO, SOURCE, fetch, git_blob_sha

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / 'public_data/fgf2_b3_source_manifest.json'
B3 = 'Inference_results/Fgf_B3/results/sus_3_20/'
FILES = {B3 + 'posterior.txt': '1000 posterior parameter samples (34 named parameters, linear scale) that generated the sim_post exports',
         B3 + 'sim_post_model_summary.txt': 'Author simulation summary: parameter order, experiment schedules and reactions for sim_post',
         B3 + 'sim_model_summary.txt': 'Author simulation summary with a single named parameter vector'}


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
    manifest = [{'repository': REPO, 'commit_sha': COMMIT, 'scope': 'Author B3 posterior and simulation summaries for b3_model.py'}]
    for path, raw in contents.items():
        target = SOURCE / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
        manifest.append({'path': path, 'blob_sha': upstream[path], 'source_size_bytes': len(raw),
                         'source_url': f'https://github.com/{REPO}/blob/{COMMIT}/{path}',
                         'retrieved_utc_date': today, 'data_role': FILES[path]})
    MANIFEST.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8', newline='\n')
    print(f'Fetched and verified {len(contents)} files')


if __name__ == '__main__':
    main()
