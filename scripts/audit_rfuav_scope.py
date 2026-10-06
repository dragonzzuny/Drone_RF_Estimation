"""Cross-check local RFUAV labels against author metadata; never read IQ payloads."""
import argparse
import collections
import datetime
import hashlib
import json
import urllib.request
from pathlib import Path

from bs4 import BeautifulSoup

PAPER = 'https://arxiv.org/html/2503.09033v2'
TREE = 'https://huggingface.co/api/datasets/kitofrank/RFUAV/tree/main?recursive=false&limit=1000'


def normalized(name):
    # The paper's Table 4 misspells the locally named AVATA2 as AVTA2.
    return name.upper().replace('DJI AVTA2', 'DJI AVATA2').strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inventory', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Preserve prior evidence; use a new output path')
    inventory_bytes = args.inventory.read_bytes()
    inventory = json.loads(inventory_bytes)
    bodies = {}
    for name, url in [('paper', PAPER), ('public_tree', TREE)]:
        with urllib.request.urlopen(url, timeout=30) as response:
            bodies[name] = response.read()
    soup = BeautifulSoup(bodies['paper'], 'html.parser')
    tables = [t for t in soup.select('table') if 'Herelink' in t.get_text()]
    if len(tables) != 1:
        raise ValueError('Ambiguous source table')
    paper_rows = []
    for tr in tables[0].select('tr'):
        cells = [c.get_text(' ', strip=True) for c in tr.select('td, th')]
        if cells and cells[0] != 'Type of UAV':
            if len(cells) != 9:
                raise ValueError(cells)
            paper_rows.append(dict(label=cells[0], video_bandwidth_mhz=None
                if cells[2] == '-' else float(cells[2])))
    paper = {normalized(r['label']): r for r in paper_rows}
    public = {normalized(Path(r['path']).stem) for r in json.loads(bodies['public_tree'])
              if r['path'].lower().endswith('.rar')}
    local = {normalized(r['category']) for r in inventory['categories']}
    if len(paper) != len(paper_rows) or len(local) != len(inventory['categories']):
        raise ValueError('Non-unique normalized category')
    rows = []
    for category in inventory['categories']:
        name = category['category']
        match = paper.get(normalized(name))
        video = match['video_bandwidth_mhz'] if match else None
        if category['aircraft_category']:
            review = 'existing_named_aircraft_cohort'
        elif video is not None:
            review = 'reconsider_video_link_capture_before_admission'
        else:
            review = 'excluded_pending_transmitter_evidence'
        packs = [p for p in inventory['packs'] if p['category'] == name]
        rows.append(dict(category=name, iq_files=category['iq_files'],
            prior_cohort_admitted=category['aircraft_category'],
            paper_label=match['label'] if match else None,
            paper_video_bandwidth_mhz=video,
            metadata_review=review,
            pack_ids=[p['pack_id'] for p in packs],
            local_sample_rates_hz=sorted({int(p['metadata']['SampleRate']) for p in packs}),
            local_centers_hz=sorted({float(p['metadata']['CenterFrequency']) for p in packs}),
            capture_direction_verified=False,
            local_payload_video_presence_verified=False,
            new_training_admission=False))
    result = dict(status='RFUAV_SCOPE_METADATA_CORRECTION',
        time_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        sources={name: dict(url=url, sha256=hashlib.sha256(bodies[name]).hexdigest())
                 for name, url in [('paper', PAPER), ('public_tree', TREE)]},
        inventory_sha256=hashlib.sha256(inventory_bytes).hexdigest(),
        local_categories=len(local), paper_categories=len(paper), public_rar_categories=len(public),
        paper_not_local=sorted(set(paper) - local), local_not_paper=sorted(local - set(paper)),
        public_not_local=sorted(public - local), local_not_public=sorted(local - public),
        paper_video_categories=sum(r['video_bandwidth_mhz'] is not None for r in paper_rows),
        local_categories_with_paper_video_evidence=sum(r['paper_video_bandwidth_mhz'] is not None for r in rows),
        metadata_review_counts=dict(collections.Counter(r['metadata_review'] for r in rows)),
        normalization_aliases={'DJI AVTA2': 'DJI AVATA2'},
        no_iq_payloads_read=True, frozen_training_and_splits_changed=False,
        interpretation='Six is an admission-list count, not proof that all other captures contain only controller emissions. Video bandwidth in the paper is not a source-direction annotation for local IQ.',
        categories=rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ('sources', 'categories')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
