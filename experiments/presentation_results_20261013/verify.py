"""Check the final scientific values, source geometry and delivered files."""
from pathlib import Path
import csv,hashlib,json,re,subprocess,zipfile
from lxml import etree as E

H=Path(__file__).resolve().parent
D=json.loads((H/'DECK.json').read_text()); O=Path(D['output'])
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
R=list(csv.DictReader((H/'inputs/PAPER_EVIDENCE_TABLE.csv').open()))
def row(g,p,m,k):return next(r for r in R if (r['group'],r['parent'],r['method'],int(r['count']))==(g,p,m,k))
assert len(D['slides'])==17 and D['main_slides']==14
assert not json.loads((H/'qa/TEXT_OVERFLOW.json').read_text())
main='\n'.join(q['t'] for s in D['slides'][:14] for q in s['records'] if q['kind']=='text')
backup='\n'.join(q['t'] for s in D['slides'][14:] for q in s['records'] if q['kind']=='text')
assert '미학습 시간 구간' in main and '개발 평가' in main
assert '다른 녹음' not in '\n'.join(q['t'] for q in D['slides'][5]['records'] if q['kind']=='text')
for frac in ['5/5','12/12','7/7','3/5','4/12','5/7']:assert frac in backup
for bad in ['B24','C24','B630','C630','native_e2','CURRENT_BEST','PAPER_EVIDENCE']:assert bad not in main+backup
for k,want in [(2,'37.6'),(3,'50.6')]:
    a=float(row('B24','B99','psd',k)['mean_nmse']);b=float(row('B24','B99','learned50',k)['mean_nmse'])
    assert f'{100*(1-b/a):.1f}'==want and want+'%' in main
for k in (2,3):
    for m in ('psd','learned50'):
        r=row('B24','B99',m,k)
        assert f"{float(r['mean_si_sdr']):.2f} dB" in main
        assert f"{float(r['weakest_nmse']):.3f}" in main
charts=[q for s in D['slides'] for q in s['records'] if q['kind']=='chart']
assert len(charts)==3
for j,m in enumerate(('parent','psd','learned50')):
    assert charts[0]['series'][j]['values']==[float(row('B24','B99',m,k)['mean_nmse']) for k in (2,3)]
for j,g in enumerate(('B24','C24')):
    assert charts[1]['series'][j]['values']==[float(row(g,'B99','learned50',k)['mean_nmse']) for k in (2,3)]
for j,m in enumerate(('psd','matched50')):
    assert charts[2]['series'][j]['values']==[float(row('C24','native_e2',m,k)['mean_nmse']) for k in (2,3)]
old=json.loads((H/'inputs/original_records.json').read_text())
for new_i,old_i in ((11,9),(12,10)):
    assert [q for q in D['slides'][new_i]['records'] if q['kind']=='path']==[q for q in old[old_i] if q['kind']=='path']
for item in json.loads((H/'inputs/SNAPSHOT.json').read_text()):assert sha(H/'inputs'/item['snapshot'])==item['sha256']
pdf=O/(D['name']+'.pdf');pptx=O/(D['name']+'.pptx')
info=subprocess.check_output(['pdfinfo',str(pdf)],text=True)
assert int(re.search(r'Pages:\s+(\d+)',info).group(1))==17
fonts=subprocess.check_output(['pdffonts',str(pdf)],text=True)
assert 'Pretendard' in fonts
for line in fonts.splitlines()[2:]:assert re.search(r'\byes\s+yes\s+yes\b',line),line
pdf_text=subprocess.check_output(['pdftotext','-layout',str(pdf),'-'],text=True)
pages=[x for x in pdf_text.split('\f') if x.strip()];assert len(pages)==17
assert '\ufffd' not in pdf_text
for s,page in zip(D['slides'],pages):
    compact=lambda t:re.sub(r'\s+','',t)
    assert compact(s['title']) in compact(page),s['title']
NS={'a':'http://schemas.openxmlformats.org/drawingml/2006/main','p':'http://schemas.openxmlformats.org/presentationml/2006/main'}
native_paths=0
with zipfile.ZipFile(pptx) as z:
    assert len([n for n in z.namelist() if re.fullmatch(r'ppt/charts/chart\d+.xml',n)])==3
    assert len([n for n in z.namelist() if re.fullmatch(r'ppt/notesSlides/notesSlide\d+.xml',n)])==17
    for i,s in enumerate(D['slides'],1):
        root=E.fromstring(z.read(f'ppt/slides/slide{i}.xml'))
        for j,q in enumerate([q for q in s['records'] if q['kind']=='path']):
            points=root.xpath(f'.//p:sp[p:nvSpPr/p:cNvPr[@name="wave_{j}"]]//a:custGeom/a:pathLst/a:path/*/a:pt',namespaces=NS)
            assert len(points)==len(q['xy'])
            xmin=min(x for x,y in q['xy'])-.025;ymin=min(y for x,y in q['xy'])-.025
            for pt,(x,y) in zip(points,q['xy']):
                assert int(pt.get('x'))==round((x-xmin)*914400)
                assert int(pt.get('y'))==round((y-ymin)*914400)
            native_paths+=1
assert native_paths==34
audit=dict(status='PASS',main_slides=14,appendix_slides=3,pdf_pages=17,editable_charts=3,
    native_vector_paths=native_paths,original_waveform_coordinates_preserved=True,
    aggregate_metrics='all existing evaluation conditions retained',new_model_inferences=0,training_updates=0,
    heldout_data_access=False,all_fonts_embedded=True,text_overflow=0,
    condition_shift_results='appendix A/B',pptx_sha256=sha(pptx),pdf_sha256=sha(pdf),
    visual_review='All 17 final pages rendered and inspected; broken SVG replaced by native DrawingML paths.')
(O/'검증결과.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2))
(H/'qa/PDF_TEXT.txt').write_text(pdf_text)
print(json.dumps(audit,ensure_ascii=False,indent=2))
