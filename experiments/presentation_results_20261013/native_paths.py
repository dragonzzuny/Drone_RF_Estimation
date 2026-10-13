"""Convert waveform placeholders to native editable DrawingML polylines.

Uses the original coordinates, without fitting, interpolation, smoothing or reordering.
"""
from pathlib import Path
import json, zipfile
from lxml import etree as E

HERE=Path(__file__).resolve().parent
D=json.loads((HERE/'DECK.json').read_text())
P=Path(D['output'])/(D['name']+'.pptx')
NS={'p':'http://schemas.openxmlformats.org/presentationml/2006/main',
    'a':'http://schemas.openxmlformats.org/drawingml/2006/main'}
A='{'+NS['a']+'}'
with zipfile.ZipFile(P) as z: payload={n:z.read(n) for n in z.namelist()}
count=0
for page,s in enumerate(D['slides'],1):
    name=f'ppt/slides/slide{page}.xml'; root=E.fromstring(payload[name])
    curves=[q for q in s['records'] if q['kind']=='path']
    for i,q in enumerate(curves):
        matches=root.xpath(f'.//p:sp[p:nvSpPr/p:cNvPr[@name="wave_{i}"]]',namespaces=NS)
        assert len(matches)==1,(page,i)
        sp=matches[0].find('p:spPr',NS)
        old=sp.find('a:prstGeom',NS);pos=list(sp).index(old);sp.remove(old)
        geom=E.Element(A+'custGeom')
        for t in ('avLst','gdLst','ahLst','cxnLst'):E.SubElement(geom,A+t)
        E.SubElement(geom,A+'rect',l='l',t='t',r='r',b='b')
        paths=E.SubElement(geom,A+'pathLst')
        xmin=min(x for x,y in q['xy'])-.025; ymin=min(y for x,y in q['xy'])-.025
        w=max(x for x,y in q['xy'])-xmin+.025;h=max(y for x,y in q['xy'])-ymin+.025
        path=E.SubElement(paths,A+'path',w=str(round(w*914400)),h=str(round(h*914400)),fill='none',stroke='1',extrusionOk='0')
        for j,(x,y) in enumerate(q['xy']):
            op=E.SubElement(path,A+('moveTo' if j==0 else 'lnTo'))
            E.SubElement(op,A+'pt',x=str(round((x-xmin)*914400)),y=str(round((y-ymin)*914400)))
        sp.insert(pos,geom)
        ln=sp.find('a:ln',NS);ln.set('cap','rnd')
        count+=1
    payload[name]=E.tostring(root,encoding='UTF-8',xml_declaration=True,standalone=True)
temp=P.with_suffix('.tmp.pptx')
with zipfile.ZipFile(temp,'w',zipfile.ZIP_DEFLATED) as z:
    for n,b in payload.items():z.writestr(n,b)
temp.replace(P)
(HERE/'qa/NATIVE_WAVEFORMS.json').write_text(json.dumps(dict(paths=count,source_coordinates='unchanged',pptx_native_editable=True),indent=2))
print('Native waveform paths:',count)
