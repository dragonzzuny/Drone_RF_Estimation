// Native PowerPoint text, shapes, notes and charts, including waveform freeform paths.
const fs = require('fs'), path = require('path');
const fallback='/home/pyj/문서/GitHub/uav_analysis/rf_detection/deep_nmf_drone_20260929/paper_update_v94/korean_manuscript_complex_iq_figures_20261005/presentation_tools/node_modules/pptxgenjs';
let PptxGenJS; try { PptxGenJS=require('pptxgenjs'); } catch { PptxGenJS=require(process.env.DRONE_PPTXGEN_MODULE || fallback); }
const d=JSON.parse(fs.readFileSync(path.join(__dirname,'DECK.json'),'utf8'));
const p=new PptxGenJS(); p.defineLayout({name:'DRONE_WIDE',width:13.333333,height:7.5});p.layout='DRONE_WIDE';
p.title='드론 RF 혼합신호의 분리 및 복원';p.subject='복소 U-Net과 신호 연결 모델의 평가';p.author='Drone RF research';p.lang='ko-KR';
p.theme={headFontFace:'Pretendard',bodyFontFace:'Pretendard',lang:'ko-KR'};
for (const info of d.slides) {
  const s=p.addSlide();s.background={color:info.dark?'17313B':'FFFFFF'};
  let curveIndex=0;
  for (const q of info.records) {
    if(q.kind==='text'){
      s.addText(q.lines.join('\n'),{x:q.x,y:q.y,w:q.w,h:Math.max(q.h,q.size/72*1.25),fontFace:'Pretendard',fontSize:q.size,bold:q.bold,color:q.color,margin:0,
        align:({1:'left',2:'center',3:'right'})[q.align],valign:'top',breakLine:false,paraSpaceAfterPt:0,lineSpacingMultiple:1.0,lang:'ko-KR'});
    } else if(q.kind==='box') {
      s.addShape(q.round?p.ShapeType.roundRect:p.ShapeType.rect,{x:q.x,y:q.y,w:q.w,h:q.h,radius:.10,
        rectRadius:.10,fill:{color:q.fill},line:{color:q.line||q.fill,width:q.line?1:0}});
    } else if(q.kind==='circle') {
      s.addShape(p.ShapeType.ellipse,{x:q.x,y:q.y,w:q.d,h:q.d,fill:{color:q.fill},line:{color:q.fill,width:0}});
    } else if(q.kind==='line') {
      const down=q.y2>=q.y1,right=q.x2>=q.x1;
      s.addShape(p.ShapeType.line,{x:Math.min(q.x1,q.x2),y:Math.min(q.y1,q.y2),w:Math.abs(q.x2-q.x1),h:Math.abs(q.y2-q.y1),
        flipH:!right,flipV:!down,line:{color:q.color,width:q.width,...(q.arrow?{endArrowType:'triangle'}:{}),...(q.dash?{dashType:'dash'}:{})}});
    } else if(q.kind==='image') {
      s.addImage({path:path.join(__dirname,q.filename),x:q.x,y:q.y,w:q.w,h:q.h});
    } else if(q.kind==='path') {
      // Replace this placeholder with native DrawingML geometry after package creation.
      const xs=q.xy.map(z=>z[0]),ys=q.xy.map(z=>z[1]);
      const minx=Math.min(...xs)-.025,miny=Math.min(...ys)-.025;
      const w=Math.max(...xs)-minx+.025,h=Math.max(...ys)-miny+.025;
      s.addShape(p.ShapeType.rect,{x:minx,y:miny,w,h,objectName:`wave_${curveIndex++}`,
        fill:{color:'FFFFFF',transparency:100},line:{color:q.color,width:q.width,...(q.dash?{dashType:'dash'}:{})}});
    } else if(q.kind==='chart') {
      s.addChart(p.ChartType.bar,q.series.map(z=>({name:z.name,labels:q.labels,values:z.values})),{
        x:q.x,y:q.y,w:q.w,h:q.h,barDir:'col',grouping:'clustered',catAxisLabelFontFace:'Pretendard',catAxisLabelFontSize:18,
        catAxisLabelColor:'17313B',valAxisLabelFontFace:'Pretendard',valAxisLabelFontSize:12,valAxisLabelColor:'647986',
        chartColors:q.colors,showLegend:true,legendPos:'t',legendFontFace:'Pretendard',legendFontSize:13,
        showTitle:false,showValue:true,dataLabelPosition:'outEnd',dataLabelFormatCode:'0.000',dataLabelColor:'17313B',
        showBorder:false,showCatName:false,showSerName:false,showShadow:false,showMarker:false,
        valAxisMinVal:0,valAxisMaxVal:q.ymax,valAxisMajorUnit:q.unit,valGridLine:{color:'DAE3E7',size:.7},catGridLine:{style:'none'},
        catAxisLineColor:'DAE3E7',valAxisLineColor:'DAE3E7',gapSizePct:65,fontFace:'Pretendard',fontSize:14
      });
    } else throw Error('Unknown record '+q.kind);
  }
  s.addNotes(info.notes);
}
const out=path.join(d.output,d.name+'.pptx');
p.writeFile({fileName:out}).then(()=>console.log(out));
