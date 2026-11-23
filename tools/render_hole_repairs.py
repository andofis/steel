"""Render before/after evidence from the saved PNG/YOLO pairs (requires Pillow and a CJK font)."""

import argparse
import html
import json
from pathlib import Path
import tempfile

from PIL import Image, ImageChops, ImageDraw, ImageFont


BG, INK, TEXT, RULE = '#f4f5f1', '#162c32', '#42575a', '#c4d1cb'
OLD, NEW, SEAM = '#ff9b36', '#48ee91', '#6ce0ff'
FONT_FILES = ('C:/Windows/Fonts/msyh.ttc', 'C:/Windows/Fonts/msyhbd.ttc')


def font(size, bold=False):
    return ImageFont.truetype(FONT_FILES[int(bold)], size)


def text(draw, pos, message, size=16, bold=False, color=TEXT):
    draw.text(pos, message, font=font(size,bold), fill=color, anchor='lt')


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def yolo_rows(path, w, h):
    result = {}
    for n,line in enumerate(path.read_text('utf-8-sig').splitlines(),1):
        if not line.strip(): continue
        cls,x,y,bw,bh = map(float,line.split())
        if cls != 0: raise ValueError('不是 hole 类')
        result[n] = (max(0,(x-bw/2)*w),max(0,(y-bh/2)*h),min(w,(x+bw/2)*w),min(h,(y+bh/2)*h))
    return result


def crop_ring(im, x, y, width, height):
    period, ih = im.size
    if not (width <= period and 0 <= y <= ih-height): raise ValueError('取景超出有效范围')
    start=x%period; first=min(width,period-start)
    crop=Image.new('L',(width,height));crop.paste(im.crop((start,y,start+first,y+height)),(0,0))
    if first<width: crop.paste(im.crop((0,y,width-first,y+height)),(first,0))
    return crop


def overlay(chip, box, x, y, period, color, seam=False):
    rgb=chip.convert('RGB');d=ImageDraw.Draw(rgb);cw,ch=chip.size
    if box is not None:
        l,t,r,b=box
        for shift in (-2*period,-period,0,period,2*period):
            left,right=l+shift-x,r+shift-x
            if left<cw and right>0 and t-y<ch and b-y>0:
                d.rectangle((round(left),round(t-y),round(right)-1,round(b-y)-1),outline=color,width=3)
    if seam:
        pos=(-x)%period
        if 0<=pos<cw:
            for a,b in ((0,12),(ch-13,ch-1)): d.line((pos,a,pos,b),fill=SEAM,width=2)
    return rgb


def panels(source, dataset, record, annotation, cw, ch):
    alias=record['alias']; w,h=record['size'];cut=record['cut_x'];l,t,r,b=annotation['ring_xyxy']
    cross=annotation['operation']=='complete_seam'
    x=-(cw//2) if cross else round((l+r-cw)/2); y=max(0,min(h-ch,round((t+b-ch)/2)))
    with Image.open(source/(alias+'.png')) as original, Image.open(dataset/'labelimg'/(alias+'.png')) as saved:
        before=crop_ring(original,x,y,cw,ch);after=crop_ring(saved,x-cut,y,cw,ch)
    if ImageChops.difference(before,after).getbbox(): raise ValueError('前后对照没有使用同一组像素')
    old=yolo_rows(source/'标注'/(alias+'.txt'),w,h).get(annotation['source_line'])
    new=yolo_rows(dataset/'labelimg'/(alias+'.txt'),w,h)[annotation['target_line']]
    # Move ONLY the saved box into the camera's original longitude system for this comparison.
    new=(new[0]+cut,new[1],new[2]+cut,new[3])
    return overlay(before,old,x,y,w,OLD,cross),overlay(after,new,x,y,w,NEW,cross)


def make_summary(source, dataset, dest, index):
    cw,ch=320,200;xs=(24,357,690); canvas=Image.new('RGB',(1034,731),BG);d=ImageDraw.Draw(canvas)
    text(d,(24,21),'钢管孔洞标注修复对照',32,True,INK)
    text(d,(25,65),'左列是采集相机原片；中、右两列是完全相同的展开图局部，只有标注框不同。',17)
    d.line((24,99,1010,99),fill=RULE,width=1)
    for x,label,dot in zip(xs,('相机原片','修复前 · 原 YOLO 框','修复后 · 新 YOLO 框'),('#819294',OLD,'#159359')):
        d.rectangle((x,116,x+5,134),fill=dot);text(d,(x+14,115),label,18,True,INK)
    cases=[('B24',6,149,'B24　补上未标注的圆孔','cam_90 原片，同一采集批次','原 TXT 在此处没有框','新增第 6 框，覆盖可见孔口'),
           ('AOX3',5,431,'AOX3　把接缝两端的同一个孔框完整','cam_0 原片，可直接看到整个孔口','第 5 框停在原展开图的边缘','补成一个完整框，不把它计成两个孔')]
    for a,n,y,title,left,mid,right in cases:
        record,ann=index[a,n]; text(d,(24,y),title,20,True,INK)
        old,new=panels(source,dataset,record,ann,cw,ch)
        with Image.open(dataset/'audit/camera_views'/(a+'.png')) as raw:
            rw,rh=raw.size; cam=raw.crop(((rw-cw)//2,(rh-ch)//2,(rw-cw)//2+cw,(rh-ch)//2+ch)).convert('RGB')
        for x,image,cap in zip(xs,(cam,old,new),(left,mid,right)):
            canvas.paste(image,(x,y+31));text(d,(x,y+ch+41),cap,15)
    d.line((24,701,1010,701),fill=RULE,width=1)
    text(d,(24,710),'青色短线标出原来的圆周接缝；对照图未使用模型预测。',13)
    canvas.save(dest/'修复前后对照.png')


def full_view(dataset, dest, alias):
    path=dataset/'labelimg'/(alias+'.png')
    with Image.open(path) as im: canvas=im.convert('RGB')
    w,h=canvas.size; d=ImageDraw.Draw(canvas)
    for n,(l,t,r,b) in yolo_rows(path.with_suffix('.txt'),w,h).items():
        d.rectangle((round(l),round(t),round(r)-1,round(b)-1),outline=NEW,width=4)
        y=max(0,round(t)-30);x=min(w-104,max(0,round(l)))
        d.rectangle((x,y,x+99,y+26),fill=INK);text(d,(x+4,y+4),f'hole  {n:02}',17,True,NEW)
    canvas.save(dest/(alias+'_修复后全图.jpg'),quality=93,subsampling=0)


def refit_overview(previous, dataset, alias, line, target):
    """Show a single revision in two full panoramas and two identical pixel windows."""
    paths = [root/'labelimg'/(alias+'.png') for root in (previous, dataset)]
    with Image.open(paths[0]) as old, Image.open(paths[1]) as new:
        if old.size != new.size or old.mode != new.mode or ImageChops.difference(old, new).getbbox():
            raise ValueError('单框复核要求前后使用相同像素、相同圆周起点')
        w,h = new.size
        rows = [yolo_rows(p.with_suffix('.txt'), w, h) for p in paths]
        if rows[0].keys() != rows[1].keys() or {n for n in rows[0] if rows[0][n]!=rows[1][n]} != {line}:
            raise ValueError('对照文件不只是指定的一个框发生变化')
        boxes = [r[line] for r in rows]
        # One shared crop large enough for both rectangles plus surrounding pipe wall.
        l,t = [min(b[i] for b in boxes) for i in (0,1)]
        r,b = [max(b[i] for b in boxes) for i in (2,3)]
        cw,ch = max(224,round(r-l)+54),max(244,round(b-t)+48)
        cx,cy = max(0,min(w-cw,round((l+r-cw)/2))),max(0,min(h-ch,round((t+b-ch)/2)))
        chip = new.crop((cx,cy,cx+cw,cy+ch)).convert('RGB')
        th=1000;tw=round(w*th/h);thumb=new.resize((tw,th),Image.Resampling.LANCZOS).convert('RGB')
    xs=(54, 54+tw+48); right=xs[1]+tw+54; top=146; width=right+316; height=top+th+58
    canvas=Image.new('RGB',(width,height),BG);d=ImageDraw.Draw(canvas)
    text(d,(22,19),f'{alias}  /  第 {line} 个孔',32,True,INK)
    text(d,(24,67),'两列均为完整展开图。只有标注改变，图像像素与圆周起点相同。',17)
    d.line((22,99,width-22,99),fill=RULE)
    for i,(xx,labels,box,color,title) in enumerate(zip(xs,rows,boxes,(OLD,NEW),('上一版','修正后'),strict=True)):
        d.rectangle((xx,113,xx+4,133),fill=('#bc641a','#087548')[i]);text(d,(xx+12,111),title,19,True,INK)
        canvas.paste(thumb,(xx,top))
        for n,(left,high,end,low) in labels.items():
            x0=xx+round(left*tw/w);x1=xx+round(end*tw/w)-1
            y0=top+round(high*th/h);y1=top+round(low*th/h)-1
            d.rectangle((x0,y0,max(x0,x1),max(y0,y1)),outline=color,width=2 if n==line else 1)
            if n==line:
                d.line((x1+2,(y0+y1)//2,xx+tw+4,(y0+y1)//2),fill=color,width=2)
                tag_x=xx+tw+5; tag_y=(y0+y1)//2-9
                d.rounded_rectangle((tag_x,tag_y,tag_x+24,tag_y+18),radius=3,fill=INK)
                text(d,(tag_x+3,tag_y+2),f'#{n}',12,True,color)
        # Draw on native pixels before scaling: the green edge doesn't cover the opening.
        zoom=overlay(chip,box,cx,cy,w,color)
        zw=282; zh=round(chip.height*zw/chip.width)
        if zh>310: zh=310;zw=round(chip.width*zh/chip.height)
        zoom=zoom.resize((zw,zh),Image.Resampling.LANCZOS)
        yy=218+i*390
        text(d,(right,yy),title,19,True,INK)
        canvas.paste(zoom,(right,yy+34))
    for v in (0,2500,5000,7500,10000):
        yy=top+min(th-1,round(v*th/h));d.line((xs[0]-7,yy,xs[0]-2,yy),fill=RULE)
        d.text((xs[0]-9,yy),str(v),font=font(10),fill=TEXT,anchor='rm')
    text(d,(right,113),f'第 {line} 孔  /  局部放大',20,True,INK)
    text(d,(right,150),'两幅局部取自同一窗口。',16)
    delta=[round(((boxes[1][i]+boxes[1][i+2])-(boxes[0][i]+boxes[0][i+2]))/2) for i in (0,1)]
    text(d,(right,1000),'框中心位移（原图像素）',16)
    text(d,(right,1032),f'Δx {delta[0]:+d} px     Δy {delta[1]:+d} px',19,True,INK)
    d.line((22,top+th+15,width-22,top+th+15),fill=RULE)
    text(d,(24,top+th+25),f'{w} × {h} px   ·   其余 {len(rows[1])-1} 个孔框保持不变',15)
    canvas.save(target)


def render(source, dataset, previous=None):
    source,dataset=source.resolve(strict=True),dataset.resolve(strict=True);out=dataset/'visuals'
    if out.exists(): raise FileExistsError(f'不覆盖已有图册：{out}')
    ledger=read_json(dataset/'audit/repairs.json');index={(r['alias'],a['target_line']):(r,a) for r in ledger for a in r['labels']}
    changes=[(r,a) for r,a in index.values() if a['operation']!='preserved']
    changes.sort(key=lambda p:(p[1]['operation']!='add',p[0]['alias'],p[1]['target_line']))
    with tempfile.TemporaryDirectory(prefix='.proof-',dir=dataset) as scratch:
        stage=Path(scratch)/'visuals';details=stage/'details';details.mkdir(parents=True)
        make_summary(source,dataset,stage,index)
        thumb_w,thumb_h=232,146; sheet_name=f'全部{len(changes)}处修复后.png'
        sheet=Image.new('RGB',(996,111+186*((len(changes)+3)//4)),BG); d=ImageDraw.Draw(sheet)
        text(d,(24,20),f'全部 {len(changes)} 处修订后的标注',29,True,INK)
        text(d,(24,61),'绿色为修复后的框；涉及原图边缘的孔已环绕显示，每处对应一个孔。',16)
        cards=[]
        for i,(record,ann) in enumerate(changes):
            alias=record['alias'];n=ann['target_line'];extra=ann['operation']=='add'
            cw,ch=350,230;old,new=panels(source,dataset,record,ann,cw,ch)
            title=f'{alias} · '+(f'新增第 {n} 框' if extra else f'原第 {n} 框')
            detail=Image.new('RGB',(740,325),BG);dd=ImageDraw.Draw(detail);text(dd,(14,10),title,23,True,INK)
            text(dd,(14,49),'修复前',15,True);text(dd,(376,49),'修复后',15,True)
            detail.paste(old,(14,73));detail.paste(new,(376,73))
            name=f"{record['capture_id']}_{n:02}.png";detail.save(details/name)
            col,row=i%4,i//4; xx=24+col*239;yy=98+row*186
            text(d,(xx,yy),title,15,True,INK);sheet.paste(new.resize((thumb_w,thumb_h),Image.Resampling.LANCZOS),(xx,yy+24))
            cards.append(f'<article><h3>{html.escape(title)}</h3><a href="details/{name}"><img loading="lazy" src="details/{name}"></a></article>')
        sheet.save(stage/sheet_name)
        for alias in ('B24','AOX3'):full_view(dataset,stage,alias)
        refit_links=[]
        if previous is not None:
            for record,ann in changes:
                if ann['operation']!='refit': continue
                alias=record['alias']; line=ann['target_line']; name=f'{alias}_第{line}孔_整图前后.png'
                refit_overview(previous,dataset,alias,line,stage/name)
                refit_links.append(f'　<a href="{html.escape(name,quote=True)}">{html.escape(alias)} 第 {line} 孔：整图前后对照</a>')
        (stage/'index.html').write_text('''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>钢管孔洞标注修复图册</title><style>body{margin:24px auto;padding:0 20px;max-width:1150px;font:16px/1.6 system-ui,sans-serif;color:#193138;background:#f4f5f1}img{display:block;max-width:100%;height:auto}main{display:grid;grid-template-columns:repeat(auto-fit,minmax(315px,1fr));gap:15px}article{background:white;padding:10px}h3{margin:2px 0 8px}a{color:#076c50}</style><h1>钢管孔洞标注修复图册</h1><p>橙色为原 TXT，绿色为新版 TXT。每项前后都取自同一片原始像素，点图可放大。</p><img src="修复前后对照.png"><p>整根查看：<a href="B24_修复后全图.jpg">B24（6 框）</a>　<a href="AOX3_修复后全图.jpg">AOX3（5 框）</a>　<a href="'''+sheet_name+'''">全部修订总览</a>'''+''.join(refit_links)+'''</p><main>'''+''.join(cards)+'''</main><h2>未强行补框：AOX5</h2><p>下图为该位置的相机原片局部。亮圈内部可见管壁纹理；保留原标注，仍从负样本中隔离。</p><img src="../audit/camera_views/AOX5.png"></html>''',encoding='utf-8')
        if stage.resolve().parent!=Path(scratch).resolve() or stage.resolve().parent.parent!=dataset or out.exists():
            raise RuntimeError('图册发布路径已改变')
        stage.rename(out)
    return out


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('source',type=Path);parser.add_argument('dataset',type=Path)
    parser.add_argument('--previous',type=Path,help='相同圆周起点的上一版数据集，用于单框整管复核')
    args=parser.parse_args();print(render(args.source,args.dataset,args.previous))
