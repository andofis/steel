"""Prepare/audit labeled panoramas offline. No model, optimizer, or camera SDK is used."""

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import html
import json
import math
from pathlib import Path
import shutil
import tempfile

import cv2
import numpy as np


PROFILE = Path(__file__).with_name("dataset_profiles") / "pipe_holes_v1.json"
SPLITS = ("train", "val")
EPS = 1e-8
YOLO_YAML = 'train: images/train\nval: images/val\nnames:\n  0: hole\n'


def check(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def load_json(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def load_gray(path):
    image = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_UNCHANGED)
    check(image is not None and image.ndim == 2 and image.dtype == np.uint8, f'不是 Mono8 图像：{path}')
    return image


def save_image(path, image):
    ok, buf = cv2.imencode(path.suffix, image)
    check(ok, f'编码失败：{path}')
    path.write_bytes(buf.tobytes())


@dataclass(frozen=True)
class Box:
    l: float
    t: float
    r: float
    b: float

    def xyxy(self):
        return [self.l, self.t, self.r, self.b]

    def coco(self):
        return [self.l, self.t, self.r - self.l, self.b - self.t]

    def area(self):
        return (self.r - self.l) * (self.b - self.t)

    def scaled(self, sx, sy):
        return Box(self.l * sx, self.t * sy, self.r * sx, self.b * sy)

    def yolo(self, w, h):
        return ((self.l + self.r) / (2*w), (self.t + self.b) / (2*h), (self.r - self.l) / w, (self.b - self.t) / h)


def read_labels(path, w, h):
    """YOLO center xywh -> continuous pixel edges; tolerate only six-decimal exporter rounding."""
    records, seen = [], set()
    for line_no, line in enumerate(path.read_text('utf-8-sig').splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split()
        where = f'{path.name}:{line_no}'
        check(len(parts) == 5 and parts[0] == '0', f'{where}: 只接受原始单类 YOLO 检测框')
        x, y, bw, bh = map(float, parts[1:])
        check(all(math.isfinite(v) for v in (x,y,bw,bh)) and 0 <= x <= 1 and 0 <= y <= 1 and
              0 < bw <= 1 and 0 < bh <= 1, f'{where}: 坐标或框尺寸无效')
        raw = (x-bw/2, y-bh/2, x+bw/2, y+bh/2)
        check(min(raw) >= -1.5e-6 and max(raw) <= 1+1.5e-6, f'{where}: 超过小数舍入容差，须复核')
        clamped = [min(1., max(0., v)) for v in raw]
        b = Box(*clamped).scaled(w,h)
        check(b.area() > 0 and b not in seen, f'{where}: 空框或重复框')
        seen.add(b)
        records.append((line_no, b, max(abs(v-u)*s for v,u,s in zip(raw,clamped,(w,h,w,h)))))
    return records


def seam_mask(box, w, h):
    """Mask both sides of a reviewed split opening, without inventing a second label."""
    l, r = (box.l-w-16, 108.) if box.l+box.r > w else (-108., box.r+16)
    return Box(l, max(0., box.t-32), r, min(float(h), box.b+32))


def pieces(box, x, y, side, period):
    """Intersect periodic horizontal / nonperiodic vertical source edges with one tile."""
    t, b = max(0., box.t-y), min(side, box.b-y)
    if b-t <= EPS:
        return []
    out = []
    for shift in (-period, 0, period, 2*period):
        l, r = max(0., box.l+shift-x), min(side, box.r+shift-x)
        if r-l > EPS:
            out.append(Box(l,t,r,b))
    return out


def tile_labels(x, y, w, cfg, annotations, masks):
    side = cfg['size']
    if any(pieces(b,x,y,side,w) for b in masks):
        return 'review', []
    rows = []
    for ann_id, box in annotations:
        hits = pieces(box,x,y,side,w)
        if len(hits) > 1:
            return 'two_fragments', []
        if not hits:
            continue
        b, = hits
        fraction = min(1., b.area()/box.area())
        if fraction + EPS < cfg['min_fraction'] or min(b.r-b.l, b.b-b.t)+EPS < cfg['min_visible_side_px']:
            return 'sliver', []  # Do not leave visible unlabelled fragments in positive OR negative tiles.
        rows.append({'annotation_id':ann_id, 'bbox':b.xyxy(), 'visible_fraction':fraction})
    return ('positive' if rows else 'negative'), rows


def background_subset(items, count, capture):
    """A deterministic sample spread over axial position; no random split of tiles."""
    ordered = sorted(items, key=lambda r: (r['y'],r['x']))
    n = min(count, len(ordered))
    def rank(r):
        return hashlib.sha256(f"{capture}:{r['x']}:{r['y']}:v1".encode()).digest()
    return [min(ordered[i*len(ordered)//n:(i+1)*len(ordered)//n], key=rank) for i in range(n)]


def grid(capture, split, w, h, cfg, annotations, masks):
    side = cfg['size']
    check(0 < side <= min(w,h) and 0 < cfg['step_x'] <= side and 0 < cfg['step_y'] <= side, '切片参数无效')
    ys = sorted({*range(0,h-side+1,cfg['step_y']), h-side})
    pos, neg, stats = [], [], Counter()
    for y in ys:
        for x in range(0,w,cfg['step_x']):
            status, rows = tile_labels(x,y,w,cfg,annotations,masks)
            stats[status] += 1
            if status in ('positive','negative'):
                (pos if rows else neg).append({'x':x,'y':y,'wrap':x+side>w,'labels':rows})
    keep_neg = background_subset(neg,len(pos),capture) if split == 'train' else neg
    stats['background_not_sampled'] = len(neg)-len(keep_neg)
    return sorted(pos+keep_neg, key=lambda r:(r['y'],r['x'])), stats


def crop_ring(image, x, y, width, height):
    check(0 <= y and y+height <= image.shape[0], '轴向禁止环绕或填充')
    if 0 <= x and x+width <= image.shape[1]:
        return image[y:y+height,x:x+width]
    return np.take(image[y:y+height], np.arange(x,x+width) % image.shape[1], axis=1)


def chip_window(shape, issue):
    h,w = shape
    region = Box(*issue['native_xyxy'])
    seam = issue['reason']=='seam_cut'
    cw = min(w, max(400, math.ceil(region.r-region.l)+80)); ch = min(h, max(320,math.ceil(region.b-region.t)+64))
    x = -(cw//2) if seam else round((region.l+region.r-cw)/2)
    y = max(0,min(h-ch,round((region.t+region.b-ch)/2)))
    return x,y,cw,ch


def write_review_chip(out, im, issue, box):
    """Only review PNGs carry overlays. The image is rolled at a seam, never vertically."""
    _,w = im.shape
    x,y,cw,ch = chip_window(im.shape,issue)
    crop = cv2.cvtColor(crop_ring(im,x,y,cw,ch),cv2.COLOR_GRAY2BGR)
    if issue['reason']=='seam_cut':
        for y0 in (0,ch-18): cv2.line(crop,(-x,y0),(-x,y0+17),(0,185,245),2)
    if box is not None:
        for b in (Box(box.l+d-x,box.t-y,box.r+d-x,box.b-y) for d in (-w,0,w)):
            if b.r>=0 and b.l<cw:
                cv2.rectangle(crop,(round(b.l),round(b.t)),(round(b.r),round(b.b)),(64,214,100),2)
    save_image(out, crop)


def coco(images, annotations):
    return {'info':{'description':'Steel pipe panoramas; supplied human detection boxes, not masks.'},'licenses':[],
            'images':images,'annotations':annotations,'categories':[{'id':1,'name':'hole','supercategory':'opening'}]}


def review_page(dest, issues):
    cards = []
    for r in issues:
        title = f"{r['source']} · " + (f"原标注第 {r['line']} 行" if r.get('line') else '无原标注框')
        cards.append(f'<article><img loading="lazy" src="{html.escape(r["preview"])}" alt="复核局部图">'
                     f'<h3>{html.escape(title)}</h3><p>{html.escape(r["note"])}</p>'
                     f'<small>原始坐标区域：{html.escape(str(r["native_xyxy"]))}</small><br>'
                     f'<a href="../master/{html.escape(r["master_image"])}">查看原尺寸展开图</a></article>')
    (dest/'index.html').write_text('''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>钢管孔洞标注复核</title>
<style>body{margin:32px auto;max-width:1240px;padding:0 20px;font:16px/1.55 system-ui,sans-serif;color:#1b333a;background:#f5f8f7}main{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:14px}article{padding:12px;background:white;border:1px solid #ccd9d5;border-radius:6px}img{width:100%;height:auto}h3{margin:7px 0}p{margin:5px 0 12px}small{color:#455c63}a{color:#066a5c}</style>
<h1>钢管孔洞标注复核</h1><p>绿框是用户原标注；黄色短线为展开图左右接缝。接缝图已把左右两侧接在一起以便查看。下面各区域未用于训练版或验证版切片，也没有自动添加目标。坐标以原图左上角为原点；接缝复核区的负横坐标表示绕到右端。要修改框，请使用<a href="to_label/">无叠框的可编辑局部图包</a>，不要在以下截图上标注。</p><main>'''
        + ''.join(cards)+'</main></html>', encoding='utf-8')


def render_dataset(source, profile_path, dest, *, labels_dir=None, capture_root=None):
    plan = load_json(profile_path)
    w,h = plan['working_size']; cfg = plan['tile']; side = cfg['size']
    images, annotations, sources, windows, issues = [], [], [], [], []
    classes = (source/'标注' if labels_dir is None else labels_dir)/'classes.txt'
    check(classes.read_text('utf-8-sig').split() == ['hole'], '类别表不是单类 hole')
    inputs = plan['sources']; aliases = {s['alias'] for s in inputs}; captures = {s['capture_id'] for s in inputs}
    check(len(aliases)==len(captures)==len(inputs) and len({s['png_sha256'] for s in inputs})==len(inputs), '重复采集、图像或文件名')
    check({p.stem for p in source.glob('*.png')}==aliases and
          {p.stem for p in classes.parent.glob('*.txt') if p!=classes}==aliases,'图片/标注与审查清单不一致，不可静默忽略新增文件')
    master, tiles, audit, review = (dest/n for n in ('master','yolo_tiles','audit','review'))
    for folder in [master/'annotations', audit, review/'chips', *(
            a/b/c for a in (master,tiles) for b in ('images','labels') for c in SPLITS)]:
        folder.mkdir(parents=True,exist_ok=True)
    (tiles/'data.yaml').write_text('# Clean tiles; intentionally no independent test split.\n'+YOLO_YAML,encoding='utf-8')
    (master/'data.yaml').write_text('# Panorama master; use the clean tiles for detection.\n'+YOLO_YAML,encoding='utf-8')
    shutil.copyfile(classes,master/'classes.txt');shutil.copyfile(profile_path,audit/'preparation_profile.json')
    for image_id, entry in enumerate(inputs,1):
        alias, cap, family = entry['alias'],entry['capture_id'],entry['family']; split = plan['families'][family]
        check(split in SPLITS and all(Path(s).name==s and s not in ('.','..') for s in (alias,cap,family)), '无效分组或路径')
        png=source/(alias+'.png'); txt=classes.parent/(alias+'.txt')
        check(sha(png)==entry['png_sha256'] and sha(txt)==entry['label_sha256'], f'{alias}: 文件已变化，须刷新复核清单')
        original=(source.parent if capture_root is None else capture_root)/family/cap/'unwrapped.png'; confirmed=False
        if original.exists():
            checked=load_json(original.with_name('capture.json'))
            check(sha(original)==entry.get('original_png_sha256',entry['png_sha256']) and checked['pipe_id']==cap,
                  f'{alias}: 对应采集包校验失败')
            confirmed=True
        im=load_gray(png); sh,sw=im.shape; sx,sy=w/sw,h/sh
        mimage=f'images/{split}/{cap}.png'; mlabel=f'labels/{split}/{cap}.txt'
        shutil.copyfile(png,master/mimage);shutil.copyfile(txt,master/mlabel)
        images.append({'id':image_id,'file_name':mimage,'width':sw,'height':sh})
        source_info={'image_id':image_id,'alias':alias,'capture_id':cap,'family':family,'split':split,
                     'physical_pipe_id':None,'origin_capture_match':confirmed,'original_png':png.name,
                     'original_label':txt.relative_to(source).as_posix(),'capture_relative':f'{family}/{cap}/unwrapped.png',
                     'png_sha256':entry['png_sha256'],'label_sha256':entry['label_sha256'],'annotations':[]}
        sources.append(source_info); clean, masks, near = [],[],set()

        def add_issue(reason, region, note, line=None, ann_id=None, box=None):
            check(reason in ('seam_cut','possible_missing','uncertain_ring'), '未知复核类型')
            check(region.area()>0 and 0<=region.t<region.b<=sh, '无效复核区域')
            key=f'{cap}_{reason}_{line or 0:02d}'
            item={'id':key,'reason':reason,'image_id':image_id,'source':alias+'.png','master_image':mimage,
                  'line':line,'annotation_id':ann_id,'native_xyxy':region.xyxy(),'note':note,'preview':f'chips/{key}.png'}
            write_review_chip(review/item['preview'],im,item,box);issues.append(item);masks.append(region.scaled(sx,sy))

        for line, box, rounding in read_labels(txt,sw,sh):
            edge=entry['edge_review'].get(str(line)); ann_id=len(annotations)+1
            if min(box.l,sw-box.r)<=3:
                near.add(str(line));check(edge in ('cut','whole'),f'{alias}:{line} 新贴边框未经复核')
            else:
                check(edge is None,f'{alias}:{line} 的接缝决策已失效')
            annotations.append({'id':ann_id,'image_id':image_id,'category_id':1,'bbox':box.coco(),'area':box.area(),'iscrowd':0})
            source_info['annotations'].append({'id':ann_id,'line':line,'edge_review':edge,'eligible':edge!='cut',
                                               'rounding_clip_px':round(rounding,9)})
            if edge=='cut':
                add_issue('seam_cut',seam_mask(box,sw,sh),'左右接合后可见同一个孔，原框只覆盖一侧。请在环绕视图确认完整框；不要标成两个孔。',line,ann_id,box)
            else:
                clean.append((ann_id,box.scaled(sx,sy)))
        check(near==set(entry['edge_review']),f'{alias}: 接缝行号不匹配')
        for r in entry.get('review_regions',[]):
            add_issue(r['reason'],Box(*r['xyxy']),r['note'])
        scaled=cv2.resize(im,(w,h),interpolation=cv2.INTER_AREA)
        keep, stat=grid(cap,split,w,h,cfg,clean,masks);source_info['sampling']=dict(stat)
        for tile in keep:
            x,y=tile['x'],tile['y']; stem=f'{cap}__y{y:05d}_x{x:04d}'
            image_path=f'images/{split}/{stem}.png'; label_path=f'labels/{split}/{stem}.txt'
            save_image(tiles/image_path,crop_ring(scaled,x,y,side,side))
            rows=['0 '+' '.join(f'{v:.9f}' for v in Box(*r['bbox']).yolo(side,side)) for r in tile['labels']]
            (tiles/label_path).write_text(''.join(s+'\n' for s in rows),encoding='utf-8')
            windows.append(dict(tile,image_id=image_id,split=split,image=image_path,label=label_path))
        print(f'{alias}: {split}, {len(clean)} eligible boxes, {len(keep)} tiles',flush=True)
    save_json(master/'annotations'/'instances_all.json',coco(images,annotations))
    for split in SPLITS:
        ids={s['image_id'] for s in sources if s['split']==split}
        save_json(master/'annotations'/f'instances_{split}.json',coco([i for i in images if i['id'] in ids],
                                                                      [a for a in annotations if a['image_id'] in ids]))
    save_json(audit/'sources.json',sources)
    (audit/'tiles.jsonl').write_text(''.join(json.dumps(r,separators=(',',':'))+'\n' for r in windows),encoding='utf-8')
    issues.sort(key=lambda r:(r['reason']!='possible_missing',r['reason']=='seam_cut',r['source'],r['line'] or 0))
    save_json(review/'issues.json',issues);review_page(review,issues)


def verify_dataset(dest):
    """Reopen emitted files and reproduce pixels and all intersecting boxes from the master."""
    master,tiles,audit,review=(dest/n for n in ('master','yolo_tiles','audit','review'))
    profile=load_json(audit/'preparation_profile.json');w,h=profile['working_size'];cfg=profile['tile'];side=cfg['size']
    sources=load_json(audit/'sources.json');co=load_json(master/'annotations'/'instances_all.json');issues=load_json(review/'issues.json')
    for path in (master,tiles):
        config='\n'.join(line for line in (path/'data.yaml').read_text('utf-8').splitlines() if not line.startswith('#'))+'\n'
        check(config==YOLO_YAML,'数据配置中的类别或相对路径失配')
    check((master/'classes.txt').read_text('utf-8-sig').split()==['hole'],'原始类名表失配')
    edit=review/'to_label'; edit_map={v['issue_id']:v for v in load_json(edit/'mapping.json')}
    check(set(edit_map)=={r['id'] for r in issues} and len(edit_map)==len(issues),'复核局部图清单不完整')
    windows=[json.loads(line) for line in (audit/'tiles.jsonl').read_text('utf-8').splitlines()];by_image=defaultdict(list)
    for tile in windows:by_image[tile['image_id']].append(tile)
    imgs={r['id']:r for r in co['images']}; anns={a['id']:a for a in co['annotations']}
    check(len(imgs)==len(sources)==len(co['images']) and len(anns)==len(co['annotations']),'母版 ID 重复或丢失')
    frozen={s['alias']:s for s in profile['sources']}
    check(set(imgs)=={s['image_id'] for s in sources} and set(frozen)=={s['alias'] for s in sources},'图像索引不完整')
    listed_ids=[m['id'] for s in sources for m in s['annotations']]
    check(len(listed_ids)==len(anns) and set(listed_ids)==set(anns),'源框索引不完整')
    for s in sources:
        check(all(s[k]==frozen[s['alias']][k] for k in ('capture_id','family','png_sha256','label_sha256')),
              '来源与冻结的审查清单不一致')
    check(co['categories']==[{'id':1,'name':'hole','supercategory':'opening'}], 'COCO 类别错误')
    file_set=set(); hashes={sp:set() for sp in SPLITS}; split_stats={}; family_splits=defaultdict(set);full_seen=set(); eligible_all=set();seen=set()
    for sp in SPLITS:
        subset=load_json(master/'annotations'/f'instances_{sp}.json'); ids={s['image_id'] for s in sources if s['split']==sp}
        check(subset==coco([i for i in co['images'] if i['id'] in ids],[a for a in co['annotations'] if a['image_id'] in ids]),f'COCO 划分错误：{sp}')
        split_stats[sp]=Counter()
    for s in sources:
        iid=s['image_id'];sp=s['split'];family_splits[s['family']].add(sp);counts=split_stats[sp];counts['panoramas']+=1
        item=imgs[iid]; png=master/item['file_name']; txt=master/'labels'/sp/(s['capture_id']+'.txt'); im=load_gray(png);sh,sw=im.shape
        check(sha(png)==s['png_sha256'] and sha(txt)==s['label_sha256'] and (sw,sh)==(item['width'],item['height']),f'母版校验失败：{png}')
        check(item['file_name']==f"images/{sp}/{s['capture_id']}.png" and profile['families'][s['family']]==sp,'源图分组错误')
        original=read_labels(txt,sw,sh); clean=[];sx,sy=w/sw,h/sh
        for (line,b,_),meta in zip(original,s['annotations'],strict=True):
            a=anns[meta['id']];check(a['image_id']==iid and a['category_id']==1 and a['bbox']==b.coco() and
                                  a['area']==b.area() and line==meta['line'],'COCO 与原 YOLO 框不一致')
            counts['source_boxes']+=1
            if meta['eligible']:clean.append((a['id'],b.scaled(sx,sy)));eligible_all.add(a['id']);counts['eligible_boxes']+=1
        image_issues=[r for r in issues if r['image_id']==iid]
        masks=[Box(*r['native_xyxy']).scaled(sx,sy) for r in image_issues]
        for r in image_issues:
            v=edit_map[r['id']]; x,y,cw,ch=chip_window(im.shape,r)
            check(v['master_image']==item['file_name'] and v['original_annotation_id']==r['annotation_id'], '复核原图/原框不匹配')
            check((x,y,cw,ch)==tuple(v['native_origin_xy']+v['chip_wh']) and v['source_wh']==[sw,sh] and
                  np.array_equal(load_gray(edit/v['file']),crop_ring(im,x,y,cw,ch)), '复核局部图与原始像素不一致')
            check((edit/v['file']).with_suffix('.txt').is_file(),'复核标签缺失')  # Operator edits are not training labels.
        work=cv2.resize(im,(w,h),interpolation=cv2.INTER_AREA)
        sampling=s['sampling']
        check(len(by_image[iid])==sampling.get('positive',0)+sampling.get('negative',0)-sampling['background_not_sampled'], '抽样数不匹配')
        check(sp!='val' or sampling['background_not_sampled']==0,'验证背景不得下采样')
        for tile in by_image[iid]:
            x,y=tile['x'],tile['y']; path=tiles/tile['image']; lab=tiles/tile['label']; tm=load_gray(path)
            check(tile['split']==sp and path.parent.name==lab.parent.name==sp and path.stem==lab.stem and
                  tile['image'] not in file_set and tile['wrap']==(x+side>w),'切片配对、坐标或划分错误')
            file_set.add(tile['image']);hashes[sp].add(hashlib.sha256(memoryview(tm)).digest())
            expected=work[y:y+side][:,(np.arange(side)+x)%w]
            check(tm.shape==(side,side) and np.array_equal(expected,tm),f'切片像素不匹配：{path}')
            check(not any(pieces(b,x,y,side,w) for b in masks),f'复核区进入了切片：{path}')
            # Independent projection: enumerate original copies overlapping this tile and compare to saved YOLO rows.
            visible={}
            for aid,b in clean:
                for d in (0,w):
                    clip=Box(max(0.,b.l+d-x),max(0.,b.t-y),min(side,b.r+d-x),min(side,b.b-y))
                    if clip.r-clip.l<=EPS or clip.b-clip.t<=EPS:continue
                    frac=clip.area()/b.area();check(frac+EPS>=cfg['min_fraction'] and min(clip.r-clip.l,clip.b-clip.t)+EPS>=cfg['min_visible_side_px'] and aid not in visible,'存在未标记的小碎片/重复目标')
                    visible[aid]=(clip,frac)
            actual=read_labels(lab,side,side);targets=tile['labels']
            check(len(actual)==len(targets)==len(visible) and set(visible)=={r['annotation_id'] for r in targets},'切片漏框或多框')
            for (_,local,_),t in zip(actual,targets):
                aid=t['annotation_id'];box,frac=visible[aid];check(np.allclose(local.xyxy(),box.xyxy(),rtol=0,atol=.00001) and np.allclose(t['bbox'],box.xyxy(),rtol=0,atol=.00001) and abs(t['visible_fraction']-frac)<EPS,'归一化或裁切坐标错误')
                seen.add(aid);counts['tile_boxes']+=1
                if frac>=1-EPS:full_seen.add(aid)
                else:counts['truncated_tile_boxes']+=1
            counts['tiles']+=1;counts['positive_tiles' if actual else 'negative_tiles']+=1
            if tile['wrap']:counts['wrapped_tiles']+=1
    check(all(len(sp)==1 for sp in family_splits.values()) and not (hashes['train']&hashes['val']), '跨集系列或像素重复')
    check(seen==eligible_all,f'干净标注没有全覆盖：{sorted(eligible_all-seen)}');check(len(file_set)==len(windows),'切片清单条目未核验')
    check(file_set=={p.relative_to(tiles).as_posix() for p in (tiles/'images').glob('*/*.png')} and
          {t['label'] for t in windows}=={p.relative_to(tiles).as_posix() for p in (tiles/'labels').glob('*/*.txt')},'切片文件清单不匹配')
    check(all((review/r['preview']).is_file() for r in issues),'复核截图缺失')
    return {'version':profile['version'],'classes':['hole'],'source_images':len(imgs),'source_boxes':len(anns),
            'eligible_unique_boxes':len(eligible_all),'represented_unique_boxes':len(seen),'unique_boxes_with_full_tile':len(full_seen),
            'seen_only_in_partial_tiles':sorted(seen-full_seen),'review_by_reason':dict(Counter(r['reason'] for r in issues)),
            'subpixel_rounding_clips':sum(m['rounding_clip_px']>1e-8 for s in sources for m in s['annotations']),
            'capture_hash_matches':sum(s['origin_capture_match'] for s in sources),'splits':{k:dict(v) for k,v in split_stats.items()},
            'same_pixels_across_splits':0,'verified_tile_images':len(file_set),'training_run':False}


def make_review_pack(dest):
    """Export clean (unpainted) review chips and editable YOLO seeds; never overwrite operator edits."""
    review=dest/'review'; output=review/'to_label'
    check(not output.exists(),'可编辑复核包已存在，不覆盖人工修改')
    co=load_json(dest/'master/annotations/instances_all.json'); anns={a['id']:Box(a['bbox'][0],a['bbox'][1],
        a['bbox'][0]+a['bbox'][2],a['bbox'][1]+a['bbox'][3]) for a in co['annotations']}
    mapping=[]
    with tempfile.TemporaryDirectory(prefix='.review-',dir=review) as scratch:
        stage=Path(scratch)/'to_label'; stage.mkdir()
        for issue in load_json(review/'issues.json'):
            im=load_gray(dest/'master'/issue['master_image']); h,w=im.shape
            x,y,cw,ch=chip_window(im.shape,issue); stem=issue['id']
            save_image(stage/(stem+'.png'),crop_ring(im,x,y,cw,ch))
            text=''; aid=issue['annotation_id']
            if aid is not None:
                old=anns[aid]; copies=[Box(old.l+d-x,old.t-y,old.r+d-x,old.b-y) for d in (-w,0,w)]
                fitted=[b for b in copies if b.l>=-EPS and b.r<=cw+EPS and b.t>=-EPS and b.b<=ch+EPS]
                check(len(fitted)==1,'原框不能完整映射至复核局部图')
                text='0 '+' '.join(f'{v:.9f}' for v in fitted[0].yolo(cw,ch))+'\n'
            (stage/(stem+'.txt')).write_text(text,encoding='utf-8')
            mapping.append({'file':stem+'.png','issue_id':stem,'source':issue['source'],'master_image':issue['master_image'],
                            'original_annotation_id':aid,'native_origin_xy':[x,y],'chip_wh':[cw,ch],'source_wh':[w,h]})
        (stage/'classes.txt').write_text('hole\n',encoding='utf-8');save_json(stage/'mapping.json',mapping)
        (stage/'README.md').write_text('''# 待人工确认的孔洞局部图\n\n这些 PNG 无叠框，可用矩形框标注工具打开，与同名 YOLO TXT 在同一目录。跨缝图初始框为原来标出的一侧，请修改为覆盖同一个完整孔的单框。其他疑点的初始 TXT 为空，表示待判，不是负样本。请另外记下已确认非孔的文件名，避免把未审和真阴性混淆。这里的文件不会自动进入训练或验证。\n\n`mapping.json` 保存原图和裁切原点。原图 x = (局部 x + 原点 x) % 原宽，原图 y = 局部 y + 原点 y。请保留文件名与映射，复核后另行发布新版本。\n''',encoding='utf-8')
        check(stage.resolve().parent.parent==review.resolve() and not output.exists(),'复核输出路径已改变')
        stage.rename(output)


def write_examples(dest):
    rows=[json.loads(s) for s in (dest/'audit/tiles.jsonl').read_text('utf-8').splitlines()]
    sources={s['image_id']:s for s in load_json(dest/'audit/sources.json')}
    sheet=np.full((2*270, 3*255, 3), 35, np.uint8)
    for i,split in enumerate(SPLITS):
        options=[r for r in rows if r['split']==split]
        samples=[next(r for r in options if r['labels'] and not r['wrap'] and all(v['visible_fraction']>.999 for v in r['labels'])),
                 next(r for r in options if r['wrap'] and r['labels']), next(r for r in options if not r['labels'])]
        for j,tile in enumerate(samples):
            im=cv2.cvtColor(load_gray(dest/'yolo_tiles'/tile['image']),cv2.COLOR_GRAY2BGR)
            for a in tile['labels']:
                l,t,r,b=map(round,a['bbox']); color=(70,230,90) if a['visible_fraction']>=1-EPS else (0,177,255)
                cv2.rectangle(im,(l,t),(r-1,b-1),color,3)
            thumb=cv2.resize(im,(240,240),interpolation=cv2.INTER_AREA);x=j*255+6;y=i*270+23
            sheet[y:y+240,x:x+240]=thumb
            label=sources[tile['image_id']]['alias'].replace('未分类','U')
            cv2.putText(sheet,f'{split} / {label} / '+('background' if not tile['labels'] else 'wrap' if tile['wrap'] else 'hole'),
                        (x,y-6),cv2.FONT_HERSHEY_SIMPLEX,.45,(230,236,235),1,cv2.LINE_AA)
    save_image(dest/'audit/tile_examples.jpg',sheet)


def write_readme(dest, report, profile):
    tr,va=(report['splits'][sp] for sp in SPLITS)
    (dest/'README.md').write_text(f'''# 钢管孔洞检测数据 v1

**后续微调入口：`yolo_tiles/data.yaml`**（请使用文件的绝对路径）。仅离线准备和验证数据，未下载或运行模型、未训练。训练类别只有 `hole`；没有 `damage`，也没有将几何算法的候选孔数当作人工标注。

| 项目 | 训练 | 验证 |
| --- | ---: | ---: |
| 原始采集次数 | {tr['panoramas']} | {va['panoramas']} |
| 原始人工框 | {tr['source_boxes']} | {va['source_boxes']} |
| 切片中覆盖的原始目标（去重） | {tr['eligible_boxes']} | {va['eligible_boxes']} |
| 640 × 640 切片 | {tr['tiles']} | {va['tiles']} |
| 有孔 / 空标签切片 | {tr['positive_tiles']} / {tr['negative_tiles']} | {va['positive_tiles']} / {va['negative_tiles']} |
| 切片上的框（含重叠视图） | {tr['tile_boxes']} | {va['tile_boxes']} |

## 原始母版

`master/images/` 和 `master/labels/` 按采集号重命名。PNG 和原始 YOLO TXT 均为独立副本，与输入文件字节一致。`master/classes.txt` 保留原类表。`master/annotations/instances_all.json` 是完整 COCO Detection 母版，另有 `instances_train.json`、`instances_val.json`；`file_name` 相对于 `master/`。COCO 用类别 1 和像素 `[左, 上, 宽, 高]`，`area` 为矩形面积；YOLO 用类别 0 和归一化 `[中心X, 中心Y, 宽, 高]`。没有生成分割掩膜。COCO 中 {report['subpixel_rounding_clips']} 处六位小数引起的亚像素越界被夹到边界，原 TXT 未改动。

## 模型输入切片

从人工实际标注的 1536 × 10000 PNG 等比缩到 1152 × 7500（当前均衡档），再切 640 × 640；不用整管缩到 640，也不二值化、锐化或画框到训练图上。这里对已标注 PNG 做面积重采样，而不是用新版本相机几何重新展开四路原图，避免旧框漂移；正式验收须补当前产线原生输出。

圆周横向起点 0/288/576/864，超过右端从左端接续；纵向步长 384，最后一片贴管尾，不纵向环绕。保留与窗口相交且可见面积至少 50%、可见短边至少 12 像素的框，并换算为局部坐标。若某框仅余更小碎片、同一孔在一张切片内出现两段，或碰到复核区，整张切片都不使用，不把残孔当背景。训练背景按管长分层、确定性选取，每张原图的背景切片不多于正切片；验证保留所有通过质量筛查的固定网格背景。`audit/tiles.jsonl` 给出原图 ID、圆周/轴向起点、是否环绕、原框 ID 及可见比例。

{report['eligible_unique_boxes']} 个可用原框全部出现在切片中，其中 {report['unique_boxes_with_full_tile']} 个至少有一个完整框；其余只能在当前干净网格中局部看到，ID 见 `audit/report.json`。重叠切片不是新的独立孔或钢管。

## 划分与待复核项

{profile['split_note']} 每个源图的所有切片留在同一集合。没有已证实的实物管编号，不能把这些采集视为同样数量的独立钢管，也不应把此验证集的成绩解释成全产线检出率。正式评估请补按实物编号独立留出的新管，包含真实跨缝、小孔、异常件。

打开 [待复核图册](review/index.html)。{report['review_by_reason']['seam_cut']} 个已标注跨缝孔只框了原展开图的一侧；母版保留，相关区域从默认切片中隔离，未擅自补全。另有 B24 原图约 (465, 2602) 的疑似漏标孔、AOX5 约 (370, 8680) 的未定性圆环，也隔离而非当作负样本。图册绿框为原标注，黄色短线为环绕接缝。修订前请确认完整孔边界或明确其不是孔。原标注部分框含较宽背景，没有自动收紧。

## 核验和追溯

- [样例切片](audit/tile_examples.jpg)：仅此质检预览画框，绿框为完整框、橙框为裁切框。
- `audit/sources.json`：原文件名、采集包、SHA-256、划分、原标注行号、是否用于切片以及采样记录。
- `review/issues.json`：每项待复核区域和对应的截图；负横坐标表示绕到图像右侧。`review/to_label/` 是无叠框的局部 PNG、可编辑 YOLO TXT 与回到原图的坐标映射，**不在默认训练配置中**。
- `audit/preparation_profile.json` 冻结本批的分组与复核决定；输入有变化时必须重新核对，不覆盖本版。`audit/report.json` 是发布前逐图解码、逐框回算的报告。
''',encoding='utf-8')


def build(source, output, profile):
    source=source.resolve(strict=True); output=output.resolve(); profile=profile.resolve(strict=True)
    check(not output.exists(),f'输出已存在，不覆盖：{output}')
    check(source!=output and source not in output.parents and output not in source.parents,'源目录和输出目录不能嵌套')
    output.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.pipe-hole-build-',dir=output.parent) as scratch:
        stage=Path(scratch)/output.name
        render_dataset(source,profile,stage)
        make_review_pack(stage);report=verify_dataset(stage);save_json(stage/'audit/report.json',report)
        write_examples(stage);write_readme(stage,report,load_json(profile))
        check(stage.resolve().parent==Path(scratch).resolve() and stage.resolve().parent.parent==output.parent
              and not output.exists(),'发布路径已改变')
        stage.rename(output)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__);commands=parser.add_subparsers(dest='command',required=True)
    prepare=commands.add_parser('build');prepare.add_argument('source',type=Path);prepare.add_argument('output',type=Path)
    prepare.add_argument('--profile',type=Path,default=PROFILE)
    verify=commands.add_parser('verify');verify.add_argument('dataset',type=Path)
    args=parser.parse_args()
    if args.command=='build':
        report=build(args.source,args.output,args.profile)
    else:
        report=verify_dataset(args.dataset);check(report==load_json(args.dataset/'audit/report.json'),'与发布时的核验报告不一致')
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
