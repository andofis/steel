"""Create an independent reviewed dataset, preserving both the source annotations and v1."""

import argparse
from copy import deepcopy
import json
from pathlib import Path
import shutil
import sys
import tempfile
import xml.etree.ElementTree as ET

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import prepare_hole_dataset as data
from tools.prepare_hole_dataset import Box, check, load_gray, load_json, read_labels, save_image, save_json, sha


PROFILE = data.PROFILE.with_name('pipe_holes_v2_repairs.json')


def clearance(cut, box, period):
    if (cut - box.l) % period <= box.r - box.l:
        return 0.
    return min((box.l - cut) % period, (cut - box.r) % period)


def choose_cut(boxes, period):
    """Use the original cut when possible. Quarter turns permute, not move, the fixed tile grid."""
    check(all(b.l < b.r and 0 < b.r-b.l < period for b in boxes), '无效圆周框')
    if all(0 <= b.l and b.r <= period for b in boxes):
        return 0
    check(period % 4 == 0, '圆周宽度不能精确四等分')
    candidates = range(0, period, period // 4)
    cut = max(candidates, key=lambda x: min(clearance(x, b, period) for b in boxes))
    check(min(clearance(cut, b, period) for b in boxes) >= 4, '没有安全的四分之一周期切口，不能裁掉或拆分目标')
    return cut


def planar(box, cut, width):
    left = (box.l - cut) % width
    right = left + (box.r - box.l)
    check(right <= width + 1e-7, '选中的新接缝穿过了目标')
    return Box(left, box.t, min(float(width), right), box.b)


def save_labels(path, boxes, width, height):
    path.write_text(''.join('0 ' + ' '.join(f'{v:.9f}' for v in b.yolo(width, height)) + '\n'
                            for b in boxes), encoding='utf-8')


def index_patches(original_profile, corrections):
    patches = {(p['alias'], p['source_line']): p for p in corrections['patches']}
    cuts = {(s['alias'], int(k)) for s in original_profile['sources']
            for k,v in s['edge_review'].items() if v == 'cut'}
    additions = {(s['alias'], None) for s in original_profile['sources']
                 if any(r['reason'] == 'possible_missing' for r in s.get('review_regions', []))}
    aliases = {s['alias'] for s in original_profile['sources']}
    refits = {key for key, patch in patches.items() if patch['operation'] == 'refit'}
    check(refits.isdisjoint(cuts | additions) and set(patches) == cuts | additions | refits
          and len(patches) == len(corrections['patches']), '修订清单与复核对象不一致')
    for key, p in patches.items():
        b = Box(*p['ring_xyxy'])
        check(b.l < b.r and b.t < b.b, '修订框无效')
        if key in cuts:
            check(p['operation'] == 'complete_seam' and b.l < 0 < b.r, '接缝框未跨过原切口')
        elif key in refits:
            check(key[0] in aliases and type(key[1]) is int and key[1] > 0
                  and len(p.get('expected_source_xyxy', ())) == 4, '重画须指向具体原框')
        else:
            check(p['operation'] == 'add', '新增目标操作无效')
    return patches


def check_refit_rows(alias, old, patches):
    for (name, line), patch in patches.items():
        if name == alias and patch['operation'] == 'refit':
            check(line in old and np.allclose(old[line].xyxy(), patch['expected_source_xyxy'],
                                              atol=.01, rtol=0), f'{alias} #{line}: 待重画的原框不匹配')


def xml_inventory(source):
    result = []
    for path in sorted(source.rglob('*.xml')):
        root = ET.parse(path).getroot()
        result.append({'file':path.relative_to(source).as_posix(),'sha256':sha(path),
                       'image':root.findtext('filename'),'objects':len(root.findall('object'))})
    return result


def prepare_working_images(source, stage, base, repairs):
    edit = stage/'labelimg'; edit.mkdir(parents=True)
    shutil.copyfile(source/'标注/classes.txt', edit/'classes.txt')
    patches = index_patches(base, repairs)
    profile = deepcopy(base); profile['version'] = repairs['version']
    ledger = []
    for entry in profile['sources']:
        alias = entry['alias']; png = source/(alias+'.png'); txt = source/'标注'/(alias+'.txt')
        check((sha(png), sha(txt)) == (entry['png_sha256'], entry['label_sha256']), f'{alias}: 原图或标注已变化，请重新复核')
        im = load_gray(png); h,w = im.shape
        old = read_labels(txt,w,h)
        check_refit_rows(alias, {n:b for n,b,_ in old}, patches)
        items = [(line, b, patches.get((alias,line))) for line,b,_ in old]
        if (alias,None) in patches:
            items.append((None,None,patches[alias,None]))
        boxes = [Box(*patch['ring_xyxy']) if patch else b for _,b,patch in items]
        pending = [r for r in entry.get('review_regions',[]) if r['reason'] != 'possible_missing']
        cut = choose_cut(boxes + [Box(*r['xyxy']) for r in pending], w)
        check(all(0 <= b.t < b.b <= h for b in boxes), '轴向边界无效')
        if cut:
            save_image(edit/png.name, np.roll(im, -cut, axis=1))
        else:
            shutil.copyfile(png, edit/png.name)
        corrected = [planar(b,cut,w) for b in boxes]
        if cut or any(q is not None for _,_,q in items):
            save_labels(edit/txt.name, corrected,w,h)
        else:
            shutil.copyfile(txt, edit/txt.name)
        log = []
        for n, ((line, before, patch), ring, out) in enumerate(zip(items,boxes,corrected,strict=True),1):
            log.append({'target_line':n, 'source_line':line, 'operation':patch['operation'] if patch else 'preserved',
                        'before_xyxy':before.xyxy() if before else None,'ring_xyxy':ring.xyxy(),'output_xyxy':out.xyxy()})
        ledger.append({'alias':alias,'capture_id':entry['capture_id'],'split':base['families'][entry['family']],
                       'size':[w,h],'cut_x':cut,'original_png_sha256':entry['png_sha256'],
                       'original_txt_sha256':entry['label_sha256'],'labels':log})
        # Flattened sources are used by the unchanged COCO/YOLO tiler, with original acquisition provenance.
        entry['original_png_sha256'] = entry['png_sha256']; entry['original_txt_sha256'] = entry['label_sha256']
        entry.update(png_sha256=sha(edit/png.name), label_sha256=sha(edit/txt.name), cut_x=cut,
                     edge_review={str(i):'whole' for i,b in enumerate(corrected,1) if min(b.l,w-b.r)<=3},
                     review_regions=[dict(r,xyxy=planar(Box(*r['xyxy']),cut,w).xyxy()) for r in pending])
    return profile, ledger


def verify_repairs(source, dest):
    report = data.verify_dataset(dest)
    ledger = load_json(dest/'audit/repairs.json')
    repairs = load_json(dest/'audit/reviewed_corrections.json')
    base = load_json(dest/'audit/original_profile.json')
    patches = index_patches(base, repairs)
    windows = [json.loads(s) for s in (dest/'audit/tiles.jsonl').read_text('utf-8').splitlines()]
    full = {a['annotation_id'] for t in windows for a in t['labels'] if a['visible_fraction'] >= 1-data.EPS}
    sources = {s['alias']:s for s in load_json(dest/'audit/sources.json')}
    check(set(sources)=={s['alias'] for s in ledger}=={s['alias'] for s in base['sources']}, '源图映射不完整')
    edit=dest/'labelimg'
    check({p.name for p in edit.iterdir() if p.is_file()}=={'classes.txt'}|{
        alias+suffix for alias in sources for suffix in ('.png','.txt')}, 'LabelImg 文件夹里有额外或缺少的同名标注/图像')
    native_grid = {0,288,576,864}; full_repairs=0; max_error=0.
    counts = dict.fromkeys(['preserved', *(p['operation'] for p in patches.values())], 0)
    tiles_by_image = {}
    for t in windows: tiles_by_image.setdefault(t['image_id'],[]).append(t)
    for record in ledger:
        alias = record['alias']; cut = record['cut_x']; s = sources[alias];w,h = record['size']
        original = source/(alias+'.png'); old_txt = source/'标注'/(alias+'.txt')
        check(sha(original)==record['original_png_sha256'] and sha(old_txt)==record['original_txt_sha256'], '原文件签名失配')
        old = {n:b for n,b,_ in read_labels(old_txt,w,h)}
        check_refit_rows(alias, old, patches)
        img = dest/'labelimg'/(alias+'.png'); txt = dest/'labelimg'/(alias+'.txt')
        check(sha(img)==s['png_sha256'] and sha(txt)==s['label_sha256'], 'LabelImg 副本与冻结母版不同')
        original_im = load_gray(original); recut_im=load_gray(img)
        check(np.array_equal(np.roll(recut_im,cut,axis=1),original_im),'反向环移后原始像素不一致')
        # The native fixed grid is preserved, even though the start column in LabelImg changed.
        offset = cut*1152/w
        check(offset.is_integer() and int(offset)%288==0, '切片网格角度改变')
        check(all((t['x']+int(offset))%1152 in native_grid for t in tiles_by_image[s['image_id']]), '切片原图角度错误')
        if cut:
            resized_original=cv2.resize(original_im,(1152,7500),interpolation=cv2.INTER_AREA)
            resized_recut=cv2.resize(recut_im,(1152,7500),interpolation=cv2.INTER_AREA)
            check(np.array_equal(np.roll(resized_recut,int(offset),axis=1),resized_original), '切片用的均衡档像素发生漂移')
        observed = read_labels(txt,w,h); lines_seen=set()
        for target, (line,b,_), meta in zip(record['labels'],observed,s['annotations'],strict=True):
            n=target['source_line']; patch=patches.get((alias,n)); op=patch['operation'] if patch else 'preserved'
            ref=Box(*patch['ring_xyxy']) if patch else old[n]
            check(target['operation']==op and target['target_line']==line==meta['line'], '框或行号映射错误')
            check(target['before_xyxy']==(old[n].xyxy() if n is not None else None), '修前框与源 TXT 不符')
            check(target['ring_xyxy']==ref.xyxy(), '冻结框与审核坐标不符')
            err=max(abs((b.l+cut-ref.l+w/2)%w-w/2), abs((b.r-b.l)-(ref.r-ref.l)), abs(b.t-ref.t), abs(b.b-ref.b))
            check(err<.0011 and np.allclose(b.xyxy(),target['output_xyxy'],atol=.0011,rtol=0),f'{alias} #{line}: 反投影框失配')
            max_error=max(max_error,err);counts[op]+=1
            if n is not None:
                check(n not in lines_seen, '原框被重复标注');lines_seen.add(n)
            if patch:
                check(meta['id'] in full, f'{alias} #{line}: 修好的孔没有完整切片');full_repairs+=1
        check(lines_seen==set(old),'原孔丢失')
    check(full_repairs==len(patches) and counts['preserved']+full_repairs==report['source_boxes'],'修订数量失配')
    check(xml_inventory(source)==load_json(dest/'audit/source_xml_inventory.json'), '源 XML 文件已变化')
    check(sha(source/'标注/classes.txt')==sha(dest/'labelimg/classes.txt'), '类别顺序被改变')
    check(report['source_boxes']==report['eligible_unique_boxes']==140 and report['review_by_reason']=={'uncertain_ring':1},'仍有未预期的漏框、重复或复核区')
    return dict(counts, fully_visible_repaired_annotations=full_repairs,
                panoramas_recut=sum(r['cut_x']!=0 for r in ledger), inverse_pixel_matches=len(ledger),
                max_label_roundtrip_px=max_error, tile_verification=report)


def write_readme(stage, proof):
    core=proof['tile_verification']; tr=core['splits']['train']; va=core['splits']['val']
    refits = [f"{p['alias']} 第 {p['source_line']} 框" for p in load_json(stage/'audit/reviewed_corrections.json')['patches'] if p['operation']=='refit']
    note = ('另按原像素孔口重画：'+'、'.join(refits)+'。') if refits else ''
    (stage/'README.md').write_text(f'''# 钢管孔洞标注修订版 {core['version'].removeprefix('pipe_holes_')}

**在 LabelImg 中看图：** 切换到 YOLO 格式，打开 `labelimg/`。图、同名 TXT 和 `classes.txt` 在同一目录；如以前设过别的保存目录，也把标注/保存目录指回这里。此目录是可编辑副本，不含可能优先被读到的旧 XML。不要把这里的 TXT 单独拿去配旧 PNG：有 {proof['panoramas_recut']} 张图换了圆周起始列，图和框是成对移动的。LabelImg 中改动的内容不会静默改写冻结的训练切片。

**改动：** 以未修改的 20 张灰度原图和 139 行 YOLO 原标注为基准。保留 {proof['preserved']} 个原框的物理位置和大小，补全 {proof['complete_seam']} 个跨缝孔（一个孔一个框）；给 B24 原图约 (465, 2602) 的暗圆补 1 框。{note}B24 现在 6 框，全套 140 框。孔洞边界使用原始照片的暗开口和近邻边沿定位，并逐项对照原图核查；这是标注修订，不是模型预测。没有为所有图加偏移，也没有自动收紧未涉事的宽松框。

**AOX5：** 同图现有 YOLO 文件有 5 框，同目录检查到的 VOC XML 为空；空 XML 不用于清空这些框。原图约 (370, 8680) 的亮圈在相机原片上内部也有管壁纹理，不能据此认定是通孔。仍保留原有 5 框，该区域不作为训练或验证背景，见 `review/index.html`。

**图像坐标：** 仅对会截断完整孔的图做无插值水平循环换列，`x新 = (x原 − cut_x) mod 原宽`，`y新 = y原`，不旋转、不翻转、不缩放整图。每张图有独立的 `cut_x`；`audit/repairs.json` 记录原框行号、修订理由和前后坐标。原坐标中的负 x 表示绕到右端。切口选在不穿过任何目标的四分之一周期格点，因此固定检测网格的实际角度也不变。

**后续数据入口：** `yolo_tiles/data.yaml`。采集系列划分保持 v1 不变；训练 {tr['panoramas']} 次采集 / {tr['tiles']} 张切片，验证 {va['panoramas']} 次采集 / {va['tiles']} 张切片，不设独立测试集。`master/annotations/instances_all.json` 是与新版成对的 COCO 母版；`master/data.yaml` 是全展开图 YOLO。全部 140 条源标注在切片版都有对应；{proof['fully_visible_repaired_annotations']} 项修订均有至少一个完整框样本。

**检查：** 20/20 张新版全图反向换列后与原图逐像素相同。保留框和修订框逐个做了坐标回算，最大误差 {proof['max_label_roundtrip_px']:.6f} 像素（相对于已有 TXT 或修订几何，不是人工标注精度）。逐张核查切片像素、YOLO/COCO 坐标和跨集重复。报告在 `audit/repair_report.json`；B24、AOX3、AOX5 对应的无框原相机局部图在 `audit/camera_views/`。未训练模型。

**效果图：** `visuals/修复前后对照.png` 为同一区域比较；`visuals/index.html` 可逐项查看并打开完整展开图。效果图带叠框，和供训练/标注的不带框 PNG 分开保存。
''',encoding='utf-8')


def build(source, output, profile_path=PROFILE):
    source=source.resolve(strict=True); output=output.resolve(); profile_path=profile_path.resolve(strict=True)
    check(not output.exists(), f'目标已存在，不覆盖：{output}')
    check(source!=output and source not in output.parents and output not in source.parents, '原数据和输出目录不能嵌套')
    fixes=load_json(profile_path); base_path=profile_path.with_name(fixes['base_profile']); base=load_json(base_path)
    check({p.stem for p in source.glob('*.png')}=={s['alias'] for s in base['sources']}
          =={p.stem for p in (source/'标注').glob('*.txt') if p.name!='classes.txt'},'原数据与审查清单不同')
    xml=xml_inventory(source)
    check(not any(r['objects'] for r in xml), '有额外的非空 VOC 标注，须先与 YOLO 对照再执行本修订')
    output.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.pipe-label-repair-',dir=output.parent) as scratch:
        tmp=Path(scratch); stage=tmp/output.name; audit=stage/'audit'; audit.mkdir(parents=True)
        new_profile, ledger=prepare_working_images(source,stage,base,fixes)
        save_json(tmp/'recut_profile.json',new_profile);save_json(audit/'repairs.json',ledger)
        save_json(audit/'source_xml_inventory.json',xml);shutil.copyfile(profile_path,audit/'reviewed_corrections.json')
        shutil.copyfile(base_path,audit/'original_profile.json')
        views=audit/'camera_views';views.mkdir()
        for r in fixes['camera_checks']:
            cam=source.parent/r['capture_relative']; check(sha(cam)==r['source_sha256'],f'相机佐证图已变更：{cam.name}')
            x,y,cw,ch=r['camera_crop_xywh'];raw=load_gray(cam); chip=raw[y:y+ch,x:x+cw]
            check(chip.shape==(ch,cw),'相机原图局部越界');save_image(views/(r['alias']+'.png'),chip)
        data.render_dataset(stage/'labelimg',tmp/'recut_profile.json',stage,labels_dir=stage/'labelimg',capture_root=source.parent)
        data.make_review_pack(stage);proof=verify_repairs(source,stage);data.write_examples(stage)
        save_json(audit/'report.json',proof['tile_verification']);save_json(audit/'repair_report.json',proof)
        write_readme(stage,proof)
        check(stage.resolve().parent==tmp.resolve() and stage.resolve().parent.parent==output.parent and not output.exists(),'发布路径已改变')
        stage.rename(output)
    return proof


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('build','verify'));parser.add_argument('source',type=Path);parser.add_argument('dataset',type=Path)
    args=parser.parse_args()
    if args.action=='build':
        result=build(args.source,args.dataset)
    else:
        result=verify_repairs(args.source,args.dataset)
        check(result==load_json(args.dataset/'audit/repair_report.json'),'结果与发布时不一致')
    print(json.dumps(result,ensure_ascii=False,indent=2))
