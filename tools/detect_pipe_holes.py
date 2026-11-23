"""Offline review of a whole panorama using the trained YOLO model (never modifies a capture)."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import tempfile

import cv2
import numpy as np

from tube_unwrap.panorama_seam import choose_seam
from tube_unwrap.tile_detection import WORKING_SIZE, detect_on_grid, output_hole


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WEIGHTS = ROOT / 'models/pipe_hole_yolo11n_v2_1/weights/best.pt'


def read_origin(path: Path, size: tuple[int,int]) -> int:
    """Only trust metadata naming this very PNG, not a sibling camera image or old preview."""
    manifest = path.parent / 'capture.json'
    if manifest.is_file():
        record = json.loads(manifest.read_text('utf-8-sig')).get('unwrapped') or {}
        if record.get('file') and (manifest.parent / record['file']).resolve() == path.resolve():
            if (record.get('width'),record.get('height')) != size:
                raise ValueError('采集清单和展开图尺寸不匹配')
            return _checked_origin((record.get('panorama_seam') or {}).get('cut_x',0), size[0])
    previous = path.parent / 'detections.json'
    if previous.is_file():
        record = json.loads(previous.read_text('utf-8-sig'))
        if record.get('panorama_file') == path.name:
            if tuple(record['size']) != size:
                raise ValueError('复核清单和展开图尺寸不匹配')
            return _checked_origin(record['cut_x'],size[0])
    return 0  # Legacy capture / standalone image: supplied origin is the reference.


def _checked_origin(value, width):
    if type(value) is not int or not 0 <= value < width:
        raise ValueError('Invalid cut_x in panorama metadata')
    return value


def make_predictor(weights: Path, conf: float, device=None):
    if not weights.is_file() or not 0 < conf <= 1:
        raise ValueError('需要本地孔洞检测权重，且置信度应在 (0,1]')
    # The acquisition runtime does not import or install ML dependencies.
    os.environ.setdefault('YOLO_OFFLINE','true')
    os.environ.setdefault('YOLO_AUTOINSTALL','false')
    from ultralytics import YOLO
    model = YOLO(str(weights),task='detect')
    if model.task != 'detect' or model.names != {0:'hole'}:
        raise ValueError('需要单类 hole 检测模型')

    def predict(images):
        for result in model.predict(images, imgsz=640, conf=conf, iou=.70, device=device,
                                    rect=False, verbose=False, save=False):
            yield result.boxes.data.detach().cpu().numpy()[:,:5]
    return predict


def review_image(image, predict, input_cut, batch, recenter):
    h,w = image.shape
    boxes = detect_on_grid(image,predict,cut_x=input_cut,batch=batch)
    # For old PNGs with no raw frames use detector extents as an optional presentation cut.
    scale = w / WORKING_SIZE[0]
    seam = choose_seam(w, ((b.left*scale-input_cut,b.right*scale-input_cut) for b in boxes),
                       max(2,round(w/64))) if recenter else None
    delta = seam.cut_x if seam else 0
    cut = (input_cut+delta) % w
    holes = [dict(id=n,**output_hole(b,WORKING_SIZE,(w,h),cut)) for n,b in enumerate(boxes,1)]
    return (np.roll(image,-delta,axis=1) if delta else image), {
        'size':[w,h], 'working_size':list(WORKING_SIZE), 'input_cut_x':input_cut, 'cut_x':cut,
        'additional_cut':asdict(seam) if seam else None, 'hole_count':len(holes), 'holes':holes,
        'coordinates':'output PNG pixels; xyxy right/bottom exclusive; two segments may share one hole id',
    }


def wrapped_detail(image, hole):
    """A full local view for the rare hole that cannot fit on the chosen straight cut."""
    h, w = image.shape
    cx, cy = hole['center_xy']
    bw, bh = hole['size_wh']
    pad = max(5, int(np.ceil(max(bw, bh)*.12)))
    cw, ch = min(w, int(np.ceil(bw))+2*pad), min(h, int(np.ceil(bh))+2*pad)
    x0, y0 = round(cx-cw/2), max(0, min(h-ch, round(cy-ch/2)))
    return np.take(image[y0:y0+ch], np.arange(x0,x0+cw) % w, axis=1)


def _save_preview(path, image, holes):
    overlay = cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)
    for hole in holes:
        for l,t,r,b in hole['segments_xyxy']:
            x,y = int(l),int(t)
            cv2.rectangle(overlay, (x,y), (int(np.ceil(r))-1,int(np.ceil(b))-1), (114,243,83), 3)
            cv2.putText(overlay, f"#{hole['id']}", (x,max(18,y-5)),cv2.FONT_HERSHEY_SIMPLEX,.7,(114,243,83),2)
    ok,buf = cv2.imencode('.jpg',overlay,[cv2.IMWRITE_JPEG_QUALITY,92]);
    if not ok: raise OSError('标注预览编码失败')
    path.write_bytes(buf.tobytes())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('panorama',type=Path); parser.add_argument('--output',required=True,type=Path,help='新建的复核目录')
    parser.add_argument('--weights',type=Path,default=DEFAULT_WEIGHTS); parser.add_argument('--device')
    parser.add_argument('--conf',type=float,default=.40); parser.add_argument('--batch',type=int,default=8)
    parser.add_argument('--recenter',action='store_true',help='同时导出按检测孔避开切口、无叠框的展开图')
    args=parser.parse_args(); out=args.output.resolve(); source=args.panorama.resolve(strict=True)
    if out.exists(): parser.error(f'不覆盖已有路径：{out}')
    image=cv2.imdecode(np.fromfile(source,np.uint8),cv2.IMREAD_UNCHANGED)
    if image is None or image.ndim!=2 or image.dtype!=np.uint8: parser.error('仅接受原始 Mono8 展开图')
    try:
        origin=read_origin(source,(image.shape[1],image.shape[0])); predict=make_predictor(args.weights,args.conf,args.device)
        shown,report=review_image(image,predict,origin,args.batch,args.recenter)
    except (ImportError,ValueError) as exc:
        parser.exit(2, f'孔洞复核失败（请使用已安装 Ultralytics / PyTorch 的检测环境）：{exc}\n')
    report.update(source=str(source),model=str(args.weights.resolve()),confidence=args.conf,
                  panorama_file='unwrapped.png' if args.recenter else None)
    out.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.hole-review-',dir=out.parent) as scratch:
        stage=Path(scratch)/'review'; stage.mkdir()
        _save_preview(stage/'detections.jpg',shown,report['holes'])
        for hole in report['holes']:
            if len(hole['segments_xyxy']) > 1:
                name = f"hole_{hole['id']:03d}_wrap.png"
                ok, buf = cv2.imencode('.png', wrapped_detail(shown, hole))
                if not ok: raise OSError('跨缝孔局部编码失败')
                (stage/name).write_bytes(buf.tobytes())
                hole['wrapped_detail'] = name
        if args.recenter:
            ok,buf=cv2.imencode('.png',shown)
            if not ok: raise OSError('展开图编码失败')
            (stage/'unwrapped.png').write_bytes(buf.tobytes())
        (stage/'detections.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n','utf-8')
        if out.exists(): raise FileExistsError(out)
        stage.rename(out)
    print(f"{report['hole_count']} 个孔，结果：{out}")


if __name__=='__main__': main()
