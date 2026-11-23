import unittest

from tube_unwrap.tile_detection import project_tile_box, merge_holes, output_hole


class TileMergeTest(unittest.TestCase):
    def project(self, tile, box):
        return project_tile_box(tile, box, 1152, 7500, 640)

    def test_recorded_partial_inside_full_even_with_low_iou_and_higher_edge_score(self):
        # 未分类14 #10: x/y are grid origins in the 1152x7500 detector plane.
        edge = self.project((0,5376), [108,593,212,640,.50])
        full = self.project((0,5760), [84,197,224,364,.56])
        self.assertEqual(merge_holes([edge,full],1152), [full])
        stronger = self.project((0,5376), [108,593,212,640,.95])
        self.assertEqual(merge_holes([stronger,full],1152), [full])
        self.assertEqual(merge_holes([edge],1152), [edge])

    def test_wrap_and_nearby_holes_and_physical_pipe_ends(self):
        full = self.project((864,384), [248,99,329,190,.8])  # x -40..41
        edge = self.project((0,384), [0,102,38,185,.9])
        neighbour = self.project((864,384), [330,99,406,190,.7])
        self.assertEqual(merge_holes([edge,neighbour,full],1152), [full,neighbour])
        self.assertFalse(self.project((0,0), [100,0,160,30,.7]).at_edge)
        self.assertFalse(self.project((0,6860), [100,615,160,640,.7]).at_edge)
        self.assertEqual(len(output_hole(full, (1152,7500), (1536,10000), 0)['segments_xyxy']), 2)
        obj = output_hole(full,(1152,7500),(1536,10000),500)
        self.assertEqual(len(obj['segments_xyxy']), 1)
        self.assertAlmostEqual(obj['center_xy'][0], (.5*1536/1152-500)%1536)


class DetectorCoordinatesTest(unittest.TestCase):
    def test_shifted_png_uses_identical_model_inputs_and_reversible_box_coordinates(self):
        import zlib
        import numpy as np
        from tools.detect_pipe_holes import review_image, wrapped_detail
        # Non-divisible dimensions: rolling after resize would change the model inputs.
        w,h,cut = 389,911,113
        image = np.random.default_rng(142).integers(0,256,(h,w),dtype=np.uint8)
        def run(pic,origin,recenter):
            signatures=[]
            def predict(batch):
                for chip in batch:
                    n = len(signatures)
                    signatures.append(zlib.crc32(chip[::17,::13].copy()))
                    yield {0:[[0,100,28,165,.91]],3:[[250,98,321,170,.62]]}.get(n,[])
            shown,report = review_image(pic,predict,origin,5,recenter)
            self.assertEqual(report['hole_count'],1)
            return shown,report,signatures
        _,old,a = run(image,0,False)
        shifted,now,b = run(np.roll(image,-cut,axis=1),cut,False)
        self.assertEqual(a,b); self.assertEqual(len(a),76)
        self.assertEqual(len(old['holes'][0]['segments_xyxy']),2)
        self.assertEqual(len(now['holes'][0]['segments_xyxy']),1)
        self.assertAlmostEqual((old['holes'][0]['center_xy'][0]-cut)%w,now['holes'][0]['center_xy'][0])
        patch = wrapped_detail(image,old['holes'][0]); self.assertGreater(patch.shape[1],old['holes'][0]['size_wh'][0])
        improved,report,c = run(image,0,True)
        self.assertEqual(c,a); self.assertEqual(len(report['holes'][0]['segments_xyxy']),1)
        np.testing.assert_array_equal(np.roll(improved,report['cut_x'],axis=1),image)

    def test_origin_is_only_read_for_matching_file_and_resolution(self):
        import json, tempfile
        from pathlib import Path
        from tools.detect_pipe_holes import read_origin
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); im=root/'unwrapped.png'
            self.assertEqual(read_origin(im,(100,400)),0)
            rec={'unwrapped':{'file':im.name,'width':100,'height':400,'panorama_seam':{'cut_x':37}}}
            (root/'capture.json').write_text(json.dumps(rec),'utf-8')
            self.assertEqual(read_origin(im,(100,400)),37)
            self.assertEqual(read_origin(root/'cam_0.png',(100,400)),0)
            with self.assertRaises(ValueError):read_origin(im,(91,400))
            rec['unwrapped']['panorama_seam']['cut_x']=100
            (root/'capture.json').write_text(json.dumps(rec),'utf-8')
            with self.assertRaises(ValueError):read_origin(im,(100,400))
