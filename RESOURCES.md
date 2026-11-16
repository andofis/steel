# 钢管拼接学习资料

## 项目源码

- [采集服务](./hikrobot_camera.py#L885) · 读 `_finalize_capture_session_locked` 中的帧排序和字节连接，以及 `_save_pipe` 的结果保存路径。
- [采集适配层](./tube_unwrap_bridge.py#L111)和[运行配方](./tube_unwrap_recipe.json) · 查四路顺序、亮度修正、端部估计和运行参数。
- [展开模块](./tube_unwrap/__init__.py#L89)和[几何映射](./tube_unwrap/geometry.py#L170) · 查相机选择、圆柱投影、重采样和接缝。
- [孔洞候选](./tube_unwrap/holes.py#L150)和[局部融合](./tube_unwrap/hole_blending.py#L101) · 查同孔去重、轮廓保护和过渡。

## 官方文档

- [OpenCV 图像几何变换](https://docs.opencv.org/5.0/main_modules/imgproc_transform.html) · 阅读 `remap` 段，把输出像素反查原图坐标的公式和代码中的 `map_x`、`map_y` 对起来。
