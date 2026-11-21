"""Selectable output grid; camera pixels and the calibrated reference grid do not change."""

from dataclasses import dataclass


@dataclass(frozen=True)
class OutputSizes:
    width: int = 1536
    height: int = 10000

    def __post_init__(self):
        if any(type(v) is not int or not 2 <= v < 32767 for v in self.native):
            raise ValueError("配方中的展开参考尺寸无效")

    @property
    def native(self) -> tuple[int, int]:
        return self.width, self.height

    @property
    def minimum(self) -> tuple[int, int]:
        return min(128, self.width), min(128, self.height)

    def validate(self, width: int, height: int) -> tuple[int, int]:
        low_w, low_h = self.minimum
        if (type(width) is not int or type(height) is not int or
                not low_w <= width <= self.width or not low_h <= height <= self.height):
            raise ValueError(f"展开尺寸须为整数：宽 {low_w}–{self.width}，高 {low_h}–{self.height} 像素")
        return width, height

    def preset(self, quarters: int) -> tuple[int, int]:
        low_w, low_h = self.minimum
        return max(low_w, round(self.width * quarters / 4)), max(low_h, round(self.height * quarters / 4))

    def controls(self, selected: tuple[int, int]) -> dict:
        low_w, low_h = self.minimum
        presets = []
        seen = set()
        for quarters, label in ((4, "原尺寸"), (3, "均衡"), (2, "快速"), (1, "小图")):
            w, h = self.preset(quarters)
            if (w, h) not in seen:
                presets.append({"label": label, "width": w, "height": h})
                seen.add((w, h))
        return {"width": selected[0], "height": selected[1], "min_width": low_w, "min_height": low_h,
                "max_width": self.width, "max_height": self.height, "presets": presets}
