"""Deterministic drawings for the 2026-10-22 multiscale RAN disclosure."""

import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

CAPTIONS = (
    "图1 系统逻辑功能及无线数据与控制流",
    "图2 不同时间尺度数据与上下文依赖的协同处理",
    "图3 上下文安装激活及执行反馈时序",
    "图4 上下文有效性与回退状态机",
    "图5 多小区功率控制与对齐残差反馈闭环",
)


def diagram(index, font, directory=None):
    directory = Path(directory or Path(__file__).resolve().parent / "figures")
    directory.mkdir(exist_ok=True)
    heights = [590, 700, 650, 630, 660]
    canvas = Image.new("RGB", (1800, heights[index - 1]), "white")
    draw = ImageDraw.Draw(canvas)
    regular = ImageFont.truetype(str(font), 43)
    small = ImageFont.truetype(str(font), 35)

    def label(x, y, text, size=small, anchor="mm"):
        bounds = draw.multiline_textbbox(
            (x, y), text, font=size, anchor=anchor, align="center", spacing=5
        )
        draw.rectangle((bounds[0] - 5, bounds[1] - 3, bounds[2] + 5, bounds[3] + 3), fill="white")
        draw.multiline_text(
            (x, y), text, font=size, fill="black", anchor=anchor, align="center", spacing=5
        )

    def box(rect, text):
        draw.rectangle(rect, fill="white", outline="black", width=3)
        x1, y1, x2, y2 = rect
        label((x1 + x2) / 2, (y1 + y2) / 2, text, regular)

    def arrow(points, text=None, at=None, dashed=False):
        if dashed:
            for (x0, y0), (x1, y1) in zip(points, points[1:]):
                length = math.hypot(x1 - x0, y1 - y0)
                for offset in range(0, int(length), 18):
                    a = offset / length
                    b = min(offset + 10, length) / length
                    draw.line(
                        (
                            x0 + (x1 - x0) * a,
                            y0 + (y1 - y0) * a,
                            x0 + (x1 - x0) * b,
                            y0 + (y1 - y0) * b,
                        ),
                        fill="black",
                        width=3,
                    )
        else:
            draw.line(points, fill="black", width=3, joint="curve")
        x0, y0 = points[-2]
        x, y = points[-1]
        angle = math.atan2(y - y0, x - x0)
        draw.polygon(
            [
                (x, y),
                (x - 17 * math.cos(angle - 0.45), y - 17 * math.sin(angle - 0.45)),
                (x - 17 * math.cos(angle + 0.45), y - 17 * math.sin(angle + 0.45)),
            ],
            fill="black",
        )
        if text:
            label(*at, text)

    if index == 1:
        box((45, 40, 465, 180), "无线测量与配置")
        box((660, 40, 1130, 180), "分源缓冲\n时间校验")
        box((1310, 40, 1760, 180), "慢速上下文\n生成实体")
        box((660, 345, 1130, 485), "快速预测\n控制实体")
        box((1310, 345, 1760, 485), "无线执行实体")
        arrow([(465, 110), (660, 110)])
        arrow([(1130, 110), (1310, 110)])
        arrow([(895, 180), (895, 345)], "最新有效状态", (890, 260))
        arrow([(1535, 180), (1535, 270), (1100, 270), (1100, 345)], "适配上下文", (1350, 235))
        arrow([(1130, 415), (1310, 415)], "控制", (1220, 380))
        arrow(
            [(1535, 485), (1535, 550), (555, 550), (555, 155), (660, 155)],
            "实际动作与实测反馈",
            (1000, 550),
        )
    elif index == 2:
        box((35, 25, 485, 155), "快数据\n最新有效片段")
        box((35, 220, 485, 350), "中速数据\n区间统计与时间")
        box((35, 420, 485, 550), "慢数据\n配置版本与生效期")
        box((665, 25, 1130, 155), "当前状态校正")
        box((665, 380, 1130, 530), "慢上下文\n及生成依赖清单")
        box((1310, 25, 1770, 155), "候选动作预测")
        box((1310, 380, 1770, 530), "每个快步\n检查复用条件")
        box((1170, 600, 1770, 690), "请求更新与本地回退")
        arrow([(485, 90), (665, 90)])
        arrow([(1130, 90), (1310, 90)])
        arrow([(485, 285), (580, 285), (580, 425), (665, 425)])
        arrow([(485, 485), (665, 485)])
        arrow([(1130, 455), (1310, 455)])
        arrow([(1540, 380), (1540, 155)], "通过", (1610, 260))
        arrow([(1540, 530), (1540, 600)], "失效", (1630, 566))
        label(875, 280, "新测量不刷新旧上下文的依赖时间")
    elif index == 3:
        xs = [250, 900, 1550]
        names = ["上下文生成实体", "快速预测控制实体", "无线执行实体"]
        for x, name in zip(xs, names):
            box((x - 220, 20, x + 220, 105), name)
            draw.line((x, 105, x, 640), fill="black", width=2)
        arrow([(250, 160), (900, 160)], "能力配置及带序号上下文", (575, 130))
        arrow([(900, 240), (250, 240)], "安装结果或拒绝原因", (575, 210), True)
        box((640, 275, 1160, 350), "激活时重新校验并切换")
        arrow([(900, 415), (1550, 415)], "控制标识与目标生效时刻", (1225, 385))
        arrow([(1550, 510), (900, 510)], "实际动作与实际生效时刻", (1225, 480), True)
        arrow([(900, 610), (250, 610)], "对齐残差及更新请求", (575, 580))
    elif index == 4:
        box((35, 35, 365, 155), "未配置")
        box((650, 35, 1065, 155), "待激活")
        box((1370, 35, 1765, 155), "有效")
        box((650, 450, 1065, 580), "回退")
        box((1370, 450, 1765, 580), "待更新")
        arrow([(365, 95), (650, 95)], "安装通过", (505, 58))
        arrow([(1065, 95), (1370, 95)], "激活校验通过", (1220, 58))
        arrow([(1670, 155), (1670, 450)], "提前刷新", (1670, 295))
        arrow([(1450, 450), (1450, 155)], "新上下文切换", (1325, 295))
        arrow([(1370, 515), (1065, 515)], "旧上下文失效", (1220, 555))
        arrow([(780, 155), (780, 450)], "校验失败或到期", (585, 280))
        arrow([(945, 450), (945, 155)], "新上下文安装", (1110, 370))
        arrow([(1565, 155), (1565, 205), (1200, 205), (1200, 415), (1030, 415), (1030, 450)])
        label(1170, 240, "硬失效")
    else:
        box((35, 30, 465, 160), "两小区有效测量")
        box((680, 30, 1110, 160), "时效与上下文检查")
        box((1335, 30, 1770, 160), "独立候选预测\n及功率硬约束过滤")
        box((1335, 305, 1770, 435), "按预配置参数评分\n下发并记录实际功率")
        box((680, 305, 1110, 435), "对齐实测与预测\n计算残差")
        box((35, 305, 465, 435), "停用与本地回退")
        box((35, 540, 565, 645), "请求并校验新上下文")
        arrow([(465, 95), (680, 95)])
        arrow([(1110, 95), (1335, 95)], "通过", (1220, 58))
        arrow([(1550, 160), (1550, 305)])
        arrow([(1335, 370), (1110, 370)], "反馈", (1220, 330))
        arrow([(895, 305), (895, 225), (250, 225), (250, 160)], "有效反馈", (545, 225))
        arrow([(680, 370), (465, 370)], "超限或超时", (575, 333))
        arrow([(710, 160), (710, 190), (500, 190), (500, 285), (400, 285), (400, 305)])
        label(565, 265, "失效")
        arrow([(250, 435), (250, 540)])
        arrow(
            [(565, 590), (1200, 590), (1200, 260), (1040, 260), (1040, 160)],
            "新上下文重检",
            (990, 590),
        )
    target = directory / f"figure_{index}.png"
    canvas.save(target, dpi=(300, 300), optimize=True)
    return target
