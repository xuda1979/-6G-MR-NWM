#!/usr/bin/env python3
"""Export the two patent sources with the future-code native Word templates."""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import zipfile

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_TAB_ALIGNMENT, WD_TAB_LEADER
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.opc.constants import RELATIONSHIP_TYPE as RT
from docx.shared import Inches, Pt
from docx.table import Table
from docx.text.paragraph import Paragraph
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
STEMS = ('INVENTION_DISCLOSURE_CN', 'PRIOR_ART_SEARCH_CN')
TITLE = '一种多模态多速率无线网络世界模型控制方法'
FORMAT_PARTS = ('word/styles.xml', 'word/theme/theme1.xml', 'word/numbering.xml',
                'word/settings.xml', 'word/fontTable.xml', 'word/webSettings.xml')
FORMAT_HASHES = {'disclosure.docx': 'eb92d21e7858dc1e94323e44224e607d48cb5b82',
                 'search.docx': '295ca826e2467a181b12b735b521ebbbd77c4a9c'}
CAPTIONS = ('图1 共享语言主干双视图与缓存快层的数据流',
            '图2 连续窗口中的终态与摘要反馈',
            '图3 固定物理输入时的偏好干预与候选评分')


def font_file(value=None):
    options = [Path(value)] if value else [
        Path('/usr/local/share/fonts/noto-cjk/NotoSansCJKsc-Regular.otf'),
        Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')]
    for p in options:
        if p.is_file():
            return p
    raise SystemExit('Pass --font-file with a Chinese font for the diagrams.')


def diagram(index, font):
    """Deterministic, black-and-white technical drawings; no generated imagery."""
    directory = HERE / 'figures'
    directory.mkdir(exist_ok=True)
    size = (1800, 630 if index == 1 else 430)
    canvas = Image.new('RGB', size, 'white')
    draw = ImageDraw.Draw(canvas)
    regular = ImageFont.truetype(str(font), 43)
    small = ImageFont.truetype(str(font), 35)

    def box(rect, words, gray=False):
        draw.rectangle(rect, fill='#f2f2f2' if gray else 'white', outline='black', width=4)
        x1, y1, x2, y2 = rect
        lines = words.split('\n')
        y = (y1 + y2) / 2 - len(lines) * 55 / 2
        for line in lines:
            width = draw.textbbox((0, 0), line, font=regular)[2]
            draw.text(((x1 + x2 - width) / 2, y), line, font=regular, fill='black')
            y += 55

    def arrow(points):
        draw.line(points, fill='black', width=4, joint='curve')
        x0, y0 = points[-2]
        x, y = points[-1]
        angle = math.atan2(y - y0, x - x0)
        head = [(x, y), (x - 18 * math.cos(angle - .45), y - 18 * math.sin(angle - .45)),
                (x - 18 * math.cos(angle + .45), y - 18 * math.sin(angle + .45))]
        draw.polygon(head, fill='black')

    if index == 1:
        box((25, 45, 395, 190), '连续观测\n状态与外生条件')
        box((25, 360, 395, 470), '运营偏好文字')
        draw.rectangle((470, 20, 890, 515), outline='black', width=4)
        draw.text((500, 35), '共享预训练语言主干', font=small, fill='black')
        box((495, 120, 865, 230), '独立物理视图', True)
        box((495, 350, 865, 460), '独立语义视图', True)
        box((975, 70, 1360, 215), '物理上下文\n缓存调制参数包')
        box((1435, 70, 1770, 215), '快层递推\n状态与记忆')
        box((975, 355, 1360, 465), '目标权重')
        box((1435, 355, 1770, 500), '预测指标\n候选配置评分')
        arrow([(395, 115), (450, 115), (450, 175), (495, 175)])
        arrow([(395, 150), (425, 150), (425, 300), (675, 300), (675, 350)])
        arrow([(395, 415), (495, 415)])
        arrow([(865, 175), (920, 175), (920, 140), (975, 140)])
        arrow([(1360, 140), (1435, 140)])
        arrow([(865, 405), (975, 405)])
        arrow([(1360, 410), (1435, 410)])
        arrow([(1770, 140), (1790, 140), (1790, 420), (1770, 420)])
        box((1450, 255, 1770, 330), '候选动作')
        arrow([(1610, 255), (1610, 215)])
        draw.text((60, 555), '窗口摘要与终态反馈至下一窗口物理输入；偏好不进入物理视图', font=small, fill='black')
    elif index == 2:
        for y, labels in [(30, ['窗口T输入\n状态与上窗摘要', '慢层更新\n生成参数包', '缓存包执行K步\n得到终态与摘要']),
                          (270, ['窗口T+1输入\n承接终态与摘要', '慢层重新更新\n生成新参数包', '继续K步\n状态自由推进'])]:
            for x, label in zip([35, 660, 1285], labels):
                box((x, y, x + 480, y + 135), label)
            arrow([(515, y + 67), (660, y + 67)])
            arrow([(1140, y + 67), (1285, y + 67)])
        arrow([(1525, 165), (1525, 235), (275, 235), (275, 270)])
        draw.text((590, 170), '窗口内不调用语言主干', font=small, fill='black')
    else:
        box((25, 30, 525, 155), '固定物理输入\n状态  动作  外生条件')
        box((660, 30, 1140, 155), '物理视图与参数包\n保持同一预测')
        box((1275, 30, 1775, 155), '同一候选配置\n预测指标保持一致')
        box((25, 275, 525, 400), '偏好文字替换\n吞吐优先或能耗优先')
        box((660, 275, 1140, 400), '语义视图\n目标权重可改变')
        box((1275, 275, 1775, 400), '候选配置重新评分\n推荐动作可以变化')
        for y in [92, 337]:
            arrow([(525, y), (660, y)])
            arrow([(1140, y), (1275, y)])
        arrow([(1525, 155), (1525, 275)])
        draw.text((35, 192), '不变性以动作固定且偏好未混入外生输入为前提', font=small, fill='black')
    target = directory / f'figure_{index}.png'
    canvas.save(target, dpi=(300, 300), optimize=True)
    return target


def add_link(p, url):
    url = url.rstrip('。，；、.,;')
    if '/patent/' in url:
        label = re.search(r'/patent/([^/]+)', url).group(1) + ' 文献页'
    elif '/abs/' in url:
        label = 'arXiv ' + url.rsplit('/', 1)[1]
    elif 'cnipa.gov.cn' in url:
        label = 'CNIPA 专利法' if '/2020/' in url else 'CNIPA 局令第84号'
    elif '/commit/' in url:
        label = '项目版本 ' + url.rsplit('/', 1)[1][:12]
    else:
        label = url.split('//', 1)[-1]
    link = OxmlElement('w:hyperlink')
    link.set(qn('r:id'), p.part.relate_to(url, RT.HYPERLINK, is_external=True))
    run = OxmlElement('w:r')
    pr = OxmlElement('w:rPr')
    style = OxmlElement('w:rStyle')
    style.set(qn('w:val'), 'Hyperlink')
    pr.append(style)
    run.append(pr)
    text = OxmlElement('w:t')
    text.text = label
    run.append(text)
    link.append(run)
    p._p.append(link)


def inline(p, value, properties=None):
    position = 0
    for m in re.finditer(r'https?://[^\s。；）)]+|\*\*[^*]+\*\*|`[^`]+`', value):
        if m.start() > position:
            p.add_run(value[position:m.start()])
        s = m.group()
        if s.startswith('http'):
            add_link(p, s)
        else:
            run = p.add_run(s[2:-2] if s.startswith('**') else s[1:-1])
            if s.startswith('**'):
                run.bold = True
        position = m.end()
    p.add_run(value[position:])
    if properties is not None:
        for run in p.runs:
            run._r.insert(0, deepcopy(properties))


def reset(element):
    for child in list(element):
        if child.tag != qn('w:pPr'):
            element.remove(child)


def math_run(text):
    run = OxmlElement('m:r')
    t = OxmlElement('m:t')
    t.text = text
    run.append(t)
    return run


def script(base, suffix, superscript=False):
    el = OxmlElement('m:sSup' if superscript else 'm:sSub')
    e = OxmlElement('m:e')
    e.append(math_run(base))
    el.append(e)
    e = OxmlElement('m:sup' if superscript else 'm:sub')
    e.append(math_run(suffix))
    el.append(e)
    return el


def equation(paragraph, index):
    """Native, editable OMML with actual subscript/superscript/fraction/radical nodes."""
    container = OxmlElement('m:oMathPara')
    expr = OxmlElement('m:oMath')
    container.append(expr)
    r = math_run
    sub = script
    if index == 0:
        nodes = [sub('u', 't'), r(' = γ ⊙ SiLU('), sub('W', 'x'), r('['), sub('s', 't'),
                 r('; '), sub('a', 't'), r('; c] + '), sub('b', 'x'), r(') + β')]
    elif index == 1:
        frac = OxmlElement('m:f')
        numerator = OxmlElement('m:num')
        for n in [r('Norm('), sub('u', 't'), r(') A '), script('B', 'T', True)]:
            numerator.append(n)
        denominator = OxmlElement('m:den')
        radical = OxmlElement('m:rad')
        props = OxmlElement('m:radPr')
        hide = OxmlElement('m:degHide')
        hide.set(qn('m:val'), '1')
        props.append(hide)
        radical.append(props)
        radical.append(OxmlElement('m:deg'))
        val = OxmlElement('m:e')
        val.append(r('r'))
        radical.append(val)
        denominator.append(radical)
        frac.append(numerator)
        frac.append(denominator)
        nodes = [sub('y', 't'), r(' = '), sub('G', 'θ'), r('('), sub('u', 't'), r(', '),
                 sub('m', 't'), r(') + '), frac]
    else:
        nodes = [sub('s', 't+1'), r(' = '), sub('s', 't'), r(' + '), sub('W', 'δ'),
                 sub('y', 't'), r(' + '), sub('b', 'δ')]
    for n in nodes:
        expr.append(n)
    paragraph._p.append(container)
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER


class Native:
    def __init__(self, path):
        self.path = path
        data = path.read_bytes()
        blob = hashlib.sha1(f'blob {len(data)}\0'.encode() + data).hexdigest()
        assert blob == FORMAT_HASHES[path.name], 'Native template bytes changed'
        self.doc = Document(path)
        self.prototypes = {}
        self.heading_sequence = []
        for p in self.doc.paragraphs:
            self.prototypes.setdefault(p.style.name, deepcopy(p._p))
            if re.fullmatch(r'Heading [123]', p.style.name):
                self.heading_sequence.append(p.style.name)
        self.tables = [deepcopy(t._tbl) for t in self.doc.tables]
        for e in list(self.doc._element.body):
            if e.tag != qn('w:sectPr'):
                self.doc._element.body.remove(e)

    def append(self, element):
        self.doc._element.body.insert(len(self.doc._element.body) - 1, element)

    def paragraph(self, style, value=''):
        element = deepcopy(self.prototypes[style])
        reset(element)
        self.append(element)
        p = Paragraph(element, self.doc._body)
        inline(p, value)
        return p

    def table(self, index, rows):
        element = deepcopy(self.tables[index])
        table = Table(element, self.doc._body)
        assert len(table.columns) == len(rows[0])
        widths = [int(col.get(qn('w:w'))) for col in element.tblGrid]
        # The source grid is authoritative. Explicit widths prevent the
        # empty-cell prototypes from collapsing header cells in LibreOffice.
        layout = element.tblPr.find(qn('w:tblLayout'))
        if layout is None:
            layout = OxmlElement('w:tblLayout')
            element.tblPr.append(layout)
        layout.set(qn('w:type'), 'fixed')
        total = element.tblPr.find(qn('w:tblW'))
        if total is None:
            total = OxmlElement('w:tblW')
            element.tblPr.append(total)
        total.set(qn('w:type'), 'dxa')
        total.set(qn('w:w'), str(sum(widths)))
        head = deepcopy(table.rows[0]._tr)
        body = deepcopy(table.rows[1]._tr)
        for row in list(element.findall(qn('w:tr'))):
            element.remove(row)
        for i, record in enumerate(rows):
            tr = deepcopy(head if i == 0 else body)
            tr.get_or_add_trPr().append(OxmlElement('w:cantSplit'))
            for cell, value, width in zip(tr.findall(qn('w:tc')), record, widths):
                cell_props = cell.find(qn('w:tcPr'))
                if cell_props is None:
                    cell_props = OxmlElement('w:tcPr')
                    cell.insert(0, cell_props)
                cell_width = cell_props.find(qn('w:tcW'))
                if cell_width is None:
                    cell_width = OxmlElement('w:tcW')
                    cell_props.append(cell_width)
                cell_width.set(qn('w:type'), 'dxa')
                cell_width.set(qn('w:w'), str(width))
                proto = deepcopy(cell.find(qn('w:p')))
                r = proto.find(qn('w:r'))
                props = deepcopy(r.find(qn('w:rPr'))) if r is not None else None
                for child in list(cell):
                    if child.tag != qn('w:tcPr'):
                        cell.remove(child)
                reset(proto)
                cell.append(proto)
                inline(Paragraph(proto, self.doc._body), value, props)
            element.append(tr)
        self.append(element)

    def title(self, search, headings, toc):
        p = self.paragraph('Title')
        p.add_run('现有技术检索报告' if search else '发明专利技术交底书').bold = True
        p.add_run().add_break()
        if search:
            p.add_run('Prior Art Search Report').add_break()
            p.add_run('多模态多速率无线网络世界模型控制')
        else:
            p.add_run(TITLE)
        self.paragraph('Author', '中国移动通信有限公司研究院')
        self.paragraph('Date', '2026-10-16')
        p = self.doc.add_paragraph('目录')
        p.paragraph_format.space_before = Pt(15)
        p.paragraph_format.space_after = Pt(6)
        p.runs[0].bold = True
        for h in headings:
            label = h.split(' / ')[0]
            p = self.doc.add_paragraph()
            p.paragraph_format.space_before = Pt(0)
            p.paragraph_format.space_after = Pt(2)
            p.paragraph_format.line_spacing = 1
            p.paragraph_format.tab_stops.add_tab_stop(Inches(5.5), WD_TAB_ALIGNMENT.RIGHT, WD_TAB_LEADER.DOTS)
            run = p.add_run(label + ('\t' + str(toc[h]) if h in toc else ''))
            run.font.size = Pt(11)
        self.doc.add_page_break()


def preserve_parts(target, template):
    with zipfile.ZipFile(target) as z:
        entries = {n: z.read(n) for n in z.namelist()}
    with zipfile.ZipFile(template) as z:
        for n in FORMAT_PARTS:
            entries[n] = z.read(n)
    with zipfile.ZipFile(target, 'w', zipfile.ZIP_DEFLATED) as z:
        for name, data in entries.items():
            z.writestr(name, data)


def export(stem, font, toc):
    search = stem == STEMS[1]
    native = Native(HERE / 'formats' / ('search.docx' if search else 'disclosure.docx'))
    lines = (HERE / (stem + '.md')).read_text().splitlines()
    roots = [line[2:] for line in lines if line.startswith('# ')]
    native.title(search, roots, toc.get(stem, {}))
    i = next(i for i, line in enumerate(lines) if line.startswith('# '))
    headings = []
    table_count = figure_count = equation_count = 0
    first = True
    while i < len(lines):
        line = lines[i].strip()
        i += 1
        if not line:
            continue
        if line.startswith('```mermaid'):
            while i < len(lines) and not lines[i].startswith('```'):
                i += 1
            i += 1
            figure_count += 1
            p = native.paragraph('Captioned Figure')
            pic = p.add_run().add_picture(str(diagram(figure_count, font)), width=Inches(5.5))
            pic._inline.docPr.set('descr', CAPTIONS[figure_count - 1])
            p.paragraph_format.keep_with_next = True
            caption = native.paragraph('Image Caption', CAPTIONS[figure_count - 1])
            caption.paragraph_format.keep_with_next = False
            continue
        if line.startswith('$$'):
            p = native.paragraph('First Paragraph')
            equation(p, equation_count)
            equation_count += 1
            continue
        if line.startswith('|'):
            block = [line]
            while i < len(lines) and lines[i].strip().startswith('|'):
                block.append(lines[i].strip())
                i += 1
            rows = [[v.strip() for v in row.strip('|').split('|')]
                    for row in block if not re.fullmatch(r'\|[\s:|\-]+\|', row)]
            assert all(len(row) == len(rows[0]) for row in rows)
            native.table(table_count, rows)
            table_count += 1
            first = False
            continue
        match = re.fullmatch(r'(#{1,3})\s+(.+)', line)
        if match:
            style = f'Heading {len(match.group(1))}'
            native.paragraph(style, match.group(2))
            headings.append(style)
            first = True
            continue
        native.paragraph('First Paragraph' if first else 'Body Text', line)
        first = False
    assert headings == native.heading_sequence, 'Heading levels/order differ from native template'
    assert table_count == len(native.tables)
    assert figure_count == (0 if search else 3)
    assert equation_count == (0 if search else 3)
    props = native.doc.core_properties
    props.author = props.last_modified_by = ''
    props.title = '现有技术检索报告' if search else '发明专利技术交底书'
    props.subject = TITLE
    props.comments = ''
    props.created = props.modified = datetime(2026, 10, 16, tzinfo=timezone.utc)
    section = native.doc.sections[0]
    p = section.footer.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    for val in ['第 ', 'PAGE', ' 页']:
        if val == 'PAGE':
            field = OxmlElement('w:fldSimple')
            field.set(qn('w:instr'), 'PAGE')
            p._p.append(field)
        else:
            p.add_run(val).font.size = Pt(10)
    target = HERE / (stem + '.docx')
    native.doc.save(target)
    preserve_parts(target, native.path)
    with zipfile.ZipFile(target) as z, zipfile.ZipFile(native.path) as t:
        assert all(z.read(part) == t.read(part) for part in FORMAT_PARTS)
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--font-file')
    parser.add_argument('--toc-map', type=Path)
    args = parser.parse_args()
    toc = json.loads(args.toc_map.read_text()) if args.toc_map else {}
    font = font_file(args.font_file)
    for stem in STEMS:
        print(export(stem, font, toc))


if __name__ == '__main__':
    main()
