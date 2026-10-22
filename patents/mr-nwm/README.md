# 无线接入网多时间尺度数据协同处理专利交底

交底书修订日期为2026-10-22，名称为“一种无线接入网多时间尺度数据协同处理与资源控制方法及装置”。本次按附件及补充反馈聚焦无线数据的不同采样周期、上下文有效性与多小区功率控制，业务目标解析仅保留为可选实现。

| 文件 | 内容 |
|---|---|
| [INVENTION_DISCLOSURE_CN.docx](INVENTION_DISCLOSURE_CN.docx) | 修订交底书，渲染为10页，含5幅附图、3个可编辑公式及10项候选权利要求 |
| [INVENTION_DISCLOSURE_CN.md](INVENTION_DISCLOSURE_CN.md) | 可审阅正文源文件，包含对应Mermaid图 |
| [REVISION_NOTES_20261022.md](REVISION_NOTES_20261022.md) | 反馈落实对照、现有实现与新增设计边界、标准资料核对范围 |
| [PRIOR_ART_SEARCH_CN.docx](PRIOR_ART_SEARCH_CN.docx) | 2026-10-16历史检索报告，针对原方案；不覆盖本次改写后的权利要求 |
| [PRIOR_ART_SEARCH_CN.md](PRIOR_ART_SEARCH_CN.md) | 历史检索报告正文源文件 |
| [SEARCH_RECORD.json](SEARCH_RECORD.json) | 原检索日期、查询、文献链接及边界，未重写 |
| [QA_RECORD.json](QA_RECORD.json) | 最新交底书结构、页数、公式、附图和散列校验；保留检索报告历史记录 |

本次代码核对基线为[`d403a7249d97`](https://github.com/xuda1979/-6G-MR-NWM/commit/d403a7249d97ab08b2a3c5257ee8a3b841638f85)。已有时间元数据、过期掩码和快慢预测接口可在代码中核查；分源并发缓冲、上下文依赖清单、分布式安装、实际执行反馈及回退状态机为本次补充的技术设计，尚未形成完整实现。本次没有新增训练或真实无线实验，也未作标准采纳、标准必要性或新颖性已确认的判断。

格式基于[future-code原生专利模板](https://github.com/xuda1979/future-code/tree/b9e5426b6f5cefa521c3ea51f7dba8842d743c7e/patents/future-code)。保留交底书11个一级章节及其顺序、申请人和发明人字段、原生Word公式；子标题和第三张表按本次无线方案调整。`formats/`原始模板保持不变，六个基础格式部件继续原样保留，正文使用显式段落、字体及表格布局，保证中文与附图可读。

发明人及个人身份信息沿用空白字段，由院内按实际创造性贡献确定；拟申请人及职务发明归属由院内核定。本目录为交底材料，未提交专利申请。新增设计的原始支持及首次公开时间需核实，历史检索报告不能作为新增组合的检索结论。

## 重新导出

需要Python 3、python-docx、Pillow以及中文字体。默认只导出交底书，防止把修订后的标题或流程带入历史检索报告。

```bash
python patents/mr-nwm/export_docx.py \
  --only disclosure \
  --font-file /path/to/NotoSansCJKsc-Regular.otf \
  --toc-map patents/mr-nwm/page_numbers.json
```

需要单独重建原检索报告时使用`--only search`；本次未重新生成该文件。五幅附图由`disclosure_figures.py`生成。原生公式导出器会校验对应源表达式，修改公式时须同步更新其OMML实现。

修改正文后，重新渲染Word、核对各页图片并更新`page_numbers.json`与`QA_RECORD.json`。本次使用Letter纸张、11 pt正文、Noto Sans CJK SC及LibreOffice渲染。不同字体可能改变分页，交底书上限为10页。
