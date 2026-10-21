# MR-NWM 专利交底与现有技术检索

本目录对应“一种多模态多速率无线网络世界模型控制方法”，编制日期为2026-10-16。

| 文件 | 内容 |
|---|---|
| [INVENTION_DISCLOSURE_CN.docx](INVENTION_DISCLOSURE_CN.docx) | 技术交底书，渲染为10页，含3幅附图、3个可编辑公式及8项候选权利要求 |
| [PRIOR_ART_SEARCH_CN.docx](PRIOR_ART_SEARCH_CN.docx) | 初步检索报告，渲染为15页，含8件相关专利和5篇原论文的比较 |
| 对应的 `.md` | 可审阅的正文源文件 |
| [SEARCH_RECORD.json](SEARCH_RECORD.json) | 检索日期、实际查询、文献链接及检索边界 |
| [QA_RECORD.json](QA_RECORD.json) | 模板、文档结构及页数校验记录 |

技术复核基线为[`c22ec3c979194`](https://github.com/xuda1979/-6G-MR-NWM/commit/c22ec3c97919463878470ad131b4473e6cd1b25b)。预训练语言路径的接入记录与轻量参考模型的合成实验分别说明；没有新增或宣称真实无线性能实验。

后续研究更新：新增 [UCC4G、UCC5G、CICV5G 公开实测数据训练](../../docs/PUBLIC_DATA.md)
及[独立结果记录](../../results/public_data/SUMMARY.md)，验证范围为观测预测。
本目录 2026-10-16 编制的交底书、检索报告及其复核记录保留当时的证据基线；
其中“尚未训练”等表述对应所引用版本。最新预测结果不构成实网控制收益或专利法律结论。

格式来自[future-code的专利目录](https://github.com/xuda1979/future-code/tree/b9e5426b6f5cefa521c3ea51f7dba8842d743c7e/patents/future-code)。保留全部一级章节顺序、各级标题结构和表格列结构；子标题按无线网络主题改写。`formats/`中的原生模板与所引用版本的Git blob一致，导出文档的样式、主题、编号、设置、字体表和网页设置六个格式部件与原模板逐字节相同。导出器明确表格网格宽度，并生成目录页码及页脚。

发明人及个人身份信息留空，由院内按本项目的实际创造性贡献确定；拟申请人及职务发明归属仍需院内核定。

本目录是技术交底与初步检索材料，不代表已经提交专利申请。项目代码和本文档均已公开，须核实有效申请日及首次公开时间；不应将普通仓库公开视为当然享有中国六个月新颖性宽限。报告没有作授权概率或自由实施结论。

重新导出需要Python 3、python-docx、Pillow及中文字体：

```bash
python export_docx.py --font-file /path/to/chinese-font.otf --toc-map page_numbers.json
```

修改正文后，应重新渲染并更新`page_numbers.json`，逐页检查，确认交底书仍不超过10页。QA使用Letter纸张、原模板字体层级和LibreOffice渲染；字体替换可能改变分页。
