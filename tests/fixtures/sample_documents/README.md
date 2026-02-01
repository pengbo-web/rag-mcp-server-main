# Sample Documents / 样例文档

这个目录用于存放测试用的样例文档。

## 目录说明

- `sample.txt` - 简单的文本文档
- `sample.md` - Markdown 格式文档
- 未来可以添加：
  - PDF 文档 (用于测试 PDF Loader)
  - 代码文件 (用于测试代码解析)
  - 图片文件 (用于测试多模态处理)

## 使用方式

在测试中通过 pytest fixture 访问：

```python
from pathlib import Path

def test_load_sample_document(project_root):
    sample_path = project_root / "tests/fixtures/sample_documents/sample.txt"
    assert sample_path.exists()
    with open(sample_path, "r", encoding="utf-8") as f:
        content = f.read()
    assert len(content) > 0
```
