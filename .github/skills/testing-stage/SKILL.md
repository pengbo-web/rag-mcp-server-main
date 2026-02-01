---
name: testing-stage
description: 在 implement 阶段完成后通过系统测试验证实现。根据任务性质确定测试类型（unit/integration/e2e），运行 pytest 并报告结果。dev-workflow 流水线的阶段 4。当用户说"运行测试"、"run tests"、"test"或实现后使用。
metadata:
  category: testing
  triggers: "run tests, test, validate, 运行测试"
allowed-tools: Read Bash(pytest:*) Bash(python:*)
---

# Testing Stage Skill（测试阶段技能）

你是 Modular RAG MCP Server 的**质量保证工程师**。在实现完成后，你必须在进入下一阶段之前通过系统测试验证工作。

> **前置条件**: 此技能在 `implement` 完成后运行。
> 规范文件位于: `.github/skills/spec-sync/specs/`

---

## 测试策略决策矩阵

**关键**: 测试类型应由**当前任务的性质**决定。从 `specs/06-schedule.md` 中读取任务的"测试方法"来决定。

| 任务特征 | 推荐测试类型 | 理由 |
|---------------------|----------------------|----------|
| 单个模块，无外部依赖 | **Unit Tests** | 快速、隔离、可重复 |
| 仅工厂/接口定义 | **Unit Tests**（使用 mocks/fakes） | 验证路由逻辑而不需要真实后端 |
| 模块需要真实 DB/文件系统 | **Integration Tests** | 需要验证与真实依赖项的交互 |
| 流水线/工作流程编排 | **Integration Tests** | 需要验证多模块协调 |
| CLI 脚本或终端用户入口点 | **E2E Tests** | 验证完整的用户工作流程 |
| 跨模块数据流（Ingestion→Retrieval） | **Integration/E2E** | 验证数据在模块间正确流动 |

---

## 测试目标

1. **验证实现完整性**: 确保规范中的所有需求都已实现。
2. **运行 Unit Tests**: 为实现的模块执行相关的 pytest unit tests。
3. **验证集成点**: 检查新代码是否与现有模块正确集成。
4. **报告问题**: 如果测试失败，提供可操作的反馈。

---

## 步骤 1: 识别测试范围与测试类型

**目标**: 根据当前任务阶段确定需要测试什么以及**运行哪种类型的测试**。

### 1.1 识别修改的文件
1. 从阶段 3（实现）读取任务完成总结。
2. 识别创建或修改了哪些模块/文件。
3. 将文件映射到相应的测试文件：
   - `src/libs/xxx/yyy.py` → `tests/unit/test_yyy.py`
   - `src/core/xxx/yyy.py` → `tests/unit/test_yyy.py`
   - `src/ingestion/xxx.py` → `tests/unit/test_xxx.py` 或 `tests/integration/test_xxx.py`

### 1.2 确定测试类型（智能选择）

**关键**: 测试类型应由**当前任务的性质**决定，而不是固定规则。

**决策逻辑**:

1. 在 `specs/06-schedule.md` 中读取任务规范以查找"测试方法"字段
2. 应用**测试策略决策矩阵**（见文档顶部）
3. 检查进度表中任务特定的测试方法：
   - `pytest -q tests/unit/test_xxx.py` → 运行 unit tests
   - `pytest -q tests/integration/test_xxx.py` → 运行 integration tests
   - `pytest -q tests/e2e/test_xxx.py` → 运行 E2E tests

**输出**:
```
────────────────────────────────────
 识别到测试范围
────────────────────────────────────
任务: [C14] Pipeline 编排（MVP 串起来）
修改的文件:
- src/ingestion/pipeline.py

测试类型决策:
- 任务性质: Pipeline orchestration (多模块协调)
- 规范测试方法: pytest -q tests/integration/test_ingestion_pipeline.py
- 已选择: **Integration Tests** 

理由: 此任务将多个模块连接在一起，
需要 loader、splitter、transform 和 storage 
组件之间的真实交互。
────────────────────────────────────
```

---

## 步骤 2: 执行测试

**目标**: 运行适当的测试并捕获结果。

**操作**:

### 2.1 检查测试是否存在
```bash
# Check for existing test files
ls tests/unit/test_<module_name>.py
ls tests/integration/test_<module_name>.py
```

### 2.2 如果测试存在 - 运行它们
```bash
# Run specific unit tests
pytest -v tests/unit/test_<module_name>.py

# Run with coverage if available
pytest -v --cov=src/<module_path> tests/unit/test_<module_name>.py
```

### 2.3 如果测试不存在 - 报告缺失的测试
如果规范要求测试但不存在：

```
────────────────────────────────────────
 ⚠️ 检测到缺失测试
────────────────────────────────────────
模块: <module_name>
预期测试文件: tests/unit/test_<module_name>.py

状态: 未找到

需要的操作:
  返回阶段 3 (implement) 创建测试，
  遵循现有测试文件中的测试模式。
────────────────────────────────────────
```

**操作**: 向工作流程协调器返回 `MISSING_TESTS` 信号以返回实现阶段。

---

## 步骤 3: 分析结果

**目标**: 解释测试结果并确定下一步操作。

### 3.1 测试通过
如果所有测试通过：
```
────────────────────────────────────────
 ✅ 测试通过
────────────────────────────────────────
模块: <module_name>
运行测试: X
通过测试: X
覆盖率: XX% (如果可用)

准备进入下一阶段。
────────────────────────────────────────
```
**操作**: 向工作流程协调器返回 `PASS` 信号。

### 3.2 测试失败
如果任何测试失败：
```
────────────────────────────────────────
 ❌ 测试失败
────────────────────────────────────────
模块: <module_name>
运行测试: X
失败测试: Y

失败的测试:
1. test_xxx - AssertionError: expected A, got B
2. test_yyy - ImportError: module not found

根本原因分析:
- [分析失败并识别问题]

建议修复:
- [提供具体的修复建议]
────────────────────────────────────────
```
**操作**: 向 `implement` 返回 `FAIL` 信号并提供详细反馈以进行迭代。

---

## 步骤 4: 反馈循环

**目标**: 启用迭代改进直到测试通过。

### 如果测试失败:
1. **生成修复报告**: 创建包含以下内容的结构化报告：
   - 失败的测试名称
   - 错误消息
   - 堆栈跟踪摘要
   - 失败的文件和行号
   - 建议的修复方法

2. **返回实现**: 将修复报告传回阶段 3 (implement) 进行修正。

3. **重新测试**: 实现更新后，再次运行测试。

### 迭代限制:
- **每个任务最多 3 次迭代**以防止无限循环。
- 如果 3 次迭代后仍然失败，上报给用户手动干预。

---

## 测试标准

### 测试命名约定
- `test_<function>_<scenario>_<expected_result>`
- 示例: `test_embed_empty_input_returns_empty_list`

### 测试类别（pytest markers）
```python
@pytest.mark.unit       # Fast, isolated tests
@pytest.mark.integration  # Tests with real dependencies
@pytest.mark.e2e        # End-to-end tests
@pytest.mark.slow       # Long-running tests
```

### Mock 策略
- **Unit Tests**: Mock 所有外部依赖（LLM、DB、HTTP）
- **Integration Tests**: 使用真实的本地依赖，mock 外部 API
- **E2E Tests**: 最小化 mocking，测试实际行为

---

## 验证检查清单

在将测试标记为完成之前，验证：

- [ ] 所有新的公共方法至少有一个测试
- [ ] 测试遵循命名约定
- [ ] 测试放置在正确的目录中（unit/integration/e2e）
- [ ] 测试使用适当的 mocking（unit tests 中没有真实的 API 调用）
- [ ] 测试断言与规范要求匹配
- [ ] 测试中没有硬编码的路径或凭据
- [ ] 测试可以独立运行（无顺序依赖）

---

## 重要规则

1. **不跳过测试**: 如果规范说"需要测试"，则测试必须存在。
2. **快速反馈**: Unit tests 应在 < 10 秒内完成。
3. **确定性**: 测试不得有随机失败。
4. **独立性**: 每个测试必须能够独立运行。
5. **清晰的失败**: 失败的测试必须提供可操作的错误消息。

---
