---
name: checkpoint
description: 总结已完成工作，更新 DEV_SPEC.md 中的进度跟踪，并为下一次迭代做准备。dev-workflow 流程的最终阶段。当任务实现和测试完成时使用，或当用户说"完成检查点"、"checkpoint"、"保存进度"、"save progress"、"任务完成"时使用。
metadata:
  category: progress-tracking
  triggers: "checkpoint, save progress, 完成检查点, 保存进度, 任务完成"
allowed-tools: Bash(python:*) Bash(git:*) Read Write
---

# Checkpoint & Progress Persistence（检查点与进度持久化）

本技能处理**任务完成总结**和**进度跟踪同步**。它确保已完成的工作被正确记录，并且 `DEV_SPEC.md` 中的项目进度保持最新状态。

> **单一职责**: 总结 → 持久化 → 准备下一步

---

## 何时使用本技能

- 当任务实现和测试**已完成**时
- 当你需要**手动更新 DEV_SPEC.md 中的进度**时
- 当你想要**为已完成的工作生成提交信息**时
- 作为 `dev-workflow` 流程的**最终阶段**

---

## 工作流程

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  步骤 1            步骤 1.5                步骤 2              步骤 3       │
│  ────────         ────────                 ────────            ────────     │
│  总结工作   →   用户确认 (WHAT)  →     持久化进度 →      提交准备          │
│  (Summarize)     (验证完成内容)        (更新 DEV_SPEC)   (WHETHER)         │
└─────────────────────────────────────────────────────────────────────────────┘

                    ┌──────────────────┐
                    │   测试通过       │
                    └────────┬─────────┘
                             ▼
                  ┌──────────────────────┐
                  │  步骤 1: 总结工作    │
                  │  生成总结            │
                  └────────┬─────────────┘
                           ▼
                  ┌──────────────────────┐
                  │ 步骤 1.5: 用户       │
                  │ 确认                 │
                  │ 等待用户确认         │
                  └────────┬─────────────┘
                           │
                    用户确认? ──否──→ 修改总结 → 返回步骤1
                           │
                       是 ▼
                  ┌──────────────────────┐
                  │ 步骤 2: 持久化       │
                    进度                 │
                  │ 更新 DEV_SPEC.md     │
                  └────────┬─────────────┘
                           ▼
                  ┌──────────────────────┐
                  │ 步骤 3: 提交准备     │
                  │ 生成提交信息         │
                  │ 等待用户确认         │
                  └────────┬─────────────┘
                           │
                    用户确认? ──否──→ 跳过提交 → 流程结束
                           │
                       是 ▼
                  ┌──────────────────────┐
                  │  执行 git commit     │
                  └────────┬─────────────┘
                           ▼
                  ┌──────────────────────┐
                  │   检查点完成         │
                  └──────────────────────┘
```

---

## 步骤 1: 工作总结

**目标**: 生成清晰、结构化的已完成工作总结。

### 1.1 收集信息

从当前会话中收集以下信息：
- **任务 ID**: 例如 `A3`, `B1`, `C5`
- **任务名称**: 例如 "配置加载与校验"
- **创建/修改的文件**: 列出所有文件更改
- **测试结果**: 通过/失败状态和覆盖率（如果可用）
- **实现迭代次数**: 发生了多少次测试-修复循环

### 1.2 生成总结报告

**输出格式**:
```
────────────────────────────────────────────────────
 任务完成: [Task ID] [Task Name]
────────────────────────────────────────────────────

 修改的文件:
  创建:
    - src/xxx/yyy.py
    - tests/unit/test_yyy.py
  修改:
    - src/xxx/zzz.py

 测试结果:
    - tests/unit/test_yyy.py: 5/5 通过 
    - tests/unit/test_zzz.py: 3/3 通过 
    - 覆盖率: 85% (如果可用)

 迭代次数: [N] (1 = 首次尝试成功)

 规范引用: DEV_SPEC.md Section [X.Y]
────────────────────────────────────────────────────
```

---

## 步骤 1.5: 用户确认（验证完成了什么）

**目标**: 在持久化进度之前，向用户展示总结以供验证。

**这确认了完成了什么工作** - 验证总结的准确性，而不是是否保存它。

### 1.5.1 确认提示

**输出格式**:
```
════════════════════════════════════════════════════
 请验证完成总结 / Please Verify Completion Summary
════════════════════════════════════════════════════

 任务: [Task ID] [Task Name]
 规范引用: DEV_SPEC.md Section [X.Y]

 修改的文件:
  创建:
    - src/xxx/yyy.py
    - tests/unit/test_yyy.py
  修改:
    - src/xxx/zzz.py

 测试结果:
    - tests/unit/test_yyy.py: 5/5 通过 
    - tests/unit/test_zzz.py: 3/3 通过 

 迭代次数: [N]

════════════════════════════════════════════════════
 以上总结是否准确？
 Is this summary accurate?

   请回复: "confirm" / "确认" 将进度保存到 DEV_SPEC.md
          "revise" / "修改" 重新生成总结
                
 注意: 这只是验证总结。DEV_SPEC.md 将在确认后更新。
 Git commit 决定将在稍后进行。
════════════════════════════════════════════════════
```

### 1.5.2 处理用户响应

| 用户响应 | 操作 |
|---------------|--------|
| "confirm" / "yes" / "确认" / "是" | 进入步骤 2 |
| "revise" / "no" / "修改" / "否" | 询问用户需要修正什么，然后重新生成总结 |

**重要**: 在用户明确确认之前，不要进入步骤 2。

---

## 步骤 2: 持久化进度

**目标**: 更新 `DEV_SPEC.md` 以标记任务为已完成。

> **自动执行**: 此步骤在步骤 1.5 用户确认后自动运行。无需额外的用户输入。

### 2.1 在 DEV_SPEC.md 中定位任务

1. 读取 `DEV_SPEC.md`（**全局**文件，不是章节文件）
2. 通过标识符模式查找任务：
   - 查找 `### [Task ID]：[Task Name]`（例如 `### A3：配置加载与校验`）
   - 或查找复选框模式: `- [ ] [Task ID] [Task Name]`

### 2.2 更新进度标记

**支持的标记样式**:

| 更新前 | 更新后 | 样式 |
|--------|-------|-------|
| `[ ]` | `[x]` | Checkbox |
| ⏳ | ✅ | Emoji |
| `### A3：任务名` | `### A3：任务名 ✅` | Title suffix |
| `(进行中)` | `(已完成)` | Chinese status |
| `(In Progress)` | `(Completed)` | English status |

**更新逻辑**:
```python
# 更新逻辑的伪代码
if task_line contains "[ ]":
    replace "[ ]" with "[x]"
elif task_line contains "⏳":
    replace "⏳" with "✅"
elif task_line contains "(进行中)" or "(In Progress)":
    replace with "(已完成)" or "(Completed)"
else:
    append " ✅" to task title
```

### 2.3 步骤 2 输出格式

**更新 DEV_SPEC.md 后的输出**:
```
────────────────────────────────────
DEV_SPEC.md 进度已更新
────────────────────────────────────
任务: [Task ID] [Task Name]
状态: [ ] -> [x]
────────────────────────────────────
```

---

## 步骤 3: 提交准备

**目标**: 生成结构化的提交信息并询问用户是否提交。

### 3.1 Commit Message 模板

**Subject 格式**:
```
<type>(<scope>): [Phase X.Y] <brief description>
```

**模板定义**:
| 字段 | 说明 | 示例 |
|-------|-------------|---------|
| `<type>` | Commit 类型（见下表） | `feat`, `fix`, `test` |
| `<scope>` | 模块/组件名称 | `config`, `retriever`, `pipeline` |
| `[Phase X.Y]` | DEV_SPEC 阶段编号 | `[Phase 2.3]`, `[Phase A3]` |
| `<brief description>` | 完成的内容（< 50 字符） | `implement config loader` |

**Commit 类型指南**:
| 更改类型 | Commit 前缀 |
|-------------|---------------|
| 新功能 | `feat:` |
| Bug 修复 | `fix:` |
| 重构 | `refactor:` |
| 仅测试 | `test:` |
| 文档 | `docs:` |
| 配置 | `chore:` |

### 3.2 生成 Commit Message

**输出格式**:
```
════════════════════════════════════════════════════
 COMMIT MESSAGE / 提交信息
════════════════════════════════════════════════════

【Subject】
feat(<module>): [Phase X.Y] implement <feature name>

【Description】
Completed DEV_SPEC.md Phase X.Y: <Task Name>

Changes:
- Added <component 1> implementation
- Added <component 2> implementation
- Added unit tests test_xxx.py

Testing:
- Command: pytest tests/unit/test_xxx.py -v
- Results: X/X passed 
- Coverage: XX% (if available)

Refs: DEV_SPEC.md Section X.Y
Task: [Task ID] <Task Name>

════════════════════════════════════════════════════
```

### 3.3 用户提交确认（决定是否提交）

**这确认是否提交** - 决定更改是否应该现在提交到 git 还是稍后手动提交。

**提示用户**:
```
────────────────────────────────────
 是否需要帮您执行 git commit？
 Do you want me to commit these changes?
────────────────────────────────────

请回复 / Please reply:
  "yes" / "commit" / "是" → 执行 git add + git commit
  "no" / "skip" / "否"   → 结束流程，您可以稍后手动提交
────────────────────────────────────
```

### 3.4 执行提交（如果确认）

**如果用户确认**:
```bash
# Stage all changed files
git add <list of changed files>

# Commit with generated message
git commit -m "<subject>" -m "<description>"
```

**成功输出**:
```
────────────────────────────────────
 提交成功 / COMMIT SUCCESSFUL
────────────────────────────────────
Commit: <short hash>
Branch: <current branch>

进度已保存，任务 [Task ID] 已完成！
Progress saved, task [Task ID] completed!
────────────────────────────────────
```

### 3.5 跳过提交（如果拒绝）

**如果用户拒绝**:
```
────────────────────────────────────
 工作流程已完成（未提交）
 WORKFLOW COMPLETED (No Commit)
────────────────────────────────────
 ✓ DEV_SPEC.md 已更新
 ⊘ Git commit 已跳过

您可以稍后手动提交:
  git add .
  git commit -m "<subject>" -m "<description>"

任务 [Task ID] 检查点完成！
Task [Task ID] checkpoint completed!
────────────────────────────────────
```

---

## 快速命令

| 用户说 | 行为 |
|-----------|---------|
| "checkpoint" / "完成检查点" | 完整工作流程（步骤 1-3）带确认 |
| "save progress" / "保存进度" | 仅步骤 1.5-2（确认 + 持久化） |
| "commit message" / "生成提交信息" | 仅步骤 3（生成提交信息） |
| "commit for me" / "帮我提交" | 步骤 3 + 执行 git commit |

---

## 重要规则

1. **始终更新全局 DEV_SPEC.md**: 这是进度跟踪的唯一真实来源。

2. **保留现有格式**: 匹配文档中已使用的标记样式（复选框 vs emoji vs 文本）。

3. **原子更新**: 一次更新一个任务。不要批量更新多个任务。

4. **需要两次用户确认**: 
   - 步骤 1.5: 用户必须在持久化之前确认工作总结
   - 步骤 3.3: 用户必须在 git commit 之前确认
   - **永远不要跳过这些确认！**

5. **可追溯性**: 每个检查点必须引用定义该任务的具体规范章节。

---
