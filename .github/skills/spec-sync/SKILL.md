---
name: spec-sync
description: 同步 DEV_SPEC.md 并将其拆分为 specs/ 目录下的章节特定文件。运行 sync_spec.py 进行更新，然后阅读 SPEC_INDEX.md 进行导航。所有基于规范的操作的基础。当用户说"同步规范"、"sync spec"或在任何依赖规范的任务之前使用。
metadata:
  category: documentation
  triggers: "sync spec, update spec, 同步规范"
allowed-tools: Bash(python:*) Read
---

# Spec Sync（规范同步）

此技能同步主规范文档（`DEV_SPEC.md`）并将其拆分为存储在 `specs/` 目录中的较小的、特定于章节的文件。

> **这是所有基于规范的操作的前提条件。** 其他技能依赖于拆分的规范文件来执行其任务。

---

## 如何使用

### 在 dev-workflow 中使用（自动）

当你触发 dev-workflow（例如"下一阶段"或"继续开发"）时，**spec-sync 会自动作为阶段 1 运行**。无需手动操作。

### 手动同步（仅边缘情况）

仅在以下情况下手动运行：
- 你在工作流程之外编辑了 `DEV_SPEC.md`
- 规范文件已损坏或缺失
- 单独测试某个技能

```bash
# 正常同步
python .github/skills/spec-sync/sync_spec.py

# 强制重新生成（即使未检测到更改）
python .github/skills/spec-sync/sync_spec.py --force
```

---

### 同步脚本的功能

脚本执行以下操作：
1. 从项目根目录读取 `DEV_SPEC.md`
2. 计算哈希以检测更改
3. 将文档拆分为 `specs/` 下的章节文件
4. 生成 `SPEC_INDEX.md` 作为导航索引

---

### 同步后：使用 SPEC_INDEX.md 导航

**使用 `SPEC_INDEX.md` 作为入口点**来了解每个规范文件包含什么：

```
Read: .github/skills/spec-sync/SPEC_INDEX.md
```

此索引文件提供：
- 每个章节内容的摘要
- 快速参考以定位所需的规范

然后从 `specs/` 目录读取所需的特定规范文件：

```
Read: .github/skills/spec-sync/specs/05-architecture.md
```

---

## 目录结构

```
.github/skills/spec-sync/
├── SKILL.md              ← 本文件
├── SPEC_INDEX.md         ← 自动生成的索引（导航索引）
├── sync_spec.py          ← 同步脚本
├── .spec_hash            ← 用于变更检测的哈希文件
└── specs/                ← 拆分的规范文件（章节文件）
    ├── 01-overview.md
    ├── 02-features.md
    ├── 03-tech-stack.md
    ├── 04-testing.md
    ├── 05-architecture.md
    ├── 06-schedule.md
    └── 07-future.md
```

---

## 重要提示

- **永远不要直接编辑 `specs/` 中的文件** — 它们是自动生成的
- **始终编辑 `DEV_SPEC.md`** 并重新运行同步脚本
- 使用 `--force` 标志即使未检测到更改也重新生成：
  ```bash
  python .github/skills/spec-sync/sync_spec.py --force
  ```
