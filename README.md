# Ascend Agent Skills

Ascend Agent 定制的 Claude Code 技能集合，专注于 NPU 内核开发等领域。

## 技能列表

| 技能名称 | 描述 |
|---------|------|
| **ta-kernel-butler** | Ascend NPU 内核开发助手 - 提供昇腾 NPU 硬件架构、triton-ascend 内核开发以及 GPU 到 NPU 迁移方面的专业指导 |

## 如何将 Marketplace 中的 Skill 引入 Claude Code

本指南将帮助你将此 marketplace 中的技能安装到你自己的 Claude Code 环境中。

### 方法一：在 Claude Code 客户端中添加（推荐）

你可以在 Claude Code 客户端中直接添加此 marketplace 并安装技能。

1. **添加 Marketplace**

   在 Claude Code 中运行以下命令：

   ```
   /plugin marketplace add cosdt/skills
   ```

2. **浏览并安装技能**

   添加 marketplace 后，你可以通过以下两种方式安装技能：

   **方式 A：交互式安装**

   在 Claude Code 中输入 `/plugin` 或点击插件图标，然后：
   - 选择 `Browse and install plugins`（浏览并安装插件）
   - 选择 `ascend-agent-skills` marketplace
   - 选择你想要安装的技能（例如 `ta-kernel-butler`）
   - 点击 `Install now`（立即安装）

   **方式 B：命令行直接安装**

   ```
   /plugin install ta-kernel-butler@ascend-agent-skills
   ```

3. **验证安装**

   安装完成后，你可以在 Claude Code 中直接使用这些技能。例如：
   > "请使用 ta-kernel-butler 帮我分析这个 NPU 内核的性能"

   Claude 会自动调用相应的技能来帮助你完成任务。

### 方法二：手动编辑配置文件

如果你更喜欢手动配置，可以按照以下步骤操作：

1. **编辑 known_marketplaces.json**

   打开或创建 `~/.claude/plugins/known_marketplaces.json` 文件，添加以下内容：

   ```json
   {
     "ascend-agent-skills": {
       "source": {
         "source": "github",
         "repo": "cosdt/skills"
       },
       "installLocation": "~/.claude/plugins/marketplaces/ascend-agent-skills",
       "lastUpdated": "2026-02-13T00:00:00.000Z"
     }
   }
   ```

   > **注意**：如果你的 `known_marketplaces.json` 文件中已经有其他 marketplace，请确保添加逗号并保持 JSON 格式正确。

2. **安装技能**

   在终端中运行安装命令：

   ```bash
   claude plugin install ta-kernel-butler@ascend-agent-skills
   ```


## 配置文件位置

- **Linux/macOS**: `~/.claude/plugins/known_marketplaces.json`
- **Windows**: `%USERPROFILE%\.claude\plugins\known_marketplaces.json`

## 故障排查

**问题**：找不到 `known_marketplaces.json` 文件
- **解决方案**：该文件会在首次添加 marketplace 时自动创建，你也可以手动创建它

**问题**：安装失败，提示权限错误
- **解决方案**：确保你有权限访问和修改 `~/.claude/` 目录

**问题**：技能安装成功但无法使用
- **解决方案**：重启 Claude Code 或检查技能的依赖是否满足

## 作者

Ascend Agent Team