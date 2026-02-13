# Zhudada's Skills

Custom Claude Code skills for NPU kernel development and more.

## Skills

### ta-kernel-butler
Ascend NPU kernel development assistant - Provides expert guidance on Ascend NPU hardware architecture, triton-ascend kernel development, and GPU to NPU migration.

## Installation

Add this marketplace to your `~/.claude/plugins/known_marketplaces.json`:

```json
{
  "zhudada-skills": {
    "source": {
      "source": "github",
      "repo": "zhudada0120/skills"
    },
    "installLocation": "/home/zhudada/.claude/plugins/marketplaces/zhudada-skills",
    "lastUpdated": "2026-02-13T00:00:00.000Z"
  }
}
```

Then install via Claude Code:
```bash
claude plugin install ta-kernel-butler --from zhudada-skills
```

## Author

zhudada0120
