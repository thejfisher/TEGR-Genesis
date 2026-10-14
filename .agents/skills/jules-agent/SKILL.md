---
name: jules-agent
description: Dispatches tasks, inspects sessions, and fetches PRs from Google Jules autonomous coding agent.
---

# Google Jules Autonomous Agent Integration

Use this skill whenever the user asks to offload tasks to Google Jules, run autonomous test writing, or check Jules background sessions.

## Commands

### 1. Dispatch a Task to Jules
Run the bridge script from `z:\TEGR Genesis\tools\jules_bridge.py`:
```bash
python "z:\TEGR Genesis\tools\jules_bridge.py" start --repo <repo_name> --prompt "<prompt>"
```
Examples:
- `python "z:\TEGR Genesis\tools\jules_bridge.py" start --repo thejfisher/TEGR-Genesis --prompt "Write unit tests for core/engine.py"`
- `python "z:\TEGR Genesis\tools\jules_bridge.py" start --repo thejfisher/TEGR-Predictions --prompt "Add regression tests for SINDy algorithms"`

### 2. Check Session Status
```bash
python "z:\TEGR Genesis\tools\jules_bridge.py" sessions
```
Or for a specific session:
```bash
python "z:\TEGR Genesis\tools\jules_bridge.py" status <session_id>
```

### 3. List Connected Repositories
```bash
python "z:\TEGR Genesis\tools\jules_bridge.py" repos
```
