# 第 15 周：RAG 项目收尾（前端 + 部署 + 文档）

2026.10.19-10.25 ｜ 把 Agentic RAG 完整版上线，MCP Server 开源，产出面试与实习材料

## 一、本周目标

- 前端从 `static/index.html` 简易版迁移到 Next.js，完成登录、文档列表、会话历史、删除文档
- CI/CD（GitHub Actions）自动跑测试并部署
- Agentic RAG 完整版上线，拿到线上 URL
- MCP Server 独立发布到 GitHub
- 完整 README、3 分钟演示视频、面试讲稿、实习申请材料包

## 二、任务清单

### 1. 前端（Next.js）

- [x] 初始化 Next.js 项目（`web/`，`src/` 目录，`output: "standalone"`）
- [x] 问答页骨架：提问、展示答案、引用来源、LangFuse Trace 链接
- [x] 文档列表页骨架：列表、删除
- [ ] 登录页（后端暂无鉴权接口，待定）
- [ ] 会话历史页（后端 `sessions` 表已存在，待补接口）
- [x] 对接后端 API：`next.config.ts` 将 `/api/v1/*` 代理到 `BACKEND_URL`，同源调用，无需 CORS
- [ ] 引用点击后定位原文片段（`/qa/chunks/{id}` 已存在，前端未接）

### 2. 后端与部署

- [ ] 确认后端接口清单，补齐登录/删除文档/会话历史接口
- [ ] `docker-compose.yml` 补充 MCP 相关配置（如需要）
- [ ] 线上环境变量与密钥管理（`.env.example` 同步更新）
- [x] Dockerfile（后端）与 `web/Dockerfile`（前端，standalone 构建）
- [x] `docker-compose.prod.yml`：postgres + redis + backend + web，端口仅绑定 127.0.0.1
- [x] `.github/workflows/deploy.yml`：推送 master 后构建镜像到 GHCR，SSH 到服务器 `docker compose pull && up -d`
- [ ] 服务器初始化：安装 Docker、放置 `.env` 与 `docker-compose.prod.yml`
- [ ] Nginx 反向代理与 HTTPS
- [ ] 线上 URL 可访问，记录在本文件「验收记录」

### 3. CI/CD

- [x] `.github/workflows/ci.yml`：安装依赖 + `pytest`（本地基线 368 passed；尚未在 GitHub 上跑过）
- [x] `.github/workflows/deploy.yml`：推送 master 后自动构建并部署（需先配置下方 Secrets）
- [x] 所需 Secrets 清单：`DEPLOY_HOST`、`DEPLOY_USER`、`DEPLOY_SSH_KEY`、`DEPLOY_DIR`、`GHCR_USER`、`GHCR_TOKEN`（PAT，需 `read:packages`），环境名 `production`
- [ ] Secrets 实际配置到 GitHub（待你操作）

### 4. MCP Server 开源

- [ ] 拆出独立目录或仓库（当前在 `app/mcp_server/`）
- [ ] README：安装、Claude Desktop 配置、两个 Tool 的示例、截图
- [ ] 发布到 GitHub，写明与主仓库的关系

### 5. 完整 README（主仓库）

- [ ] 项目背景
- [ ] 架构图（复用 `docs/week13/graph_agentic.md` 的流程图）
- [ ] 技术栈
- [ ] 部署方法
- [ ] 指标：引用 `docs/week10/strategy_comparison.md` 与 `docs/week13/ab_report.md` 的数据
- [ ] 失败案例：引用 `docs/week13/` 中的边界测试结果
- [ ] 已知问题

### 6. 演示与面试材料

- [ ] 3 分钟演示视频（含 LangFuse Trace 展示 Agent 决策过程）
- [ ] 面试讲稿：1 分钟版
- [ ] 面试讲稿：3 分钟版
- [ ] 实习简历
- [ ] 作品集链接汇总
- [ ] 1 分钟自我介绍

### 7. 每周固定动作

- [ ] 代码提交 ≥ 5 次，至少 1 次有明确功能增量
- [ ] 刷题/复习 ≥ 3 次，每次 45-90 分钟
- [ ] 实习投递：本周起每周 5-8 个（第 16 周前可先行开始）
- [ ] 项目周报

## 三、验收产出对照

| 计划要求 | 状态 | 链接 / 备注 |
|---|---|---|
| RAG 项目 GitHub 完整版 | 待完成 | |
| 线上 URL | 待完成 | |
| 演示视频 | 待完成 | |
| MCP Server 开源仓库 | 待完成 | |
| LangFuse Dashboard 公开截图 | 待完成 | |
| Next.js 前端上线 | 待完成 | |
| 实习申请材料包 | 待完成 | |

## 四、验证记录

> 每项验证填写：日期、命令或操作、结果、截图路径。

- 全量测试：2026-10-12 本地 `pytest -q`，368 passed
- CI 首次绿灯：待推送到 GitHub 后验证
- 线上 URL 访问：
- MCP Server 在 Claude Desktop 中调用：

## 五、已知限制

- 待补充

## 六、风险与降级方案

- 若 Next.js 前端在周三前仍未完成主要页面：先上线现有 `static/index.html`，Next.js 版本作为第 23 周前的增量迭代，并在简历中如实描述。
- 若云服务器部署受阻：本地 `docker-compose up` 录屏 + 公网隧道作为临时演示，同时保留部署问题记录。
- 若演示视频时间不足：先录 2 分钟版本，细节放进 README 的 Trace 截图。
