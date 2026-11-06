# D-BRIDGE 语义标签网络实验

本项目按 `docs/202611061622` 的设计实现层次化、增量式语义标签网络构建：时间与热度联合分层抽样、LLM 原子概念提取、Embedding 相似节点检索、LLM 等价/层级关系判定，以及 Neo4j 图谱增量写入。正式实验会真实调用 `.env` 中配置的聊天模型和 Embedding 服务。

## 环境准备

Python 3.11+、Docker Compose v2，以及可访问的 OpenAI 兼容聊天和 Embedding API。

```bash
cp .env.example .env
# 编辑 .env：填写 API_KEY、BASE_URL、MODEL、Embedding 参数和 Neo4j 密码
docker compose up -d neo4j
uv venv
uv pip install --python .venv/bin/python -e '.[dev]'
```

本仓库不提交 `.env`。聊天模型和 Embedding 服务可以使用同一个或不同的兼容服务地址；`EMBEDDING_DIMENSIONS` 必须与服务实际返回维度一致。Neo4j 默认使用本地持久化 volume，程序不会清空既有图谱。

## 运行实验

```bash
.venv/bin/d-bridge
# 或指定输入目录和 env 文件
.venv/bin/d-bridge --data-dir data --env-file .env
```

默认示例位于 `data/posts.json`，包含 5 条可复现中文帖子。每次运行会在 `result/<本地时间戳>/` 新建独立目录，生成：

- `config.json`：本次公开实验参数快照，不包含 API key、Embedding key 或数据库密码。
- `input_posts.json`：实际加载的原始帖子。
- `experiment.log`：结构化 Loguru 运行日志，包含业务步骤和 LLM 输入/输出。
- `result.json`：抽样、概念、决策、最终标签和图谱导出汇总。
- `graph.json`：本次完成后 Neo4j 中的概念与关系快照。

进度条显示概念处理进度。运行需要外部 API 和正在运行、凭据匹配的 Neo4j 服务。

## 数据格式

数据目录内可放置一个或多个 `.json` 文件。文件根节点可为单个对象或对象数组，每条帖子格式如下：

```json
{
  "post_id": "post-001",
  "created_at": "2026-10-01T09:00:00+08:00",
  "hotness": 92,
  "text": "帖子正文"
}
```

ID 必须非空且全局唯一，时间必须为 ISO-8601，热度需为数值。抽样按时间和热度的秩分位分层，在每层按时间/ID 确定性轮转抽取；数据不足配置数量时使用全部数据。

## 配置

所有超参数可在 `.env` 覆盖，默认值见 `.env.example`：`SAMPLE_SIZE`、`TIME_BINS`、`HOTNESS_BINS`、`SIMILARITY_TOP_K`、`SIMILARITY_THRESHOLD`、`NEO4J_BATCH_SIZE`。模型通过 `API_KEY`、`BASE_URL`、`MODEL`、`TEMPERATURE`、`TIMEOUT`、`MAX_RETRIES` 配置；向量服务通过 `EMBEDDING_*` 配置；Neo4j 通过 `NEO4J_*` 配置。

LangGraph 包含 `extract_atomic_concepts` 与 `classify_equivalence_and_relation` 两个真实聊天模型调用节点，并在每个概念上循环执行相似候选检索、判定与图谱更新。支持的关系为“上位”“下位”“相似”“无关”；Neo4j 中用 `IS_A` 表示上下位，用 `RELATED_TO` 表示相似，帖子来源用 `MENTIONS` 表示，等价概念合并后转移到规范节点。

## 测试与排查

```bash
.venv/bin/python -m pytest
```

测试仅使用可注入替身模型和内存图存储，不请求外部 API。真实实验若失败，先检查 `experiment.log`，确认模型端点支持结构化 JSON 输出，Embedding 维度匹配，并检查 Neo4j：

```bash
docker compose ps
docker compose logs neo4j
```

Neo4j Browser 默认地址为 `http://localhost:7474`，Bolt 为 `bolt://localhost:7687`。
