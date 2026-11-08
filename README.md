# D-BRIDGE 语义标签网络实验

本项目按 `docs/202611061622` 的设计实现层次化、增量式语义标签网络构建：时间与热度联合分层抽样、LLM 原子概念提取、Embedding 相似节点检索、LLM 等价/层级关系判定，以及 Neo4j 图谱增量写入。正式实验会真实调用 `.env` 中配置的聊天模型和 Embedding 服务。

## 环境准备

Python 3.11+、Docker Compose v2，以及可访问的 OpenAI 兼容聊天和 Embedding API。

```bash
cp .env.example .env
# 编辑 .env：填写 API_KEY、BASE_URL、MODEL、Embedding 参数和 Neo4j 密码
make install
```

本仓库不提交 `.env`。聊天模型和 Embedding 服务可以使用同一个或不同的兼容服务地址；`EMBEDDING_DIMENSIONS` 必须与服务实际返回维度一致。Neo4j 默认使用本地持久化 volume，但实验为保证各轮隔离，会在每个 trial 开始前删除目标 database 中所有带 `Post` 或 `Concept` 标签的节点及其关系。请为 D-BRIDGE 使用专用 database，不要与其他业务共用这些标签。

`BASE_URL` 和 `EMBEDDING_BASE_URL` 必须是以 `http://` 或 `https://` 开头的完整 OpenAI 兼容 API 地址（通常包含 `/v1`）；API key 只能填写在对应的 `*_API_KEY` 字段，不能与地址互换。

## 运行实验

```bash
make run
```

`make run` 会自动启动 Neo4j，等待 healthcheck 通过后运行实验。需要指定输入目录或环境文件时可使用：

```bash
DATA_DIR=data ENV_FILE=.env make run
```

也可以单独管理服务：`make neo4j` 启动并等待 Neo4j，`make stop` 停止容器，`make test` 运行离线测试。`make run` 会让 Compose 和 Python 都以 `ENV_FILE` 为配置来源，避免终端中残留的同名环境变量造成两边凭据不一致。

默认示例位于 `data/posts.json`，包含 20 条可复现的新能源车企模拟舆情，兼有正面、负面及混合观点。每次运行会在 `result/<本地时间戳>/` 新建批次目录，生成：

- `config.json`：本次公开实验参数快照，不包含 API key、Embedding key 或数据库密码。
- `prompts.json`：本次实际使用的完整提示词模板快照及各模板版本。
- `input_posts.json`：本批次实际加载的规范化输入快照。
- `experiment.log`：批次级结构化 Loguru 日志。
- `summary.json`：各轮状态、耗时、token、抽样覆盖率汇总和跨轮稳定性指标。
- `trials/trial-<序号>-seed-<种子>/`：每轮独立的 `config.json`、`experiment.log`、`sampling.json`、`result.json` 和 `graph.json`；失败轮保存 `error.json`。

`sampling.json` 记录总体数量、实际样本量、总体覆盖率、有效分箱数，以及每个联合层的原始数量、抽样数量、层内覆盖率、取值范围和样本 ID。`result.json` 同时包含该抽样摘要；其中的 `llm_usage` 保存每次调用的 input/output/total tokens、模型、响应 ID、后端 fingerprint 和耗时。兼容服务不返回 usage 时会明确标记不可用，不进行本地估算。运行时先显示逐帖概念提取进度，再显示合并后概念的处理进度。运行需要外部 API 和正在运行、凭据匹配的 Neo4j 服务。

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

ID 必须非空且全局唯一，时间必须为 ISO-8601，热度需为数值。抽样按时间和热度做等频秩分箱；并列值按当前 trial seed 随机打散，联合层先保证最低覆盖，再按剩余层容量使用最大余数法分配名额，层内无放回随机抽取。数据不足配置数量时使用全部数据。

## 配置

所有超参数可在 `.env` 覆盖，默认值见 `.env.example`：`SAMPLE_SIZE`、`TIME_BINS`、`HOTNESS_BINS`、`SIMILARITY_TOP_K`、`SIMILARITY_THRESHOLD`、`NEO4J_BATCH_SIZE`。模型通过 `API_KEY`、`BASE_URL`、`MODEL`、`TEMPERATURE`、`TIMEOUT`、`MAX_RETRIES` 配置；默认温度为 `0`。`ENABLE_THINKING` 只接受 `true` 或 `false`，默认 `false`；模型名包含 `qwen` 时会作为兼容扩展字段发送，其他模型不会收到该字段。向量服务通过 `EMBEDDING_*` 配置；Neo4j 通过 `NEO4J_*` 配置。

概念提取按帖子分别调用模型，`CONCEPT_EXTRACTION_CONCURRENCY` 控制最大并发数，默认 `3`。每篇帖子内先规范化概念，跨帖子再按移除空白并忽略大小写后的名称确定性合并，同时合并来源帖子 ID；不会额外调用 LLM 或 Embedding 做语义近义合并。

合并后的概念按 `CONCEPT_PROCESSING_CONCURRENCY` 分批处理，控制等价/关系判定 LLM 的最大并发数，默认 `3`。每批概念基于该批开始时的图谱检索候选，并在判定完成后按原顺序写入图谱；下一批会看到前一批的更新。设置为 `1` 时概念处理保持逐个顺序执行。

`EXPERIMENT_SEEDS` 使用 JSON 整数数组，数组长度就是 trial 数量：默认 `[42]` 只运行一次，设置为 `[42,43,44]` 会运行三次，也允许使用重复种子。每轮会设置 Python `random` 和 NumPy 的全局随机状态；Pandas 随机操作依赖 NumPy，全新随机代码仍应显式传入 `random_state`。这些 seed 不会传给 LLM，只控制本地 Python 随机性，包括分箱并列值、配额平局和层内样本；相同 seed 可复现同一批样本，不同 seed 用于观察抽样与后续输出的变化。

每轮开始前清空 D-BRIDGE 图谱，完成后先导出该轮快照。多轮成功时 Neo4j 最终保留最后一轮图谱；任一轮失败会立即停止后续轮次，写出失败摘要并清理失败轮的残留数据。`summary.json` 对概念、标签和关系集合计算两两 Jaccard 与完全一致率；只有一轮时稳定性状态为 `insufficient_samples`。

提示词模板默认从 `config/prompt_templates.json` 加载，也可通过 `PROMPT_TEMPLATE_PATH` 指向其他 JSON 文件，以便在不同模板版本之间切换。概念提取与关系判定模板各自维护 `version`；修改任一模板正文时应同步升级对应版本。模板中的 `allowed_variables` 必须与正文占位符一致，程序会在调用模型前校验必需变量、未知变量和消息结构。每次运行会复制实际模板到结果目录，并在 `config.json` 和 `result.json` 中记录版本，因此后续修改源模板不会影响历史实验的追溯。

LangGraph 包含 `extract_atomic_concepts` 与 `classify_equivalence_and_relation` 两个真实聊天模型调用节点；概念处理按配置并发判定，并按批次顺序增量更新图谱。支持的关系为“上位”“下位”“相似”“无关”；Neo4j 中用 `IS_A` 表示上下位，用 `RELATED_TO` 表示相似，帖子来源用 `MENTIONS` 表示，等价概念合并后转移到规范节点。

## 测试与排查

```bash
uv run pytest
```

测试仅使用可注入替身模型和内存图存储，不请求外部 API。真实实验若失败，先检查 `experiment.log`，确认模型端点支持结构化 JSON 输出，Embedding 维度匹配，并检查 Neo4j：

```bash
docker compose ps
docker compose logs neo4j
```

Neo4j Browser 默认地址为 `http://localhost:7474`，Bolt 为 `bolt://localhost:7687`。

Neo4j 的 `NEO4J_AUTH` 只在持久化 volume 首次初始化时设置密码。若修改过 `.env` 中的 `NEO4J_PASSWORD`，已有数据库不会自动同步；应恢复初始化时的密码。仅当现有本地图谱可以全部丢弃时，才运行 `make neo4j-reset` 删除 volume 并按当前 `ENV_FILE` 重新初始化。该命令不可恢复已有 Neo4j 数据。
