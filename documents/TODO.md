# API 文档索引功能待办

当前项目已经定义文档导入与文档列表的 API 契约，但尚未实现 Handler、Service 或 Web 框架路由。本文件记录后续服务化前需要完成的设计与实现工作。

## 当前已具备的契约

- `POST /documents/ingest`
  - 请求：`DocumentIngestRequest`
  - 响应：`DocumentIngestResponse`
- `GET /documents`
  - 响应：`DocumentListResponse`

契约位于 `app/api/contracts/documents.py`，目前不应直接调用 ingest Pipeline 或 Repository。

## 待实现的应用服务

### 1. IndexBuildService

- 放置位置：`app/indexing/services/build.py`
- 职责：接收应用级构建命令，调用 `IndexBuilder` 或 `IndexLoader`，返回 `IndexBuildResult`。
- 边界：Service 不依赖 Pydantic Request/Response，也不读取 HTTP 请求。
- 需要明确 `rebuild` 的语义：
  - `false`：若存在兼容且处于 ready 状态的索引，直接加载；否则构建。
  - `true`：强制重新执行构建流程。
- 注意：`IndexBuilderConfig.skip_existing` 只控制同一构建中的 chunk 去重，不能替代 API 请求的 `rebuild` 语义。

### 2. DocumentCatalogService

- 放置位置：`app/indexing/services/catalog.py`
- 职责：从 `DocumentRepository` 与 `ChunkRepository` 读取索引产物，聚合成文档摘要。
- 输出应包含文档身份、版本、来源路径、标题、内容哈希、chunk 数量与必要 metadata。
- 不应把 Repository 返回的数据结构直接暴露给 API。

## 待实现的 API 适配层

### 1. Documents Handler

- 放置位置：`app/api/handlers/documents.py`
- 依赖注入 `IndexBuildService` 与 `DocumentCatalogService`。
- 仅负责把 API Request 转换为应用命令、调用 Service、交给 Presenter 返回 Response。
- 不在 Handler 内构造 Builder、Loader、Repository 或 Factory。

### 2. Documents Presenter

- 放置位置：`app/api/presenters/documents.py`
- 负责将 indexing 的流程结果和目录摘要转换为 `DocumentIngestResponse`、`DocumentListResponse`。
- Manifest 对外输出前应选择稳定字段；不要直接泄露内部对象或本机绝对路径。

### 3. 路由接入

- 当项目引入 FastAPI 或其他 Web 框架后，将 `api/routes/catalog.py` 中的契约映射为真实路由。
- HTTP 层应统一处理 `AppError`，在错误响应和响应头中返回 `trace_id`。
- 对导入任务设置合理的超时、并发控制和请求体限制；大规模导入最终应改为后台任务而非同步 HTTP 请求。

## 验收标准

- API Handler 不直接依赖 `IndexBuilder`、`IndexLoader`、Repository 或 ingest Pipeline。
- `rebuild` 的行为有单元测试，并与已有索引版本状态一致。
- 文档列表只读取持久化索引，不触发 embedding 或重新切分。
- 所有成功与失败响应均可关联到 `trace_id`。
- API 契约、Presenter 与 indexing 应用服务之间不存在反向依赖。
