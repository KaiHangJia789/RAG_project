# FastAPI 后端工程实践

## 应用生命周期管理

FastAPI 使用 lifespan 上下文管理器处理启动和关闭逻辑，替代了旧版的事件装饰器写法。

启动阶段要初始化的典型资源包括：数据库连接池、Redis 连接、各类服务单例、缓存预热。关闭阶段要按相反顺序释放这些资源。

关键设计原则是**优雅降级**：任何一个外部依赖连不上，都不应该让整个应用启动失败。数据库连不上时，应用仍应启动并提供不依赖数据库的接口（比如健康检查、静态页面），同时把相关服务标记为不可用，等真正被调用时返回明确的错误。

这个原则的实际价值在于排查效率。如果数据库故障导致应用直接退出，运维看到的是"服务挂了"；如果应用能启动并在日志里明确打出"数据库连接失败，文档服务不可用"，问题定位就是秒级的。

连接重试应当带超时控制。`asyncio.wait_for` 配合有限次数的重试（如 3 次、每次 5 秒超时）既能容忍瞬时抖动，又不会让启动无限期挂起。

## 依赖注入的组织方式

FastAPI 的 `Depends` 机制让路由函数声明自己需要什么，由框架负责提供。依赖可以声明为可调用对象或类。

对于需要跨请求复用的重量级对象（数据库连接池、模型客户端），常见做法是用**模块级全局服务定位器**：定义一个模块级变量存放单例，在 lifespan 启动时赋值，再写一个 getter 函数给 `Depends` 使用。

这种做法的好处是简单直接，测试时可以直接覆盖模块级变量注入替身。代价是隐式的全局状态，模块间存在隐式耦合。规模再大一些时应该换成正式的 DI 容器。

getter 函数必须处理"服务未初始化"的情况。此时抛 `HTTPException(503)` 而不是让 `None` 传播下去——后者会在某个不确定的位置抛出 `AttributeError`，返回 500，让人以为是代码 bug 而非服务未就绪。

依赖类型别名可以显著提升路由函数签名的可读性：`DocumentServiceDep = Annotated[DocumentService, Depends(get_document_service)]`。

## Pydantic v2 模型设计

请求和响应模型用 Pydantic 定义。v2 相对 v1 有几个必须注意的变化。

配置从内部类 `class Config` 改为 `model_config = ConfigDict(...)`。示例字段从 `example=` 改为 `json_schema_extra={"example": ...}` 或 `Field(examples=[...])`。校验器从 `@validator` 改为 `@field_validator`。

字段约束直接写在 `Field` 里：`ge` 和 `le` 控制数值范围，`min_length` 和 `max_length` 控制字符串长度，`pattern` 用正则约束格式。校验失败时 FastAPI 自动返回 422，错误信息包含字段路径和失败原因。

响应模型应该用泛型包装统一格式。定义 `APIResponse[T]` 包含 `code`、`message`、`data` 三个字段，路由声明 `response_model=APIResponse[DocumentResponse]`。这样所有接口的响应结构一致，前端可以写一个通用的解包函数。

分页响应用 `PaginatedData[T]` 包装，包含 `items`、`total`、`page`、`page_size`、`total_pages`。

## 异常处理体系

设计一个应用异常基类，携带 HTTP 状态码、用户可读消息和详细说明：

```python
class AppException(Exception):
    def __init__(self, message: str, code: int = 400, detail: str | None = None):
        self.message = message
        self.code = code
        self.detail = detail
```

具体异常继承它：`DocumentNotFoundError` 返回 404，`FileValidationError` 返回 400，`IndexNotReadyError` 返回 503。

关键点在于**只需注册基类的处理器**。Starlette 的异常处理器查找会沿类继承链向上匹配，所以 `app.add_exception_handler(AppException, handler)` 一处注册就能覆盖所有子类，新增异常不需要改 main.py。

错误信息的设计要有可操作性。"索引未构建"应该附带具体的修复命令（`python scripts/build_index.py`），而不是只说"服务不可用"。这条经验在实践中节省的排查时间非常可观。

除了自定义异常处理器，还应该注册一个 `Exception` 的兜底处理器返回 500，避免未捕获异常泄露堆栈。

## 中间件与请求日志

中间件以洋葱模型包裹请求处理。常见的两个是：

**计时中间件**记录每个请求的耗时，在响应头里加上 `X-Process-Time`。

**请求日志中间件**记录方法、路径、状态码、耗时、客户端 IP。实现时要注意——`await call_next(request)` 返回响应后读取 `response.status_code`，但不能读取响应体（会消耗流导致客户端收不到内容）。

中间件的注册顺序决定了执行顺序，先注册的在外层。

## 文件上传的最佳实践

上传接口要处理的边界比想象中多。

**大小限制**：不能等文件全部读进内存再检查大小。应该在读取过程中累计字节数，超限立即中断并返回 413。对于必须完整读取的场景，至少要在读取后立即检查，避免超大文件占用内存。

**扩展名校验**：从文件名提取后缀，与白名单比对。注意要用 `Path(filename).suffix.lower()` 处理大小写，并且只信任最后一个后缀（`evil.pdf.exe` 实际是 exe）。

**文件名安全**：永远不要用用户提供的文件名作为磁盘路径。正确做法是生成 UUID 作为文件名，原始名只存数据库用于展示。这 simultaneously 防御了路径穿越（`../../etc/passwd`）和重名覆盖。

**原子性**：先写文件再写数据库。如果数据库写入失败，必须删除已落盘的文件，否则会留下孤儿文件。反过来（先写库再写文件）更糟——事务回滚会留下"记录在但文件没了"的坏数据，详情页能查到但文件读不出来。

**状态一致性**：异步处理（如解析、索引）完成后要把状态写回数据库。常见 bug 是只更新了内存里的响应对象，数据库里的状态永远停在初始值，导致列表页显示的状态永远不对。

## 测试策略

FastAPI 应用用 `httpx.AsyncClient` 配合 `ASGITransport` 做测试，不需要真正启动服务器。

内存替身是测试的核心。数据库层用一个按表名分桶的字典模拟，Redis 用一个内存字典实现常用命令。替身通过构造函数注入或覆盖模块级变量接入。

替身有个隐蔽风险：**SQL 解析型的替身会在匹配失败时静默返回错误结果**。比如替身只识别 `WHERE user_id = $1` 这一种过滤，遇到 `WHERE document_id = $1` 时不报错，而是返回全表。这样写错的 SQL 在单元测试里会"通过"，问题只在生产环境暴露。

缓解办法有三：给未识别的 SQL 模式加警告日志；对新增的查询补显式断言（断言返回条数而不仅仅断言不报错）；补充针对真实数据库的集成测试。

## 常见坑

**`Annotated[..., Depends(...)]` 与默认值的冲突**：写了 `x: SomeDep = None` 之后，`None` 其实不会生效（`Depends` 优先），但会误导读者。要么去掉 `= None`，要么明确知道这是占位写法。

**异常处理器只注册了自定义的**：FastAPI 默认的 `HTTPException` 处理器会返回 `{"detail": ...}` 格式，与自定义的 `APIError` 格式不一致。前端需要同时兼容两种格式（`data.message || data.detail`）。

**后台任务与请求生命周期**：用 `BackgroundTasks` 执行的任务在响应返回后运行，此时请求作用域的资源（如数据库会话）可能已释放。长耗时任务应该用独立的任务队列。

**多进程部署与内存状态冲突**：`uvicorn --workers N` 会启动 N 个进程。如果应用持有进程内可变状态（如内存索引、内存缓存），各进程状态不一致且会互相覆盖持久化文件。这类应用必须用 `--workers 1`，或者把状态外置到 Redis 等共享存储。
