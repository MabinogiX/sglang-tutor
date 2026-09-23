# 最小 Axum API 服务器

一个使用 Axum 与 libtorch Rust bindings 的 Rust HTTP 服务示例。

## 运行

```bash
cargo run
```

服务默认监听 `http://127.0.0.1:3000`。

在另一个终端请求接口：

```bash
curl http://127.0.0.1:3000/
# Hello from Axum!

curl http://127.0.0.1:3000/health
# ok
```

`controllers::router` 负责注册路由；`hello` 和 `health` 是异步处理函数，返回的字符串会自动成为 HTTP 响应体。

## 校验与 API 文档示例

启动服务后访问 `http://127.0.0.1:3000/swagger-ui/`，可以查看和调用自动生成的 Swagger UI；原始 OpenAPI 文档位于 `http://127.0.0.1:3000/api-docs/openapi.json`。

`POST /users` 使用 `serde` 反序列化 JSON、`validator` 声明字段规则，并用 `axum-valid` 在进入 controller 前自动执行校验：

```bash
curl -X POST http://127.0.0.1:3000/users \
  -H 'content-type: application/json' \
  -d '{"name":"alice","email":"alice@example.com"}'
```

`name` 必须有 3 至 30 个字符，`email` 必须是合法邮箱。响应为 `201 Created`；非法 JSON 或不符合规则的字段会得到 `400 Bad Request`（字段规则错误会以 JSON 返回），且 controller / service 不会执行。

## 目录结构

```text
src/
├── main.rs                         # 启动 HTTP 服务
├── controllers/
│   ├── mod.rs                      # 路由注册
│   └── health_controller.rs         # HTTP 请求处理
└── services/
    ├── mod.rs
    └── health_service.rs            # 业务逻辑
```

controller 只处理 HTTP 层的输入和输出，并调用 service；service 保持与 Axum 解耦，放置业务规则。当前服务没有数据库或外部依赖，因此不额外引入 repository / data-access 层。

## 已迁移的 KV Cache 基础模块

`src/engine/kvcache/` 已包含 Rust 版本的 `BaseCacheHandle`、`CacheManager` trait、`KVCachePool` 与 `KVCacheAllocator`：

- 页表、空闲页栈、内存页数计算由 Rust 管理；
- `KVCachePool::new` 使用 `tch::Tensor::f_empty` 创建 `(2, layers, pages, page_size, kv_heads, head_dim)` Tensor；
- `get_all_kv_cache` 返回 `tch::Tensor` K/V 切片，供之后迁移的 attention 层使用；
- `RadixCacheManager` 已迁移，支持页对齐前缀匹配、共享前缀引用计数、插入回滚、请求释放和页粒度驱逐；
- Naive cache manager、Rust attention layer 绑定，以及加速器空闲显存查询尚未迁移；调用相应入口会返回 `未实现` 错误。

本机使用 `tch-rs` 构建时，需将 `LIBTORCH_USE_PYTORCH=1` 指向含 libtorch 的 Python 环境。`tch 0.26` 的官方目标版本是 PyTorch/libtorch 2.13，本项目当前环境为 2.13.0。macOS 运行时还需设置 `DYLD_LIBRARY_PATH`，让动态链接器找到 libtorch：

```bash
VIRTUAL_ENV=/Users/dp/code/sglang-rust/.venv \
PATH=/Users/dp/code/sglang-rust/.venv/bin:$PATH \
LIBTORCH_USE_PYTORCH=1 \
DYLD_LIBRARY_PATH=/Users/dp/code/sglang-rust/.venv/lib/python3.12/site-packages/torch/lib \
cargo test --lib
```
