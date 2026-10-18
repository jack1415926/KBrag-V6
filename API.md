# API 接口入口

完整、唯一维护的API协议见：

- [项目技术架构说明文档/01_API接口说明.md](项目技术架构说明文档/01_API接口说明.md)

当前端点：

| 方法与路径 | 用途 |
| --- | --- |
| `GET /health` | 健康检查与运行状态 |
| `POST /chat` | 文本或图片问答 |
| `GET /manual-images/{image_id}` | 返回回答引用的本地手册图片 |
| `GET /` | 内置图文问答网页 |
| `GET /latency-baseline/status` | 读取本地延迟实验进度 |

快速启动：

```bash
pip install -r requirements.txt
./run_api.sh
```

本文件只作为根目录入口，避免与正式API文档重复维护请求字段、响应结构、错误码和示例。

