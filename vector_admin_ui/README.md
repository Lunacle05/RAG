# 向量库文档管理前端

一个独立的 React + TypeScript 管理页面，用于：

- 上传一个或多个文档到向量数据库
- 查看已入库文档（按 `source_path` 聚合）
- 多选删除文档对应向量（可选同步删除磁盘文件）

## 1. 启动后端

确保 `rag_service/run_rag.py` 已启动，并监听 `http://127.0.0.1:6006`。

## 2. 启动前端

```bash
cd vector_admin_ui
# npm install
npm run dev
```

默认访问：`http://127.0.0.1:6008`

## 3. 配置后端地址

可通过环境变量指定 API 地址：

```bash
VITE_API_BASE=http://127.0.0.1:6006 npm run dev
```

## 4. 后端接口

- `GET /admin/documents`
- `POST /admin/upload`（multipart，字段名 `files`）
- `POST /admin/delete`
  - body: `{ "source_paths": string[], "remove_files": boolean }`
