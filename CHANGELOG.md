# Changelog

## v2.0.0 (2026-04)

### 🚀 重大更新

- 项目全面迁移至 TypeScript 技术栈
  - 后端：Hono + Drizzle ORM + better-sqlite3
  - 前端：Nuxt 3 + Vue 3
  - AI Agent：Mastra 框架
- 重做单集工作台 UI 和生产流程
  - 更紧凑的控制台布局
  - 重做分镜编辑区
  - 重做配音、镜头图、视频、合成、导出界面
- 新增 Docker 部署支持，前后端合并为单镜像
- 增加运行时 Skill 加载机制
- 扩展多厂商媒体 Adapter
  - 图片：OpenAI、Gemini、MiniMax、火山引擎、阿里
  - 视频：MiniMax、火山引擎/Seedance、Vidu、阿里
  - TTS：MiniMax
- 增加宫格图生成、切分和重新分配流程
- 优化本地文件处理与参考图按需转码

## v1.0.4 (2026-01-27)

- 引入本地存储策略，规避外部资源链接失效
- Base64 参考图嵌入式传输
- 修复镜头切换状态重置问题
- 添加场景迁移至章节

## v1.0.3 (2026-01-16)

- SQLite 纯 Go 驱动，支持 CGO_ENABLED=0 跨平台编译
- 优化并发性能（WAL 模式）
- Docker 跨平台支持 host.docker.internal

## v1.0.2 (2026-01-14)

- 修复视频生成 API 响应解析问题
- 添加 OpenAI Sora 视频端点配置
- 优化错误处理和日志输出
