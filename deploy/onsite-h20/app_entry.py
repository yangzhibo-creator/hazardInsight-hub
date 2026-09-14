"""离网现场单容器入口：聚类网关 API + 前端静态资源。

为什么需要这一层：现场只报备了一个端口（6006），而演示既要 API 又要页面。
这里把两者装进同一个 FastAPI 应用，避免在基础镜像里额外引入 Node 或 nginx
（h20-inference:v1.0 只保证 Python 运行时）。

挂载顺序很重要：`app.main:app` 先把 `/api/*` 路由注册完，再挂载 `StaticFiles`
到 `/`。FastAPI 按注册顺序匹配，因此 API 不会被静态资源吞掉。

找不到前端产物时**不报错**：现场可能只想验证 API（`/api/health`），
把"页面缺失"变成启动失败会凭空增加一个故障点。
"""

from __future__ import annotations

import os
from pathlib import Path

from fastapi.staticfiles import StaticFiles

from app.main import app

WEB_DIR = Path(os.environ.get("WEB_DIR", "/app/web"))
#: 离线演示的维护性开关：`SERVE_WEB=0` 时只提供 API（便于排查前端问题）。
SERVE_WEB = os.environ.get("SERVE_WEB", "1").strip().lower() not in {"0", "false", "no", "off"}

if SERVE_WEB and WEB_DIR.is_dir():
    # html=True：目录请求返回 index.html，SPA 的客户端路由才能工作
    app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web")
