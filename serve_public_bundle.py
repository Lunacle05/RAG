# cd RAG
# python serve_public_bundle.py --port 6020 --public-base-url "https://uu809261-n0jj-0de602da.westd.seetacloud.com:8443"
import argparse
import fnmatch
import html
import os
import secrets
import tempfile
import threading
import time
import urllib.parse
import zipfile
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Iterable


def _to_posix_relpath(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _should_exclude(rel_posix_path: str, patterns: Iterable[str]) -> bool:
    for pat in patterns:
        if fnmatch.fnmatch(rel_posix_path, pat):
            return True
    return False


def build_zip_bundle(
    *,
    repo_root: Path,
    output_zip_path: Path,
    exclude_patterns: list[str],
) -> tuple[int, int]:
    file_count = 0
    total_bytes = 0

    repo_root = repo_root.resolve()
    output_zip_path.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(output_zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for dirpath, dirnames, filenames in os.walk(repo_root):
            current_dir = Path(dirpath)
            rel_dir_posix = _to_posix_relpath(current_dir, repo_root)
            if rel_dir_posix == ".":
                rel_dir_posix = ""
            if rel_dir_posix and _should_exclude(f"{rel_dir_posix}/", exclude_patterns):
                dirnames[:] = []
                continue

            pruned_dirnames: list[str] = []
            for d in dirnames:
                rel_subdir = f"{rel_dir_posix}/{d}" if rel_dir_posix else d
                if _should_exclude(f"{rel_subdir}/", exclude_patterns):
                    continue
                pruned_dirnames.append(d)
            dirnames[:] = pruned_dirnames

            for name in filenames:
                src_path = current_dir / name
                rel_file = f"{rel_dir_posix}/{name}" if rel_dir_posix else name
                if _should_exclude(rel_file, exclude_patterns):
                    continue
                if src_path.resolve() == output_zip_path.resolve():
                    continue
                try:
                    st = src_path.stat()
                except FileNotFoundError:
                    continue
                if not src_path.is_file():
                    continue
                zf.write(src_path, arcname=rel_file)
                file_count += 1
                total_bytes += int(st.st_size)

    return file_count, total_bytes


def _format_bytes(n: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    x = float(n)
    for u in units:
        if x < 1024 or u == units[-1]:
            if u == "B":
                return f"{int(x)} {u}"
            return f"{x:.2f} {u}"
        x /= 1024
    return f"{n} B"


class BundleServer:
    def __init__(
        self,
        *,
        repo_root: Path,
        host: str,
        port: int,
        public_base_url: str | None,
        token: str,
        exclude_patterns: list[str],
    ) -> None:
        self.repo_root = repo_root.resolve()
        self.host = host
        self.port = port
        self.public_base_url = public_base_url
        self.token = token
        self.exclude_patterns = exclude_patterns

        self._lock = threading.Lock()
        self._zip_path: Path | None = None
        self._zip_file_count = 0
        self._zip_total_bytes = 0
        self._zip_built_at = 0.0

    def ensure_zip(self) -> None:
        with self._lock:
            if self._zip_path and self._zip_path.exists():
                return
            tmp_dir = Path(tempfile.gettempdir())
            ts = time.strftime("%Y%m%d_%H%M%S")
            zip_path = tmp_dir / f"rag_bundle_{ts}.zip"
            file_count, total_bytes = build_zip_bundle(
                repo_root=self.repo_root,
                output_zip_path=zip_path,
                exclude_patterns=self.exclude_patterns,
            )
            self._zip_path = zip_path
            self._zip_file_count = file_count
            self._zip_total_bytes = total_bytes
            self._zip_built_at = time.time()

    def get_zip_info(self) -> tuple[Path, int, int, float]:
        self.ensure_zip()
        assert self._zip_path is not None
        return self._zip_path, self._zip_file_count, self._zip_total_bytes, self._zip_built_at

    def cleanup(self) -> None:
        with self._lock:
            if self._zip_path and self._zip_path.exists():
                try:
                    self._zip_path.unlink()
                except OSError:
                    pass
            self._zip_path = None


def make_handler(server: BundleServer) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            parsed = urllib.parse.urlparse(self.path)
            path = parsed.path
            query = urllib.parse.parse_qs(parsed.query)

            if path == "/":
                self._handle_index()
                return

            if path == "/download.zip":
                self._handle_download(query)
                return

            self.send_error(HTTPStatus.NOT_FOUND, "Not Found")

        def _handle_index(self) -> None:
            zip_path, file_count, total_bytes, built_at = server.get_zip_info()
            built_at_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(built_at))
            size_str = _format_bytes(total_bytes)
            token = urllib.parse.quote(server.token, safe="")
            dl_url = f"/download.zip?token={token}"

            base = server.public_base_url
            public_hint = ""
            if base:
                base = base.rstrip("/")
                public_hint = f"<p>公网地址：<a href='{html.escape(base)}'>{html.escape(base)}</a></p>"

            body = f"""<!doctype html>
<html lang="zh-CN">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>一键下载打包</title>
    <style>
      body {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; padding: 24px; }}
      .card {{ max-width: 760px; border: 1px solid #e5e7eb; border-radius: 12px; padding: 16px 18px; }}
      .btn {{ display: inline-block; padding: 10px 14px; background: #111827; color: #fff; border-radius: 10px; text-decoration: none; }}
      code {{ background: #f3f4f6; padding: 2px 6px; border-radius: 6px; }}
      .meta {{ color: #374151; }}
      .warn {{ color: #991b1b; }}
    </style>
  </head>
  <body>
    <div class="card">
      <h2>项目打包下载</h2>
      <p class="meta">已生成压缩包：<code>{html.escape(zip_path.name)}</code></p>
      <p class="meta">文件数：{file_count}，体积约：{html.escape(size_str)}，生成时间：{html.escape(built_at_str)}</p>
      {public_hint}
      <p><a class="btn" href="{html.escape(dl_url)}">一键下载 ZIP</a></p>
      <p class="warn">注意：链接包含访问令牌，不要公开转发。</p>
    </div>
  </body>
</html>
"""
            body_bytes = body.encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body_bytes)))
            self.end_headers()
            self.wfile.write(body_bytes)

        def _handle_download(self, query: dict[str, list[str]]) -> None:
            token = (query.get("token") or [""])[0]
            if token != server.token:
                self.send_error(HTTPStatus.FORBIDDEN, "Forbidden")
                return

            zip_path, _, _, _ = server.get_zip_info()
            try:
                size = zip_path.stat().st_size
                f = open(zip_path, "rb")
            except OSError:
                self.send_error(HTTPStatus.INTERNAL_SERVER_ERROR, "Bundle not available")
                return

            try:
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "application/zip")
                self.send_header("Content-Disposition", f'attachment; filename="{zip_path.name}"')
                self.send_header("Content-Length", str(size))
                self.end_headers()
                while True:
                    chunk = f.read(1024 * 1024)
                    if not chunk:
                        break
                    self.wfile.write(chunk)
            finally:
                try:
                    f.close()
                except OSError:
                    pass

        def log_message(self, fmt: str, *args: object) -> None:
            return

    return Handler


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=6020)
    parser.add_argument("--public-base-url", default=None)
    parser.add_argument("--token", default=None)
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent

    token = (args.token or "").strip() or secrets.token_urlsafe(16)

    default_excludes = [
        ".git/**",
        "**/__pycache__/**",
        "**/*.pyc",
        "**/.ipynb_checkpoints/**",
        "chroma_db/**",
        "Qwen3-*/**",
        "vector_admin_ui/node_modules/**",
        "vector_admin_ui/dist/**",
        "vector_admin_ui/.env*",
    ]

    exclude_patterns = [*default_excludes, *(args.exclude or [])]

    bundle_server = BundleServer(
        repo_root=repo_root,
        host=args.host,
        port=args.port,
        public_base_url=args.public_base_url,
        token=token,
        exclude_patterns=exclude_patterns,
    )
    bundle_server.ensure_zip()

    httpd = ThreadingHTTPServer((args.host, args.port), make_handler(bundle_server))
    local_url = f"http://127.0.0.1:{args.port}/"

    if args.public_base_url:
        print(f"公网访问：{args.public_base_url.rstrip('/')}/")
    print(f"本机访问：{local_url}")
    print(f"下载页面：{local_url}?token={token}")
    print(f"下载链接：{local_url}download.zip?token={token}")

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        try:
            httpd.server_close()
        finally:
            bundle_server.cleanup()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
