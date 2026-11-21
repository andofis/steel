# -*- coding: utf-8 -*-
"""Local web service for the Hikrobot four-camera acquisition console."""

from __future__ import annotations

import argparse
import atexit
import functools
import http.server
import json
import socket
import threading
import webbrowser
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from hikrobot_camera import HikRobotService

ROOT = Path(__file__).resolve().parent
SERVICE = HikRobotService()


def find_available_port(preferred: int, attempts: int = 20) -> int:
    for port in range(preferred, preferred + attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind(("127.0.0.1", port))
            except OSError:
                continue
            return port
    raise OSError(f"端口 {preferred}-{preferred + attempts - 1} 均被占用")


class CaptureHandler(http.server.SimpleHTTPRequestHandler):
    server_version = "HikrobotCapture/1.0"

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store, max-age=0")
        self.send_header("X-Content-Type-Options", "nosniff")
        super().end_headers()

    def log_message(self, fmt: str, *args) -> None:
        print("[HTTP] " + fmt % args)

    def send_json(self, data: object, status: int = 200) -> None:
        payload = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def send_error_json(self, message: str, status: int = 500) -> None:
        self.send_json({"ok": False, "error": message}, status=status)

    def read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", "0") or "0")
        if length <= 0:
            return {}
        raw = self.rfile.read(length)
        try:
            value = json.loads(raw.decode("utf-8"))
            return value if isinstance(value, dict) else {}
        except Exception:
            raise ValueError("请求 JSON 格式错误")

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        if path == "/api/state":
            self.send_json({"ok": True, "state": SERVICE.get_state()})
            return
        if path.startswith("/api/camera-xml/"):
            try:
                slot = int(path.split("/")[-1])
                xml = SERVICE.get_gencim_xml(slot)
            except Exception as exc:
                self.send_error_json(str(exc), status=409)
                return
            payload = xml.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/xml; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        if path == "/api/health":
            state = SERVICE.get_state()
            self.send_json({
                "ok": True,
                "sdk_ready": state["sdk_ready"],
                "sdk_version": state["sdk_version"],
                "camera_count": len([camera for camera in state["cameras"] if camera["status"] == "online"]),
            })
            return
        if path == "/api/unwrapped/wait":
            after = parse_qs(parsed.query).get("after", [""])[0]
            try:
                self.send_json({"result": SERVICE.wait_for_unwrapped(after)})
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass  # The tab closed during the bounded wait.
            return
        if path == "/api/unwrapped/latest.png":
            try:
                expected = parse_qs(parsed.query).get("pipe", [""])[0]
                payload, info = SERVICE.get_latest_unwrapped(expected)
            except Exception as exc:
                self.send_error_json(str(exc), status=404)
                return
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("X-Unwrapped-Width", str(info.get("width", "")))
            self.send_header("X-Unwrapped-Height", str(info.get("height", "")))
            self.end_headers()
            self.wfile.write(payload)
            return
        if path.startswith("/api/preview/") and path.endswith(".bmp"):
            try:
                slot = int(path.split("/")[-1].split(".")[0])
                data, info = SERVICE.get_preview(slot)
            except Exception:
                data, info = None, {}
            if not data:
                self.send_error_json("当前没有可用预览帧", status=404)
                return
            self.send_response(200)
            self.send_header("Content-Type", "image/bmp")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("X-Frame-Width", str(info.get("width", "")))
            self.send_header("X-Frame-Height", str(info.get("height", "")))
            self.end_headers()
            self.wfile.write(data)
            return
        super().do_GET()

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        try:
            body = self.read_json()
            if path == "/api/refresh":
                count = SERVICE.refresh_devices(quiet=False)
                self.send_json({"ok": True, "device_count": count, "state": SERVICE.get_state()})
                return
            if path == "/api/connect":
                self.send_json({"ok": True, "state": SERVICE.connect_all()})
                return
            if path == "/api/connect/one":
                self.send_json({"ok": True, "state": SERVICE.connect_one(int(body.get("slot", -1)))})
                return
            if path == "/api/disconnect":
                self.send_json({"ok": True, "state": SERVICE.disconnect_all()})
                return
            if path == "/api/disconnect/one":
                self.send_json({"ok": True, "state": SERVICE.disconnect_one(int(body.get("slot", -1)))})
                return
            if path == "/api/preview/start":
                self.send_json({"ok": True, "state": SERVICE.start_preview()})
                return
            if path == "/api/preview/stop":
                self.send_json({"ok": True, "state": SERVICE.stop_preview()})
                return
            if path == "/api/acquisition/start":
                self.send_json({"ok": True, "state": SERVICE.start_acquisition()})
                return
            if path == "/api/acquisition/stop":
                self.send_json({"ok": True, "state": SERVICE.stop_acquisition()})
                return
            if path == "/api/acquisition/trigger":
                self.send_json({"ok": True, "state": SERVICE.trigger_once()})
                return
            if path == "/api/config":
                self.send_json({"ok": True, "state": SERVICE.apply_config(body)})
                return
            if path == "/api/unwrap-size":
                self.send_json({"ok": True, "state": SERVICE.set_unwrap_size(body.get("width"), body.get("height"))})
                return
            if path == "/api/open-output":
                output = SERVICE.open_output_dir()
                self.send_json({"ok": True, "path": output})
                return
            if path == "/api/select-output-dir":
                selected = SERVICE.select_output_dir_dialog()
                self.send_json({"ok": True, "path": selected, "state": SERVICE.get_state()})
                return
            self.send_error_json("接口不存在", status=404)
        except ValueError as exc:
            self.send_error_json(str(exc), status=400)
        except RuntimeError as exc:
            self.send_error_json(str(exc), status=409)
        except Exception as exc:
            self.send_error_json(str(exc), status=500)


def shutdown_service() -> None:
    try:
        SERVICE.shutdown()
    except Exception:
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description="海康四相机图像采集控制台")
    parser.add_argument("--port", type=int, default=8765, help="首选端口，默认 8765")
    parser.add_argument("--no-browser", action="store_true", help="不自动打开浏览器")
    args = parser.parse_args()

    atexit.register(shutdown_service)
    port = find_available_port(args.port)
    handler = functools.partial(CaptureHandler, directory=str(ROOT))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", port), handler)
    url = f"http://127.0.0.1:{port}"

    print("=" * 68)
    print("四相机钢管图像采集控制台")
    print(f"访问地址: {url}")
    state = SERVICE.get_state()
    print(f"MVS SDK: {'就绪' if state['sdk_ready'] else '不可用'}  {state['sdk_version']}")
    print("关闭此窗口即可停止采集服务")
    print("=" * 68)

    if not args.no_browser:
        threading.Timer(0.45, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n服务已停止")
    finally:
        server.server_close()
        shutdown_service()


if __name__ == "__main__":
    main()
