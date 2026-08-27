import sys
import os
import subprocess
import signal
import threading
import time
import uvicorn


def start_frontend():
    """启动前端 Vite 开发服务器"""
    frontend_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontend")
    if not os.path.exists(frontend_dir):
        print("[!] 前端目录不存在，跳过前端启动")
        return None

    print("[*] Starting frontend dev server (Vite)...")
    try:
        # Windows 使用 npm.cmd，Linux/Mac 使用 npm
        npm_cmd = "npm.cmd" if sys.platform == "win32" else "npm"
        process = subprocess.Popen(
            [npm_cmd, "run", "dev"],
            cwd=frontend_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
            if sys.platform == "win32"
            else 0,
        )

        # 打印前端日志
        def print_frontend_output():
            for line in process.stdout:
                print(f"[Frontend] {line.strip()}")

        thread = threading.Thread(target=print_frontend_output, daemon=True)
        thread.start()

        return process
    except FileNotFoundError:
        print("[!] 未找到 npm，请确保已安装 Node.js")
        return None


def start_backend():
    """启动后端 FastAPI 服务器"""
    print(f"[*] Starting backend server (FastAPI) with {sys.executable}")
    uvicorn.run(
        "app.main:app",
        host="127.0.0.1",
        port=5000,
        reload=True,
        log_level="info",
    )


def cleanup(processes):
    """清理所有子进程"""
    print("\n[*] Shutting down all servers...")
    for proc in processes:
        if proc and proc.poll() is None:
            try:
                if sys.platform == "win32":
                    # Windows: 使用 taskkill 强制终止进程树，避免 npm.cmd 的 Ctrl+C 确认提示
                    subprocess.run(
                        ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                        capture_output=True,
                        timeout=5,
                    )
                else:
                    proc.terminate()
                proc.wait(timeout=5)
            except Exception:
                proc.kill()
    print("[*] All servers stopped.")


if __name__ == "__main__":
    processes = []

    try:
        # 启动前端
        frontend_proc = start_frontend()
        if frontend_proc:
            processes.append(frontend_proc)

        # 等待前端启动
        if frontend_proc:
            print("[*] Waiting for frontend to initialize...")
            time.sleep(2)

        # 打印访问地址
        print("\n" + "=" * 50)
        print("[*] Servers started:")
        print("    Frontend: http://127.0.0.1:5173")
        print("    Backend:  http://127.0.0.1:5000")
        print("    API Docs: http://127.0.0.1:5000/docs")
        print("=" * 50 + "\n")

        # 启动后端（阻塞）
        start_backend()

    except KeyboardInterrupt:
        print("\n[*] Received interrupt signal")
    finally:
        cleanup(processes)
