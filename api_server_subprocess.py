#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
IndexTTS2 Web API 服务 - 子进程模式
批量处理，完全释放显存，支持SRT字幕
"""
import os
import sys
import io
import time
import uuid
import subprocess
import json
import logging
import hashlib
import signal
from pathlib import Path
from datetime import datetime
from typing import Optional
from queue import Queue
import threading
import traceback
from logging.handlers import RotatingFileHandler

# 设置环境变量
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

# 日志配置
LOG_DIR = Path("./logs")
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE_PATH = LOG_DIR / f"api_server_{datetime.now().strftime('%Y%m%d')}.log"

def setup_logging():
    """配置日志系统（安全处理已关闭的stdout）"""
    # Windows 控制台 UTF-8 支持（如果可用）
    if sys.platform == 'win32':
        for stream_name in ("stdout", "stderr"):
            try:
                stream_obj = getattr(sys, stream_name, None)
                if stream_obj and not getattr(stream_obj, "closed", False) and hasattr(stream_obj, "buffer"):
                    setattr(sys, stream_name, io.TextIOWrapper(stream_obj.buffer, encoding='utf-8', errors='replace'))
            except Exception:
                # 如果流已关闭或不可包装，忽略
                pass
    
    # 创建日志格式
    log_format = logging.Formatter(
        '[%(asctime)s] [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 创建根日志记录器
    logger = logging.getLogger('API')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    
    # 控制台处理器（仅当可用流存在时添加）
    console_stream = None
    if getattr(sys, "stdout", None) and not getattr(sys.stdout, "closed", False):
        console_stream = sys.stdout
    elif getattr(sys, "__stdout__", None) and not getattr(sys.__stdout__, "closed", False):
        console_stream = sys.__stdout__
    if console_stream:
        console_handler = logging.StreamHandler(console_stream)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(log_format)
        logger.addHandler(console_handler)
    
    # 文件处理器
    # 文件处理器
    try:
        file_handler = RotatingFileHandler(
            LOG_FILE_PATH,
            maxBytes=50*1024*1024,  # 50MB
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)
    except Exception as e:
        # 如果文件处理器初始化失败，至少告知用户
        fallback_stream = getattr(sys, "__stderr__", sys.stderr)
        try:
            print(f"[LOG] 无法创建日志文件 {LOG_FILE_PATH}: {e}", file=fallback_stream)
        except Exception:
            pass
    
    return logger

# 初始化日志
logger = setup_logging()

def _ensure_logger():
    """确保logger的handler未被关闭，必要时重新初始化"""
    global logger
    try:
        for handler in list(logger.handlers):
            stream = getattr(handler, "stream", None)
            if stream is not None and getattr(stream, "closed", False):
                logger.removeHandler(handler)
        if not logger.handlers:
            raise ValueError("logger handler closed")
    except Exception:
        logger = setup_logging()
    return logger

def log(message, level='INFO'):
    """统一日志函数，自动恢复被关闭的handler"""
    l = _ensure_logger()
    try:
        if level == 'INFO':
            l.info(message)
        elif level == 'ERROR':
            l.error(message)
        elif level == 'WARNING':
            l.warning(message)
        elif level == 'DEBUG':
            l.debug(message)
        else:
            l.info(message)
    except Exception:
        # 如果logger完全不可用，最后尝试输出到stderr
        logger = setup_logging()
        try:
            if level == 'INFO':
                logger.info(message)
            elif level == 'ERROR':
                logger.error(message)
            elif level == 'WARNING':
                logger.warning(message)
            elif level == 'DEBUG':
                logger.debug(message)
            else:
                logger.info(message)
        except Exception:
            try:
                fallback_stream = getattr(sys, "__stderr__", sys.stderr)
                print(message, file=fallback_stream)
            except Exception:
                pass

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field
import uvicorn
import shutil

# 配置
WORKER_SCRIPT = "tts_worker.py"
OUTPUT_DIR = Path("./outputs")
OUTPUT_DIR.mkdir(exist_ok=True)
SOURCE_DIR = Path("./source")
SOURCE_DIR.mkdir(exist_ok=True)
WEB_DIR = Path("./assets/web")
WEB_DIR.mkdir(parents=True, exist_ok=True)
WEB_INDEX_FILE = WEB_DIR / "index.html"

# API 请求模型
class TTSRequest(BaseModel):
    text: str = Field(..., description="要合成的文本内容", min_length=1, max_length=2000)
    target_duration: Optional[float] = Field(None, description="目标音频时长（秒）", ge=0.5, le=120.0)
    prompt_audio: Optional[str] = Field("tests/sample_prompt.wav", description="参考音频路径或音色名称（可通过/voices接口获取可用音色列表）")
    emo_text: Optional[str] = Field(None, description="情绪描述文本，用于控制语音的情感表达")
    force_split_at_newline: Optional[bool] = Field(False, description="是否在换行符处强制断句")
    device_mode: Optional[str] = Field("auto", description="设备模式：auto/gpu/gpu_cuda/cpu")

class TTSResponse(BaseModel):
    task_id: str
    status: str
    message: str
    audio_url: Optional[str] = None
    srt_url: Optional[str] = None

# 任务队列
task_queue = Queue()
task_results = {}
is_processing = False
processing_lock = threading.Lock()
current_worker_process = None  # 当前worker进程，用于强制释放
worker_process_lock = threading.Lock()  # worker进程锁
QUEUE_IDLE_TIMEOUT = 5  # 队列空闲后等待几秒再释放（秒）

# 缓存：存储相同请求的结果（作弊功能）
request_cache = {}
cache_lock = threading.Lock()
MAX_CACHE_SIZE = 500  # 最大缓存数量

def start_tts_worker():
    """启动TTS子进程"""
    cmd = [sys.executable, WORKER_SCRIPT]
    env = os.environ.copy()
    env['PYTHONIOENCODING'] = 'utf-8'
    
    process = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding='utf-8',
        errors='replace',
        env=env,
        bufsize=1
    )
    
    log(f"[WORKER] 子进程已启动，PID: {process.pid}")
    
    def forward_stderr():
        try:
            for line in iter(process.stderr.readline, ''):
                if not line:
                    break
                log(f"[WORKER] {line.strip()}", "ERROR")
        except Exception as e:
            log(f"[WORKER] 转发日志失败: {e}", "WARNING")
    
    threading.Thread(target=forward_stderr, daemon=True).start()
    time.sleep(0.5)  # 等待启动
    return process

def send_task_to_worker(process, task):
    """发送任务到子进程"""
    if process.poll() is not None or process.stdin.closed:
        raise RuntimeError("Worker进程不可用或stdin已关闭")
    task_data = {
        "task_id": task["task_id"],
        "text": task["text"],
        "output": str(OUTPUT_DIR / f"{task['task_id']}.wav"),
        "prompt": task.get("prompt_audio", "tests/sample_prompt.wav"),
        "target_duration": task.get("target_duration"),
        "emo_text": task.get("emo_text"),
        "force_split_at_newline": task.get("force_split_at_newline", False),
        "device_mode": task.get("device_mode", "auto")
    }
    
    task_json = json.dumps(task_data, ensure_ascii=False)
    process.stdin.write(task_json + '\n')
    process.stdin.flush()
    log(f"[WORKER] 已发送任务 {task['task_id']}")

def read_result_from_worker(process, timeout=600):
    """从子进程读取结果"""
    import queue as queue_module
    result_queue = queue_module.Queue()
    
    def read_with_timeout():
        try:
            start_time = time.time()
            max_lines = 1000
            
            for lines_read in range(max_lines):
                # 检查超时
                if time.time() - start_time > timeout:
                    log(f"[WORKER] 读取超时", 'ERROR')
                    result_queue.put(None)
                    return
                
                # 检查进程状态
                if process.poll() is not None:
                    log(f"[WORKER] 子进程已退出", 'ERROR')
                    result_queue.put(None)
                    return
                
                # 读取一行
                line = process.stdout.readline()
                if not line:
                    time.sleep(0.1)
                    continue
                
                line = line.strip()
                if not line:
                    continue
                
                # 尝试解析JSON
                try:
                    parsed_result = json.loads(line)
                    if isinstance(parsed_result, dict) and "task_id" in parsed_result:
                        result_queue.put(parsed_result)
                        return
                except json.JSONDecodeError:
                    continue
            
            result_queue.put(None)
            
        except Exception as e:
            log(f"[WORKER] 读取失败: {e}", 'ERROR')
            result_queue.put(None)
    
    # 创建读取线程
    read_thread = threading.Thread(target=read_with_timeout, daemon=True)
    read_thread.start()
    read_thread.join(timeout)
    
    if read_thread.is_alive():
        log(f"[WORKER] 读取超时", 'ERROR')
        raise RuntimeError("Worker timeout")
    
    try:
        result = result_queue.get_nowait()
    except:
        result = None
    
    if result is None:
        raise RuntimeError("Failed to read result from worker")
    
    return result

def shutdown_worker(process):
    """关闭子进程"""
    try:
        shutdown_cmd = json.dumps({"command": "shutdown"})
        process.stdin.write(shutdown_cmd + '\n')
        process.stdin.flush()
        process.wait(timeout=30)
        log("[WORKER] 子进程已退出")
    except:
        try:
            process.terminate()
            process.wait(timeout=5)
        except:
            process.kill()
        log("[WORKER] 子进程已强制终止")

def force_shutdown_worker_process():
    """强制终止当前worker进程（用于退出时）"""
    global current_worker_process
    with worker_process_lock:
        proc = current_worker_process
        current_worker_process = None
    if proc and proc.poll() is None:
        try:
            proc.terminate()
            proc.wait(timeout=3)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass

def process_queue():
    """处理任务队列"""
    global is_processing, current_worker_process
    
    worker_process = None
    try:
        def start_worker():
            global current_worker_process
            proc = start_tts_worker()
            with worker_process_lock:
                current_worker_process = proc
            return proc

        log("[QUEUE] 启动子进程")
        worker_process = start_worker()
        
        while True:
            try:
                task = task_queue.get(timeout=1.0)
            except:
                # 队列空，检查是否真的结束
                if task_queue.empty():
                    time.sleep(0.5)
                    if task_queue.empty():
                        log(f"[QUEUE] 队列已清空，等待 {QUEUE_IDLE_TIMEOUT} 秒后释放内存")
                        # 等待指定时间，期间如果有新任务会继续处理
                        idle_start = time.time()
                        should_exit = False
                        while time.time() - idle_start < QUEUE_IDLE_TIMEOUT:
                            time.sleep(0.5)
                            # 检查是否有新任务
                            if not task_queue.empty():
                                log("[QUEUE] 检测到新任务，继续处理")
                                break
                            # 检查worker进程是否被外部强制终止
                            with worker_process_lock:
                                if current_worker_process is None:
                                    log("[QUEUE] Worker进程已被外部终止，退出处理")
                                    should_exit = True
                                    break
                                if current_worker_process != worker_process:
                                    log("[QUEUE] Worker进程已被替换，退出处理")
                                    should_exit = True
                                    break
                            # 检查worker进程是否已退出
                            if worker_process.poll() is not None:
                                log("[QUEUE] Worker进程已退出，退出处理")
                                should_exit = True
                                break
                        else:
                            # 等待时间到，正常关闭
                            if not should_exit:
                                log("[QUEUE] 等待时间到，释放内存")
                                shutdown_worker(worker_process)
                                with processing_lock:
                                    is_processing = False
                                with worker_process_lock:
                                    if current_worker_process == worker_process:
                                        current_worker_process = None
                                break
                        
                        # 如果被外部终止或进程退出，清理状态并退出
                        if should_exit:
                            # 确保worker进程已关闭
                            if worker_process.poll() is None:
                                try:
                                    worker_process.terminate()
                                    worker_process.wait(timeout=2)
                                except:
                                    try:
                                        worker_process.kill()
                                    except:
                                        pass
                            with processing_lock:
                                is_processing = False
                            with worker_process_lock:
                                if current_worker_process == worker_process:
                                    current_worker_process = None
                            break
                continue
            
            task_id = task["task_id"]
            log(f"[QUEUE] 处理任务 {task_id}")
            
            try:
                task_results[task_id]["status"] = "processing"
                try:
                    send_task_to_worker(worker_process, task)
                except RuntimeError as re:
                    log(f"[QUEUE] Worker不可用，尝试重启: {re}", "WARNING")
                    # 尝试重启worker并再次发送
                    try:
                        if worker_process and worker_process.poll() is None:
                            worker_process.terminate()
                    except Exception:
                        pass
                    worker_process = start_worker()
                    send_task_to_worker(worker_process, task)
                result = read_result_from_worker(worker_process)
                
                if result["status"] == "success":
                    task_results[task_id]["status"] = "completed"
                    task_results[task_id]["result"] = result
                    task_results[task_id]["completed_at"] = time.time()
                    
                    # 保存到缓存（作弊功能）
                    cache_key = task_results[task_id].get("cache_key")
                    if cache_key:
                        with cache_lock:
                            request_cache[cache_key] = {
                                "task_id": task_id,
                                "completed_at": task_results[task_id]["completed_at"],
                                "cached_at": time.time()  # 缓存时间戳，用于清理
                            }
                            # 清理缓存（如果超过最大数量）
                            cleanup_cache()
                        log(f"[CACHE] 任务 {task_id} 结果已保存到缓存: {cache_key}")
                    
                    log(f"[QUEUE] 任务 {task_id} 完成")
                else:
                    task_results[task_id]["status"] = "failed"
                    error_msg = result.get("error", "").strip()
                    # 清理错误信息中的路径，避免暴露绝对路径
                    if error_msg:
                        # 移除可能的路径信息
                        if os.path.sep in error_msg or os.path.altsep in error_msg:
                            # 如果包含路径分隔符，只保留错误描述部分
                            error_msg = "处理失败，请检查输入参数"
                    task_results[task_id]["error"] = error_msg if error_msg else "Unknown error"
                    log(f"[QUEUE] 任务 {task_id} 失败: {task_results[task_id]['error']}", 'ERROR')
                    
            except Exception as e:
                log(f"[QUEUE] 任务 {task_id} 异常: {e}", 'ERROR')
                log(traceback.format_exc(), 'ERROR')
                task_results[task_id]["status"] = "failed"
                # 清理异常信息中的路径
                error_str = str(e)
                if os.path.sep in error_str or os.path.altsep in error_str:
                    error_str = "处理异常，请稍后重试"
                task_results[task_id]["error"] = error_str
                
                # 子进程异常，退出
                if "timeout" in str(e).lower() or "worker" in str(e).lower():
                    try:
                        worker_process.terminate()
                        worker_process.wait(timeout=5)
                    except:
                        worker_process.kill()
                    with processing_lock:
                        is_processing = False
                    with worker_process_lock:
                        current_worker_process = None
                    break
            finally:
                task_queue.task_done()
                
    except Exception as e:
        log(f"[QUEUE] 队列异常: {e}", 'ERROR')
        with processing_lock:
            is_processing = False
        with worker_process_lock:
            current_worker_process = None
    finally:
        if worker_process and worker_process.poll() is None:
            try:
                worker_process.terminate()
                worker_process.wait(timeout=5)
            except:
                worker_process.kill()
        with worker_process_lock:
            if current_worker_process == worker_process:
                current_worker_process = None

def cleanup_cache():
    """清理缓存：当缓存数量超过最大值时，删除最早的缓存项（LRU策略）"""
    if len(request_cache) <= MAX_CACHE_SIZE:
        return
    
    # 按缓存时间排序，获取最早的缓存项（LRU：最近最少使用）
    # 对于没有cached_at的旧缓存项，使用completed_at或0作为默认值
    sorted_cache = sorted(
        request_cache.items(),
        key=lambda x: x[1].get("cached_at", x[1].get("completed_at", 0))
    )
    
    # 计算需要删除的数量
    to_remove = len(request_cache) - MAX_CACHE_SIZE
    
    # 删除最早的缓存项
    removed_count = 0
    for cache_key, cache_info in sorted_cache[:to_remove]:
        task_id = cache_info.get("task_id")
        
        # 删除缓存项
        if cache_key in request_cache:
            del request_cache[cache_key]
            removed_count += 1
        
        # 可选：删除对应的文件（如果需要释放磁盘空间）
        # 注意：这里不删除文件，因为可能有其他任务引用
        # 如果需要删除文件，可以取消下面的注释
        # audio_path = OUTPUT_DIR / f"{task_id}.wav"
        # srt_path = OUTPUT_DIR / f"{task_id}.srt"
        # try:
        #     if audio_path.exists():
        #         audio_path.unlink()
        #     if srt_path.exists():
        #         srt_path.unlink()
        # except Exception as e:
        #     log(f"[CACHE] 删除缓存文件失败: {e}", 'WARNING')
    
    if removed_count > 0:
        log(f"[CACHE] 清理缓存: 删除了 {removed_count} 个最早的缓存项，当前缓存数量: {len(request_cache)}")

def cleanup_outputs():
    """删除生成的音频与字幕文件"""
    removed_files = 0
    for file_path in OUTPUT_DIR.glob("*"):
        # 只删除我们生成的wav/srt，其他文件夹也允许删除以清理嵌套输出
        if file_path.is_file() and file_path.suffix.lower() in {".wav", ".srt"}:
            try:
                file_path.unlink()
                removed_files += 1
            except Exception as e:
                log(f"[CLEAN] 删除文件失败 {file_path}: {e}", "WARNING")
        elif file_path.is_dir():
            try:
                shutil.rmtree(file_path)
                removed_files += 1
            except Exception as e:
                log(f"[CLEAN] 删除目录失败 {file_path}: {e}", "WARNING")
    return removed_files

def reset_task_data():
    """清空任务和缓存记录"""
    cleared_tasks = len(task_results)
    task_results.clear()
    request_cache.clear()
    return cleared_tasks

def add_task(task_id: str, text: str, target_duration: Optional[float], prompt_audio: str, emo_text: Optional[str] = None, force_split_at_newline: bool = False, cache_key: Optional[str] = None, device_mode: str = "auto"):
    """添加任务"""
    global is_processing
    
    log(f"[ADD] 任务 {task_id}: {text[:50]}...")
    
    task = {
        "task_id": task_id,
        "text": text,
        "target_duration": target_duration,
        "prompt_audio": prompt_audio,
        "emo_text": emo_text,
        "force_split_at_newline": force_split_at_newline,
        "device_mode": device_mode
    }
    
    task_results[task_id] = {
        "status": "pending",
        "text": text,
        "target_duration": target_duration,
        "created_at": time.time(),
        "result": None,
        "error": None,
        "cache_key": cache_key,  # 保存缓存键
        "device_mode": device_mode
    }
    
    task_queue.put(task)
    log(f"[ADD] 任务 {task_id} 已加入队列，队列长度: {task_queue.qsize()}")
    
    # 启动处理线程
    with processing_lock:
        if not is_processing:
            is_processing = True
            log("[ADD] 启动处理线程")
            thread = threading.Thread(target=process_queue, daemon=True)
            thread.start()

# FastAPI 应用
app = FastAPI(
    title="IndexTTS2 API",
    description="文本转语音 API 服务（子进程模式，支持SRT字幕）",
    version="2.0"
)

@app.get("/")
async def root():
    return {
        "service": "IndexTTS2 TTS API",
        "version": "2.0",
        "status": "running",
        "mode": "subprocess",
        "features": ["audio", "srt_subtitle", "voices"],
        "queue_size": task_queue.qsize()
    }

@app.get("/web", response_class=HTMLResponse)
async def web_ui():
    """提供简单的网页界面，不影响现有API"""
    try:
        if WEB_INDEX_FILE.exists():
            return HTMLResponse(WEB_INDEX_FILE.read_text(encoding="utf-8"))
        
        fallback = """
        <!doctype html>
        <html><body><h2>IndexTTS2 Web 页面未找到</h2><p>请检查 assets/web/index.html 是否存在。</p></body></html>
        """
        return HTMLResponse(fallback, status_code=200)
    except Exception as e:
        log(f"[WEB] 加载页面失败: {e}", 'ERROR')
        raise HTTPException(500, "加载网页失败")

@app.get("/status")
async def get_status():
    return {
        "queue_size": task_queue.qsize(),
        "is_processing": is_processing,
        "total_tasks": len(task_results)
    }

@app.post("/tts", response_model=TTSResponse)
async def text_to_speech(request: TTSRequest):
    try:
        log(f"[API] TTS请求: {request.text[:50]}...")
        allowed_modes = {"auto", "gpu", "gpu_cuda", "cpu"}
        device_mode = (request.device_mode or "auto").lower()
        if device_mode not in allowed_modes:
            raise HTTPException(400, f"设备模式不支持: {request.device_mode}")
        
        # 尝试查找音色文件（支持音色名称或路径）
        prompt_audio_path = find_voice_file(request.prompt_audio)
        
        if not prompt_audio_path:
            # 如果找不到，尝试使用原始路径
            if os.path.exists(request.prompt_audio):
                prompt_audio_path = os.path.abspath(request.prompt_audio)
            else:
                # 清理路径信息，避免暴露绝对路径
                safe_path = sanitize_path_for_client(request.prompt_audio)
                raise HTTPException(400, f"参考音频不存在: {safe_path}。请使用/voices接口查看可用音色列表")
        
        log(f"[API] 使用参考音频: {prompt_audio_path}")
        
        # 检查缓存（作弊功能）
        cache_key = generate_cache_key(
            request.text,
            prompt_audio_path,
            request.target_duration,
            request.emo_text,
            request.force_split_at_newline,
            device_mode
        )
        
        with cache_lock:
            if cache_key in request_cache:
                cached_result = request_cache[cache_key]
                cached_task_id = cached_result["task_id"]
                
                # 检查缓存的文件是否还存在
                audio_path = OUTPUT_DIR / f"{cached_task_id}.wav"
                srt_path = OUTPUT_DIR / f"{cached_task_id}.srt"
                
                if audio_path.exists() and srt_path.exists():
                    # 更新缓存时间戳（LRU策略：最近使用的缓存不会被删除）
                    request_cache[cache_key]["cached_at"] = time.time()
                    log(f"[CACHE] 缓存命中，直接返回结果: {cached_task_id}")
                    return TTSResponse(
                        task_id=cached_task_id,
                        status="completed",
                        message="从缓存返回",
                        audio_url=f"/download/{cached_task_id}",
                        srt_url=f"/download/{cached_task_id}/srt"
                    )
                else:
                    # 缓存的文件不存在，删除缓存项
                    log(f"[CACHE] 缓存文件不存在，删除缓存项: {cache_key}")
                    del request_cache[cache_key]
        
        # 缓存未命中，创建新任务
        task_id = str(uuid.uuid4())
        
        # 在任务信息中保存缓存键
        add_task(task_id, request.text, request.target_duration, prompt_audio_path, request.emo_text, request.force_split_at_newline, cache_key, device_mode)
        
        return TTSResponse(
            task_id=task_id,
            status="pending",
            message="任务已提交",
            audio_url=None,
            srt_url=None
        )
    except HTTPException:
        raise
    except Exception as e:
        log(f"[API] 错误: {e}", 'ERROR')
        raise HTTPException(500, str(e))

@app.get("/task/{task_id}")
async def get_task(task_id: str):
    if task_id not in task_results:
        raise HTTPException(404, "任务不存在")
    
    info = task_results[task_id]
    response = {
        "task_id": task_id,
        "status": info["status"],
        "created_at": info["created_at"],
        "text": info["text"][:100],
        "device_mode": info.get("device_mode", "auto")
    }
    
    if info["status"] == "completed":
        result = info["result"]
        response.update({
            "completed_at": info.get("completed_at"),
            "audio_url": f"/download/{task_id}",
            "srt_url": f"/download/{task_id}/srt",
            "audio_duration": result["audio_duration"],
            "generation_time": result["generation_time"],
            "rtf": result["rtf"]
        })
    elif info["status"] == "failed":
        response["error"] = info.get("error", "Unknown")
    
    return response

@app.get("/tasks")
async def list_tasks():
    tasks = []
    for tid, info in task_results.items():
        tasks.append({
            "task_id": tid,
            "status": info["status"],
            "text": info["text"][:50] + "..." if len(info["text"]) > 50 else info["text"],
            "created_at": info["created_at"],
            "device_mode": info.get("device_mode", "auto")
        })
    
    tasks.sort(key=lambda x: x["created_at"], reverse=True)
    return {"total": len(tasks), "tasks": tasks}

@app.post("/cleanup")
async def cleanup_data(clear_cache: bool = True, clear_outputs_flag: bool = True):
    """
    清理缓存与生成文件
    - clear_cache: 清空任务记录和缓存
    - clear_outputs_flag: 删除输出目录中的音频/字幕文件
    """
    summary = {}
    if clear_cache:
        removed_tasks = reset_task_data()
        summary["cleared_tasks"] = removed_tasks
        log(f"[CLEAN] 已清空任务和缓存，共 {removed_tasks} 条")
    
    if clear_outputs_flag:
        removed_files = cleanup_outputs()
        summary["removed_files"] = removed_files
        log(f"[CLEAN] 已删除输出文件 {removed_files} 个")
    
    return {"status": "success", "summary": summary}

@app.post("/free")
async def free_memory():
    """立即释放内存，终止worker进程（不等待队列）"""
    global current_worker_process, is_processing
    
    with worker_process_lock:
        if current_worker_process is None:
            return {
                "status": "success",
                "message": "没有运行中的worker进程",
                "memory_freed": False
            }
        
        worker_process = current_worker_process
        current_worker_process = None
    
    try:
        log("[FREE] 立即释放内存，终止worker进程")
        
        # 尝试优雅关闭
        try:
            shutdown_cmd = json.dumps({"command": "shutdown"})
            worker_process.stdin.write(shutdown_cmd + '\n')
            worker_process.stdin.flush()
            worker_process.wait(timeout=2)  # 只等待2秒
            log("[FREE] Worker进程已优雅关闭")
        except:
            # 优雅关闭失败，强制终止
            try:
                worker_process.terminate()
                worker_process.wait(timeout=2)
                log("[FREE] Worker进程已终止")
            except:
                worker_process.kill()
                log("[FREE] Worker进程已强制终止")
        
        with processing_lock:
            is_processing = False
        
        return {
            "status": "success",
            "message": "内存已立即释放",
            "memory_freed": True
        }
    except Exception as e:
        log(f"[FREE] 释放内存失败: {e}", 'ERROR')
        return {
            "status": "error",
            "message": f"释放内存失败: {str(e)}",
            "memory_freed": False
        }

@app.get("/download/{task_id}")
async def download_audio(task_id: str):
    """下载音频文件"""
    audio_path = OUTPUT_DIR / f"{task_id}.wav"
    if not audio_path.exists():
        raise HTTPException(404, f"音频文件不存在")
    
    return FileResponse(audio_path, media_type="audio/wav", filename=f"{task_id}.wav")

@app.get("/download/{task_id}/srt")
async def download_srt(task_id: str):
    """下载SRT字幕文件"""
    srt_path = OUTPUT_DIR / f"{task_id}.srt"
    if not srt_path.exists():
        raise HTTPException(404, f"字幕文件不存在")
    
    return FileResponse(srt_path, media_type="text/plain", filename=f"{task_id}.srt")

@app.get("/voice/{voice_name}")
async def download_voice_sample(voice_name: str):
    """根据音色名称下载语音样本"""
    try:
        # 查找音色文件
        voice_file_path = find_voice_file(voice_name)
        
        if not voice_file_path:
            raise HTTPException(404, f"音色 '{voice_name}' 不存在。请使用/voices接口查看可用音色列表")
        
        voice_path = Path(voice_file_path)
        
        if not voice_path.exists():
            raise HTTPException(404, f"音色文件不存在: {voice_name}")
        
        # 确定媒体类型
        media_types = {
            '.wav': 'audio/wav',
            '.mp3': 'audio/mpeg',
            '.flac': 'audio/flac',
            '.ogg': 'audio/ogg',
            '.m4a': 'audio/mp4',
            '.aac': 'audio/aac',
            '.wma': 'audio/x-ms-wma'
        }
        media_type = media_types.get(voice_path.suffix.lower(), 'audio/wav')
        
        log(f"[API] 下载音色样本: {voice_name} -> {voice_path.name}")
        
        return FileResponse(
            voice_path,
            media_type=media_type,
            filename=voice_path.name
        )
    except HTTPException:
        raise
    except Exception as e:
        log(f"[API] 下载音色样本错误: {e}", 'ERROR')
        # 避免在错误消息中暴露路径信息
        error_msg = str(e)
        # 移除可能的路径信息
        if os.path.sep in error_msg or os.path.altsep in error_msg:
            error_msg = "文件访问错误"
        raise HTTPException(500, f"下载音色样本失败: {error_msg}")

def find_voice_file(voice_input: str) -> Optional[str]:
    """
    根据音色名称或路径查找音频文件
    
    Args:
        voice_input: 音色名称（不含扩展名）或文件路径
        
    Returns:
        找到的文件路径，如果未找到则返回None
    """
    if not voice_input or not voice_input.strip():
        return None
    
    voice_input = voice_input.strip()
    
    # 如果已经是完整路径且文件存在，直接返回
    if os.path.exists(voice_input):
        return os.path.abspath(voice_input)
    
    # 在source目录中查找音色
    if not SOURCE_DIR.exists():
        return None
    
    audio_extensions = {'.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac', '.wma'}
    voice_name = voice_input
    
    # 移除可能的扩展名
    if '.' in voice_name:
        voice_name = Path(voice_name).stem
    
    # 在source目录中递归查找匹配的文件
    for file_path in SOURCE_DIR.rglob('*'):
        if file_path.is_file() and file_path.suffix.lower() in audio_extensions:
            # 匹配文件名（不含扩展名）
            if file_path.stem.lower() == voice_name.lower():
                return str(file_path.absolute())
            # 也匹配完整文件名
            if file_path.name.lower() == voice_input.lower():
                return str(file_path.absolute())
    
    return None

@app.post("/voices/upload")
async def upload_voice(file: UploadFile = File(...)):
    """
    上传音色文件，仅支持wav且大小限制为2MB
    """
    MAX_SIZE = 2 * 1024 * 1024  # 2MB
    
    try:
        if not file.filename:
            raise HTTPException(400, "未提供文件名")
        
        if Path(file.filename).suffix.lower() != ".wav":
            raise HTTPException(400, "仅支持wav文件")
        
        safe_name = sanitize_filename(file.filename)
        target_path = SOURCE_DIR / f"{safe_name}_{uuid.uuid4().hex[:8]}.wav"
        
        size = 0
        with target_path.open("wb") as f_out:
            while True:
                chunk = await file.read(512 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                if size > MAX_SIZE:
                    f_out.close()
                    target_path.unlink(missing_ok=True)
                    raise HTTPException(400, "文件超过2MB限制")
                f_out.write(chunk)
        
        await file.close()
        log(f"[API] 上传音色: {target_path.name} ({size} bytes)")
        return {
            "status": "success",
            "voice": target_path.stem,
            "filename": target_path.name,
            "size": size
        }
    except HTTPException:
        raise
    except Exception as e:
        log(f"[API] 上传音色失败: {e}", 'ERROR')
        raise HTTPException(500, f"上传失败: {e}")

def generate_cache_key(text: str, prompt_audio_path: str, target_duration: Optional[float], 
                      emo_text: Optional[str], force_split_at_newline: bool, device_mode: str = "auto") -> str:
    """
    生成请求的缓存键
    
    Args:
        text: 文本内容
        prompt_audio_path: 参考音频路径（已标准化）
        target_duration: 目标时长
        emo_text: 情绪文本
        force_split_at_newline: 是否强制换行断句
        
    Returns:
        缓存键（MD5哈希）
    """
    # 标准化参数
    cache_data = {
        "text": text.strip(),
        "prompt_audio": prompt_audio_path,
        "target_duration": target_duration,
        "emo_text": emo_text.strip() if emo_text else None,
        "force_split_at_newline": force_split_at_newline,
        "device_mode": device_mode
    }
    
    # 生成JSON字符串并计算MD5
    cache_str = json.dumps(cache_data, sort_keys=True, ensure_ascii=False)
    cache_key = hashlib.md5(cache_str.encode('utf-8')).hexdigest()
    
    return cache_key

def sanitize_path_for_client(path: str) -> str:
    """
    清理路径信息，避免向客户端暴露绝对路径
    
    Args:
        path: 文件路径
        
    Returns:
        清理后的路径（只显示文件名或相对路径）
    """
    if not path:
        return ""
    
    # 如果是绝对路径，只返回文件名
    if os.path.isabs(path):
        return os.path.basename(path)
    
    # 如果是相对路径，返回相对路径（但限制深度）
    # 避免暴露太多目录结构
    path_parts = Path(path).parts
    if len(path_parts) > 3:
        # 如果路径太深，只显示最后3层
        return str(Path(*path_parts[-3:]))
    
    return path

def sanitize_filename(filename: str) -> str:
    """
    生成安全的文件名（仅保留字母、数字、下划线和中划线）
    """
    name = Path(filename).stem
    safe = ''.join(ch for ch in name if ch.isalnum() or ch in ('_', '-'))
    return safe or "voice"

def calculate_file_hash(file_path: Path) -> str:
    """
    计算文件的MD5哈希值
    
    Args:
        file_path: 文件路径
        
    Returns:
        MD5哈希值（十六进制字符串）
    """
    hash_md5 = hashlib.md5()
    try:
        with open(file_path, "rb") as f:
            # 分块读取，避免大文件占用过多内存
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    except Exception as e:
        log(f"[HASH] 计算文件哈希失败 {file_path}: {e}", 'WARNING')
        return ""

@app.get("/voices")
async def list_available_voices():
    """获取当前服务可用的音色列表（source目录下的所有语音文件）"""
    try:
        voices = []
        audio_extensions = {'.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac', '.wma'}
        
        if not SOURCE_DIR.exists():
            return {
                "voices": []
            }
        
        for file_path in SOURCE_DIR.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in audio_extensions:
                # 使用文件名（不含扩展名）作为音色名称
                voice_name = file_path.stem
                # 计算文件哈希值
                file_hash = calculate_file_hash(file_path)
                voices.append({
                    "name": voice_name,
                    "hash": file_hash
                })
        
        # 按音色名称排序
        voices.sort(key=lambda x: x["name"])
        
        return {
            "voices": voices
        }
    except Exception as e:
        log(f"[API] 获取音色列表错误: {e}", 'ERROR')
        raise HTTPException(500, f"获取音色列表失败: {str(e)}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="IndexTTS2 API 服务")
    parser.add_argument("--host", default="0.0.0.0", help="服务地址")
    parser.add_argument("--port", type=int, default=8001, help="服务端口")
    
    args = parser.parse_args()
    
    log("=" * 60)
    log("IndexTTS2 API 服务启动")
    log("=" * 60)
    log(f"服务地址: http://{args.host}:{args.port}")
    log(f"API文档: http://{args.host}:{args.port}/docs")
    log(f"运行模式: 子进程批量处理")
    log(f"输出目录: {OUTPUT_DIR.absolute()}")
    log(f"日志目录: {LOG_DIR.absolute()}")
    log(f"日志文件: {LOG_FILE_PATH.absolute()}")
    log("=" * 60)
    
    def handle_exit(sig=None, frame=None):
        log("收到退出信号，正在停止服务...", "WARNING")
        force_shutdown_worker_process()
        sys.exit(0)
    
    # 捕获Ctrl+C / SIGTERM（改进的 Windows 兼容性）
    try:
        signal.signal(signal.SIGINT, handle_exit)
        signal.signal(signal.SIGTERM, handle_exit)
        # Windows 特定：CTRL+BREAK
        if hasattr(signal, 'SIGBREAK'):
            signal.signal(signal.SIGBREAK, handle_exit)
    except Exception as e:
        log(f"设置信号处理器失败: {e}", "WARNING")
    
    try:
        # 使用 uvicorn.Server 而不是 uvicorn.run，以便更好地处理退出
        import asyncio
        config = uvicorn.Config(
            "api_server_subprocess:app",
            host=args.host,
            port=args.port,
            reload=False,
            access_log=False,
            log_level="info"
        )
        server = uvicorn.Server(config)
        
        # 运行服务器
        asyncio.run(server.serve())
    except KeyboardInterrupt:
        log("检测到 KeyboardInterrupt", "WARNING")
        handle_exit()
    except Exception as e:
        log(f"服务器异常: {e}", "ERROR")
        handle_exit()
