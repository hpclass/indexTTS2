#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TTS Worker - 子进程推理脚本
在独立进程中运行，批量处理任务，队列清空后释放所有显存
"""
import os
import sys
import io
import json
import time
from pathlib import Path
from datetime import datetime

# 设置日志目录
LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

# 创建worker专用日志文件
worker_log_file = LOG_DIR / f"tts_worker_{datetime.now().strftime('%Y%m%d')}.log"
worker_log = open(worker_log_file, 'a', encoding='utf-8', buffering=1)

# 重定向stderr到日志文件（保留stdout用于JSON输出）
sys.stderr = worker_log

# Windows UTF-8 支持
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# 设置环境变量
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

import torch
import gc
import re

def log_worker(msg):
    """Worker日志函数，带时间戳"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{timestamp}] [WORKER] {msg}", file=sys.stderr, flush=True)

def split_text_to_segments(text, min_chars=3, max_chars=12):
    """
    将文本分割成小段（3-12字）
    在停顿符号处切分，同时控制长度
    """
    # 一级停顿符号（句子结束）
    primary_punctuation = '。！？；!?;'
    # 二级停顿符号（短暂停顿）
    secondary_punctuation = '，、,、'
    # 所有标点符号（用于清理末尾）
    all_punctuation = primary_punctuation + secondary_punctuation + '….'
    
    segments = []
    current_segment = ""
    
    for char in text:
        current_segment += char
        current_len = len(current_segment.strip())
        
        # 遇到一级停顿符号
        if char in primary_punctuation:
            if current_len >= min_chars:
                segments.append(current_segment.strip())
                current_segment = ""
            # 如果太短，继续积累
            
        # 遇到二级停顿符号
        elif char in secondary_punctuation:
            # 如果已经积累了足够的字符
            if current_len >= min_chars:
                segments.append(current_segment.strip())
                current_segment = ""
            # 如果太短，继续积累
            
        # 如果当前段落太长，强制在二级停顿符号后切分
        elif current_len > max_chars:
            # 尝试在最近的停顿符号处切分
            last_punct_pos = -1
            for i in range(len(current_segment)-1, -1, -1):
                if current_segment[i] in secondary_punctuation:
                    last_punct_pos = i
                    break
            
            if last_punct_pos > 0:
                # 在停顿符号处切分
                segments.append(current_segment[:last_punct_pos+1].strip())
                current_segment = current_segment[last_punct_pos+1:]
            else:
                # 没有找到停顿符号，强制切分
                segments.append(current_segment[:max_chars].strip())
                current_segment = current_segment[max_chars:]
    
    # 处理剩余部分
    if current_segment.strip():
        if segments and len(current_segment.strip()) < min_chars:
            # 如果最后一段太短，合并到前一段
            segments[-1] = segments[-1] + current_segment.strip()
        else:
            segments.append(current_segment.strip())
    
    # 清理每段末尾的标点符号
    cleaned_segments = []
    for seg in segments:
        # 移除末尾的所有标点符号
        while seg and seg[-1] in all_punctuation:
            seg = seg[:-1]
        # 只保留非空的段
        if seg:
            cleaned_segments.append(seg)
    
    return cleaned_segments if cleaned_segments else [text.rstrip(all_punctuation)]

def format_srt_time(seconds):
    """将秒数格式化为SRT时间格式 (HH:MM:SS,mmm)"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millisecs = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millisecs:03d}"

def clean_subtitle_text(text):
    """
    清理字幕文本：
    1. 替换引号、逗号、冒号等符号为空格
    2. 多个空格合并成一个空格
    """
    # 需要替换为空格的符号
    symbols_to_replace = [
        '"', "'", '"', '"', ''', ''',  # 各种引号
        ',', '，',  # 逗号
        ':', '：',  # 冒号
        ';', '；',  # 分号
        '(', ')', '（', '）',  # 括号
        '[', ']', '【', '】',  # 方括号
        '{', '}',  # 花括号
        '<', '>',  # 尖括号
        '/', '\\',  # 斜杠
        '|',  # 竖线
        '-', '—', '–',  # 各种横线
        '+', '=',  # 加号等号
        '*', '&', '%', '$', '#', '@', '!', '~', '`',  # 其他符号
        '…',  # 省略号
    ]
    
    # 替换所有符号为空格
    for symbol in symbols_to_replace:
        text = text.replace(symbol, ' ')
    
    # 合并多个空格为一个空格
    text = re.sub(r'\s+', ' ', text)
    
    # 去除首尾空格
    text = text.strip()
    
    return text

def cleanup_model(model):
    """释放模型并清理显存"""
    try:
        del model
    except Exception:
        pass
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        except Exception:
            pass

def load_model_for_mode(device_mode: str):
    """根据设备模式加载模型"""
    from indextts.infer_indextts2 import IndexTTS2

    mode = (device_mode or "auto").lower()
    log_worker(f"加载模型，模式: {mode}")

    device = None
    use_cuda_kernel = None

    if mode == "cpu":
        device = "cpu"
        use_cuda_kernel = False
    elif mode == "gpu":
        if not torch.cuda.is_available():
            raise RuntimeError("当前环境无可用CUDA")
        device = "cuda:0"
        use_cuda_kernel = False
    elif mode == "gpu_cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("当前环境无可用CUDA")
        device = "cuda:0"
        use_cuda_kernel = True
    else:
        # auto：交给IndexTTS2自行判断
        device = None
        use_cuda_kernel = None

    model = IndexTTS2(
        cfg_path="checkpoints/config.yaml",
        model_dir="checkpoints",
        is_fp16=False,
        device=device,
        use_cuda_kernel=use_cuda_kernel
    )
    log_worker(f"模型加载完成，device={model.device}, use_cuda_kernel={model.use_cuda_kernel}")
    return model, mode

def generate_srt_with_tts_sentences(text, audio_duration, output_path, tts_sentence_texts):
    """
    生成SRT字幕文件（直接使用IndexTTS的分句结果）
    
    Args:
        text: 原始完整文本
        audio_duration: 音频总时长（秒）
        output_path: 输出音频文件路径
        tts_sentence_texts: IndexTTS实际使用的分句列表（已经从token转回文本）
    """
    try:
        # 字幕提前显示的时间（秒）
        SUBTITLE_ADVANCE = 0.3
        
        log_worker(f"使用IndexTTS的 {len(tts_sentence_texts)} 个实际分句生成字幕")
        
        # 计算每个TTS句子应该占用的时间（按字符数比例）
        total_chars = sum(len(s) for s in tts_sentence_texts)
        if total_chars == 0:
            total_chars = len(text)
        
        srt_content = []
        subtitle_index = 1
        current_time = 0.0
        
        for idx, tts_sentence in enumerate(tts_sentence_texts):
            # 该句子应占用的总时间（按字符比例）
            sentence_duration = (len(tts_sentence) / total_chars) * audio_duration
            
            # 将TTS句子再细分成3-12字的小段
            segments = split_text_to_segments(tts_sentence, min_chars=3, max_chars=12)
            log_worker(f"TTS句子[{idx+1}] '{tts_sentence[:20]}...' ({len(tts_sentence)}字) -> {len(segments)}个字幕段, 分配时长{sentence_duration:.2f}秒")
            
            # 在该句子的时间范围内，按字符数比例分配时间给每个小段
            sentence_total_chars = sum(len(seg) for seg in segments)
            if sentence_total_chars == 0:
                sentence_total_chars = 1
            
            for segment in segments:
                # 计算该段的时长
                segment_duration = (len(segment) / sentence_total_chars) * sentence_duration
                
                # 计算时间戳（提前0.3秒）
                start_time = max(0, current_time - SUBTITLE_ADVANCE)
                end_time = max(0, current_time + segment_duration - SUBTITLE_ADVANCE)
                
                # 确保结束时间不超过总时长
                if end_time > audio_duration:
                    end_time = audio_duration
                
                # 清理字幕文本：替换符号为空格，合并多个空格
                cleaned_segment = clean_subtitle_text(segment)
                
                # 跳过空的字幕段
                if not cleaned_segment:
                    current_time += segment_duration
                    continue
                
                # SRT格式
                srt_content.append(f"{subtitle_index}")
                srt_content.append(f"{format_srt_time(start_time)} --> {format_srt_time(end_time)}")
                srt_content.append(cleaned_segment)
                srt_content.append("")  # 空行分隔
                
                subtitle_index += 1
                current_time += segment_duration
        
        # 保存SRT文件
        srt_path = Path(output_path).with_suffix('.srt')
        with open(srt_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(srt_content))
        
        log_worker(f"✅ SRT字幕已生成: {srt_path}")
        log_worker(f"   共{subtitle_index-1}个字幕段, 字幕提前{SUBTITLE_ADVANCE}秒显示")
        log_worker(f"   与IndexTTS的{len(tts_sentence_texts)}个实际分句完全对应")
        return str(srt_path)
    
    except Exception as e:
        log_worker(f"生成SRT字幕失败: {e}")
        import traceback
        log_worker(f"错误堆栈:\n{traceback.format_exc()}")
        return None

def process_single_task(model, task_data):
    """处理单个任务"""
    task_id = task_data["task_id"]
    text = task_data["text"]
    output_path = task_data["output"]
    prompt_audio = task_data["prompt"]
    target_duration = task_data.get("target_duration")
    emo_text = task_data.get("emo_text")  # 情绪描述文本
    force_split_at_newline = task_data.get("force_split_at_newline", False)  # 是否在换行处强制断句

    log_worker(f"处理任务 {task_id}")
    log_worker(f"文本: {text[:50]}...")
    if emo_text:
        log_worker(f"情绪描述: {emo_text}")
    if force_split_at_newline:
        log_worker(f"启用换行符强制断句")
    
    # 验证prompt_audio文件是否存在
    if not os.path.exists(prompt_audio):
        error_msg = f"参考音频文件不存在: {prompt_audio}"
        log_worker(f"错误: {error_msg}")
        result = {
            "task_id": task_id,
            "status": "failed",
            "error": error_msg
        }
        print(json.dumps(result), flush=True)
        return False
    
    start_time = time.time()

    try:
        # 如果启用换行符强制断句，先预处理文本
        if force_split_at_newline:
            # 将换行符替换为句号，这样会在换行处断句
            # 保留换行符前后的内容，但用句号分隔
            processed_text = text.replace('\r\n', '。').replace('\n', '。').replace('\r', '。')
            # 清理多余的句号
            # 将英文句号等符号也替换为中文句号，可扩展
            # 可配置需要替换的符号列表（如后续需要可自行添加）
            sentence_delimiters = ['。', r'\.', '．']  # 中文句号、英文点、日文句号等
            # 构造正则表达式（转义英文点）
            pattern = '|'.join(sentence_delimiters)
            # 替换成中文句号，合并连续符号
            processed_text = re.sub(f'(?:{pattern})+', '。', processed_text)
            log_worker(f"预处理文本: 将换行符替换为句号以强制断句")
            text = processed_text
        
        # 在推理前，先获取IndexTTS的分句信息（复用）
        log_worker("获取IndexTTS分句信息...")
        text_tokens_list = model.tokenizer.tokenize(text)
        max_tokens = target_duration and 500 or 120
        
        # 使用与IndexTTS相同的分句逻辑
        if target_duration and len(text_tokens_list) > max_tokens:
            # 时长控制模式下，如果太长会禁用分句
            tts_sentences = [text_tokens_list]
        else:
            tts_sentences = model.tokenizer.split_sentences(text_tokens_list, max_tokens)
        
        # 将token转回文本，得到实际的TTS分句
        tts_sentence_texts = []
        for sent_tokens in tts_sentences:
            sent_text = model.tokenizer.decode(model.tokenizer.convert_tokens_to_ids(sent_tokens))
            tts_sentence_texts.append(sent_text)
        
        log_worker(f"IndexTTS将文本分为 {len(tts_sentence_texts)} 个句子")
        
        # 构建参数
        kwargs = {
            "spk_audio_prompt": prompt_audio,
            "text": text,
            "output_path": output_path,
            "verbose": False
        }

        # 情绪描述参数（如果提供）
        use_emo_text_param = False
        if emo_text:
            kwargs["use_emo_text"] = True
            kwargs["emo_text"] = emo_text
            use_emo_text_param = True
            log_worker(f"使用情绪文本: {emo_text}")

        if target_duration:
            kwargs["use_speed"] = True
            kwargs["target_dur"] = target_duration
            kwargs["max_text_tokens_per_sentence"] = 500
            log_worker(f"时长控制: {target_duration}秒")

        # 执行推理
        log_worker("开始推理...")
        log_worker(f"调用 model.infer() - kwargs: {list(kwargs.keys())}")
        
        try:
            model.infer(**kwargs)
            log_worker("model.infer() 完成")
        except RuntimeError as e:
            error_msg = str(e)
            # 检查是否是emo_matrix维度不匹配的错误
            if use_emo_text_param and ("size of tensor" in error_msg.lower() or "must match" in error_msg.lower()):
                log_worker(f"警告: 情绪文本导致维度不匹配错误，回退到不使用情绪文本模式")
                log_worker(f"错误信息: {error_msg}")
                # 移除情绪文本参数，重试
                kwargs.pop("use_emo_text", None)
                kwargs.pop("emo_text", None)
                log_worker("重新尝试推理（不使用情绪文本）...")
                model.infer(**kwargs)
                log_worker("model.infer() 完成（不使用情绪文本）")
            else:
                # 其他错误，直接抛出
                raise

        generation_time = time.time() - start_time
        log_worker(f"任务 {task_id} 推理完成 (耗时: {generation_time:.2f}秒)")

        # 获取音频时长
        import torchaudio
        waveform, sample_rate = torchaudio.load(output_path)
        audio_duration = waveform.shape[1] / sample_rate
        rtf = generation_time / audio_duration

        # 生成SRT字幕文件（直接使用已获取的分句信息）
        log_worker("开始生成SRT字幕...")
        srt_path = generate_srt_with_tts_sentences(text, audio_duration, output_path, tts_sentence_texts)

        # 输出结果（JSON格式，用于父进程解析）
        result = {
            "task_id": task_id,
            "output_path": output_path,
            "audio_duration": audio_duration,
            "generation_time": generation_time,
            "rtf": rtf,
            "srt_path": srt_path,
            "status": "success"
        }

        # 打印到 stdout（父进程会读取）
        print(json.dumps(result), flush=True)

        return True

    except Exception as e:
        import traceback
        error_msg = str(e)
        error_trace = traceback.format_exc()

        log_worker(f"任务 {task_id} 错误: {error_msg}")
        log_worker(f"错误堆栈:\n{error_trace}")

        # 输出错误结果
        result = {
            "task_id": task_id,
            "status": "failed",
            "error": error_msg
        }
        print(json.dumps(result), flush=True)

        return False

def main():
    log_worker(f"子进程启动: PID={os.getpid()}")

    try:
        # 设置环境变量，尝试禁止各种库的输出
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
        
        log_worker("等待从stdin读取任务...")

        # 批量处理循环
        task_count = 0
        model = None
        current_mode = None
        while True:
            # 从 stdin 读取任务数据
            log_worker(f"等待读取任务 #{task_count + 1}...")
            line = sys.stdin.readline().strip()
            
            if not line:
                # 空行表示没有更多任务
                log_worker("收到空行，准备退出")
                break

            log_worker(f"收到任务数据: {line[:100]}...")
            
            try:
                task_data = json.loads(line)
                if task_data.get("command") == "shutdown":
                    log_worker("收到关闭命令")
                    break

                # 设备模式切换（默认auto）
                requested_mode = task_data.get("device_mode", "auto") or "auto"
                if model is None or requested_mode != current_mode:
                    if model is not None:
                        log_worker(f"切换设备模式: {current_mode} -> {requested_mode}")
                        cleanup_model(model)
                        model = None
                    try:
                        model, current_mode = load_model_for_mode(requested_mode)
                    except Exception as e:
                        error_msg = f"无法加载设备模式[{requested_mode}]: {e}"
                        log_worker(error_msg)
                        result = {
                            "task_id": task_data.get("task_id", "unknown"),
                            "status": "failed",
                            "error": error_msg
                        }
                        print(json.dumps(result), flush=True)
                        continue

                task_count += 1
                log_worker(f"开始处理第 {task_count} 个任务")
                process_single_task(model, task_data)
                log_worker(f"第 {task_count} 个任务处理完成")

            except json.JSONDecodeError as e:
                log_worker(f"JSON解析错误: {e}")
                continue
            except Exception as e:
                log_worker(f"处理任务时出错: {e}")
                import traceback
                log_worker(f"异常堆栈:\n{traceback.format_exc()}")
                continue

        # 清理模型
        log_worker("清理模型...")
        if model is not None:
            cleanup_model(model)
            model = None

        log_worker(f"子进程即将退出，显存将完全释放（共处理 {task_count} 个任务）")
        worker_log.close()
        sys.exit(0)

    except Exception as e:
        import traceback
        error_msg = str(e)
        error_trace = traceback.format_exc()

        log_worker(f"初始化错误: {error_msg}")
        log_worker(f"异常堆栈:\n{error_trace}")

        worker_log.close()
        sys.exit(1)

if __name__ == "__main__":
    main()

