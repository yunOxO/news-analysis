import uvicorn
import torch
import os
import tempfile
import time
import json
from typing import List, Dict, Any
from fastapi import FastAPI, Request, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager
import torchaudio
from json_repair import repair_json
import re
from difflib import SequenceMatcher

# 请确保 model.py 中的 FunASRNano 可以正常导入
from model import FunASRNano
from openai import OpenAI

model_context = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- 启动时加载模型 ---
    print("正在加载 FunASRNano 模型...")
    try:
        model_dir = "/root/Fun-ASR/Fun-ASR-Nano-2512"  # 请确保路径正确
        m, kwargs = FunASRNano.from_pretrained(model=model_dir, device="cuda:0")
        m.eval()
        model_context["model"] = m
        model_context["kwargs"] = kwargs
        print("模型加载完成。")
    except Exception as e:
        print(f"模型加载失败: {e}")
    yield
    # --- 关闭时清理 (如有需要) ---
    model_context.clear()


app = FastAPI(lifespan=lifespan)

import os

# 初始化大模型客户端
# client_llm = OpenAI(base_url="https://ark.cn-beijing.volces.com/api/v3", api_key="f67e1b66-b73c-4a5b-8997-e6596e2be7ad")
client_llm = OpenAI(
    # If the environment variable is not configured, replace the following line with: api_key="sk-xxx"
    api_key=os.getevn("ALIYUN_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

SYSTEM_PROMPT = "帮我根据下面内容新闻内容进行拆分，为每个新闻内容进行摘要，并根据新闻片段标注对应的起始与结束时间。摘要信息尽可能简洁\n返回结果请依照下面list格式：\n[ { 'start': '00:00', 'end': '00:44', 'title': '标题1', 'summary': '摘要1' } ]"

# --- 限制上传文件大小 ---
app.add_middleware(
    CORSMiddleware,  # 跨域按需添加
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def limit_upload_size(request: Request, call_next):
    if request.method == "POST" and request.url.path == "/asr":
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > 1000 * 1024 * 1024:  # 10MB
            raise HTTPException(status_code=413, detail="File too large (max 10MB)")
    response = await call_next(request)
    return response


# --- 请求与响应模型定义 ---

class AsrSentence(BaseModel):
    start: str
    end: str
    asr_sentence: str


class ChapterSummary(BaseModel):
    start: str
    end: str
    title: str
    summary: str
    asr_content: str


# 接口1：ASR 的响应格式
class AsrOnlyResponse(BaseModel):
    asr_body: List[AsrSentence]


# 接口2：大模型切片的请求格式
class LlmSegmentRequest(BaseModel):
    asr_body: List[AsrSentence]


# 接口2：大模型切片的响应格式
class LlmSegmentResponse(BaseModel):
    news_clip: List[ChapterSummary] = []


def arabic_to_zh(num_str: str) -> str:
    """将阿拉伯数字字符串转换为中文读法"""
    # 场景 1：如果是以 0 开头的数字（如区号），或者超过 8 位的长数字（如电话），通常逐字直读
    if num_str.startswith("0") or len(num_str) >= 8:
        digits_map = "零一二三四五六七八九"
        return "".join(digits_map[int(c)] for c in num_str)

    # 场景 2：常规数值转换为带量词的读法
    num = int(num_str)
    if num == 0:
        return "零"

    digits = "零一二三四五六七八九"
    units = ["", "十", "百", "千", "万", "十", "百", "千", "亿"]
    result = ""
    s = str(num)
    n = len(s)

    for i, char in enumerate(s):
        d = int(char)
        unit_index = n - 1 - i

        if d != 0:
            result += digits[d] + units[unit_index]
        else:
            # 处理中间的 0 和连续的 0
            if unit_index % 4 == 0:  # 遇到万、亿位
                if result.endswith("零"):
                    result = result[:-1]
                if not result.endswith(units[unit_index]) and len(result) > 0:
                    result += units[unit_index]
            else:
                if not result.endswith("零"):
                    result += "零"

    # 收尾清理
    if result.endswith("零"):
        result = result[:-1]

    # 习惯修正："一十五" -> "十五"
    if result.startswith("一十"):
        result = result[1:]

    # 习惯修正：ASR 通常将 2000 识别为 "两千" 而非 "二千"
    result = result.replace("二千", "两千").replace("二万", "两万")

    return result


def replace_arabic_numbers(text: str) -> str:
    """使用正则匹配所有的阿拉伯数字，并进行批量中文替换"""
    pattern = re.compile(r'\d+')
    return pattern.sub(lambda x: arabic_to_zh(x.group()), text)


# --- 核心处理逻辑 ---

def format_timestamp(seconds: float) -> str:
    """将秒数转换为 MM:SS 格式"""
    m = int(seconds // 60)
    # s = int(seconds % 60)
    s = seconds % 60
    return f"{m:02d}:{s:06.3f}"


def format_timestamp_2(seconds: float) -> str:
    """将秒数转换为 MM:SS 格式"""
    m = int(seconds // 60)
    s = int(seconds % 60)
    # s = seconds % 60
    return f"{m:02d}:{s:02d}"


def time_to_seconds(t) -> float:
    """将时间字符串转换为秒数，具备超强容错能力"""
    if isinstance(t, (int, float)):
        return float(t)
    if not t:
        return 0.0

    # 替换中文冒号，去除首尾空格
    t_str = str(t).strip().replace("：", ":")

    try:
        parts = t_str.split(":")
        if len(parts) == 2:
            return float(parts[0]) * 60 + float(parts[1])
        elif len(parts) == 3:
            return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
    except Exception:
        pass

    # 终极兜底：如果 split 依然失败，用正则硬抠数字
    nums = re.findall(r"[\d\.]+", t_str)
    if len(nums) >= 2:
        return float(nums[-2]) * 60 + float(nums[-1])

    return 0.0


def process_funasr_output(raw_res) -> List[dict]:
    # --- 1. 数据提取与解包 (保持不变) ---
    if isinstance(raw_res, tuple):
        raw_res = raw_res[0]
    if isinstance(raw_res, list) and len(raw_res) > 0:
        data = raw_res[0][0] if isinstance(raw_res[0], list) else raw_res[0]
    else:
        data = raw_res

    full_text = data.get('text', '')
    ctc_timestamps = data.get('ctc_timestamps', [])

    if not full_text or not ctc_timestamps:
        return []

    # --- 2. 建立 Token 索引映射 (保持不变) ---
    token_str = ""
    char_to_token_idx = []

    for i, item in enumerate(ctc_timestamps):
        clean_token = item['token'].replace(" ", "").lower()
        for _ in clean_token:
            token_str += _
            char_to_token_idx.append(i)

    # --- 3. 智能句子切分 (解决切分粒度过大问题) ---
    # 第一层切分：用句号、问号、叹号、换行符切分
    raw_parts = re.split(r'([。？！\n])', full_text)
    temp_sentences = []
    temp_s = ""
    for p in raw_parts:
        temp_s += p
        if p in ['。', '？', '！', '\n']:
            if temp_s.strip(): temp_sentences.append(temp_s.strip())
            temp_s = ""
    if temp_s.strip(): temp_sentences.append(temp_s.strip())

    # 第二层切分：遇到极长的句子（>80字），在逗号处强行切断，防止时间跨度过大
    sentences = []
    for sen in temp_sentences:
        if len(sen) > 80 and '，' in sen:
            sub_parts = re.split(r'([，；])', sen)
            sub_s = ""
            for sp in sub_parts:
                sub_s += sp
                if sp in ['，', '；']:
                    if sub_s.strip(): sentences.append(sub_s.strip())
                    sub_s = ""
            if sub_s.strip(): sentences.append(sub_s.strip())
        else:
            sentences.append(sen)

    # --- 4. 比例锚点 + 动态窗口对齐 (解决长视频累积漂移) ---
    asr_sentences = []

    # 预计算所有干净句子的总长度，用于计算全局比例
    clean_sentences = []
    for sen in sentences:
        clean_sen = re.sub(r'[^\w]', '', sen).lower()
        clean_sen = replace_arabic_numbers(clean_sen)
        clean_sentences.append(clean_sen)

    total_clean_chars = sum(len(s) for s in clean_sentences)
    total_token_chars = len(token_str)
    current_clean_char_idx = 0

    for i, sen in enumerate(sentences):
        sen_clean = clean_sentences[i]
        expected_len = len(sen_clean)

        if expected_len == 0:
            continue

        # 【核心优化 1：计算全局比例锚点】
        # 判断这句话在全文的进度，直接推算出它在 token_str 中的绝对位置
        ratio = current_clean_char_idx / total_clean_chars if total_clean_chars > 0 else 0
        anchor_pos = int(ratio * total_token_chars)

        # 【核心优化 2：以锚点为中心展开窗口】
        # 窗口大小：句子长度 + 前后各 80 个字符的容错空间，再乱也能框住
        margin = max(80, int(expected_len * 1.5))
        window_start = max(0, anchor_pos - margin)
        window_end = min(total_token_chars, anchor_pos + expected_len + margin)

        sub_token_str = token_str[window_start:window_end]

        matcher = SequenceMatcher(None, sen_clean, sub_token_str)

        # --- 终极优化：碎片缝合算法 ---
        # 获取所有的匹配碎片块 (Match对象列表)
        blocks = matcher.get_matching_blocks()

        # 过滤掉太小的碎片（如偶然匹配上的单个字），防止拉扯边界
        min_blk_size = 2 if expected_len >= 4 else 1
        valid_blocks = [blk for blk in blocks if blk.size >= min_blk_size]

        total_matched = sum(blk.size for blk in valid_blocks)
        min_match_thresh = max(2, int(expected_len * 0.15))

        if total_matched >= min_match_thresh and valid_blocks:
            first_block = valid_blocks[0]
            last_block = valid_blocks[-1]

            # 计算这些碎片在 sub_token_str 中的跨度
            span_in_sub = (last_block.b + last_block.size) - first_block.b

            # 防御机制：如果跨度大得离谱（比如跨越了预期长度的 2.5 倍），说明首尾抓到了错误的噪声字符
            if span_in_sub > expected_len * 2.5:
                # 跨度过大，退化回使用最大的一块连续碎片
                largest_block = max(valid_blocks, key=lambda x: x.size)
                matched_start_in_global = window_start + largest_block.b
                matched_end_in_global = window_start + largest_block.b + largest_block.size - 1
            else:
                # 正常缝合：起始位置取第一个碎片的开头，结束位置取最后一个碎片的结尾
                matched_start_in_global = window_start + first_block.b
                matched_end_in_global = window_start + last_block.b + last_block.size - 1

            start_token_idx = char_to_token_idx[min(matched_start_in_global, len(char_to_token_idx) - 1)]
            end_token_idx = char_to_token_idx[min(matched_end_in_global, len(char_to_token_idx) - 1)]
        else:
            # 【核心优化 3：全局比例强制兜底】
            # 如果彻底乱码匹配不上，不依赖游标，直接用这句话的真实进度比例换算时间轴！
            start_ratio = current_clean_char_idx / total_clean_chars
            end_ratio = min(1.0, (current_clean_char_idx + expected_len) / total_clean_chars)

            start_token_idx = char_to_token_idx[min(int(start_ratio * total_token_chars), len(char_to_token_idx) - 1)]
            end_token_idx = char_to_token_idx[min(int(end_ratio * total_token_chars), len(char_to_token_idx) - 1)]

        asr_sentences.append({
            "start": format_timestamp_2(ctc_timestamps[start_token_idx]['start_time']),
            "end": format_timestamp_2(ctc_timestamps[end_token_idx]['end_time']),
            "asr_sentence": sen
        })

        # 推进全局进度
        current_clean_char_idx += expected_len

    return asr_sentences


# --- 接口 1：ASR 提取接口 ---

@app.post("/asr", response_model=AsrOnlyResponse)
async def create_asr(file: UploadFile = File(...)):
    if "model" not in model_context:
        raise HTTPException(status_code=503, detail="Model not loaded")

    model = model_context["model"]
    kwargs = model_context["kwargs"]

    try:
        # 1. 临时保存上传的文件（避免内存占用）
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp_file:
            tmp_file.write(await file.read())
            tmp_file_path = tmp_file.name

        # 获取音频真实时长
        try:
            audio_info = torchaudio.info(tmp_file_path)
            audio_duration = audio_info.num_frames / audio_info.sample_rate
        except Exception as e:
            print(f"Warning: 无法获取音频时长 ({e})")
            audio_duration = 0.0

        print(f"\n--- 开始处理文件: {file.filename} (音频时长: {audio_duration:.2f}s) ---")

        torch.cuda.empty_cache()
        asr_start_time = time.perf_counter()

        # 执行推理
        res = model.inference(data_in=[tmp_file_path], **kwargs)

        asr_cost_time = time.perf_counter() - asr_start_time
        rtf = asr_cost_time / audio_duration if audio_duration > 0 else 0.0
        print(f"[耗时监测] FunASR推理完成，耗时: {asr_cost_time:.3f}s | 实时率(RTF): {rtf:.3f}")

        # 格式化结果 (仅返回 ASR 句子列表)
        asr_sentences = process_funasr_output(res)

        os.unlink(tmp_file_path)

        return {"asr_body": asr_sentences}

    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")


# --- 接口 2：大模型切片与摘要接口 ---

@app.post("/segment", response_model=LlmSegmentResponse)
def segment_news(request: LlmSegmentRequest):
    """
    接收 ASR 提取的文本结果，通过大模型进行新闻切片与摘要
    注：因为 OpenAI client 是同步的，这里使用 def 而不是 async def，
    FastAPI 会自动将其放入线程池中执行，避免阻塞主事件循环。
    """
    asr_sentences = [item.dict() for item in request.asr_body]

    if not asr_sentences:
        raise HTTPException(status_code=400, detail="ASR body cannot be empty")

    print("[耗时监测] 开始调用大模型提取新闻摘要...")
    llm_start_time = time.perf_counter()

    try:
        completion = client_llm.chat.completions.create(
            # model="deepseek-v3-250324",
            model="qwen3.5-plus",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": str(asr_sentences)}
            ],
            stream=False,
            max_tokens=16384,
            temperature=0.1,
            top_p=0.9,
            extra_body={"enable_thinking": False}
        )
        ans = completion.choices[0].message.content
        ans = ans.replace("```json", "").replace("```", "").strip()
        llm_end_time = time.perf_counter()
        print(f"[耗时监测] 摘要提取完成，耗时: {llm_end_time - llm_start_time:.3f}s")

        try:
            ans = ans.replace("'", '"')
            ans_list = json.loads(repair_json(ans))
            # --- 终极逻辑：基于时间交集比例的匹配 ---
            for i, clip in enumerate(ans_list):
                clip_start_sec = time_to_seconds(clip.get("start", "00:00"))
                clip_end_sec = time_to_seconds(clip.get("end", "00:00"))

                # 容错：如果大模型抽风没给结束时间，默认给个极大值，但仅限于最后一个切片
                if clip_end_sec <= clip_start_sec:
                    clip_end_sec = clip_start_sec + 9999.0

                content_parts = []

                for sentence in asr_sentences:
                    # 兼容不同字段命名习惯
                    sen_start = time_to_seconds(sentence.get("start", sentence.get("raw_start", "00:00")))
                    sen_end = time_to_seconds(sentence.get("end", sentence.get("raw_end", "00:00")))

                    # 防御极端的 0 时长句子
                    if sen_end <= sen_start:
                        sen_end = sen_start + 0.1

                        # 计算两个时间段的重叠部分 (Intersection)
                    overlap_start = max(clip_start_sec, sen_start)
                    overlap_end = min(clip_end_sec, sen_end)
                    overlap_duration = max(0.0, overlap_end - overlap_start)

                    sen_duration = sen_end - sen_start

                    # 判断条件：重叠时间超过句子总时长的 50% (说明这句话的"主体"在这个切片里)
                    if overlap_duration / sen_duration >= 0.5:
                        text = sentence.get("asr_sentence", sentence.get("asr_text", ""))
                        content_parts.append(text)

                clip["asr_content"] = "".join(content_parts)

                # 调试打印（你可以看终端里的输出，一眼就能发现大模型的时间有没有给错）
                # print(f"切片 {i+1} [{clip.get('start')}-{clip.get('end')}]: 转换后为 {clip_start_sec}s - {clip_end_sec}s, 提取了 {len(content_parts)} 句话")
            # ----------------------------------------------------
        except Exception as e:
            print(f"解析大模型返回结果失败: {e}, 原始字符串: {ans}")
            ans_list = []

        return {"news_clip": ans_list}

    except Exception as e:
        print(f"LLM 调用失败: {e}")
        raise HTTPException(status_code=500, detail=f"LLM processing failed: {str(e)}")


if __name__ == "__main__":
    # 启动服务
    # uvicorn.run(app, host="0.0.0.0", port=8800)
    import asyncio
    import uvicorn

    config = uvicorn.Config(app, host="0.0.0.0", port=8802)
    server = uvicorn.Server(config)
    asyncio.run(server.serve())
