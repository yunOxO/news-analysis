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

# 初始化大模型客户端
# client_llm = OpenAI(base_url="https://ark.cn-beijing.volces.com/api/v3", api_key="f67e1b66-b73c-4a5b-8997-e6596e2be7ad")
client_llm = OpenAI(
    # If the environment variable is not configured, replace the following line with: api_key="sk-xxx"
    api_key="",
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


import re


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
    """
    将 FunASR 的输出转换为句子级分段格式（移除了大模型调用逻辑）
    """
    data = {}
    # --- 1. 数据提取 ---
    # --- 核心修复：自动解包逻辑 ---
    if isinstance(raw_res, tuple):
        raw_res = raw_res[0]

    if isinstance(raw_res, list) and len(raw_res) > 0:
        data = raw_res[0][0] if isinstance(raw_res[0], list) else raw_res[0]
    else:
        data = raw_res

    full_text = data.get('text', '')
    ctc_timestamps = data.get('ctc_timestamps', [])

    if not isinstance(data, dict):
        print(f"ERROR: 无法解析数据结构: {raw_res}")
        return []

    full_text = data.get('text', '')
    ctc_timestamps = data.get('ctc_timestamps', [])

    if not full_text:
        return []

    # --- 2. 预处理：建立 Token 索引映射 ---
    # 我们把所有原始 Token 拼成一串纯文本，并记录每个字符对应的 Token 索引
    token_str = ""
    char_to_token_idx = []

    for i, item in enumerate(ctc_timestamps):
        clean_token = item['token'].replace(" ", "").lower()
        for _ in clean_token:
            token_str += _
            char_to_token_idx.append(i)

    # --- 3. 句子切分 ---
    # 仅按 。？！ 切分，保留标点
    raw_parts = re.split(r'([。？！])', full_text)
    sentences = []

    temp_s = ""
    for p in raw_parts:
        temp_s += p
        if p in ['。', '？', '！']:
            if temp_s.strip(): sentences.append(temp_s.strip())
            temp_s = ""
    if temp_s.strip(): sentences.append(temp_s.strip())

    # --- 4. 模糊对齐寻找时间轴 ---
    asr_sentences = []
    search_start_pos = 0

    for sen in sentences:
        # 清洗句子：去掉空格、标点，转小写，用于匹配
        sen_clean = re.sub(r'[^\w]', '', sen).lower()
        if not sen_clean: continue

        # 在 token_str 中寻找与当前句子最相似的片段
        # 我们只在当前游标之后的片段里找，提高速度和准确度
        sub_token_str = token_str[search_start_pos:]
        matcher = SequenceMatcher(None, sen_clean, sub_token_str)
        match = matcher.find_longest_match(0, len(sen_clean), 0, len(sub_token_str))

        if match.size > 0:
            # 找到匹配片段在全局 token_str 中的起止位置
            matched_start_in_global = search_start_pos + match.b
            matched_end_in_global = matched_start_in_global + match.size - 1

            # 映射回 ctc_timestamps 的索引
            start_token_idx = char_to_token_idx[matched_start_in_global]
            end_token_idx = char_to_token_idx[matched_end_in_global]

            # 更新下一次搜索的起始位置
            search_start_pos = matched_end_in_global + 1
        else:
            # 如果完全没匹配到（极端情况），使用比例兜底
            start_token_idx = char_to_token_idx[min(search_start_pos, len(char_to_token_idx) - 1)]
            end_token_idx = start_token_idx

        asr_sentences.append({
            "start": format_timestamp_2(ctc_timestamps[start_token_idx]['start_time']),
            "end": format_timestamp_2(ctc_timestamps[end_token_idx]['end_time']),
            "asr_sentence": sen
        })

    return asr_sentences

    # punctuations = set(['。', '？', '！', '，', '、'])
    #
    # asr_sentences = []
    # current_seg_start = None
    # current_seg_end = 0.0
    # ts_index = 0
    # segment_text_accumulator = ""
    #
    # for char in full_text:
    #     segment_text_accumulator += char
    #     is_punct = char in punctuations
    #
    #     if not is_punct:
    #         if ts_index < len(ctc_timestamps):
    #             ts_item = ctc_timestamps[ts_index]
    #             if current_seg_start is None:
    #                 current_seg_start = ts_item['start_time']
    #             current_seg_end = ts_item['end_time']
    #             ts_index += 1
    #
    #     if is_punct:
    #         if current_seg_start is not None:
    #             start_str = format_timestamp(current_seg_start)
    #             end_str = format_timestamp(current_seg_end)
    #
    #             asr_sentences.append({
    #                 "start": start_str,
    #                 "end": end_str,
    #                 "asr_sentence": segment_text_accumulator
    #             })
    #
    #         current_seg_start = None
    #         current_seg_end = 0.0
    #         segment_text_accumulator = ""
    #
    # # 处理最后一句
    # if segment_text_accumulator and current_seg_start is not None:
    #     start_str = format_timestamp(current_seg_start)
    #     end_str = format_timestamp(current_seg_end)
    #     asr_sentences.append({
    #         "start": start_str,
    #         "end": end_str,
    #         "asr_sentence": segment_text_accumulator
    #     })
    #
    # return asr_sentences


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

    config = uvicorn.Config(app, host="0.0.0.0", port=8800)
    server = uvicorn.Server(config)
    asyncio.run(server.serve())