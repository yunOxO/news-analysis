# news-analysis
基于中文新闻的语音信息，进行内容切分，生成新闻内容的切片与结果。

## 🎬 快速开始

### 📝 前提条件

> 环境配置
>
> ```bash
> # 需要至少4GB的GPU显存资源来启动asr模型
> pip install -r requirements.txt
> ```
> ASR模型下载
> ```bash
> modelscope download --model FunAudioLLM/Fun-ASR-Nano-2512 --local_dir ./Fun-ASR-Nano-2512
> ```
> 服务启动
>
> ```bash 
> python tool_news_clip_audio.py
> ```

### 接口1-音频asr接口

```bash
POST /asr
支持mp3、mp4等音频文件

# response
{
    "asr_body": [
        {
            "start": "00:18",
            "end": "00:26",
            "asr_sentence": "各位观众晚上好晚上好，今天是二月五号星期四，农历十二月十八，欢迎收看新闻联播节目。"
        }
    ]
}
```

### 接口2-内容切片接口
```shell
# 请求
POST /segment
## 传参
--data-raw '{
    "asr_body": [ { "start": "00:18", "end": "00:26", "asr_sentence": "各位观众晚上好晚上好，今天是二月五号星期四，农历十二月十八，欢迎收看新闻联播节目。" }, { "start": "00:26", "end": "00:29", "asr_sentence": "首先为您介绍今天节目的主要内容。" }]
}'

# response
{
    "news_clip": [
        {
            "start": "00:18",
            "end": "01:48",
            "title": "新闻联播开场及内容提要",
            "summary": "主持人开场介绍日期，并简要播报今日主要新闻内容，包括中美通话、中老友好年启动、乡村振兴、市场监管成效及国际热点等。"
        }
    ]
}
```