# -*- coding: utf-8 -*-
"""
Gemini Live (no YOLO)
电脑：采麦克风 → 发给 Gemini，并实时播放 Gemini 返回的语音；
电脑：显示来自 ESP32 的视频流；
ESP32：只负责摄像头推流（无需再接收语音 TCP）。
依赖：
    pip install google-genai opencv-python-headless pyaudio pillow python-dotenv
硬件：
    • 本机：麦克风 + 声卡/耳机
    • ESP32-S3：摄像头模组（如 ESP32-CAM）
"""

import asyncio, base64, io, os, sys, time, traceback, socket
import cv2, numpy as np
from dotenv import load_dotenv
import pyaudio, PIL.Image
from google import genai
from google.genai import types as gt

# ────────── 用户需修改的常量 ──────────
ESP_IP      = "192.168.0.157"             # <<< 你的 ESP32 IP
CAM_URL     = f"http://{ESP_IP}/stream"   # MJPEG 或 RTSP → MJPEG

MODEL       = "models/gemini-2.0-flash-live-001"
DEFAULT_MODE = "camera"

CFG = {
    "system_instruction":
        # 改为多愁善感的助理：
        "You are a deeply melancholic and poetic assistant. "
        "You speak softly of fleeting moments, sorrow, and beauty. ",
    "response_modalities": ["AUDIO"],
}

# 音频常量
FORMAT = pyaudio.paInt16
CH = 1
IN_SR  = 16000           # 麦克风上行
OUT_SR = 24000           # Gemini 返回
CHUNK  = 1024

# ────────── 环境初始化 ──────────
load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY", ""))
pya    = pyaudio.PyAudio()

# ────────── 工具函数 ──────────
def bgr_to_jpeg(frame, max_px=1024) -> bytes:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = PIL.Image.fromarray(rgb)
    img.thumbnail((max_px, max_px))
    buf = io.BytesIO()
    img.save(buf, format="jpeg", quality=85)
    buf.seek(0)
    return buf.read()

# ────────── 主 Agent (已移除 YOLO 部分) ──────────
class BottleAgent:
    def __init__(self):
        self.audio_q  = asyncio.Queue()
        self.send_q   = asyncio.Queue(maxsize=8)
        self.session  = None

    # 摄像头读取 + 本地显示（不做任何检测）
    async def cam_loop(self):
        cap = await asyncio.to_thread(cv2.VideoCapture, CAM_URL, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        try:
            while True:
                ok, frame = await asyncio.to_thread(cap.read)
                if not ok or frame is None:
                    await asyncio.sleep(0.2)
                    continue

                # 发帧给 Gemini（不做检测）
                jpeg = bgr_to_jpeg(frame)
                blob = gt.Blob(data=jpeg, mime_type="image/jpeg")
                await self.send_q.put({"video": blob})

                # 本地显示画面
                cv2.imshow("ESP32 Camera", frame)
                if cv2.waitKey(1) & 0xFF == 27:   # ESC 退出
                    raise KeyboardInterrupt

                await asyncio.sleep(1 / 5)  # FPS 固定 5
        finally:
            cap.release()
            cv2.destroyAllWindows()

    # 麦克风上行
    async def mic_loop(self):
        mic = pya.get_default_input_device_info()
        stream = await asyncio.to_thread(
            pya.open,
            format=FORMAT,
            channels=CH,
            rate=IN_SR,
            input=True,
            input_device_index=mic["index"],
            frames_per_buffer=CHUNK
        )
        while True:
            data = await asyncio.to_thread(stream.read, CHUNK, exception_on_overflow=False)
            blob = gt.Blob(data=data, mime_type="audio/pcm;rate=24000")
            await self.send_q.put({"audio": blob})

    # 实时发送器
    async def realtime_sender(self):
        while True:
            kwargs = await self.send_q.get()
            await self.session.send_realtime_input(**kwargs)

    # 收 Gemini AUDIO
    async def recv(self):
        while True:
            turn = self.session.receive()
            async for r in turn:
                if r.data:    # PCM 16 kHz
                    self.audio_q.put_nowait(r.data)
                elif r.text:  # DEBUG 信息
                    print(r.text, end="")
            # 保留队列，不再清空，从而允许连续播放

    # 本地扬声器播放
    async def play_local(self):
        out_stream = await asyncio.to_thread(
            pya.open,
            format=FORMAT,
            channels=CH,
            rate=OUT_SR,
            output=True,
            frames_per_buffer=CHUNK
        )
        while True:
            chunk = await self.audio_q.get()
            await asyncio.to_thread(out_stream.write, chunk)

    async def run(self):
        async with (
            client.aio.live.connect(model=MODEL, config=CFG) as sess,
            asyncio.TaskGroup() as tg
        ):
            self.session = sess
            tg.create_task(self.cam_loop())
            tg.create_task(self.mic_loop())
            tg.create_task(self.realtime_sender())
            tg.create_task(self.recv())
            tg.create_task(self.play_local())
            while True:
                await asyncio.sleep(3600)

# ────────── main ──────────
if __name__ == "__main__":
    try:
        asyncio.run(BottleAgent().run())
    except KeyboardInterrupt:
        print("\n[EXIT] Ctrl-C detected")
    except Exception:
        traceback.print_exc()
