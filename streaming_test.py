# -*- coding: utf-8 -*-
"""
Gemini Live API + YOLOv8 侦测水瓶 → 触发 “My name is Omo.”
依赖：
    pip install google-genai opencv-python ultralytics pyaudio pillow python-dotenv
"""

import asyncio, base64, io, os, sys, time, traceback
import cv2, numpy as np
from ultralytics import YOLO
from dotenv import load_dotenv
import pyaudio, PIL.Image
from google import genai
from google.genai import types as gt           # ← 官方类型别名

# ---------- 基本参数 ----------
MODEL = "models/gemini-2.0-flash-live-001"

DEFAULT_MODE = "camera"
YOLO_W = "yolov8n.pt"
LABEL  = "bottle"
COOLDOWN = 0.01        # 秒
FPS   = 5             # 摄像头帧率

CFG = {
    "system_instruction":
        "You are a visual assistant. "
        "If the user sends exactly the text 'trigger_omo', "
        "you must immediately reply: “My name is Omo.”",
    "response_modalities": ["AUDIO"],
}

# 音频常量
FORMAT=pyaudio.paInt16; CH=1; IN_SR=16000; OUT_SR=24000; CHUNK=1024

# ---------- 环境 ----------
load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY", ""))
pya = pyaudio.PyAudio()

# ---------- 工具函数 ----------
def bgr_to_jpeg(frame, max_px=1024)->bytes:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = PIL.Image.fromarray(rgb); img.thumbnail((max_px,max_px))
    buf=io.BytesIO(); img.save(buf, format="jpeg", quality=85); buf.seek(0); return buf.read()

# ---------- Agent ----------
class BottleAgent:
    def __init__(self):
        self.yolo = YOLO(YOLO_W)
        self.last = 0
        self.audio_q = asyncio.Queue()
        self.send_q  = asyncio.Queue(maxsize=8)   # 待送实时数据
        self.session = None

    # 摄像头 + YOLO
    async def cam_loop(self):
        cap = await asyncio.to_thread(cv2.VideoCapture, 0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,640); cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
        try:
            while True:
                ok, frame = await asyncio.to_thread(cap.read)
                if not ok: break

                # YOLO 推理
                res = await asyncio.to_thread(self.yolo, frame, verbose=False)
                found = any(self.yolo.names[int(b.cls[0])] == LABEL for b in res[0].boxes)

                # 发视频帧
                jpeg = bgr_to_jpeg(frame)
                blob = gt.Blob(data=jpeg, mime_type="image/jpeg")
                await self.send_q.put({"video": blob})

                # 触发文本
                now=time.time()
                if found and now-self.last>COOLDOWN:
                    self.last=now
                    await self.session.send_client_content(
                        turns=gt.Content(role="user",
                                         parts=[gt.Part(text="trigger_omo")]),
                        turn_complete=True)

                await asyncio.sleep(1/FPS)
        finally:
            cap.release()

    # 麦克风
    async def mic_loop(self):
        mic=pya.get_default_input_device_info()
        stream = await asyncio.to_thread(
            pya.open, format=FORMAT, channels=CH, rate=IN_SR,
            input=True, input_device_index=mic["index"], frames_per_buffer=CHUNK)
        while True:
            data = await asyncio.to_thread(stream.read, CHUNK, exception_on_overflow=False)
            blob = gt.Blob(data=data, mime_type="audio/pcm;rate=16000")
            await self.send_q.put({"audio": blob})

    # 发送实时流
    async def realtime_sender(self):
        while True:
            kwargs = await self.send_q.get()     # {"audio":Blob} 或 {"video":Blob}
            await self.session.send_realtime_input(**kwargs)

    # 接收、播放语音
    async def recv(self):
        while True:
            turn = self.session.receive()
            async for r in turn:
                if r.data:  self.audio_q.put_nowait(r.data)
                elif r.text: print(r.text, end="")
            while not self.audio_q.empty(): self.audio_q.get_nowait()  # 清残余，支援打断

    async def play(self):
        stream = await asyncio.to_thread(
            pya.open, format=FORMAT, channels=CH, rate=OUT_SR, output=True)
        while True:
            chunk=await self.audio_q.get()
            await asyncio.to_thread(stream.write, chunk)

    async def run(self):
        async with (
            client.aio.live.connect(model=MODEL, config=CFG) as sess,
            asyncio.TaskGroup() as tg
        ):
            self.session=sess
            tg.create_task(self.cam_loop())
            tg.create_task(self.mic_loop())
            tg.create_task(self.realtime_sender())
            tg.create_task(self.recv())
            tg.create_task(self.play())
            while True: await asyncio.sleep(3600)   # 主协程挂起

# ---------- main ----------
if __name__=="__main__":
    try: asyncio.run(BottleAgent().run())
    except KeyboardInterrupt: print("\n退出")
    except Exception: traceback.print_exc()
