"""
Download 5 different voice style samples using Edge TTS
御姐、萌萝莉、熟女、少妇、少女
"""
import asyncio
import edge_tts
import os

# Voice configurations
VOICE_STYLES = {
    "yujie": {  # 御姐 - 成熟优雅
        "voice": "zh-CN-XiaomoNeural",
        "text": "你好，我是御姐风格的AI助手，很高兴认识你。我可以帮你处理各种问题。",
        "rate": "+0%",
        "pitch": "-5Hz",
        "description": "御姐 - 成熟优雅，声音低沉性感"
    },
    "luoli": {  # 萌萝莉 - 可爱童真
        "voice": "zh-CN-XiaomengNeural",
        "text": "你好呀！我是小萝莉，今天想和你一起玩游戏呢！你喜欢什么游戏呀？",
        "rate": "+5%",
        "pitch": "+10Hz",
        "description": "萌萝莉 - 可爱童真，声音甜美"
    },
    "shunv": {  # 熟女 - 知性温柔
        "voice": "zh-CN-XiaohanNeural",
        "text": "你好，今天过得怎么样？有什么需要帮助的吗？我会耐心倾听你的想法。",
        "rate": "+0%",
        "pitch": "-3Hz",
        "description": "熟女 - 知性温柔，成熟稳重"
    },
    "shaofu": {  # 少妇 - 温柔体贴
        "voice": "zh-CN-XiaorouNeural",
        "text": "欢迎回来，辛苦了。今天工作累吗？我准备了一些轻松的话题，陪你聊聊天吧。",
        "rate": "+0%",
        "pitch": "+0Hz",
        "description": "少妇 - 温柔体贴，声音柔和"
    },
    "shaonv": {  # 少女 - 活泼开朗
        "voice": "zh-CN-XiaoyiNeural",
        "text": "嗨！好久不见啦！今天天气真好，我们一起做点有趣的事情吧！",
        "rate": "+8%",
        "pitch": "+5Hz",
        "description": "少女 - 活泼开朗，青春洋溢"
    }
}

async def download_voice_sample(style_id, config):
    """Download a single voice sample"""
    print(f"\n正在生成: {config['description']}")
    print(f"  语音引擎: {config['voice']}")
    print(f"  示例文本: {config['text']}")

    output_dir = "voice_samples"
    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(output_dir, f"{style_id}_edge.wav")

    try:
        # Create TTS communication
        communicate = edge_tts.Communicate(
            text=config['text'],
            voice=config['voice'],
            rate=config['rate'],
            pitch=config['pitch']
        )

        # Save to file
        await communicate.save(output_file)

        print(f"  [OK] 已保存: {output_file}")
        return True

    except Exception as e:
        print(f"  [ERROR] 生成失败: {e}")
        return False

async def download_all_samples():
    """Download all voice samples"""
    print("=" * 80)
    print("Edge TTS 语音样本下载器 - 5种音色风格")
    print("=" * 80)

    success_count = 0

    for style_id, config in VOICE_STYLES.items():
        result = await download_voice_sample(style_id, config)
        if result:
            success_count += 1

    print("\n" + "=" * 80)
    print(f"下载完成！成功: {success_count}/{len(VOICE_STYLES)}")
    print("=" * 80)

    print("\n【生成的音色文件】")
    print("-" * 80)
    for style_id, config in VOICE_STYLES.items():
        filename = f"{style_id}_edge.wav"
        print(f"  - {filename:25} - {config['description']}")

    print("\n【使用方法】")
    print("-" * 80)
    print("在 main_full.py 中配置：")
    print("""
# 御姐风格
tts_config = TTSConfig(engine="edge", voice="zh-CN-XiaomoNeural", pitch="-5Hz")

# 萌萝莉风格
tts_config = TTSConfig(engine="edge", voice="zh-CN-XiaomengNeural", pitch="+10Hz")

# 熟女风格
tts_config = TTSConfig(engine="edge", voice="zh-CN-XiaohanNeural", pitch="-3Hz")

# 少妇风格
tts_config = TTSConfig(engine="edge", voice="zh-CN-XiaorouNeural")

# 少女风格
tts_config = TTSConfig(engine="edge", voice="zh-CN-XiaoyiNeural", pitch="+5Hz")
    """)

    print("=" * 80)

if __name__ == "__main__":
    asyncio.run(download_all_samples())
