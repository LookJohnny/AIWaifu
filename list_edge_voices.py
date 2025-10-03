"""
List all available Edge TTS voices (Chinese + English)
"""
import asyncio
import edge_tts

async def list_voices():
    print("=" * 80)
    print("Available Edge TTS Voices (Chinese + English)")
    print("=" * 80)

    voices = await edge_tts.list_voices()

    # Filter Chinese and English voices
    chinese_voices = [v for v in voices if v["Locale"].startswith("zh-")]
    english_voices = [v for v in voices if v["Locale"].startswith("en-")]

    print("\n【中文语音包】")
    print("-" * 80)
    for v in chinese_voices:
        gender = v.get("Gender", "N/A")
        name = v.get("ShortName", "N/A")
        locale = v.get("Locale", "N/A")
        print(f"{gender:8} | {name:35} | {locale}")

    print("\n【English Voices】")
    print("-" * 80)
    for v in english_voices[:10]:  # Show first 10
        gender = v.get("Gender", "N/A")
        name = v.get("ShortName", "N/A")
        locale = v.get("Locale", "N/A")
        print(f"{gender:8} | {name:35} | {locale}")

    print("\n" + "=" * 80)
    print(f"Total Chinese voices: {len(chinese_voices)}")
    print(f"Total English voices: {len(english_voices)}")
    print("=" * 80)

    # Recommend best voices for Ani
    print("\n【推荐语音包 for Ani (Anime Companion)】")
    print("-" * 80)

    recommendations = [
        ("zh-CN-XiaoxiaoNeural", "晓晓 - 温柔甜美，适合动漫角色"),
        ("zh-CN-XiaoyiNeural", "晓伊 - 活泼可爱，少女感强"),
        ("zh-CN-YunxiNeural", "云希 - 清新自然（男声，如需男性角色）"),
        ("zh-CN-XiaohanNeural", "晓涵 - 温暖亲切"),
        ("zh-CN-XiaomengNeural", "晓梦 - 童真可爱（儿童音）"),
        ("zh-TW-HsiaoChenNeural", "晓臻 - 台湾口音，温柔"),
        ("en-US-AriaNeural", "Aria - 美式英语，友好自然"),
        ("en-US-JennyNeural", "Jenny - 美式英语，活泼开朗"),
    ]

    for voice_id, description in recommendations:
        print(f"✨ {voice_id:30} - {description}")

    print("\n" + "=" * 80)

if __name__ == "__main__":
    asyncio.run(list_voices())
