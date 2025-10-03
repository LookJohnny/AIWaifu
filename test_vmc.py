"""
Test VMC Protocol with different blend shape names
"""
import asyncio
from pythonosc import udp_client

async def test_blend_shapes():
    client = udp_client.SimpleUDPClient("127.0.0.1", 39539)

    # Test all common VRM blend shape names
    test_shapes = [
        # VRM standard
        ("Joy", 1.0),
        ("Angry", 1.0),
        ("Sorrow", 1.0),
        ("Fun", 1.0),

        # Mouth shapes
        ("A", 1.0),
        ("I", 1.0),
        ("U", 1.0),
        ("E", 1.0),
        ("O", 1.0),

        # ARKit style
        ("mouthSmile", 1.0),
        ("eyeWideLeft", 1.0),
        ("eyeWideRight", 1.0),
    ]

    print("Testing VMC blend shapes...")
    print("Watch VSeeFace for changes!\n")

    for shape_name, value in test_shapes:
        print(f"Trying: {shape_name} = {value}")
        client.send_message("/VMC/Ext/Blend/Val", [shape_name, float(value)])
        client.send_message("/VMC/Ext/Blend/Apply", [])
        await asyncio.sleep(1.5)

        # Reset
        client.send_message("/VMC/Ext/Blend/Val", [shape_name, 0.0])
        client.send_message("/VMC/Ext/Blend/Apply", [])
        await asyncio.sleep(0.5)

    print("\nTest complete!")

if __name__ == "__main__":
    asyncio.run(test_blend_shapes())
