"""
Animation Controller - VMC Protocol Integration for VSeeFace
Handles real-time expression control and lip-sync coordination
"""

import asyncio
import logging
from typing import Optional, Dict
from pythonosc import udp_client
from pythonosc.osc_message_builder import OscMessageBuilder

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AnimationController:
    """
    Controls 3D character animations via VMC (Virtual Motion Capture) protocol.
    Communicates with VSeeFace using OSC (Open Sound Control) over UDP.
    """

    # Emotion to blend shape mapping
    # VRM ARKit blend shape names (standard)
    EMOTION_BLEND_SHAPES = {
        "joy": "fun",  # VSeeFace uses "fun" for happy
        "sad": "sad",
        "anger": "angry",
        "surprise": "surprised",
        "neutral": "neutral",
        "excited": "fun",  # Map to happy/fun
        "confused": "sad",  # Map to sad/uncertain
        "curious": "surprised"  # Map to surprised/interested
    }

    def __init__(self, host: str = "127.0.0.1", port: int = 39539):
        """
        Initialize VMC OSC client

        Args:
            host: VSeeFace IP address (default: localhost)
            port: VMC protocol port (default: 39539)
        """
        self.host = host
        self.port = port
        self.client: Optional[udp_client.SimpleUDPClient] = None
        self.connected = False
        self.current_expression: Optional[str] = None
        self.current_intensity: float = 0.0

        # Expression transition settings
        self.transition_duration = 0.3  # seconds for smooth transitions
        self.reset_delay = 2.0  # seconds to hold expression before returning to neutral

        self._connect()

    def _connect(self):
        """Establish OSC connection to VSeeFace"""
        try:
            self.client = udp_client.SimpleUDPClient(self.host, self.port)
            self.connected = True
            logger.info(f"[OK] Animation controller connected to VSeeFace at {self.host}:{self.port}")
        except Exception as e:
            self.connected = False
            logger.warning(f"[WARN] Could not connect to VSeeFace: {e}")
            logger.warning("Animation features disabled - voice will continue to work")

    async def set_expression(self, emotion: str, intensity: float = 1.0):
        """
        Set character facial expression

        Args:
            emotion: Emotion type (joy, sad, anger, surprise, neutral)
            intensity: Expression intensity 0.0-1.0
        """
        if not self.connected or not self.client:
            logger.debug(f"Animation disabled - would show {emotion} at {intensity:.1f}")
            return

        # Validate emotion
        if emotion not in self.EMOTION_BLEND_SHAPES:
            logger.warning(f"Unknown emotion: {emotion}, defaulting to neutral")
            emotion = "neutral"

        # Clamp intensity
        intensity = max(0.0, min(1.0, intensity))

        # Get blend shape name
        blend_shape = self.EMOTION_BLEND_SHAPES[emotion]

        try:
            # Reset previous expression first (smooth transition)
            if self.current_expression and self.current_expression != emotion:
                await self._set_blend_shape(
                    self.EMOTION_BLEND_SHAPES[self.current_expression],
                    0.0
                )

            # Set new expression
            await self._set_blend_shape(blend_shape, intensity)

            # Update state
            self.current_expression = emotion
            self.current_intensity = intensity

            logger.info(f"[Expression] {emotion} ({intensity:.1%})")

        except Exception as e:
            logger.error(f"Failed to set expression: {e}")

    async def _set_blend_shape(self, blend_shape_name: str, value: float):
        """
        Send blend shape value via VMC protocol

        VMC Protocol Format:
        /VMC/Ext/Blend/Val <blend_shape_name> <value>

        Args:
            blend_shape_name: Name of blend shape (Joy, Sorrow, Angry, etc.)
            value: Blend value 0.0-1.0
        """
        if not self.client:
            return

        try:
            # Send OSC message
            self.client.send_message(
                "/VMC/Ext/Blend/Val",
                [blend_shape_name, float(value)]
            )

            # Apply the blend shape changes
            self.client.send_message("/VMC/Ext/Blend/Apply", [])

            logger.debug(f"Sent: /VMC/Ext/Blend/Val {blend_shape_name} {value:.2f}")

        except Exception as e:
            logger.error(f"Failed to send blend shape: {e}")
            self.connected = False

    async def reset_to_neutral(self, delay: float = None):
        """
        Return character to neutral expression

        Args:
            delay: Seconds to wait before resetting (default: self.reset_delay)
        """
        if delay is None:
            delay = self.reset_delay

        if delay > 0:
            await asyncio.sleep(delay)

        await self.set_expression("neutral", 0.3)  # Subtle neutral

    async def trigger_lipsync(self, audio_file: str = None):
        """
        Trigger lip-sync animation

        Note: VSeeFace automatically handles lip-sync from system audio,
        so this is a placeholder for future enhancements (phoneme-based sync)

        Args:
            audio_file: Path to audio file (for future phoneme analysis)
        """
        if not self.connected:
            return

        # VSeeFace handles lip-sync automatically via microphone/system audio
        # This method is here for future enhancements like phoneme-based control
        logger.debug("Lip-sync handled automatically by VSeeFace")

    async def set_mouth_open(self, value: float):
        """
        Manually control mouth opening (for advanced lip-sync)

        Args:
            value: Mouth open amount 0.0-1.0
        """
        if not self.connected or not self.client:
            return

        try:
            # VRM standard mouth blend shape
            await self._set_blend_shape("A", value)  # 'A' vowel shape
        except Exception as e:
            logger.error(f"Failed to set mouth: {e}")

    def close(self):
        """Cleanup resources"""
        if self.connected:
            logger.info("Animation controller closed")
            self.connected = False
            self.client = None

    def __del__(self):
        """Destructor"""
        self.close()


# Standalone test
async def test_animation_controller():
    """Test the animation controller with VSeeFace"""
    print("=" * 60)
    print("Animation Controller Test")
    print("=" * 60)
    print("Make sure VSeeFace is running with VMC receiver enabled!")
    print("Port: 39539")
    print()

    controller = AnimationController()

    if not controller.connected:
        print("[FAIL] Could not connect to VSeeFace")
        print("1. Launch VSeeFace")
        print("2. Load a VRM character")
        print("3. Enable VMC receiver in settings (port 39539)")
        return

    print("[OK] Connected to VSeeFace")
    print()

    # Test each emotion
    emotions = [
        ("joy", 1.0),
        ("sad", 0.8),
        ("anger", 0.9),
        ("surprise", 1.0),
        ("neutral", 0.5)
    ]

    for emotion, intensity in emotions:
        print(f"Testing: {emotion} (intensity: {intensity})")
        await controller.set_expression(emotion, intensity)
        await asyncio.sleep(2)  # Hold for 2 seconds

    # Return to neutral
    print("Returning to neutral...")
    await controller.reset_to_neutral(delay=0)

    print()
    print("[OK] Test complete!")
    controller.close()


if __name__ == "__main__":
    # Run test if executed directly
    asyncio.run(test_animation_controller())
