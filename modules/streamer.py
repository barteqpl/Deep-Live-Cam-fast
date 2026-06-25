import subprocess
import cv2
import logging

logger = logging.getLogger("DLC.STREAMER")

class UDPStreamer:
    """Helper to stream processed frames over UDP using ffmpeg subprocess.
    Encodes raw BGR24 frames to h264 and sends via MPEG-TS container."""
    def __init__(self, address: str, width: int, height: int, fps: float = 30.0):
        self.address = address
        self.width = width
        self.height = height
        self.fps = fps
        self.process = None

    def start(self) -> bool:
        # Resolve address format
        if ":" not in self.address:
            # If it's just a port, default to 127.0.0.1
            addr = f"127.0.0.1:{self.address}"
        else:
            addr = self.address

        # Build ffmpeg command line
        cmd = [
            "ffmpeg",
            "-y",
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-s", f"{self.width}x{self.height}",
            "-r", str(self.fps),
            "-i", "-",
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-tune", "zerolatency",
            "-f", "mpegts",
            f"udp://{addr}?pkt_size=1316"
        ]

        try:
            self.process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            logger.info(f"UDP Streamer started towards udp://{addr}")
            return True
        except Exception as e:
            logger.error(f"Failed to start ffmpeg for UDP streaming: {e}")
            return False

    def write(self, frame) -> None:
        if self.process is not None and self.process.stdin is not None:
            # Ensure frame size matches the expected width/height
            if frame.shape[1] != self.width or frame.shape[0] != self.height:
                frame = cv2.resize(frame, (self.width, self.height))
            try:
                self.process.stdin.write(frame.tobytes())
                self.process.stdin.flush()
            except IOError:
                # Process might have terminated/closed
                pass

    def stop(self) -> None:
        if self.process is not None:
            try:
                if self.process.stdin is not None:
                    self.process.stdin.close()
            except Exception:
                pass
            try:
                self.process.terminate()
                self.process.wait(timeout=1.0)
            except Exception:
                pass
            self.process = None
            logger.info("UDP Streamer stopped.")
