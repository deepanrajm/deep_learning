from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.utilities.types import Image
import pyautogui
import io

# Create server
mcp = FastMCP("Screenshot Demo")

@mcp.tool()
def capture_screenshot() -> Image:
    """Capture the current screen and return the image. Use this tool whenever the user requests it."""
    
    buffer = io.BytesIO()
    
    # if the file exceeds ~1MB, it will be rejected by Claude
    screenshot = pyautogui.screenshot()
    screenshot.convert("RGB").save(buffer, format="JPEG", quality=60, optimize=True)
    return Image(data=buffer.getvalue(), format="jpeg")

if __name__ == "__main__":
    mcp.run()