import pytesseract
from PIL import Image

def get_ocr(img: Image.Image) -> str:
    """
    Extract all text from a PIL Image using Tesseract OCR.
    
    Args:
        img: PIL Image object (RGB or grayscale).
    
    Returns:
        Extracted text as a string.
    """
    # Optional: convert to grayscale or RGB if needed (pytesseract handles both)
    return pytesseract.image_to_string(img)