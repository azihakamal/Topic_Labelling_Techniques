import cv2
import pytesseract
from pdf2image import convert_from_path
import os
import numpy as np
from PIL import Image
import random

#CONFIG
PDF_PATH = "./media/Pressespiegel_20200602.pdf"
OUTPUT_DIR = "./label_article/sample_articles/output_with_colored_blocks"
os.makedirs(OUTPUT_DIR, exist_ok=True)

#Generate Distinct Colors for Blocks
random.seed(42)
NUM_COLORS = 20
color_list = [tuple(random.choices(range(50, 256), k=3)) for _ in range(NUM_COLORS)]

#Convert PDF to images
pages = convert_from_path(PDF_PATH, dpi=300)

for page_num, pil_image in enumerate(pages):
    print(f"Processing page {page_num + 1}")

    # Convert PIL image to OpenCV (numpy) format
    image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    # Run Tesseract OCR with layout detection
    ocr_data = pytesseract.image_to_data(
        image,
        lang='deu',
        output_type=pytesseract.Output.DICT,
        config="--psm 1"
    )

    # Assign random colors to each block
    block_colors = {}
    n_boxes = len(ocr_data['text'])

    for i in range(n_boxes):
        word = ocr_data['text'][i]
        conf = int(ocr_data['conf'][i])
        block_num = ocr_data['block_num'][i]

        if word.strip() == "" or conf < 30:
            continue

        # Get box coordinates
        x = ocr_data['left'][i]
        y = ocr_data['top'][i]
        w = ocr_data['width'][i]
        h = ocr_data['height'][i]

        # Assign a color to the block
        if block_num not in block_colors:
            block_colors[block_num] = color_list[block_num % NUM_COLORS]
        color = block_colors[block_num]

        # Draw rectangle and label
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        cv2.putText(
            image,
            word,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            1
        )

    # Save annotated image
    out_path = os.path.join(OUTPUT_DIR, f"page_{page_num + 1}.jpg")
    cv2.imwrite(out_path, image)

print("All pages processed. Check the 'output_with_colored_blocks' folder.")
