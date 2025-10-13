import logging

from paddleocr import PaddleOCR

logger = logging.getLogger(__name__)

ocr_model = PaddleOCR(
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False)

def digit_edit_distance(a, b, confusion_map, similar_penalty=0.5):
    if len(a) != len(b):
        return float('inf')  # Don't compare if lengths are different

    distance = 0
    for char_a, char_b in zip(a, b):
        if char_a == char_b:
            continue
        elif char_b in confusion_map.get(char_a, []):
            distance += similar_penalty  # Less penalty for similar digits
        else:
            distance += 1  # Full penalty for unrelated digits
    return distance

def ocr_digit_correct(expected_values, detected_value, max_distance=1.0):
    confusion_map = {
        '0': ['6', '8', '9'],
        '1': ['7'],
        '2': ['3', '7'],
        '3': ['2', '5', '8', '9'],
        '4': ['1', '7', '9'],
        '3': ['8', '9'],
        '5': ['2', '6'],
        '6': ['5', '8', '0'],
        '7': ['1'],
        '8': ['0', '6', '9', '3'],
        '9': ['8', '0', '3'],
    }

    candidates = []
    for value in expected_values:
        dist = digit_edit_distance(detected_value, value, confusion_map)
        if dist <= max_distance:
            candidates.append((value, dist))

    if candidates:
        candidates.sort(key=lambda x: x[1])  # Sort by lowest distance
        return candidates[0][0]

    return detected_value  # No good match found


def perform_ocr(image):
    result = ocr_model.predict(input=image)

    texts = []
    for res in result:  
        rec_texts = "".join(res.get('rec_texts', []))
        texts.append(rec_texts)
    
    ocr_value = "".join(texts)

    # TODO: Replace hardcoded expected OCR values with values from RFID readings
    expected_ocr_values = ["1785", "1120", "1032", "2292", "321"]
    expected_ocr_values.extend([str(i).zfill(3) for i in range(1, 51)])
    
    return ocr_digit_correct(expected_ocr_values, ocr_value, max_distance=3)