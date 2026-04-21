# Scale bar calibration using OCR and line detection

import cv2
import numpy as np
import pytesseract
import re
import os
import platform
from typing import Tuple, Optional, Dict


# Try to auto-detect Tesseract on Windows
def _setup_tesseract():
    if platform.system() == 'Windows':
        possible_paths = [
            r'C:\Program Files\Tesseract-OCR\tesseract.exe',
            r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
            r'C:\Users\{}\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'.format(os.getenv('USERNAME')),
        ]

        for path in possible_paths:
            if os.path.exists(path):
                pytesseract.pytesseract.tesseract_cmd = path
                return True

        # Check if it's in PATH
        try:
            import subprocess
            result = subprocess.run(['tesseract', '--version'],
                                  capture_output=True,
                                  timeout=2)
            if result.returncode == 0:
                return True
        except:
            pass

    return False


_tesseract_available = _setup_tesseract()


class ScaleBarCalibrator:
    """Detects scale bars in SEM images using OCR + line detection."""

    def __init__(self, image: np.ndarray, verbose: bool = False):
        self.image = image
        self.verbose = verbose
        if len(image.shape) == 3:
            self.gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            self.gray = image.copy()

        self.scale_value = None
        self.scale_unit = None
        self.scale_length_pixels = None
        self.pixels_per_unit = None

    def calibrate(self, search_region: str = "bottom") -> Dict:
        """Main calibration: extract text + detect line."""
        # Step 1: Read the scale bar text with OCR
        text_result = self.extract_scale_text(search_region)

        if text_result is None:
            raise ValueError("Could not detect scale bar text")

        self.scale_value = text_result['value']
        self.scale_unit = text_result['unit']
        text_bbox = text_result['bbox']

        # Step 2: Find the scale bar line
        line_result = self.detect_scale_line(text_bbox, search_region)

        if line_result is None:
            raise ValueError("Could not detect scale bar line")

        self.scale_length_pixels = line_result['length']

        # Step 3: Calculate conversion (pixels per micrometer)
        self.pixels_per_unit = self.scale_length_pixels / self.scale_value

        return {
            'scale_value': self.scale_value,
            'scale_unit': self.scale_unit,
            'scale_length_pixels': self.scale_length_pixels,
            'pixels_per_unit': self.pixels_per_unit,
            'conversion_factor': 1.0 / self.pixels_per_unit,
            'text_bbox': text_bbox,
            'line_coords': line_result['coords']
        }

    def extract_scale_text(self, search_region: str = "bottom") -> Optional[Dict]:
        """Use OCR to read scale bar text (e.g., "0.3 µm")."""
        h, w = self.gray.shape

        # Define search region
        if search_region == "bottom":
            roi = self.gray[int(h * 0.85):h, :]
            y_offset = int(h * 0.85)
            x_offset = 0
        elif search_region == "top":
            roi = self.gray[0:int(h * 0.15), :]
            y_offset = 0
            x_offset = 0
        else:  # full
            roi = self.gray
            y_offset = 0
            x_offset = 0

        # Preprocess for OCR
        roi_processed = self._preprocess_for_ocr(roi)

        # Run Tesseract OCR
        try:
            ocr_data = pytesseract.image_to_data(
                roi_processed,
                output_type=pytesseract.Output.DICT,
                config='--psm 11'  # sparse text mode
            )
        except Exception as e:
            error_msg = f"OCR failed: {e}"

            if 'tesseract is not installed' in str(e).lower() or 'not in your path' in str(e).lower():
                print("\n" + "="*60)
                print("ERROR: Tesseract OCR not found!")
                print("="*60)
                print("\nQuick Fix:")
                print("  Add this at the top of your script:")
                print("     import pytesseract")
                print("     pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'")
                print("="*60 + "\n")
            else:
                print(error_msg)

            return None

        # Parse OCR results to find scale text
        if self.verbose:
            print(f"  OCR detected {len([t for t in ocr_data['text'] if t.strip()])} text items")

        for i, text in enumerate(ocr_data['text']):
            if text.strip() == '':
                continue

            if self.verbose:
                print(f"    Checking: '{text}'")

            # Try to parse (e.g., "0.3µm", "300nm")
            parsed = self._parse_scale_text(text)
            if parsed:
                if self.verbose:
                    print(f"      Parsed: {parsed['value']} {parsed['unit']}")

                # Get bounding box
                x, y, w_box, h_box = (
                    ocr_data['left'][i],
                    ocr_data['top'][i],
                    ocr_data['width'][i],
                    ocr_data['height'][i]
                )

                return {
                    'value': parsed['value'],
                    'unit': parsed['unit'],
                    'bbox': (x + x_offset, y + y_offset, w_box, h_box),
                    'raw_text': text
                }

        return None

    def _parse_scale_text(self, text: str) -> Optional[Dict]:
        """Parse text like "0.3µm", "300 nm", etc."""
        text = text.strip().replace(' ', '').lower()

        # Common patterns
        patterns = [
            r'(\d+\.?\d*)\s*µm',
            r'(\d+\.?\d*)\s*um',
            r'(\d+\.?\d*)\s*μm',
            r'(\d+\.?\d*)\s*nm',
            r'(\d+\.?\d*)\s*pm',
        ]

        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                value = float(match.group(1))

                # Determine unit
                if 'nm' in text:
                    unit = 'nm'
                elif 'pm' in text:
                    unit = 'pm'
                else:
                    unit = 'µm'

                return {'value': value, 'unit': unit}

        return None

    def _preprocess_for_ocr(self, roi: np.ndarray) -> np.ndarray:
        """Enhance image for OCR - invert if text is white-on-black."""
        mean_val = cv2.mean(roi)[0]

        # If mostly dark, text is probably white, so invert
        if mean_val < 127:
            return cv2.bitwise_not(roi)
        else:
            return roi

    def detect_scale_line(self, text_bbox: Tuple[int, int, int, int],
                         search_region: str = "bottom") -> Optional[Dict]:
        """
        Find scale bar line by looking for horizontal bright lines with
        vertical ticks at the ends (L-shaped brackets). Trim 8.7% from
        each end to exclude the bracket extensions.
        """
        x_text, y_text, w_text, h_text = text_bbox
        height, width = self.gray.shape

        # Search area around the text
        search_margin = 100
        y_start = max(5, y_text - search_margin)
        y_end = min(height - 2, y_text + h_text + search_margin)

        best_line = None
        max_score = -1

        # Scan each row for horizontal bright runs
        for y in range(y_start, y_end):
            run = 0
            start_x = 0

            for x in range(width):
                # Bright pixel (>200)
                if self.gray[y, x] > 200:
                    if run == 0:
                        start_x = x
                    run += 1
                elif run > 0:
                    # Valid candidate: 20-150px
                    if 20 < run < width * 0.45:
                        score = run

                        # Look for vertical ticks at ends (brackets)
                        has_vertex = False
                        for dy in range(-10, 11):
                            if dy == 0:
                                continue
                            ty = y + dy
                            if 0 <= ty < height:
                                # Check for vertical at start or end
                                for dx in range(-2, 3):
                                    x_left = min(width - 1, max(0, start_x + dx))
                                    x_right = min(width - 1, max(0, start_x + run - 1 + dx))
                                    if self.gray[ty, x_left] > 180 or self.gray[ty, x_right] > 180:
                                        has_vertex = True
                                        break
                                if has_vertex:
                                    break

                        # Only use lines with brackets
                        if has_vertex:
                            # Trim 8.7% from each end (brackets add ~10.5px each side)
                            trim_percentage = 0.087
                            trim_amount = int(run * trim_percentage)

                            trimmed_start = start_x + trim_amount
                            trimmed_end = start_x + run - 1 - trim_amount
                            trimmed_pixels = trimmed_end - trimmed_start + 1

                            # Minimum length check
                            if trimmed_pixels < 20:
                                trimmed_pixels = run
                                trimmed_start = start_x
                                trimmed_end = start_x + run - 1

                            score = trimmed_pixels + 10000

                            # Penalize if touching text
                            noise_neighbors = 0
                            if start_x > 2 and self.gray[y, start_x - 2] > 150:
                                noise_neighbors += 1
                            if start_x + run < width - 2 and self.gray[y, start_x + run + 2] > 150:
                                noise_neighbors += 1

                            if noise_neighbors > 0:
                                score -= 5000

                            if score > max_score:
                                max_score = score
                                best_line = {'x': trimmed_start, 'y': y, 'pixels': trimmed_pixels}

                    run = 0

            # Handle last run if at edge
            if run > 0 and 20 < run < width * 0.45:
                if run > max_score:
                    max_score = run
                    best_line = {'x': start_x, 'y': y, 'pixels': run}

        if best_line is None:
            return None

        return {
            'length': best_line['pixels'],
            'coords': (best_line['x'], best_line['y'],
                      best_line['x'] + best_line['pixels'], best_line['y'])
        }

    def pixels_to_units(self, pixels: float) -> float:
        """Convert pixels to micrometers (or whatever unit was detected)."""
        if self.pixels_per_unit is None:
            raise ValueError("Calibration not performed. Call calibrate() first.")
        return pixels / self.pixels_per_unit

    def units_to_pixels(self, units: float) -> float:
        """Convert micrometers to pixels."""
        if self.pixels_per_unit is None:
            raise ValueError("Calibration not performed. Call calibrate() first.")
        return units * self.pixels_per_unit


def calibrate_image(image_path: str, search_region: str = "bottom") -> ScaleBarCalibrator:
    """Helper function to calibrate from image file path."""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")

    calibrator = ScaleBarCalibrator(image)
    calibrator.calibrate(search_region)

    return calibrator
