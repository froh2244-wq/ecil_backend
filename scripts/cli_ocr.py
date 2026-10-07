"""Run PaddleOCR from a terminal and print or save recognized text.

Install the pinned CPU dependencies for the verified Python 3.12 / Windows x64
environment:

    python -m pip install -r requirements.txt

Examples:

    python cli_ocr.py --image receipt.png
    python cli_ocr.py -i receipt.png --lang korean --min-conf 0.5 --json
    python cli_ocr.py -i receipt.png --no-preprocess
    python cli_ocr.py -i receipt.png --no-cls --output receipt.txt
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import re
import sys
import unicodedata
from contextlib import redirect_stdout
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence


_NUTRITION_CONTEXT_RE = re.compile(
    r"영양\s*(?:성분|정보)?|열량|칼로리|나트륨|탄수화물|당류|식이섬유|"
    r"단백질|(?:트랜스|포화|총)?지방(?:산)?|콜레스테롤|칼슘|칼륨|철분|비타민|"
    r"\b(?:nutrition|energy|calories?|sodium|carbohydrates?|sugars?|fiber|"
    r"protein|fat|cholesterol|calcium|potassium|iron|vitamins?)\b",
    re.IGNORECASE,
)
_GRAM_NUTRIENT_RE = re.compile(
    r"탄수화물|당류|식이섬유|단백질|(?:트랜스|포화|총)?지방(?:산)?|"
    r"\b(?:carbohydrates?|sugars?|fiber|protein|fat)\b",
    re.IGNORECASE,
)
_OCR_NG_UNIT_RE = re.compile(r"(?P<amount>\d+(?:[.,]\d+)?)\s*ng\b", re.IGNORECASE)
_OCR_NINE_AS_GRAM_RE = re.compile(
    r"(?P<amount>\d+(?:[.,]\d+)?)\s*9"
    r"(?!\s*(?:mg|ng|g)\b)(?=$|[\s,;:)\]])",
    re.IGNORECASE,
)
_SUPPORTED_LANGUAGES = {"korean", "en"}
_TEXT_DETECTION_MODEL = "PP-OCRv6_medium_det"
_TEXT_RECOGNITION_MODEL = "korean_PP-OCRv5_mobile_rec"
_USER_ALLERGENS = ("복숭아", "오징어", "땅콩", "우유")
_ALLERGY_REPORT_SEPARATOR = "---------------------------------"


def _paddleocr_api_diagnostic(module: Any) -> str:
    """Describe the imported package when the expected 3.x API is missing."""
    try:
        version = importlib.metadata.version("paddleocr")
    except importlib.metadata.PackageNotFoundError:
        version = "unknown (distribution metadata not found)"
    module_path = getattr(module, "__file__", "unknown")
    return (
        f"Imported paddleocr version: {version}; module: {module_path}. "
        "This CLI requires the PaddleOCR 3.7 predict() API. Install the pinned "
        "dependencies with the same interpreter used to run this script: "
        "python -m pip install -r requirements.txt. Verify with: "
        "python -c \"import paddleocr; print(paddleocr.__version__, paddleocr.__file__)\""
    )


@dataclass
class OCRResult:
    """One recognized text item and its polygon coordinates."""

    text: str
    confidence: float
    box: list[list[float]] | None


def detect_allergens(
    records: Sequence[OCRResult],
    allergens: Sequence[str] = _USER_ALLERGENS,
) -> list[str]:
    """Return configured allergens found in the recognized OCR text.

    Whitespace is ignored so a recognized ingredient such as ``복 숭아`` is
    still detected. Matching is case-insensitive for any Latin-script terms.
    """
    normalized_text = "".join(
        re.sub(r"\s+", "", unicodedata.normalize("NFKC", record.text)).casefold()
        for record in records
    )
    return [
        allergen
        for allergen in allergens
        if allergen
        and re.sub(
            r"\s+", "", unicodedata.normalize("NFKC", allergen)
        ).casefold()
        in normalized_text
    ]


def print_allergy_report(
    records: Sequence[OCRResult], *, file: Any = sys.stdout
) -> None:
    """Print the test allergen assessment for the recognized text."""
    detected = detect_allergens(records)
    print(_ALLERGY_REPORT_SEPARATOR, file=file)
    print("사용자 알레르기 성분: 복숭아, 오징어, 땅콩, 우유", file=file)
    if detected:
        print("알레르기 감지: 알레르기 유발 원인 감지!", file=file)
        print(f"감지된 알레르기 성분: {', '.join(detected)}", file=file)
    else:
        print("알레르기 감지: (알레르기 성분 감지 안됨)", file=file)
    print(_ALLERGY_REPORT_SEPARATOR, file=file)


def _confidence_threshold(value: str) -> float:
    try:
        threshold = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number between 0.0 and 1.0") from exc
    if not 0.0 <= threshold <= 1.0:
        raise argparse.ArgumentTypeError("must be between 0.0 and 1.0")
    return threshold


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract text from an image with PaddleOCR."
    )
    parser.add_argument(
        "-i", "--image", required=True, help="path to the image to analyze"
    )
    parser.add_argument(
        "-l",
        "--lang",
        default="korean",
        help='recognition language supported by the Korean model: "korean" or "en" (default: "korean")',
    )
    parser.add_argument(
        "--cls",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="use text-line orientation classification (enabled by default; use --no-cls to disable)",
    )
    parser.add_argument(
        "--preprocess",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="preprocess the image with OpenCV (enabled by default; use --no-preprocess to disable)",
    )
    parser.add_argument(
        "--min-conf",
        type=_confidence_threshold,
        default=0.0,
        metavar="FLOAT",
        help="minimum confidence to include, from 0.0 to 1.0 (default: 0.0)",
    )
    parser.add_argument(
        "-o", "--output", help="optional path to save recognized text as UTF-8"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="print a JSON array with text, confidence, and box coordinates",
    )
    return parser


def _as_sequence(value: Any) -> Sequence[Any] | None:
    if isinstance(value, (list, tuple)):
        return value
    # Paddle may expose coordinates as array-like values (for example, NumPy).
    to_list = getattr(value, "tolist", None)
    if callable(to_list):
        converted = to_list()
        if isinstance(converted, (list, tuple)):
            return converted
    return None


def _number_float(value: Any) -> float | None:
    """Convert Python and NumPy scalar values to finite floats."""
    if isinstance(value, bool):
        return None
    item = getattr(value, "item", None)
    if callable(item):
        try:
            value = item()
        except (TypeError, ValueError):
            return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _parse_box(value: Any) -> list[list[float]] | None:
    points = _as_sequence(value)
    if points is None or not points:
        return None

    # PaddleX may expose rec_boxes as [x_min, y_min, x_max, y_max]. Convert
    # these rectangles to the polygon shape used by the CLI's JSON response.
    flat = [_number_float(coordinate) for coordinate in points]
    if len(points) == 4 and all(coordinate is not None for coordinate in flat):
        x_min, y_min, x_max, y_max = flat
        return [
            [x_min, y_min],
            [x_max, y_min],
            [x_max, y_max],
            [x_min, y_max],
        ]

    parsed: list[list[float]] = []
    for point in points:
        coordinates = _as_sequence(point)
        if coordinates is None or len(coordinates) < 2:
            return None
        x, y = _number_float(coordinates[0]), _number_float(coordinates[1])
        if x is None or y is None:
            return None
        parsed.append([x, y])
    return parsed or None


def _prediction_payload(prediction: Any) -> dict[str, Any]:
    """Read the JSON payload exposed by a PaddleOCR 3.x result object."""
    if isinstance(prediction, dict):
        payload: Any = prediction
    else:
        payload = getattr(prediction, "json", None)
        if callable(payload):
            payload = payload()

    if isinstance(payload, str):
        payload = json.loads(payload)
    if not isinstance(payload, dict):
        raise ValueError("PaddleOCR result does not contain a JSON object")

    result = payload.get("res", payload)
    if not isinstance(result, dict):
        raise ValueError("PaddleOCR result has an invalid 'res' object")
    return result


def parse_ocr_output(raw: Any, min_confidence: float = 0.0) -> list[OCRResult]:
    """Parse PaddleOCR 3.x result objects and correct nutrition-label units."""
    records: list[OCRResult] = []
    if raw is None:
        return records

    if isinstance(raw, dict) or hasattr(raw, "json"):
        predictions = [raw]
    else:
        try:
            predictions = list(raw)
        except TypeError:
            predictions = [raw]

    for prediction in predictions:
        result = _prediction_payload(prediction)
        texts = _as_sequence(result.get("rec_texts"))
        scores = _as_sequence(result.get("rec_scores"))
        boxes_value = result.get("rec_polys")
        if boxes_value is None:
            boxes_value = result.get("rec_boxes")
        boxes = _as_sequence(boxes_value)

        if texts is None or scores is None:
            raise ValueError("PaddleOCR result is missing rec_texts or rec_scores")
        if len(texts) != len(scores):
            raise ValueError(
                "PaddleOCR rec_texts and rec_scores have different lengths"
            )
        if boxes is not None and len(boxes) != len(texts):
            raise ValueError(
                "PaddleOCR recognition text and box arrays have different lengths"
            )

        for index, (text, raw_score) in enumerate(zip(texts, scores)):
            if not isinstance(text, str):
                raise ValueError(f"PaddleOCR rec_texts[{index}] is not text")
            score = _number_float(raw_score)
            if score is None:
                raise ValueError(f"PaddleOCR rec_scores[{index}] is not numeric")
            if score < min_confidence:
                continue
            box = _parse_box(boxes[index]) if boxes is not None else None
            records.append(OCRResult(text=text, confidence=score, box=box))

    # On nutrition rows, OCR commonly confuses a trailing "g" with "9" and
    # the first letter in "mg" with "n". Keep corrections nutrition-specific.
    for record in records:
        if not _NUTRITION_CONTEXT_RE.search(record.text):
            continue
        record.text = _OCR_NG_UNIT_RE.sub(r"\g<amount>mg", record.text)
        if _GRAM_NUTRIENT_RE.search(record.text):
            record.text = _OCR_NINE_AS_GRAM_RE.sub(r"\g<amount>g", record.text)

    return records


def preprocess_image(image_path: Path) -> Any:
    """Decode and enhance an image with OpenCV for OCR inference."""
    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        raise RuntimeError(
            "OpenCV and NumPy are required for preprocessing; install them with "
            "'pip install opencv-contrib-python numpy', or use --no-preprocess."
        ) from exc

    try:
        # fromfile + imdecode supports non-ASCII paths on Windows.
        encoded = np.fromfile(str(image_path), dtype=np.uint8)
        image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    except (OSError, cv2.error) as exc:
        raise ValueError(f"could not read image with OpenCV: {exc}") from exc
    if image is None:
        raise ValueError(f"OpenCV could not decode image: {image_path}")

    height, width = image.shape[:2]
    longest_side = max(height, width)
    if longest_side < 960:
        scale = min(2.0, 960.0 / longest_side)
        image = cv2.resize(
            image,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_CUBIC,
        )

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)


def _display_width(value: str) -> int:
    width = 0
    for char in value:
        if unicodedata.combining(char):
            continue
        width += 2 if unicodedata.east_asian_width(char) in {"W", "F"} else 1
    return width


def _pad_right(value: str, width: int) -> str:
    return value + " " * max(0, width - _display_width(value))


def print_table(records: Sequence[OCRResult]) -> None:
    index_width = max(3, len(str(len(records))))
    confidence_width = len("Confidence")
    text_width = max([len("Text"), *(_display_width(item.text.replace("\n", " ")) for item in records)])

    print(f"{'No.':>{index_width}}  {_pad_right('Confidence', confidence_width)}  Text")
    print(f"{'-' * index_width}  {'-' * confidence_width}  {'-' * text_width}")
    for index, item in enumerate(records, start=1):
        text = item.text.replace("\r", " ").replace("\n", " ")
        confidence = f"{item.confidence:.4f}"
        print(
            f"{index:>{index_width}}  "
            f"{confidence:>{confidence_width}}  "
            f"{_pad_right(text, text_width)}"
        )


def run(args: argparse.Namespace) -> int:
    language = args.lang.strip().casefold()
    if language not in _SUPPORTED_LANGUAGES:
        print(
            f"Error: unsupported language '{args.lang}'. "
            "The selected Korean recognition model supports Korean, English, "
            "and digits; use --lang korean or --lang en.",
            file=sys.stderr,
        )
        return 1
    # The explicit Korean recognition model controls the character set. --lang
    # is validated here so unsupported languages never silently use that model.

    image_path = Path(args.image).expanduser()
    if not image_path.is_file():
        print(f"Error: image file does not exist: {image_path}", file=sys.stderr)
        return 1

    ocr_input: Any = str(image_path)
    if args.preprocess:
        print("Preprocessing image with OpenCV...", file=sys.stderr)
        try:
            ocr_input = preprocess_image(image_path)
        except (RuntimeError, ValueError) as exc:
            print(f"Error: image preprocessing failed: {exc}", file=sys.stderr)
            return 1

    print("Initializing PaddleOCR model...", file=sys.stderr)
    try:
        # Some Paddle/PaddleOCR builds emit setup messages directly to stdout.
        # Redirect those messages so stdout stays suitable for piping/JSON.
        with redirect_stdout(sys.stderr):
            import paddleocr
            from paddleocr import PaddleOCR

            ocr = PaddleOCR(
                text_detection_model_name="PP-OCRv6_medium_det",
                text_recognition_model_name="korean_PP-OCRv5_mobile_rec",
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=args.cls,
                engine="paddle_static",
                enable_mkldnn=False,
            )
            if not callable(getattr(ocr, "predict", None)):
                raise RuntimeError(
                    "The imported PaddleOCR class does not provide predict(). "
                    + _paddleocr_api_diagnostic(paddleocr)
                )
            print("Running OCR...", file=sys.stderr)
            raw_result = list(ocr.predict(input=ocr_input))
    except Exception as exc:
        print(
            f"Error: PaddleOCR 3.7 initialization or inference failed: "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 1

    try:
        records = parse_ocr_output(raw_result, args.min_conf)
    except Exception as exc:
        print(
            f"Error: could not parse PaddleOCR 3.x results: "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 1

    if args.output:
        try:
            output_path = Path(args.output).expanduser()
            output_path.write_text(
                "\n".join(item.text for item in records)
                + ("\n" if records else ""),
                encoding="utf-8",
            )
        except OSError as exc:
            print(f"Error: could not write output file: {exc}", file=sys.stderr)
            return 1

    if args.json:
        print(
            json.dumps([asdict(item) for item in records], ensure_ascii=False),
            flush=True,
        )
        # Keep stdout as a single valid JSON value for scripts and pipelines.
        print_allergy_report(records, file=sys.stderr)
    else:
        print_table(records)
        print_allergy_report(records)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
