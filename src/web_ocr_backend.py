"""FastAPI OCR backend using the PaddleOCR 3.x general OCR pipeline.

The PaddleOCR model is initialized once during FastAPI lifespan startup and
reused for requests. Importing this module does not download or initialize a
model; model loading starts when the ASGI application starts.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import threading
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, Callable, Literal, Sequence

from fastapi import FastAPI, File, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from starlette.exceptions import HTTPException as StarletteHTTPException


logger = logging.getLogger("web_ocr_backend")

TEXT_DETECTION_MODEL = "PP-OCRv6_medium_det"
TEXT_RECOGNITION_MODEL = "korean_PP-OCRv5_mobile_rec"

# Bound the size of a single request and the amount of decoded image data.
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MAX_IMAGE_PIXELS = 40_000_000

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


class ErrorBody(BaseModel):
    code: str
    message: str
    details: list[dict[str, str]] | None = None


class ErrorResponse(BaseModel):
    error: ErrorBody


class ModelNames(BaseModel):
    text_detection: str = TEXT_DETECTION_MODEL
    text_recognition: str = TEXT_RECOGNITION_MODEL


class HealthResponse(BaseModel):
    status: Literal["ok", "unavailable"]
    model_ready: bool
    model_state: Literal["initializing", "ready", "failed"]
    models: ModelNames
    error: ErrorBody | None = None


class OCRItem(BaseModel):
    text: str
    confidence: float
    box: list[list[float]]


class ImageDimensions(BaseModel):
    width: int
    height: int
    processed_width: int
    processed_height: int


class OCRSuccess(BaseModel):
    success: Literal[True] = True
    filename: str
    image: ImageDimensions
    count: int
    results: list[OCRItem]
    models: ModelNames


@dataclass
class PreprocessedImage:
    image: Any
    width: int
    height: int
    processed_width: int
    processed_height: int


@dataclass
class OCRRecord:
    text: str
    confidence: float
    box: list[list[float]]


class ImageInputError(ValueError):
    """An upload is not a decodable image or exceeds an image limit."""

    def __init__(self, code: str, message: str, status_code: int = 422) -> None:
        super().__init__(message)
        self.code = code
        self.status_code = status_code


def _as_sequence(value: Any) -> Sequence[Any] | None:
    if isinstance(value, (list, tuple)):
        return value
    # Paddle results commonly expose NumPy arrays.
    to_list = getattr(value, "tolist", None)
    if callable(to_list):
        converted = to_list()
        if isinstance(converted, (list, tuple)):
            return converted
    return None


def _number_float(value: Any) -> float | None:
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


def _parse_box(value: Any, index: int) -> list[list[float]]:
    points = _as_sequence(value)
    if points is None or not points:
        raise ValueError(f"PaddleOCR box at index {index} is missing or empty")

    flat = [_number_float(coordinate) for coordinate in points]
    # rec_boxes uses [x_min, y_min, x_max, y_max]; expose the same polygon
    # representation for either rec_polys or the rectangular fallback.
    if len(points) == 4 and all(coordinate is not None for coordinate in flat):
        x_min, y_min, x_max, y_max = flat
        return [
            [x_min, y_min],
            [x_max, y_min],
            [x_max, y_max],
            [x_min, y_max],
        ]

    parsed: list[list[float]] = []
    for point_index, point in enumerate(points):
        coordinates = _as_sequence(point)
        if coordinates is None or len(coordinates) < 2:
            raise ValueError(
                f"PaddleOCR box at index {index} has invalid point {point_index}"
            )
        x = _number_float(coordinates[0])
        y = _number_float(coordinates[1])
        if x is None or y is None:
            raise ValueError(
                f"PaddleOCR box at index {index} has non-numeric coordinates"
            )
        parsed.append([x, y])
    if not parsed:
        raise ValueError(f"PaddleOCR box at index {index} is empty")
    return parsed


def _prediction_payload(prediction: Any) -> dict[str, Any]:
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


def parse_ocr_output(raw: Any) -> list[OCRRecord]:
    """Pair PaddleOCR 3.x texts, scores, and coordinates by array index."""
    if raw is None:
        raise ValueError("PaddleOCR returned no result object")

    if isinstance(raw, dict) or hasattr(raw, "json"):
        predictions = [raw]
    else:
        try:
            predictions = list(raw)
        except TypeError as exc:
            raise ValueError("PaddleOCR result is not iterable") from exc
    if not predictions:
        raise ValueError("PaddleOCR returned an empty result list")

    records: list[OCRRecord] = []
    for prediction_index, prediction in enumerate(predictions):
        result = _prediction_payload(prediction)
        texts = _as_sequence(result.get("rec_texts"))
        scores = _as_sequence(result.get("rec_scores"))
        boxes_value = result.get("rec_polys")
        if boxes_value is None:
            boxes_value = result.get("rec_boxes")
        boxes = _as_sequence(boxes_value)

        if texts is None:
            raise ValueError(
                f"PaddleOCR result {prediction_index} is missing rec_texts"
            )
        if scores is None:
            raise ValueError(
                f"PaddleOCR result {prediction_index} is missing rec_scores"
            )
        if boxes is None:
            raise ValueError(
                f"PaddleOCR result {prediction_index} is missing rec_polys/rec_boxes"
            )
        if len(texts) != len(scores):
            raise ValueError(
                f"PaddleOCR result {prediction_index}: rec_texts and rec_scores "
                "have different lengths"
            )
        if len(texts) != len(boxes):
            raise ValueError(
                f"PaddleOCR result {prediction_index}: recognition text and box "
                "arrays have different lengths"
            )

        for index, (text, raw_score) in enumerate(zip(texts, scores)):
            if not isinstance(text, str):
                raise ValueError(f"PaddleOCR rec_texts[{index}] is not text")
            score = _number_float(raw_score)
            if score is None:
                raise ValueError(f"PaddleOCR rec_scores[{index}] is not numeric")
            records.append(
                OCRRecord(
                    text=text,
                    confidence=score,
                    box=_parse_box(boxes[index], index),
                )
            )

    # Retain the CLI's nutrition-label-only unit corrections.
    for record in records:
        if not _NUTRITION_CONTEXT_RE.search(record.text):
            continue
        record.text = _OCR_NG_UNIT_RE.sub(r"\g<amount>mg", record.text)
        if _GRAM_NUTRIENT_RE.search(record.text):
            record.text = _OCR_NINE_AS_GRAM_RE.sub(r"\g<amount>g", record.text)
    return records


def preprocess_image(image_bytes: bytes) -> PreprocessedImage:
    """Decode and enhance bytes using the CLI's existing OpenCV algorithm."""
    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        raise RuntimeError(
            "OpenCV and NumPy are required; install requirements-web.txt."
        ) from exc

    try:
        encoded = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    except cv2.error as exc:
        raise ImageInputError("invalid_image", "The upload could not be decoded as an image.") from exc
    if image is None:
        raise ImageInputError("invalid_image", "The upload could not be decoded as an image.")

    height, width = image.shape[:2]
    if width * height > MAX_IMAGE_PIXELS:
        raise ImageInputError(
            "image_dimensions_too_large",
            f"Decoded image exceeds the {MAX_IMAGE_PIXELS:,}-pixel limit.",
            status_code=413,
        )

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
    processed = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    processed_height, processed_width = processed.shape[:2]
    return PreprocessedImage(
        image=processed,
        width=width,
        height=height,
        processed_width=processed_width,
        processed_height=processed_height,
    )


def _create_paddleocr() -> Any:
    from paddleocr import PaddleOCR

    return PaddleOCR(
        text_detection_model_name=TEXT_DETECTION_MODEL,
        text_recognition_model_name=TEXT_RECOGNITION_MODEL,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=True,
        device="cpu",
        engine="paddle_static",
        enable_mkldnn=False,
    )


def _configured_cors_origins() -> list[str]:
    raw = os.getenv("WEB_OCR_CORS_ORIGINS", "")
    origins = [origin.strip() for origin in raw.split(",") if origin.strip()]
    if "*" in origins:
        raise ValueError(
            "WEB_OCR_CORS_ORIGINS must list explicit origins; '*' is not allowed"
        )
    return origins


def _error_response(
    status_code: int,
    code: str,
    message: str,
    details: list[dict[str, str]] | None = None,
) -> JSONResponse:
    body = ErrorResponse(
        error=ErrorBody(code=code, message=message, details=details)
    )
    return JSONResponse(status_code=status_code, content=body.model_dump(exclude_none=True))


def create_app(ocr_factory: Callable[[], Any] | None = None) -> FastAPI:
    """Create the ASGI app; an optional factory supports embedding/custom setup."""

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        application.state.ocr_model = None
        application.state.model_state = "initializing"
        application.state.model_error_type = None
        application.state.model_lock = threading.Lock()
        try:
            model = (ocr_factory or _create_paddleocr)()
            if not callable(getattr(model, "predict", None)):
                raise RuntimeError("The initialized PaddleOCR object has no predict() method")
        except Exception as exc:
            application.state.model_state = "failed"
            application.state.model_error_type = type(exc).__name__
            logger.exception(
                "PaddleOCR initialization failed for detection=%s recognition=%s",
                TEXT_DETECTION_MODEL,
                TEXT_RECOGNITION_MODEL,
            )
        else:
            application.state.ocr_model = model
            application.state.model_state = "ready"
            logger.info(
                "PaddleOCR initialized: detection=%s recognition=%s engine=paddle_static",
                TEXT_DETECTION_MODEL,
                TEXT_RECOGNITION_MODEL,
            )
        try:
            yield
        finally:
            application.state.ocr_model = None
            application.state.model_state = "failed"

    application = FastAPI(
        title="Web OCR API",
        version="1.0.0",
        description="Image OCR API using the PaddleOCR 3.x general OCR pipeline.",
        lifespan=lifespan,
    )

    cors_origins = _configured_cors_origins()
    if cors_origins:
        application.add_middleware(
            CORSMiddleware,
            allow_origins=cors_origins,
            allow_credentials=False,
            allow_methods=["GET", "POST"],
            allow_headers=["Content-Type"],
        )

    @application.exception_handler(RequestValidationError)
    async def request_validation_error_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        del request
        details = [
            {
                "location": ".".join(str(part) for part in item.get("loc", ())),
                "message": str(item.get("msg", "Invalid request")),
            }
            for item in exc.errors()
        ]
        return _error_response(
            422,
            "invalid_request",
            "The request is invalid; POST /ocr requires multipart field 'file'.",
            details,
        )

    @application.exception_handler(StarletteHTTPException)
    async def http_error_handler(
        request: Request, exc: StarletteHTTPException
    ) -> JSONResponse:
        del request
        if isinstance(exc.detail, dict) and "code" in exc.detail:
            return _error_response(
                exc.status_code,
                str(exc.detail["code"]),
                str(exc.detail.get("message", "Request failed")),
            )
        known_errors = {
            400: ("bad_request", "The request could not be parsed."),
            404: ("not_found", "The requested endpoint was not found."),
            405: ("method_not_allowed", "The HTTP method is not allowed."),
        }
        code, message = known_errors.get(
            exc.status_code, ("http_error", str(exc.detail))
        )
        return _error_response(exc.status_code, code, message)

    @application.exception_handler(Exception)
    async def unhandled_error_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled request error for %s", request.url.path, exc_info=exc)
        return _error_response(
            500, "internal_error", "The server encountered an unexpected error."
        )

    @application.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse | JSONResponse:
        state = application.state.model_state
        if state == "ready":
            return HealthResponse(
                status="ok", model_ready=True, model_state="ready", models=ModelNames()
            )
        return JSONResponse(
            status_code=503,
            content=HealthResponse(
                status="unavailable",
                model_ready=False,
                model_state=state,
                models=ModelNames(),
                error=ErrorBody(
                    code="model_initialization_failed",
                    message="The OCR model is not ready; check the backend logs.",
                ),
            ).model_dump(exclude_none=True),
        )

    @application.post("/ocr", response_model=OCRSuccess)
    def ocr(file: UploadFile = File(...)) -> OCRSuccess:
        model = application.state.ocr_model
        if model is None:
            raise StarletteHTTPException(
                status_code=503,
                detail={
                    "code": "model_not_ready",
                    "message": "The OCR model is not available; check GET /health and server logs.",
                },
            )

        try:
            image_bytes = file.file.read(MAX_UPLOAD_BYTES + 1)
        except Exception as exc:
            logger.exception("Could not read uploaded file")
            raise StarletteHTTPException(
                status_code=400,
                detail={"code": "upload_read_failed", "message": "The uploaded file could not be read."},
            ) from exc
        if len(image_bytes) > MAX_UPLOAD_BYTES:
            raise StarletteHTTPException(
                status_code=413,
                detail={
                    "code": "upload_too_large",
                    "message": f"Uploads are limited to {MAX_UPLOAD_BYTES // (1024 * 1024)} MiB.",
                },
            )
        if not image_bytes:
            raise StarletteHTTPException(
                status_code=422,
                detail={"code": "empty_upload", "message": "The uploaded file is empty."},
            )

        try:
            processed = preprocess_image(image_bytes)
        except ImageInputError as exc:
            raise StarletteHTTPException(
                status_code=exc.status_code,
                detail={"code": exc.code, "message": str(exc)},
            ) from exc
        except Exception as exc:
            logger.exception("Image preprocessing failed for filename=%r", file.filename)
            raise StarletteHTTPException(
                status_code=500,
                detail={
                    "code": "image_processing_failed",
                    "message": "Image preprocessing failed; check the backend logs.",
                },
            ) from exc

        try:
            # Serialize access to the shared PaddleOCR pipeline, whose predictor
            # is initialized once and may not support concurrent predict calls.
            with application.state.model_lock:
                raw_result = list(model.predict(input=processed.image))
        except Exception as exc:
            logger.exception("PaddleOCR inference failed for filename=%r", file.filename)
            raise StarletteHTTPException(
                status_code=500,
                detail={
                    "code": "ocr_inference_failed",
                    "message": "OCR inference failed; check the backend logs.",
                },
            ) from exc

        try:
            records = parse_ocr_output(raw_result)
        except Exception as exc:
            logger.exception("Could not parse PaddleOCR output for filename=%r", file.filename)
            raise StarletteHTTPException(
                status_code=500,
                detail={
                    "code": "ocr_result_invalid",
                    "message": "The OCR engine returned an invalid result; check the backend logs.",
                },
            ) from exc

        results = [
            OCRItem(text=item.text, confidence=item.confidence, box=item.box)
            for item in records
        ]
        return OCRSuccess(
            filename=file.filename or "uploaded-image",
            image=ImageDimensions(
                width=processed.width,
                height=processed.height,
                processed_width=processed.processed_width,
                processed_height=processed.processed_height,
            ),
            count=len(results),
            results=results,
            models=ModelNames(),
        )

    return application


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=os.getenv("WEB_OCR_HOST", "127.0.0.1"),
        port=int(os.getenv("WEB_OCR_PORT", "8000")),
        reload=False,
    )
