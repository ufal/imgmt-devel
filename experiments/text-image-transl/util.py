from PIL import Image
from typing import List
from pydantic import BaseModel, model_validator, Field, ValidationError

class BoundingBox(BaseModel):
	x: float = Field(ge=0.0, le=1.0)
	y: float = Field(ge=0.0, le=1.0)
	w: float = Field(gt=0.0, le=1.0)
	h: float = Field(gt=0.0, le=1.0)

	@model_validator(mode="after")
	def validate_edges(self):
		if self.x + self.w > 1 or self.y + self.h > 1:
			raise ValueError("Bounding box extends outside the image")
		return self


class OCRTranslationResult(BaseModel):
	source_text_bounding_boxes: List[BoundingBox]
	source_texts: List[str]
	target_texts: List[str]

	@model_validator(mode="after")
	def validate_lengths(self):
		n = len(self.source_texts)

		if len(self.source_text_bounding_boxes) != n:
			raise ValueError(
				"source_text_bounding_boxes and source_texts must have equal length"
			)

		if len(self.target_texts) != n:
			raise ValueError(
				"target_texts and source_texts must have equal length"
			)

		return self


def denormalize_bounding_boxes(img_path: str, result: OCRTranslationResult) -> dict:
    with Image.open(img_path) as img:
        width, height = img.size

    data = result.model_dump()

    scaled_boxes = []

    for box in result.source_text_bounding_boxes:
        scaled_boxes.append(
            {
                "x": round(box.x * width),
                "y": round(box.y * height),
                "w": round(box.w * width),
                "h": round(box.h * height),
            }
        )

    data["source_text_bounding_boxes"] = scaled_boxes

    return data