import os
import re
import base64
import json
import argparse
from pydantic import ValidationError
from openai import OpenAI

from util import OCRTranslationResult, denormalize_bounding_boxes


E_INFRA_MODELS = {"kimi-k3": "kimi-k3"}
OPENROUTER_MODELS = {"kimi-k3": "moonshotai/kimi-k3", "qwen3.5": "qwen/qwen3.5-122b-a10b", "gpt-5.6-luna": "openai/gpt-5.6-luna", "claude-haiku-4.5": "anthropic/claude-haiku-4.5"}


PROVIDERS = {
    "einfra": {
        "base_url": "https://llm.ai.e-infra.cz/v1/",
        "api_key": "OPENAI_API_KEY",
    },
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "api_key": "OPENROUTER_API_KEY",
    },
}


MODEL_IDS = {
    "einfra": E_INFRA_MODELS,
    "openrouter": OPENROUTER_MODELS,
}


PROMPT_TEMPLATE = """
You are an OCR and translation system.

Task:
1. Detect all the text visible in the image.
2. For each detected text element:
   - Extract the original text exactly as it appears.
   - Translate it to natural <target_language>.
   - Estimate a normalized bounding box around the text.

Bounding box requirements:
- Use normalized coordinates in the range [0.0, 1.0].
- Coordinates are relative to the full image dimensions.
- Format:
  - x = left edge / image width
  - y = top edge / image height
  - w = box width / image width
  - h = box height / image height
- w and h must be greater than 0
- x + w and y + h must be less than or equal to 1
- Bounding boxes should tightly enclose the visible text.
- Preserve reading order from top-to-bottom and left-to-right.
- Each source text must have exactly one corresponding bounding box and one translation.

Output requirements:
- Return ONLY valid JSON.
- Do not include markdown fences.
- Do not include explanations or additional fields.
- Do not omit any detected text.
- If a translation is uncertain, provide the most likely translation.

Example output structure:

{
  "source_text_bounding_boxes": [
    {
      "x": 0.10,
      "y": 0.20,
      "w": 0.30,
      "h": 0.08
    }
  ],
  "source_texts": [
    "<source_language> text"
  ],
  "target_texts": [
    "<target_language> translation"
  ]
}

Analyze the provided image and return the JSON result.
"""


def extract_json(text: str) -> str:
	"""
	Return the first fenced block's contents, or the trimmed response if none exists.

	The returned text is then passed to ``OCRTranslationResult`` for JSON validation.
	"""

	text = text.strip()

	# Remove a Markdown code fence if present
	match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
	if match:
		return match.group(1).strip()

	return text


def create_client(provider: str) -> OpenAI:
	config = PROVIDERS[provider]

	return OpenAI(
		api_key = os.environ.get(config["api_key"]),
		base_url = config["base_url"]
	)


def resolve_model(provider: str, model: str) -> str:
    if provider not in PROVIDERS:
        raise ValueError(f"Unknown provider: {provider}")

    models = MODEL_IDS[provider]
    if model not in models:
        raise ValueError(
            f"{model!r} is not configured for {provider}. "
            f"Choose from: {', '.join(models)}"
        )

    return models[model]


def call_llm(model: str, client: OpenAI, img_path: str, source_lang: str, target_lang: str, prompt: str) -> str:
	"""
	Send the image and language-specific prompt to the model and return its raw text.

	The image is embedded as a base64 data URL in the Responses API request.
	"""

	# replace the placeholders for source and target languages in the prompt
	prompt = prompt.replace("<source_language>", source_lang)
	prompt = prompt.replace("<target_language>", target_lang)

	# convert the image to binary
	with open(img_path, "rb") as img_file:
		image_b64 = base64.b64encode(img_file.read()).decode("utf-8")
	
	# make the API call
	response = client.responses.create(
		model=model,
		input=[
			{
				"role": "user",
				"content": [
					{
						"type": "input_text",
						"text": prompt
					},
					{
						"type": "input_image",
						"image_url": f"data:image/png;base64,{image_b64}"
					}
				]
			}
		]
	)

	output = response.output_text
	return output


def run_system(img_path: str, model: str, provider: str, source_lang: str, target_lang: str, output_dir: str, id: int = 0):
	"""
	Save validated OCR and translations with pixel bounding boxes as JSON.

	Return ``{"metadata": path, "image": None}`` on success, or ``None`` if
	the model response fails schema validation. 
	The file is saved in ``output_dir/logs``.
	"""

	# prepare the client
	client = create_client(provider)

	# resolve the model ID based on the provider used
	modelID = resolve_model(provider, model)

	# call the LLM and validate the format of the result
	result = call_llm(modelID, client, img_path, source_lang, target_lang, PROMPT_TEMPLATE)
	result = extract_json(result)
	
	try:
		result = OCRTranslationResult.model_validate_json(result)
	except ValidationError as e:
		print(e)
		return

	# convert the bounding box coordinates
	denormalized_result = denormalize_bounding_boxes(img_path, result)

	# prepare output path
	output_path = os.path.join(output_dir, "logs")
	os.makedirs(output_path, exist_ok=True)

	# If an ID is passed for the image include it in the name of the saved file
	if id == 0:
		json_output_path = os.path.join(output_path, f"{target_lang}_metadata.json")
	else:
		json_output_path = os.path.join(output_path, f"{source_lang}-{target_lang}_{id}_metadata.json")

	# save JSON output
	with open(json_output_path, "w") as log_file:
		json.dump(denormalized_result, log_file, ensure_ascii=False, indent=2)
		
	return { "metadata": json_output_path, "image": None }


def verify_args(args: argparse.Namespace) -> bool:
	if args.source_lang.strip().casefold() == args.target_lang.strip().casefold():
		raise ValueError("Source and target languages must be different")

	# verify the image path
	if not args.img_path.lower().endswith('.png'):
		raise ValueError("The image path must end with .png")
	if not os.path.isfile(args.img_path):
		raise ValueError(f"The image path does not exist or is not a file: {args.img_path}")

	return True


def main(args: argparse.Namespace):
	print(f"Processing image at {args.img_path}")
	run_system(args.img_path, args.model, args.provider, args.source_lang, args.target_lang, args.output_dir)


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--provider", required=True, choices=["einfra", "openrouter"], help="The API provider from which the multimodal LLM will be called.")
	parser.add_argument("--model", type=str, required=True, choices=["kimi-k3", "qwen3.5", "gpt-5.6-luna", "claude-haiku-4.5"], help="Multimodal LLM to be called in order to translate the text. It has to be supported by the API provider.")
	parser.add_argument("--source_lang", type=str, required=True, help="The language used in the source image which is to be translated.")
	parser.add_argument("--target_lang", type=str, required=True, help="The language in which the text in the image will be translated to.")
	parser.add_argument("--img_path", type=str, required=True, help="Path to the source image which will be translated.")
	parser.add_argument("--output_dir", type=str, required=True, help="Path to the directory, where the translation result will be saved.")

	main_args = parser.parse_args([] if "__file__" not in globals() else None)

	verify_args(main_args)
	
	main(main_args)