import os
import json
import argparse
import TIT_LLM.main as TIT
from eval.imgmt_eval import PipelineDataLoader, Evaluator, download_model, load_from_checkpoint


_COMET_MODEL = None


TIT_LLM_MODELS = ["qwen3.5", "kimi-k3", "gpt-5.6-luna", "claude-haiku-4.5"]


DATA_PATH = "../../../img-mt_dataset/data"
OUTPUT_PATH = "./experiment_results"


def get_comet_model():
    global _COMET_MODEL

    if _COMET_MODEL is None:
        model_path = download_model("Unbabel/wmt22-comet-da")
        _COMET_MODEL = load_from_checkpoint(model_path)

    return _COMET_MODEL


def result_exists(metadata_path: str) -> bool:
	return os.path.exists(metadata_path)


def evaluate(outputs: dict, reference_file: str, src_lang: str, tgt_lang: str):
	'''
	Load reference and predicted data for a language pair and evaluate text
	detection and translation. 
	
	Return CER, bounding-box F1 scores, chrF, and COMET in a metrics dictionary.
	'''

	# prepare the reference and predicted files based on the experiment and evaluate the results
	loader = PipelineDataLoader(src_lang, tgt_lang)
	gold = loader.load_gold(reference_file)

	if outputs.get("metadata") is not None:
		pred = loader.load_predicted(outputs["metadata"])
	else:
		pred = { "target_PNG": outputs["image"] }

	ev = Evaluator(gold, pred)

	# load the comet model
	comet_model = get_comet_model()

	# run the evaluation
	cer, f1_area, f1_count = ev.evaluate_text_detection(merging=True, rendered=False)
	chrF = ev.evaluate_translation( "chrf", merging=True)
	comet = ev.evaluate_translation( "comet", merging=True, model=comet_model)

	metrics = {"CER": cer, "F1_Area": f1_area, "F1_Count": f1_count, "chrF": chrF, "COMET": comet}
	
	return metrics


def call_tit_system(id: str, lang_pair: list[str], provider: str, model: str, data_folder: str, system_results_path):
	'''
	Run the selected model through OpenRouter on one source image. 
	
	Return the paths reported by TIT.run_system, or None if the 
	call fails or produces no validated result.
	'''
	
	img_path = os.path.join(data_folder, "png-base", f"{lang_pair[0]}.png")

	try:
		print(f"Running system on {img_path} with {model}")
		outputs = TIT.run_system(img_path = img_path, model = model, provider = provider, source_lang = lang_pair[0], target_lang = lang_pair[1], output_dir = system_results_path, id = id)
	except Exception as e:
		print(e)
		return None

	return outputs


def run_system(id: str, lang_pair: list[str], provider: str, model: str, split_folder:str, results_folder: str):
	'''
	Obtain and evaluate a TIT result for one image group and language pair.
	Reuse saved metrics or generated metadata when available; otherwise run
	the model. 
	
	Save new metrics and return them, or None if generation fails.
	'''
	
	metrics = None

	# prepare the paths for the reference files and storing the predicted files and evaluation results
	data_folder = os.path.join(split_folder, id)
	system_results_path = os.path.join(results_folder, model.upper())
	eval_dir = os.path.join(system_results_path, "eval")
	
	os.makedirs(system_results_path, exist_ok=True)
	os.makedirs(eval_dir, exist_ok=True)
	
	eval_result_path = os.path.join(eval_dir, f"{lang_pair[0]}-{lang_pair[1]}_{id}_eval.json")
	generated_result_path = os.path.join(system_results_path, "logs", f"{lang_pair[0]}-{lang_pair[1]}_{id}_metadata.json")

	# check if evaluation results for this instance already exist and reuse them if they do
	if result_exists(eval_result_path):
		with open(eval_result_path, "r", encoding="utf-8") as file:
			metrics = json.load(file)

	# otherwise, check if generated translation exists and reuse it if it does
	elif result_exists(generated_result_path):
		outputs = { "metadata": generated_result_path, "image": None }

		if outputs == None:
			return None
		
		# Evaluate the results, save metrics
		reference_path = os.path.join(data_folder, f"{lang_pair[0]}-{lang_pair[1]}.json")
		if not os.path.exists(reference_path):
			reference_path = os.path.join(data_folder, f"{lang_pair[1]}-{lang_pair[0]}.json")

		metrics = evaluate(outputs = outputs, reference_file = reference_path, src_lang=lang_pair[0], tgt_lang=lang_pair[1])

		with open(eval_result_path, "w", encoding="utf-8") as f:
			json.dump(metrics, f, indent=2)

		return metrics

	# run the TIT system if not results exist
	else:
		outputs = call_tit_system(id, lang_pair, provider, model, data_folder, system_results_path)

		if outputs == None:
			return None

		# Evaluate the results, save metrics
		reference_path = os.path.join(data_folder, f"{lang_pair[0]}-{lang_pair[1]}.json")
		if not os.path.exists(reference_path):
			reference_path = os.path.join(data_folder, f"{lang_pair[1]}-{lang_pair[0]}.json")

		metrics = evaluate(outputs = outputs, reference_file = reference_path, src_lang=lang_pair[0], tgt_lang=lang_pair[1])

		with open(eval_result_path, "w", encoding="utf-8") as f:
			json.dump(metrics, f, indent=2)

	return metrics


def average_metrics(results: list):
	if not results:
		return None

	return {
		key: sum(r[key] for r in results) / len(results)
		for key in results[0]
	}


def main(args: argparse.Namespace):
	# Set up results directory
	current_dir = os.path.dirname(os.path.abspath(__file__))
	output_dir = os.path.join(current_dir, OUTPUT_PATH, f"TIT_LLM", args.split)
	os.makedirs(output_dir, exist_ok=True)

	# Prepare a list for storing the results
	split_results = []
	target_split_dir = os.path.join(current_dir, DATA_PATH, args.split)

	# Loop through data
	count = 0
	for id_folder in os.listdir(target_split_dir):
		if count >= args.limit:
			break

		count += 1

		# Iterate through the .json metadata files and extract image pairs from them
		for data_file in  os.listdir(os.path.join(target_split_dir, id_folder)):
			if not data_file.endswith(".json"):
				continue
			
			lang_pair = os.path.splitext(data_file)[0].split('-')

			# evaluate only translations on english source images
			if lang_pair[0] != "en":
				if lang_pair[1] != "en":
					continue
				else:
					lang_pair = lang_pair[::-1]

			print(f"{id_folder}/{lang_pair[0]}-{lang_pair[1]}")
			# Receive evaluation results from the translation system call
			metrics = run_system(id_folder, lang_pair, args.provider, args.llm_model, target_split_dir, output_dir)

			if metrics != None:
				split_results.append(metrics)
			
	split_average = average_metrics(split_results)
	print(f"Average of {args.split} split: {split_average}")


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--provider", required=True, choices=["einfra", "openrouter"], help="The API provider from which the multimodal LLM will be called.")
	parser.add_argument("--llm_model", required=True, choices=TIT_LLM_MODELS, help="Multimodal LLM to run the TIT experiments with.")
	parser.add_argument("--split", choices=["test", "dev", "train"], required=True, help="Split from the data to run the experiments on.")
	parser.add_argument("--limit", type=int, default=20, help="Number of image instances to run the experiments on. The available language pairs for each image instance may vary.")
	args = parser.parse_args()

	main(args)
