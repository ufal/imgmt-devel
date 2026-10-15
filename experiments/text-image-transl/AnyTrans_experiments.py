import os
import json
import subprocess
import pandas as pd
import argparse
from eval.imgmt_eval import PipelineDataLoader, Evaluator, download_model, load_from_checkpoint


DATA_DIR = "../../../img-mt_dataset/data"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PIPELINE_PYTHON = os.path.abspath(os.path.join(SCRIPT_DIR, "../../../AnyTrans-extended/venv/bin/python3"))
ANYTRANS_MAIN_CODE = os.path.abspath(os.path.join(SCRIPT_DIR, "../../../AnyTrans-extended/main_code.py"))

OCR_ENGINES = ["paddle"]

OUTPUT_PATH = "./experiment_results"

_COMET_MODEL = None


def get_comet_model():
    global _COMET_MODEL

    if _COMET_MODEL is None:
        model_path = download_model("Unbabel/wmt22-comet-da")
        _COMET_MODEL = load_from_checkpoint(model_path)

    return _COMET_MODEL


def results_exist(output_dir, source, target, id_folder):
    log_path = os.path.join(output_dir, "logs", f"{source}-{target}_{id_folder}_metadata.json")

    return os.path.isfile(log_path)


def eval_exists(output_dir, source, target, id_folder):
    evaluation_path = os.path.join(output_dir, "eval", f"{source}-{target}_{id_folder}_eval.json")

    return os.path.isfile(evaluation_path)


def run_pipeline(
    id_folder,
    source,
    target,
    dataset_dir,
    ocr_engine,
    use_gold,
    box_by_box,
    output_dir,
):
    '''
    Run AnyTrans on one source image with the selected OCR and translation options.

    Return the output directory when the pipeline succeeds, or None when it fails.
    '''
    
    result_name = f"{source}-{target}_{id_folder}"

    os.makedirs(output_dir, exist_ok=True)

    image_path = os.path.join(dataset_dir, id_folder, "png-base", f"{source}.png")

    cmd = [
        PIPELINE_PYTHON,
        ANYTRANS_MAIN_CODE,

        "--source_lang",
        source,

        "--target_lang",
        target,

        "--ocr_engine",
        ocr_engine,

        "--rendering",
        "deterministic",

        "--image_path",
        image_path,

        "--output_dir",
        output_dir,

        "--log",
        "True",

        "--result_name", 
        result_name,
    ]

    if use_gold:
        cmd.extend(["--use_gold", "True"])

    if box_by_box:
        cmd.extend(["--box_by_box", "True"])

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(
            f"Pipeline failed for "
            f"{id_folder}/{source}-{target}"
        )
        print(result.stderr)

        return None

    return output_dir


def evaluate(source, target, id_folder, dataset_dir, output_dir):
    '''
    Evaluate an AnyTrans prediction against the matching gold annotations.

    Reuse saved metrics when available; otherwise calculate and save CER, detection
    F1 scores, chrF, and COMET. Return the metrics, or None if an input is missing.
    '''

    # prepare path for saving the evaluation results
    evaluation_dir = os.path.join(output_dir, "eval")
    os.makedirs(evaluation_dir, exist_ok=True)

    evaluation_path = os.path.join(evaluation_dir, f"{source}-{target}_{id_folder}_eval.json")

    if os.path.exists(evaluation_path):
        with open(evaluation_path, "r", encoding="utf-8") as f:
            return json.load(f)
    
    log_path = os.path.join(output_dir, "logs", f"{source}-{target}_{id_folder}_metadata.json")
    
    if not os.path.exists(log_path):
        print(
            f"Missing system result for "
            f"{id_folder}/{source}-{target}"
        )

        return None

    # 
    gold_path = os.path.join(dataset_dir, id_folder, f"{source}-{target}.json")

    if not os.path.exists(gold_path):
        gold_path = os.path.join(dataset_dir, id_folder, f"{target}-{source}.json")

    if not os.path.exists(gold_path):
        print(
            f"Missing gold metadata for "
            f"{id_folder}/{source}-{target}"
        )

        return None

    # prepare and run the evaluation
    loader = PipelineDataLoader(source, target)
    gold = loader.load_gold(gold_path)
    predicted = loader.load_predicted(log_path)
    evaluator = Evaluator(gold, predicted)

    cer, f1_area, f1_count = evaluator.evaluate_text_detection(merging=True, rendered=False)
    chrf = evaluator.evaluate_translation("chrf", merging=True)
    comet = evaluator.evaluate_translation("comet", merging=True, model=get_comet_model())

    metrics = {"CER": cer, "F1_Area": f1_area, "F1_Count": f1_count, "chrF": chrf, "COMET": comet}
    with open(evaluation_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return metrics

def main(args):
    # Set up results directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(current_dir, OUTPUT_PATH, f"AnyTrans", args.split)
    os.makedirs(output_dir, exist_ok=True)
    
    all_results = []
    for ocr in OCR_ENGINES:
        ocr_output_dir = os.path.join(output_dir, ocr)

        ocr_results = []
        target_ocr_dir = os.path.join(current_dir, DATA_DIR, args.split)

        count = 0
        for id_folder in os.listdir(target_ocr_dir):
            if count >= args.limit:
                break

            id_path = os.path.join(target_ocr_dir, id_folder)

            if not os.path.isdir(id_path):
                continue

            count += 1

            print("\n" + "=" * 70)
            print(f"Processing image instance: {id_folder}")
            print("=" * 70)

            # Find all language-pair metadata files
            for data_file in os.listdir(id_path):
                if not data_file.endswith(".json"):
                    continue

                lang_pair = os.path.splitext(data_file)[0].split("-")

                if lang_pair[0] != "en":
                    if lang_pair[1] != "en":
                        continue
                    lang_pair = lang_pair[::-1]

                source, target = lang_pair

                print(
                    f"\nImage: {id_folder} | "
                    f"Language pair: {source}2{target}"
                )

                system_result_exists = results_exist(ocr_output_dir, source, target, id_folder)

                if not system_result_exists:
                    run_pipeline(id_folder, source, target, target_ocr_dir, ocr, False, False, ocr_output_dir,)
                else:
                    print("System result already exists, skipping pipeline.")

                print("Evaluating results")
                if not eval_exists(ocr_output_dir, source, target, id_folder):
                    metrics = evaluate(source, target, id_folder, target_ocr_dir, ocr_output_dir)
                else:
                    print("Evaluation already exists, loading saved result.")

                    evaluation_path = os.path.join(ocr_output_dir, "eval", f"{source}-{target}_{id_folder}_eval.json")

                    with open(evaluation_path, "r", encoding="utf-8") as f:
                        metrics = json.load(f)

                if metrics:
                    metrics.update({"id": id_folder, "ocr": ocr, "split": args.split, "pair": f"{source}2{target}"})

                    ocr_results.append(metrics)

        # Average over all processed image-language pairs
        if ocr_results:
            ocr_results_df = pd.DataFrame(ocr_results)

            print(f"\nAverage of {args.split} split:")
            print(ocr_results_df[ ["CER", "F1_Area", "F1_Count", "chrF", "COMET"]].mean())

        all_results.extend(ocr_results)

    if not all_results:
        print("No results collected.")
        return

    df = pd.DataFrame(all_results)

    print("\nOverall average:")
    print(df[["CER", "F1_Area", "F1_Count", "chrF", "COMET"]].mean())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["test", "dev", "train"], required=True, help="Split from the data to run the experiments on.")
    parser.add_argument("--limit", type=int, default=20, help="Number of images to process")

    args = parser.parse_args()

    main(args)
