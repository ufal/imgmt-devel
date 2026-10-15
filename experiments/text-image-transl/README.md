# Text in image translation experiments

This directory contains two scripts for running text in image translation experiments on the ImgMT dataset:

- `TIT_experiments.py` calls a multimodal language model to detect and translate text in an image.

- `AnyTrans_experiments.py` runs the AnyTrans pipeline to detect and translate text in an image.

Both scripts evaluate the detected text, bounding boxes, and translations produced by the given system against the dataset annotations.

## Setting up input data and the AnyTrans pipelined system

The experiment scripts depend on two directories outside this repository: `img-mt_dataset` for the annotated images and `AnyTrans-extended` for the AnyTrans pipeline. It is recommended to have them alongside `imgmt-devel`:

```text
workspace/
├── imgmt-devel/
│   └── experiments/text-image-transl/
├── img-mt_dataset/
│   └── data/{train,dev,test}/
└── AnyTrans-extended/
    ├── main_code.py
    └── venv/bin/python3
```

`TIT_experiments.py` reads the dataset from its `DATA_PATH` variable and writes results under `OUTPUT_PATH`. `AnyTrans_experiments.py` uses `DATA_DIR` and `OUTPUT_PATH`, plus `PIPELINE_PYTHON` and `ANYTRANS_MAIN_CODE` variables to find the separate AnyTrans environment and entry point. These paths are resolved relative to the experiment script directory. If your checkouts have a different layout, update the corresponding constants at the top of the scripts before running them.

Each dataset split contains numbered image groups. For an English to Hungarian example, the scripts expect `img-mt_dataset/data/test/2265/png-base/en.png` and a reference file such as `img-mt_dataset/data/test/2265/en-hu.json`. Both experiment scripts use English as the source language and process every available English language pair in each selected group. `--limit` counts image groups, not language pairs.

The `png-base` images must exist before running either experiment. If you have the dataset SVGs but not these PNGs, generate them with the dataset conversion script. It needs Inkscape on `PATH` and Pillow in the active Python environment. From this directory, for example:

```bash
python ../../../img-mt_dataset/utils/svg2png.py \
  --base-dir ../../../img-mt_dataset/data/test --mode base
```

Repeat with `dev` or `train` when using those splits. Check that the intended groups contain `png-base/en.png` before starting an experiment.

## Setting up the experiment environment

Run these commands from `imgmt-devel/experiments/text-image-transl/`. Python 3.12 is used for the experiment environment in this workspace:

```bash
python3.12 -m venv venv
source venv/bin/activate
python -m pip install -r requirements.txt
```

Evaluation loads the `Unbabel/wmt22-comet-da` model on its first uncached run, so the environment needs access to download that checkpoint.

### Additional AnyTrans environment setup

`AnyTrans_experiments.py` launches `AnyTrans-extended/main_code.py` with the Python executable at `AnyTrans-extended/venv/bin/python3`. Set up that separate environment from the experiment directory:

```bash
cd ../../../AnyTrans-extended
python3.11 -m venv venv
source venv/bin/activate
python -m pip install -r requirements.txt
deactivate
cd ../imgmt-devel/experiments/text-image-transl
source venv/bin/activate
```

The AnyTrans source and its own requirements file must be present at that sibling path. The experiment currently uses PaddleOCR and deterministic rendering.

### AnyTrans translation and OCR options

The call to `run_pipeline()` in `AnyTrans_experiments.py` currently passes `use_gold=False` and `box_by_box=False`. With `box_by_box=False`, AnyTrans first asks its language model to translate all detected text boxes together and falls back to translating boxes individually if that fails. To run the box by box method from the start, change the `box_by_box` argument in that call to `True`. These options are set in the script; they are not command line arguments to `AnyTrans_experiments.py`.

`run_pipeline()` can also forward `use_gold=True` to AnyTrans. This is intended for an ideal OCR experiment that supplies the reference `source_texts` and `source_text_bounding_boxes`, leaving the model's translation to evaluate. However, the current `AnyTrans-extended/main_code.py` initializes `gold_data` to `None` and does not load it when `--use_gold` is passed. Setting `use_gold=True` in the experiment script alone therefore does not enable ideal OCR; the AnyTrans entry point must first load the reference metadata for the selected translation direction and pass it as `gold_data` to its pipeline.

## API credentials

Set the relevant key in the shell before running a script. The scripts read environment variables; they do not load a `.env` file.

```bash
export OPENAI_API_KEY="<e-infra API key>"
```

`AnyTrans_experiments.py` calls e-INFRA and needs `OPENAI_API_KEY`. `TIT_experiments.py` and standalone `TIT_LLM` can use either provider: select `--provider einfra` with `OPENAI_API_KEY`, or `--provider openrouter` with `OPENROUTER_API_KEY`.

```bash
export OPENROUTER_API_KEY="<OpenRouter API key>"
```

## Running the TIT experiment

From `imgmt-devel/experiments/text-image-transl/`, with the experiment environment active, run:

```bash
python TIT_experiments.py --provider einfra --llm_model kimi-k3 --split test --limit 20
```

`--split` accepts `train`, `dev`, or `test`; `--limit` defaults to 20. Currently the only model configured for e-INFRA is `kimi-k3`. The models configured for OpenRouter are listed in `TIT_LLM/main.py`. For example, to select a configured OpenRouter model:

```bash
python TIT_experiments.py --provider openrouter --llm_model qwen3.5 --split test --limit 20
```

Predictions are saved under `experiment_results/TIT_LLM/<split>/<MODEL>/logs/`, and per-pair metrics under the corresponding `eval/` directory. The script prints the average metrics for the selected split.


## Running the AnyTrans experiment

With both environments prepared, run:

```bash
python AnyTrans_experiments.py --split test --limit 20
```

`--split` accepts `train`, `dev`, or `test`; `--limit` defaults to 20. Per-pair metadata and evaluation files are saved under `experiment_results/AnyTrans/<split>/paddle/logs/` and `eval/`. The script prints averages for per-OCR engine and overall performance. AnyTrans writes rendered images into the shared `paddle/` directory using the source image basename, so later images can overwrite earlier rendered images, since they are not being used for evaluation.

## Running TIT_LLM on individual images

For a single PNG, use the same experiment environment. The following example translates text from English to Hungarian and saves a JSON result without running evaluation:

```bash
python -m TIT_LLM.main \
  --provider einfra \
  --model kimi-k3 \
  --source_lang en \
  --target_lang hu \
  --img_path ../../../img-mt_dataset/data/test/2265/png-base/en.png \
  --output_dir ./single_results
```

`--provider` accepts `einfra` or `openrouter`. The model must be configured for the chosen provider in `TIT_LLM/main.py`. The input must be an existing `.png` file, and the source and target languages must differ. This example writes `single_results/logs/hu_metadata.json`, containing source texts, their pixel bounding boxes, and translated texts. Reusing the same output directory and target language overwrites that file.

## Configuring TIT models

The model name passed to `--llm_model` is a short name, such as `kimi-k3`. `TIT_LLM/main.py` maps that name to the identifier expected by each provider in `E_INFRA_MODELS` and `OPENROUTER_MODELS`. To add a model to an experiment:

1. Add its short name and provider-specific identifier to the appropriate mapping in `TIT_LLM/main.py`.
2. Add the same short name to `TIT_LLM_MODELS` in `TIT_experiments.py`, so `--llm_model` accepts it.
3. If you also want to call it through standalone `TIT_LLM`, add the short name to the `--model` choices in `TIT_LLM/main.py`.

A model needs an entry only in the mapping for the provider you intend to use. The scripts check whether the selected model is configured for that provider when they make a new model call. A configured identifier does not establish that your provider account has access to it.

## Evaluation results and reruns

Experiment evaluations save `CER`, `F1_Area`, `F1_Count`, `chrF`, and `COMET` for each image and language pair. Existing prediction and evaluation files are reused on later runs. If you change model behavior, experiment settings, or evaluation code, use a fresh results location or move the old results aside so the script does not reuse stale files. In particular, `TIT_experiments.py` does not include the provider in its results path: running the same short model name through another provider will reuse the first provider's files unless you move them aside.
