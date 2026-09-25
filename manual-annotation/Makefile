IMGMT_DATA_DIR ?=
MANUAL_ANNOT_DIR ?= annots
N_PAIRS_PER_ANNOTATOR ?= 10
ANNOTATORS ?= annotator1
SEED ?=

.PHONY: check-imgmt-data-dir manual-eval prepare-batch sort-bbs distribute-svgs check-svgs

check-imgmt-data-dir:
	@if [ -z "$(strip $(IMGMT_DATA_DIR))" ]; then \
		echo "Warning: IMGMT_DATA_DIR is not set; set it before running make." >&2; \
	fi

manual-eval: check-imgmt-data-dir manual-eval-test

# manual annotation on the dev set has not been finished, so evaluating on it does not make sense

manual-eval-%: check-imgmt-data-dir
	python eval/manual_eval.py --imgmt-dir "$(IMGMT_DATA_DIR)/$*" --manual-dir "$(MANUAL_ANNOT_DIR)/$*"

prepare-batch: check-imgmt-data-dir prepare-batch-test

prepare-batch-%: check-imgmt-data-dir
	python utils/prepare_batch.py "$(IMGMT_DATA_DIR)/$*" "$(N_PAIRS_PER_ANNOTATOR)" "$(MANUAL_ANNOT_DIR)/$*" --annotators $(ANNOTATORS) $(if $(SEED),--seed $(SEED),)

sort-bbs: sort-bbs-test

sort-bbs-%:
	python utils/sort_bbs.py "$(MANUAL_ANNOT_DIR)/$*"

# SVGs are optional for evaluation because the annotation JSON contains all required data.
distribute-svgs: check-imgmt-data-dir distribute-svgs-test

distribute-svgs-%: check-imgmt-data-dir
	python utils/distribute_svgs.py "$(IMGMT_DATA_DIR)/$*" "$(MANUAL_ANNOT_DIR)/$*"

check-svgs: check-imgmt-data-dir check-svgs-test

check-svgs-%: check-imgmt-data-dir
	python utils/distribute_svgs.py "$(IMGMT_DATA_DIR)/$*" "$(MANUAL_ANNOT_DIR)/$*" --check
