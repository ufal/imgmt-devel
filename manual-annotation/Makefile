IMGMT_DATA_DIR=
MANUAL_ANNOT_DIR=annots

.PHONY: manual-eval distribute-svgs check-svgs

manual-eval: manual-eval-test

# manual annotation on the dev set has not been finished, so evaluating on it does not make sense

manual-eval-%:
	@if [ -z "$(strip $(IMGMT_DATA_DIR))" ]; then \
		echo "Warning: IMGMT_DATA_DIR is not set; set it before running $@." >&2; \
	fi
	python eval/manual_eval.py --imgmt-dir "$(IMGMT_DATA_DIR)/$*" --manual-dir "$(MANUAL_ANNOT_DIR)/$*"

# SVGs are optional for evaluation because the annotation JSON contains all required data.
distribute-svgs: distribute-svgs-test

distribute-svgs-%:
	@if [ -z "$(strip $(IMGMT_DATA_DIR))" ]; then \
		echo "Warning: IMGMT_DATA_DIR is not set; set it before running $@." >&2; \
	fi
	python utils/distribute_svgs.py "$(IMGMT_DATA_DIR)/$*" "$(MANUAL_ANNOT_DIR)/$*"

check-svgs: check-svgs-test

check-svgs-%:
	@if [ -z "$(strip $(IMGMT_DATA_DIR))" ]; then \
		echo "Warning: IMGMT_DATA_DIR is not set; set it before running $@." >&2; \
	fi
	python utils/distribute_svgs.py "$(IMGMT_DATA_DIR)/$*" "$(MANUAL_ANNOT_DIR)/$*" --check
