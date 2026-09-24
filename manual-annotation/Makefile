IMGMT_DATA_DIR=
MANUAL_ANNOT_DIR=annots

manual-eval : manual-eval-test

# manual annotation on the dev set has not been finished, so evaluating on it does not make sense

manual-eval-% :
	python eval/manual_eval.py --imgmt-dir $(IMGMT_DATA_DIR)/$* --manual-dir $(MANUAL_ANNOT_DIR)/$*
