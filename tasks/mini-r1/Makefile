help:
	@echo "Usage: make [target]"
	@echo "Targets:"
	@echo "  dist_train: Run distributed training"
	@echo "  train: Run training"

dist_train:
	export CUDA_VISIBLE_DEVICES=0,1,2,3,4 \
	accelerate launch --num_processes 3 \
		--config_file ./deepspeed_zero3.yaml \
		run_r1_grpo.py \
		--config ./grpo-qwen-2.5-3b-deepseek-r1-countdown.yaml

train:
	python run_r1_grpo.py \
		--config ./grpo-qwen-2.5-3b-deepseek-r1-countdown.yaml
