ROOT_DIR=~/labs/202504-longcot/OlympiadBench/inference/code
cd code
# python evaluate_all.py --model_name ~/labs/all-models/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
#     --cuda_device 1 \
#     --dataset_path ${ROOT_DIR}/Dataset/data/OE_TO_maths_en_COMP.json
python evaluate_all.py --model_name ~/labs/all-models/Qwen2-0.5B \
    --cuda_device 1 \
    --dataset_path ${ROOT_DIR}/Dataset/data/OE_TO_maths_en_COMP.json

# OE_TO_maths_en_COMP.json
# OE_TO_maths_zh_CEE.json
# OE_TO_maths_zh_COMP.json
# OE_TO_physics_en_COMP.json
# OE_TO_physics_zh_CEE.json
# deepseek-ai/deepseek-math-7b-rl
# ~/labs/202504-longcot/OlympiadBench/inference/code/Dataset/data/OE_TO_maths_en_COMP.json
