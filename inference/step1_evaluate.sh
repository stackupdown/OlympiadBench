ROOT_DIR=~/labs/202504-longcot/OlympiadBench/inference/code
cd code
# python evaluate_all.py --model_name ~/labs/all-models/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
#     --cuda_device 1 \
#     --dataset_path ${ROOT_DIR}/Dataset/data/OE_TO_maths_en_COMP.json
# python evaluate_all.py --model_name ~/labs/all-models/Qwen2-0.5B \
#     --cuda_device 1 \
#     --dataset_path ${ROOT_DIR}/Dataset/data/OE_TO_maths_en_COMP.json
# 5.14
# python evaluate_all.py --model_name ~/labs/all-models/deepseek-ai/deepseek-math-7b-rl \
#     --cuda_device 1 \
#     --save_dir ~/labs/202504-longcot/OlympiadBench/inference/generated \
#     --dataset_path ${ROOT_DIR}/Dataset/data/OE_TO_maths_en_COMP.json \
#     > evaluate.log 2>&1

# OE_TO_maths_en_COMP.json
# OE_TO_maths_zh_CEE.json
# OE_TO_maths_zh_COMP.json
# OE_TO_physics_en_COMP.json
# OE_TO_physics_zh_CEE.json
# deepseek-ai/deepseek-math-7b-rl
# ~/labs/202504-longcot/OlympiadBench/inference/code/Dataset/data/OE_TO_maths_en_COMP.json

# 5.14 cancel
# model_name=qwen1000
# export CUDA_VISIBLE_DEVICES="4,5"
# python evaluate_all.py --model_name ~/labs/202504-longcot/open-r1/data/Qwen2.5-0.5B-Open-R1-Distill \
#     --cuda_device 1 \
#     --save_dir ~/labs/202504-longcot/OlympiadBench/inference/generated \
#     --dataset_path ${ROOT_DIR}/Dataset/data/OE_TO_maths_en_COMP.json \
#     > evaluate_${model_name}.log 2>&1

# qwen 0.5b
# model_name=qwen1000
# export CUDA_VISIBLE_DEVICES="4,5"
# python evaluate_all.py --model_name /home/xiaojianha/labs/202504-longcot/open-r1/data/s1/Qwen2.5-0.5B-Open-R1-Distill \
#     --cuda_device 1 \
#     --save_dir ~/labs/202504-longcot/OlympiadBench/inference/generated \
#     --dataset_path ${ROOT_DIR}/Dataset/data/OE_TO_maths_en_COMP.json \
#     --saving_name s1_distill \
#     > evaluate_${model_name}.log 2>&1

# qwen 0.3B
model_name=qwen_s1_epoch3
export CUDA_VISIBLE_DEVICES="4,5,6,7"
python evaluate_all.py --model_name /home/xiaojianha/labs/202504-longcot/open-r1/data/s1-epoch3/Qwen2.5-3B-Open-R1-Distill \
    --cuda_device 1 \
    --save_dir ~/labs/202504-longcot/OlympiadBench/inference/generated \
    --dataset_path ${ROOT_DIR}/Dataset/data/OE_TO_maths_en_COMP.json \
    --saving_name s1_epoch3_distill \
    > evaluate_${model_name}.log 2>&1
