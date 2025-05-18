# python judge.py --path ~/labs/202504-longcot/OlympiadBench/inference/generated \
#     --model_dir deepseek-math-7b-rl --single_model 1 \
#     > judge_output.log 2>&1

python judge.py --path ~/labs/202504-longcot/OlympiadBench/inference/generated \
    --model_dir s1_epoch3_distill --single_model 1 \
    > judge_output.log 2>&1

# step3.
# ROOT_DIR=/home/xiaojianha/labs/202504-longcot/OlympiadBench/inference/code
# python calculate_accuracy.py --answer_dir merged_result --ref_dir ${ROOT_DIR}/Dataset/data/
