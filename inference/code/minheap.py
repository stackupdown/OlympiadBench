import pynvml
import os

def select_gpus(min_memory_gb, min_gpu=1, max_gpu=4, memory_type='free'):
    assert max_gpu > 0
    pynvml.nvmlInit()
    valid_gpus = []
    left_gpus = []
    try:
        device_count = pynvml.nvmlDeviceGetCount()
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            target_mem = mem_info.free if memory_type == 'free' else mem_info.total
            if target_mem >= min_memory_gb * (1024 ** 3):  # Convert GB to bytes
                valid_gpus.append((str(i), target_mem))
            left_gpus.append(target_mem // (1024 ** 3))
    finally:
        pynvml.nvmlShutdown()

    valid_gpus.sort(key=lambda x:x[1], reverse=True)
    valid_gpus = [x[0] for x in valid_gpus]
    if not valid_gpus or len(valid_gpus) < min_gpu:
        raise RuntimeError("No GPUs meet the memory requirement.{}{}".format(valid_gpus, left_gpus))
    valid_gpus = valid_gpus[:max_gpu]
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(valid_gpus)
    return valid_gpus, left_gpus


import logging

def get_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter('[%(levelname)s][%(asctime)s-%(name)s]-[%(filename)s:%(lineno)d]:%(message)s')
    console_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    return logger
