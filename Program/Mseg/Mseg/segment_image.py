import cv2
import numpy as np
from mseg_semantic.utils import config
from mseg.utils.names_utils import load_class_names, get_universal_class_names
from mseg_semantic.tool.inference_task import InferenceTask


cv2.ocl.setUseOpenCL(False)


def segment_image(
    input_path,
    args,
    device_type,
    output_path
):
    args.u_classes         = get_universal_class_names()
    args.print_freq        = 10
    args.num_model_classes = len(args.u_classes)

    itask = InferenceTask(
        args,
        base_size       = args.base_size,
        crop_h          = args.test_h,
        crop_w          = args.test_w,
        input_file      = input_path,
        output_taxonomy = 'universal',
        scales          = args.scales,
        device_type     = device_type,
        output_path     = output_path
    )
    itask.execute()


if __name__ == '__main__':
    import sys
    import os

    input_path      = sys.argv[1]
    cfg_path        = sys.argv[2]
    device_type     = sys.argv[3]
    output_path     = sys.argv[4]

    args = config.load_cfg_from_cfg_file(cfg_path)

    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)

    segment_image(input_path, args, device_type, output_path)
