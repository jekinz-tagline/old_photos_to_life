import os
import shutil
import sys
from subprocess import call


def run_cmd(command):
    try:
        call(command, shell=True)
    except KeyboardInterrupt:
        print("Process interrupted")
        sys.exit(1)


def run_pipeline(
    input_folder: str,
    output_folder: str,
    model_root: str = ".",
    gpu_ids: str = "-1",
    checkpoint_name: str = "Setting_9_epoch_100",
    with_scratch: bool = False,
    hr: bool = False,
):
    input_folder = os.path.abspath(input_folder)
    output_folder = os.path.abspath(output_folder)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    python_exec = sys.executable  # use correct Python interpreter path

    # === Stage 1: Overall Quality Improve ===
    print("Running Stage 1: Overall restoration")
    os.chdir(os.path.join(model_root, "Global"))

    stage_1_output_dir = os.path.join(output_folder, "stage_1_restore_output")
    os.makedirs(stage_1_output_dir, exist_ok=True)

    if not with_scratch:
        stage_1_command = (
            f"{python_exec} test.py --test_mode Full --Quality_restore "
            f"--test_input {input_folder} --outputs_dir {stage_1_output_dir} --gpu_ids {gpu_ids}"
        )
        run_cmd(stage_1_command)
    else:
        mask_dir = os.path.join(stage_1_output_dir, "masks")
        new_input = os.path.join(mask_dir, "input")
        new_mask = os.path.join(mask_dir, "mask")

        stage_1_command_1 = (
            f"{python_exec} detection.py --test_path {input_folder} --output_dir {mask_dir} "
            f"--input_size full_size --GPU {gpu_ids}"
        )
        run_cmd(stage_1_command_1)

        hr_flag = " --HR" if hr else ""
        stage_1_command_2 = (
            f"{python_exec} test.py --Scratch_and_Quality_restore "
            f"--test_input {new_input} --test_mask {new_mask} "
            f"--outputs_dir {stage_1_output_dir} --gpu_ids {gpu_ids}{hr_flag}"
        )
        run_cmd(stage_1_command_2)

    # Move Stage 1 results
    stage_1_results = os.path.join(stage_1_output_dir, "restored_image")
    final_output_dir = os.path.join(output_folder, "final_output")
    os.makedirs(final_output_dir, exist_ok=True)
    for x in os.listdir(stage_1_results):
        shutil.copy(os.path.join(stage_1_results, x), final_output_dir)

    print("Finish Stage 1 ...\n")

    # === Stage 2: Face Detection ===
    print("Running Stage 2: Face Detection")
    os.chdir(os.path.join(model_root, "Face_Detection"))

    stage_2_input_dir = stage_1_results
    stage_2_output_dir = os.path.join(output_folder, "stage_2_detection_output")
    os.makedirs(stage_2_output_dir, exist_ok=True)

    script = "detect_all_dlib_HR.py" if hr else "detect_all_dlib.py"
    stage_2_command = f"{python_exec} {script} --url {stage_2_input_dir} --save_url {stage_2_output_dir}"
    run_cmd(stage_2_command)
    print("Finish Stage 2 ...\n")

    # === Stage 3: Face Enhancement ===
    print("Running Stage 3: Face Enhancement")
    os.chdir(os.path.join(model_root, "Face_Enhancement"))

    stage_3_input_face = stage_2_output_dir
    stage_3_output_dir = os.path.join(output_folder, "stage_3_face_output")
    os.makedirs(stage_3_output_dir, exist_ok=True)

    if hr:
        checkpoint_name = "FaceSR_512"
        load_size = 512
        batch_size = 1
    else:
        load_size = 256
        batch_size = 4

    stage_3_command = (
        f"{python_exec} test_face.py --old_face_folder {stage_3_input_face} "
        f"--old_face_label_folder ./ --tensorboard_log --name {checkpoint_name} --gpu_ids {gpu_ids} "
        f"--load_size {load_size} --label_nc 18 --no_instance --preprocess_mode resize "
        f"--batchSize {batch_size} --results_dir {stage_3_output_dir} --no_parsing_map"
    )
    run_cmd(stage_3_command)
    print("Finish Stage 3 ...\n")

    # === Stage 4: Blending / Warp Back ===
    print("Running Stage 4: Blending")
    os.chdir(os.path.join(model_root, "Face_Detection"))

    stage_4_input_image_dir = stage_1_results
    stage_4_input_face_dir = os.path.join(stage_3_output_dir, "each_img")
    stage_4_output_dir = os.path.join(output_folder, "final_output")
    os.makedirs(stage_4_output_dir, exist_ok=True)

    blend_script = (
        "align_warp_back_multiple_dlib_HR.py"
        if hr
        else "align_warp_back_multiple_dlib.py"
    )
    stage_4_command = (
        f"{python_exec} {blend_script} --origin_url {stage_4_input_image_dir} "
        f"--replace_url {stage_4_input_face_dir} --save_url {stage_4_output_dir}"
    )
    run_cmd(stage_4_command)

    print("Finish Stage 4 ...\n")
    print("All the processing is done. Please check the results.")
