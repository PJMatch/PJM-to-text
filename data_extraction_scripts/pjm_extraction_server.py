"""Module for extraction from PJM dataset oprimized for the Threadripper."""

import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from pjm_extraction import (
    DATASET_FPS,
    OUTPUT_PATH,
    get_files_to_process,
    get_processed_filenames,
    init_mediapipe,
    load_models_to_memory,
    process_sequence,
)


def process_file_worker(args):
    """Multiprocessing worker."""
    pjm_file, model_buffers, fps = args

    detectors = None
    try:
        detectors = init_mediapipe(model_buffers)
        sequence_data, sequence_name = process_sequence(pjm_file, fps, detectors)

        if sequence_data is None:
            return (pjm_file, False, "Sequence data is empty", 0)

        np.save(OUTPUT_PATH / f"{sequence_name}.npy", np.array(sequence_data, dtype=object))

        frame_count = len(sequence_data)
        return (pjm_file, True, None, frame_count)

    except Exception as e:
        return (pjm_file, False, str(e), 0)

    finally:
        detectors["pose"].close()
        detectors["hands"].close()
        detectors["face"].close()


def process_pjm():
    """Multiprocessing menager."""
    processed = get_processed_filenames()
    files_to_process = get_files_to_process(processed)

    print("Loading models into memory buffer...")
    model_buffers = load_models_to_memory()

    err_file_set = set()
    processed_log = OUTPUT_PATH / "processed_log.txt"

    tasks = [(f, model_buffers, DATASET_FPS) for f in files_to_process]

    MAX_WORKERS = 12

    total_frames = 0
    successful_videos = 0
    print(f"Igniting ProcessPool with {MAX_WORKERS} workers...")

    start_time = time.perf_counter()
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_file_worker, task): task for task in tasks}

        for future in as_completed(futures):
            pjm_file, success, error_msg, frame_count = future.result()

            if success:
                total_frames += frame_count
                successful_videos += 1
                with open(processed_log, "a", encoding="utf-8") as f:
                    f.write(f"{str(pjm_file)}\n")
                print(f"Successfully processed file {pjm_file.name} ({frame_count} frames)")
            else:
                print(f"ERROR processing {pjm_file.name}: {error_msg}")
                err_file_set.add(str(pjm_file))

    end_time = time.perf_counter()
    total_time = end_time - start_time

    overall_fps = total_frames / total_time if total_time > 0 else 0

    print("\n" + "=" * 40)
    print("PIPELINE BENCHMARK RESULTS")
    print("=" * 40)
    print(f"Workers Used    : {MAX_WORKERS}")
    print(f"Videos Processed: {successful_videos} / {len(tasks)}")
    print(f"Total Frames    : {total_frames}")
    print(f"Total Time      : {total_time:.2f} seconds")
    print(f"Overall Speed   : {overall_fps:.2f} FPS")
    print("=" * 40 + "\n")
    return err_file_set


def main():
    """Main function."""
    res_err = process_pjm()
    if res_err:
        print(f"Error occured in files {res_err}")
    else:
        print("Succesfully extracted features from all videos")


if __name__ == "__main__":
    main()
