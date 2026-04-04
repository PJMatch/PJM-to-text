"""Module for extraction from PJM dataset oprimized for the Threadripper."""

import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager

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


@contextmanager
def suppress_cpp_warnings():
    """Temporarily redirects OS-level stderr to a black hole to silence C++ prints."""
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    saved_stderr_fd = os.dup(sys.stderr.fileno())

    try:
        os.dup2(devnull_fd, sys.stderr.fileno())
        yield
    finally:
        os.dup2(saved_stderr_fd, sys.stderr.fileno())
        os.close(devnull_fd)
        os.close(saved_stderr_fd)


def process_file_worker(args):
    pjm_file, model_buffers, fps = args

    detectors = None
    try:
        with suppress_cpp_warnings():
            detectors = init_mediapipe(model_buffers)
        sequence_data, sequence_name = process_sequence(pjm_file, fps, detectors)

        if sequence_data is None:
            return (pjm_file, False, "Sequence data is empty")

        np.save(OUTPUT_PATH / f"{sequence_name}.npy", np.array(sequence_data, dtype=object))

        return (pjm_file, True, None)

    except Exception as e:
        return (pjm_file, False, str(e))

    finally:
        detectors["pose"].close()
        detectors["hands"].close()
        detectors["face"].close()


def process_pjm():
    processed = get_processed_filenames()
    files_to_process = get_files_to_process(processed)

    print("Loading models into memory buffer...")
    model_buffers = load_models_to_memory()

    err_file_set = set()
    processed_log = OUTPUT_PATH / "processed_log.txt"

    tasks = [(f, model_buffers, DATASET_FPS) for f in files_to_process]

    MAX_WORKERS = 12

    print(f"Igniting ProcessPool with {MAX_WORKERS} workers...")

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_file_worker, task): task for task in tasks}

        for future in as_completed(futures):
            pjm_file, success, error_msg = future.result()

            if success:
                with open(processed_log, "a", encoding="utf-8") as f:
                    f.write(f"{str(pjm_file)}\n")
                print(f"Successfully processed file {pjm_file.name}")
            else:
                print(f"ERROR processing {pjm_file.name}: {error_msg}")
                err_file_set.add(str(pjm_file))

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
