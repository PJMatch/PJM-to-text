"""Module for extraction from PJM dataset oprimized for the Threadripper."""

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
    pjm_file, model_buffers, fps = args

    detectors = None
    try:
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
