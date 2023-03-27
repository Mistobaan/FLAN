import os
import threading
from typing import List
import queue
from concurrent.futures import ThreadPoolExecutor
import simplejson as json
import logging
import tqdm

# Function to write messages to the queue
def add_message_to_queue(msg, q):
    q.put(msg)

# Function to process messages from the queue
def process_messages(thread_id, total_workers, q, dataset):

    partition_number = thread_id
    partition_file = os.path.join(dataset, f"{dataset}_{partition_number:04d}.jsonl")

    f = open(partition_file, "a", encoding='utf-8')
    size = 0
    MAX_SIZE_250MB = 1024 * 1024 * 250

    while True:
        idx, msg = q.get()
        if msg == "STOP":
            f.close()
            break

        item = json.dumps({
            "inputs": msg["inputs_pretokenized"].decode(),
            "targets": msg["targets_pretokenized"].decode(),
            "task_source": msg["_task_source"].decode().lower(),
            "task_name": msg["_task_name"].decode().lower(),
        })
        size += f.write(item)
        size += f.write("\n")

        if size > MAX_SIZE_250MB:
            f.close()
            size = 0
            partition_number = thread_id + total_workers
            partition_file = os.path.join(dataset, f"{dataset}_{partition_number:04d}.jsonl")
            f = open(partition_file, "a", encoding='utf-8')


def load_dataset(name:str, tasks:List[str]):
    from flan.v2 import mixtures
    import seqio
    import tensorflow_datasets as tdfs

    seqio.MixtureRegistry.add(
        name,
        tasks,
    )
    selected_mixture = seqio.get_mixture_or_task('t0_submix')

    return selected_mixture.get_dataset(
        sequence_length={"inputs": 4096, "targets": 4096},
        split=tdfs.Split.TRAIN,
        num_epochs=1,
        shuffle=False,
        copy_pretokenized=True,
        passthrough_features=["_template_idx", "_task_source", "_task_name", "_template", "_template_type"]
    )


def main(dataset='t0_submix'):
    # Create a directory for partition files if it doesn't exist
    if not os.path.exists(dataset):
        os.makedirs(dataset)

    # Initialize the message queue
    message_queue = queue.Queue()

    num_workers = max(os.cpu_count() - 1, 1)
    # Create the thread pool
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Start the consumer threads
        logging.info(f"starting {num_workers} consumer threads")
        consumers = [executor.submit(process_messages, dataset, idx, message_queue) for idx in range(num_workers)]

        logging.info(f"reading {dataset} dataset")
        ds = load_dataset(dataset, tasks=[
            ('t0_zsopt', 1),      # mixing weight = 25%
            # ('t0_fsopt', 1),      # mixing weight = 25%
            # ('t0_zsnoopt', 1),    # mixing weight = 25%
            # ('t0_fsnoopt', 1),    # mixing weight = 25%
        ])

        # Add messages to the queue
        for idx, message in enumerate(tqdm.tqdm(ds)):
            message_queue.put((idx, message))

        # Add 'STOP' signals to the queue to stop the consumer threads
        for _ in range(num_workers):
            add_message_to_queue((-1, "STOP"), message_queue)

        # Wait for all threads to finish
        for future in consumers:
            future.result()


if __name__ == "__main__":
    main()