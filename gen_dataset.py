import json
import seqio
import tqdm as tqdm
import concurrent.futures

import sys

def write_chunk(chunk_idx, size):
    dataset = selected_mixture.get_dataset(
        sequence_length={"inputs": 4096, "targets": 4096},
        split="train",
        num_epochs=1,
        shuffle=True,
        copy_pretokenized=True,
        passthrough_features=["_task_source", "_task_name"]
    )

    # write out the data to a new json line file
    with open("niv2_zsopt_%04d.jsonl" % chunk_idx, "w") as f:
        for ex in dataset[chunk_idx: chunk_idx+size].as_numpy_iterator():
            json.dump({
                    "inputs": ex["inputs_pretokenized"].decode().replace("\n", " "),
                    "targets": ex["targets_pretokenized"].decode().replace("\n", " "),
                    "task_source": ex["_task_source"].decode(),
                    "task_name": ex["_task_name"].decode(),
                }, f)
            f.write('\n')


def main():
    selected_mixture = seqio.get_mixture_or_task('t0')

    dataset = selected_mixture.get_dataset(
        sequence_length={"inputs": 4096, "targets": 4096},
        split="train",
        num_epochs=1,
        shuffle=False,
        copy_pretokenized=False,
        passthrough_features=["_template_idx", "_task_source", "_task_name", "_template", "_template_type"]
    )

    cpu_count = max(sys.cpu_count()-1, 1)
    size = len(dataset) // cpu_count

    # We can use a with statement to ensure threads are cleaned up promptly
    with concurrent.futures.ThreadPoolExecutor(max_workers=cpu_count) as executor:
        # Start the load operations and mark each future with its URL
        future_to_url = {executor.submit(write_chunk, url, 60): url for url in URLS}
        for future in concurrent.futures.as_completed(future_to_url):
            url = future_to_url[future]
            try:
                data = future.result()
            except Exception as exc:
                print('%r generated an exception: %s' % (url, exc))
            else:
                print('%r page is %d bytes' % (url, len(data)))

if __name__ == "__main__":
    main()