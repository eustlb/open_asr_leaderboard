import argparse
import os
import re
import torch
from transformers import AutoProcessor, CohereAsrForConditionalGeneration
import evaluate
from normalizer import data_utils
import time
from tqdm import tqdm

wer_metric = evaluate.load("wer")


def remove_brackets(text):
    text = text.replace("(", " ").replace(")", " ")
    text = re.sub(r'\s+', ' ', text)
    return text


@torch.inference_mode()
def main(args):
    args.model_id = os.path.normpath(args.model_id)

    device = f"cuda:{args.device}" if args.device >= 0 else "cpu"

    print(f"Loading model: {args.model_id}")
    processor = AutoProcessor.from_pretrained(args.model_id)
    model = CohereAsrForConditionalGeneration.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    model.eval()

    def build_records(dataset_iter, desc):
        records = []
        for sample in tqdm(dataset_iter, desc=desc):
            audio_array = sample["audio"]["array"]
            sampling_rate = sample["audio"]["sampling_rate"]
            records.append(
                {
                    "audio_array": audio_array,
                    "sampling_rate": sampling_rate,
                    "reference": sample["original_text"],
                    "reference_norm_en": sample["norm_text"],
                    "audio_length_s": len(audio_array) / sampling_rate,
                }
            )
        return records

    def transcribe_batch(records):
        audios = [record["audio_array"] for record in records]
        sampling_rates = [record["sampling_rate"] for record in records]

        inputs = processor(
            audios,
            sampling_rate=sampling_rates[0],
            return_tensors="pt",
        )
        inputs = inputs.to(model.device, dtype=model.dtype)

        start_time = time.time()
        outputs = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
        runtime = time.time() - start_time

        batch_predictions = processor.batch_decode(outputs, skip_special_tokens=True)
        return batch_predictions, runtime

    def transcribe_records(records, collect_results) -> tuple[list[float], list[float], list[str], list[str]] | None:
        all_predictions = []
        total_runtime = 0.0

        for i in tqdm(range(0, len(records), args.batch_size), desc="Transcribing..."):
            batch = records[i:i + args.batch_size]
            preds, runtime = transcribe_batch(batch)
            all_predictions.extend(preds)
            total_runtime += runtime

        per_sample_runtime = total_runtime / len(records)

        if not collect_results:
            return None

        audio_lengths = [record["audio_length_s"] for record in records]
        transcription_times = [per_sample_runtime] * len(records)

        normalizer = data_utils.normalizer if args.language == 'en' else data_utils.ml_normalizer
        predictions = [normalizer(remove_brackets(pred)) for pred in all_predictions]
        if args.language == 'en':
            references = [record["reference_norm_en"] for record in records]
        else:
            references = [normalizer(record["reference"]) for record in records]

        return audio_lengths, transcription_times, predictions, references

    if args.warmup_steps is not None:
        warmup_dataset = data_utils.load_data(args)
        warmup_dataset = data_utils.prepare_data(warmup_dataset)

        num_warmup_samples = args.warmup_steps * args.batch_size
        if args.streaming:
            warmup_dataset = warmup_dataset.take(num_warmup_samples)
        else:
            warmup_dataset = warmup_dataset.select(range(min(num_warmup_samples, len(warmup_dataset))))
        warmup_records = build_records(iter(warmup_dataset), desc="Preparing warm-up samples...")
        transcribe_records(warmup_records, collect_results=False)

    dataset = data_utils.load_data(args)
    dataset = data_utils.prepare_data(dataset)

    if args.max_eval_samples is not None and args.max_eval_samples > 0:
        print(f"Subsampling dataset to first {args.max_eval_samples} samples!")
        if args.streaming:
            dataset = dataset.take(args.max_eval_samples)
        else:
            dataset = dataset.select(range(min(args.max_eval_samples, len(dataset))))

    records = build_records(iter(dataset), desc="Preparing samples...")
    records.sort(key=lambda record: record["audio_length_s"], reverse=True)
    (
        audio_lengths,
        transcription_times,
        predictions,
        references,
    ) = transcribe_records(records, collect_results=True)

    manifest_path = data_utils.write_manifest(
        references,
        predictions,
        args.model_id,
        args.dataset_path,
        args.dataset,
        args.split,
        audio_length=audio_lengths,
        transcription_time=transcription_times,
        basedir=args.basedir,
    )
    print("Results saved at path:", os.path.abspath(manifest_path))

    wer = wer_metric.compute(
        references=references, predictions=predictions
    )
    wer = round(100 * wer, 2)
    rtfx = round(sum(audio_lengths) / sum(transcription_times), 2)
    print("WER:", wer, "%", "RTFx:", rtfx)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_id", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, default="esb/datasets")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--language", type=str, default="en")
    parser.add_argument("--device", type=int, default=-1)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--max_new_tokens", type=int, default=500)
    parser.add_argument("--max_eval_samples", type=int, default=None)
    parser.add_argument("--no-streaming", dest="streaming", action="store_false")
    parser.add_argument("--warmup_steps", type=int, default=4)
    parser.add_argument("--basedir", type=str, default="./results/")
    args = parser.parse_args()
    parser.set_defaults(streaming=False)

    main(args)
