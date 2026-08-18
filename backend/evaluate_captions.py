#!/usr/bin/env python3
"""Evaluate JSONL caption predictions.

Example line:
{"image":"images/1.jpg","prediction":"a dog on grass","references":["a dog runs in a field"]}
"""

import argparse
import json

from backend.evaluation.caption_metrics import evaluate_records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("jsonl", help="JSONL file with prediction/references")
    parser.add_argument("--clip", action="store_true", help="also compute CLIP similarity")
    args = parser.parse_args()

    records = []
    with open(args.jsonl, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    print(json.dumps(evaluate_records(records, include_clip=args.clip), indent=2))


if __name__ == "__main__":
    main()
