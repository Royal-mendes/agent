from __future__ import annotations

import argparse
import json
from typing import List

from agent.learning.gt_teacher import GTTeacher
from agent.learning.baseline_teacher import BaselineTeacher
from agent.learning.hindsight_labeler import HindsightLabeler
from agent.learning.lesson_builder import LessonBuilder
from agent.learning.tool_call_dataset import ToolCallDataset
from agent.learning.trajectory_ingestor import TrajectoryIngestor
from agent.schemas import AgentConfig


def build_samples(args: argparse.Namespace) -> List[object]:
    ingestor = TrajectoryIngestor()
    student = ingestor.load(args.student_log, source=args.student_source)
    if args.split:
        student.split = args.split
    samples = HindsightLabeler().label(student)
    if args.baseline_log:
        baseline = ingestor.load(args.baseline_log, source="baseline_apexnav")
        samples.extend(BaselineTeacher().build_samples(student, baseline))
    if args.gt_log:
        gt = ingestor.load(args.gt_log, source="gt")
        samples.extend(GTTeacher().build_samples(student, gt))
    return samples


def main() -> int:
    parser = argparse.ArgumentParser(description="Build reflective tool-use lessons from ApexNav traces.")
    parser.add_argument("--student-log", required=True, help="Reflective episode JSON or text log.")
    parser.add_argument("--baseline-log", help="Optional ApexNav baseline text log.")
    parser.add_argument("--gt-log", help="Optional simple GT trajectory JSON.")
    parser.add_argument("--student-source", default="self_reflection")
    parser.add_argument("--split", help="Override split for generated samples.")
    parser.add_argument("--output-dataset", default="data/tool_call_learning_samples.jsonl")
    parser.add_argument("--memory-path", default="data/reflection_memory.jsonl")
    parser.add_argument("--policy-patch-path", default="data/policy_patches.json")
    parser.add_argument("--write-memory", action="store_true")
    parser.add_argument("--memory-write-mode", default="train_only")
    args = parser.parse_args()

    samples = build_samples(args)
    dataset = ToolCallDataset(args.output_dataset)
    dataset_written = dataset.append_many(samples)
    result = {
        "samples": [sample.to_dict() for sample in samples],
        "dataset_written": dataset_written,
        "output_dataset": args.output_dataset,
    }
    if args.write_memory:
        cfg = AgentConfig(
            memory_path=args.memory_path,
            policy_patch_path=args.policy_patch_path,
            memory_write_mode=args.memory_write_mode,
            enable_policy_patch_table=True,
        )
        result["persistence"] = LessonBuilder(cfg).persist(samples)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
