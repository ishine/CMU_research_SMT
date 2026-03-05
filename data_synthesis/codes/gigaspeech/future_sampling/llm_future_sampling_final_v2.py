#!/usr/bin/env python3
"""
用 simalign 对齐的入口：直接调 core，传 --align-method simalign。
Align 与 base 都在 GPU 0 上跑（默认 --align-device cuda:0）。

用法: 同 llm_future_sampling_final.py，依赖: pip install simalign jieba
"""

import os
import sys


def main() -> None:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    core = os.path.join(script_dir, "llm_future_sampling_core.py")
    argv = [sys.executable, core] + sys.argv[1:]
    # 强制 v2：simalign + GPU 0；若未指定再补 selection-mode / consensus-ratio
    argv.extend(["--align-method", "simalign", "--align-device", "cuda:0"])
    if "--selection-mode" not in argv:
        argv.extend(["--selection-mode", "semantic_merge_vote"])
    if "--consensus-ratio" not in argv:
        argv.extend(["--consensus-ratio", "0.7"])
    os.execv(sys.executable, argv)


if __name__ == "__main__":
    main()
