"""GPU verification: MiniMax-VL Continuous Token builder with real processor.

Tests that build_initial_tokens and merge_non_assistant_tokens produce token_ids
identical to full re-render (legacy path) for MiniMax-VL-01.

Usage:
    cd /workspace/verl && python scripts/test_minimax_vl_ct_gpu.py 2>&1 | tee /workspace/minimax_vl_test.txt
"""

import sys
sys.path.insert(0, ".")

from PIL import Image
import numpy as np

from transformers import AutoTokenizer, AutoProcessor
from verl.utils.tokenizer.continuous_token_wiring import create_continuous_token_builder


def make_dummy_image(w=64, h=64):
    return Image.fromarray(np.random.randint(0, 255, (h, w, 3), dtype=np.uint8))


def legacy_full_render(processor, messages, images):
    """Legacy path: tokenizer.apply_chat_template + processor (ground truth).

    Replicates the official MiniMax-VL inference pattern:
      1. Flatten content blocks to string with <image> placeholders
      2. processor.tokenizer.apply_chat_template(...)
      3. processor(images=..., text=prompt)
    """
    flat_messages = []
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, list):
            parts = []
            for block in content:
                if isinstance(block, dict):
                    btype = block.get("type", "")
                    if btype in ("image", "image_url"):
                        parts.append("<image>")
                    elif btype == "text":
                        parts.append(block.get("text", ""))
            flat_messages.append({**msg, "content": "".join(parts)})
        else:
            flat_messages.append(msg)

    prompt = processor.tokenizer.apply_chat_template(
        flat_messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(
        images=images if images else None,
        text=prompt,
        return_tensors="pt",
    )
    return inputs["input_ids"][0].tolist()


def extract_images_from_messages(messages):
    """Extract PIL images from OpenAI-style content blocks."""
    images = []
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") in ("image", "image_url"):
                    img = block.get("image")
                    if img is not None:
                        images.append(img)
    return images


def test_minimax_vl(model_name):
    print(f"\n{'='*60}")
    print(f"Testing: {model_name}")
    print(f"{'='*60}")

    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    tokenizer = processor.tokenizer

    builder = create_continuous_token_builder(
        tokenizer,
        model_family="auto",
        model_path=model_name,
        processor=processor,
    )
    print(f"  Builder class: {builder.__class__.__name__}")
    assert hasattr(builder, 'merge_non_assistant_tokens'), "merge_non_assistant_tokens missing!"
    assert hasattr(builder, 'processor'), "processor not set!"

    img1 = make_dummy_image()
    img2 = make_dummy_image(80, 80)

    # --- Scenario 1: build_initial_tokens with image ---
    print("\n  [Scenario 1] build_initial_tokens with 1 image")
    messages_1 = [
        {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
        {"role": "user", "content": [
            {"type": "image", "image": img1},
            {"type": "text", "text": "Describe this image."},
        ]},
    ]
    ct_ids = builder.build_initial_tokens(messages_1)
    legacy_ids = legacy_full_render(processor, messages_1, [img1])

    if ct_ids == legacy_ids:
        print(f"    PASS - {len(ct_ids)} tokens match")
    else:
        print(f"    FAIL - CT={len(ct_ids)} vs Legacy={len(legacy_ids)}")
        for i, (a, b) in enumerate(zip(ct_ids, legacy_ids)):
            if a != b:
                print(f"    First diff at pos {i}: CT={a} vs Legacy={b}")
                ctx_start = max(0, i - 3)
                ctx_end = min(len(ct_ids), i + 4)
                print(f"    CT context [{ctx_start}:{ctx_end}]: {ct_ids[ctx_start:ctx_end]}")
                print(f"    Legacy context [{ctx_start}:{ctx_end}]: {legacy_ids[ctx_start:ctx_end]}")
                break
        if len(ct_ids) != len(legacy_ids):
            print(f"    Length diff: {len(ct_ids) - len(legacy_ids)} tokens")
        return False

    # --- Scenario 2: merge_non_assistant_tokens (text-only append) ---
    print("\n  [Scenario 2] merge_non_assistant_tokens (text-only append)")
    assistant_text = "This is a colorful image with various patterns."
    assistant_ids = tokenizer.encode(assistant_text, add_special_tokens=False)
    assistant_merge = builder.merge_assistant_tokens(ct_ids, assistant_ids)
    runtime_after_asst = assistant_merge.token_ids

    prev_messages = messages_1 + [{"role": "assistant", "content": assistant_text}]
    updated_messages = prev_messages + [
        {"role": "user", "content": "What colors do you see?"},
    ]

    merge_result = builder.merge_non_assistant_tokens(
        prev_messages, updated_messages, runtime_after_asst,
    )
    ct_ids_2 = merge_result.token_ids

    legacy_ids_2 = legacy_full_render(
        processor, updated_messages, [img1]
    )

    if ct_ids_2 == legacy_ids_2:
        print(f"    PASS - {len(ct_ids_2)} tokens match")
    else:
        print(f"    FAIL - CT={len(ct_ids_2)} vs Legacy={len(legacy_ids_2)}")
        diff_pos = next(
            (i for i, (a, b) in enumerate(zip(ct_ids_2, legacy_ids_2)) if a != b),
            None,
        )
        if diff_pos is not None:
            print(f"    First diff at pos {diff_pos}: CT={ct_ids_2[diff_pos]} vs Legacy={legacy_ids_2[diff_pos]}")
        if len(ct_ids_2) != len(legacy_ids_2):
            print(f"    Length diff: {len(ct_ids_2) - len(legacy_ids_2)} tokens")
        return False

    # --- Scenario 3: merge_non_assistant_tokens WITH new image ---
    print("\n  [Scenario 3] merge_non_assistant_tokens (new image in appended turn)")
    messages_3_prev = messages_1 + [{"role": "assistant", "content": assistant_text}]
    messages_3_updated = messages_3_prev + [
        {"role": "user", "content": [
            {"type": "image", "image": img2},
            {"type": "text", "text": "Now describe this second image."},
        ]},
    ]

    merge_result_3 = builder.merge_non_assistant_tokens(
        messages_3_prev, messages_3_updated, runtime_after_asst,
    )
    ct_ids_3 = merge_result_3.token_ids

    legacy_ids_3 = legacy_full_render(processor, messages_3_updated, [img1, img2])

    if ct_ids_3 == legacy_ids_3:
        print(f"    PASS - {len(ct_ids_3)} tokens match (image tokens correctly expanded)")
    else:
        print(f"    FAIL - CT={len(ct_ids_3)} vs Legacy={len(legacy_ids_3)}")
        diff_pos = next(
            (i for i, (a, b) in enumerate(zip(ct_ids_3, legacy_ids_3)) if a != b),
            None,
        )
        if diff_pos is not None:
            print(f"    First diff at pos {diff_pos}")
        if len(ct_ids_3) != len(legacy_ids_3):
            print(f"    Length diff: {len(ct_ids_3) - len(legacy_ids_3)} tokens")
        return False

    print(f"\n  ALL 3 SCENARIOS PASS for {model_name}")
    return True


def main():
    models_to_test = [
        "MiniMaxAI/MiniMax-VL-01",
    ]

    results = {}
    for model in models_to_test:
        try:
            results[model] = test_minimax_vl(model)
        except Exception as e:
            print(f"\n  ERROR testing {model}: {e}")
            import traceback
            traceback.print_exc()
            results[model] = False

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for model, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {model}")

    if all(results.values()):
        print("\nAll tests passed!")
        sys.exit(0)
    else:
        print("\nSome tests failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
