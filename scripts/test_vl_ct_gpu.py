"""GPU verification: VL Continuous Token builder with real processor.

Tests that merge_non_assistant_tokens correctly goes through the processor path
when images are present, producing token_ids identical to full re-render (legacy).
"""

import sys
sys.path.insert(0, ".")

from PIL import Image
import numpy as np

from transformers import AutoTokenizer, AutoProcessor
from verl.utils.tokenizer.continuous_token_wiring import create_continuous_token_builder


def make_dummy_image(w=64, h=64):
    return Image.fromarray(np.random.randint(0, 255, (h, w, 3), dtype=np.uint8))


def legacy_full_render(processor, messages, *, tools=None):
    """Legacy path: full apply_chat_template with processor (ground truth)."""
    from qwen_vl_utils import process_vision_info
    images, _ = process_vision_info(messages)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=images, return_tensors="pt", padding=True)
    return inputs["input_ids"][0].tolist()


def test_qwen_vl(model_name):
    print(f"\n{'='*60}")
    print(f"Testing: {model_name}")
    print(f"{'='*60}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

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
        {"role": "user", "content": [
            {"type": "image", "image": img1},
            {"type": "text", "text": "Describe this image."},
        ]},
    ]
    ct_ids = builder.build_initial_tokens(messages_1)
    legacy_ids = legacy_full_render(processor, messages_1)
    if ct_ids == legacy_ids:
        print(f"    PASS - {len(ct_ids)} tokens match")
    else:
        print(f"    FAIL - CT={len(ct_ids)} vs Legacy={len(legacy_ids)}")
        # Show first diff
        for i, (a, b) in enumerate(zip(ct_ids, legacy_ids)):
            if a != b:
                print(f"    First diff at pos {i}: CT={a} vs Legacy={b}")
                break
        return False

    # --- Scenario 2: merge_non_assistant_tokens (text only, no new images) ---
    print("\n  [Scenario 2] merge_non_assistant_tokens (text-only append)")
    messages_2 = messages_1 + [
        {"role": "assistant", "content": "This is a colorful image."},
        {"role": "user", "content": "What colors do you see?"},
    ]
    # Simulate runtime: after initial + assistant + new user turn
    runtime_ids = ct_ids + tokenizer.encode(
        "This is a colorful image.", add_special_tokens=False
    )
    # For merge, we need assistant tokens merged first
    assistant_text = "This is a colorful image."
    assistant_merge = builder.merge_assistant_tokens(
        ct_ids,
        tokenizer.encode(assistant_text, add_special_tokens=False),
    )
    runtime_after_asst = assistant_merge.token_ids

    # Now merge the new user message
    prev_messages = messages_1 + [{"role": "assistant", "content": assistant_text}]
    updated_messages = prev_messages + [{"role": "user", "content": "What colors do you see?"}]
    merge_result = builder.merge_non_assistant_tokens(
        prev_messages, updated_messages, runtime_after_asst,
    )
    ct_ids_2 = merge_result.token_ids

    legacy_ids_2 = legacy_full_render(processor, updated_messages)
    if ct_ids_2 == legacy_ids_2:
        print(f"    PASS - {len(ct_ids_2)} tokens match")
    else:
        print(f"    FAIL - CT={len(ct_ids_2)} vs Legacy={len(legacy_ids_2)}")
        diff_pos = next((i for i, (a, b) in enumerate(zip(ct_ids_2, legacy_ids_2)) if a != b), None)
        if diff_pos is not None:
            print(f"    First diff at pos {diff_pos}: CT={ct_ids_2[diff_pos]} vs Legacy={legacy_ids_2[diff_pos]}")
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

    legacy_ids_3 = legacy_full_render(processor, messages_3_updated)
    if ct_ids_3 == legacy_ids_3:
        print(f"    PASS - {len(ct_ids_3)} tokens match (image tokens correctly expanded)")
    else:
        print(f"    FAIL - CT={len(ct_ids_3)} vs Legacy={len(legacy_ids_3)}")
        diff_pos = next((i for i, (a, b) in enumerate(zip(ct_ids_3, legacy_ids_3)) if a != b), None)
        if diff_pos is not None:
            print(f"    First diff at pos {diff_pos}")
        # Check if it's a length issue (image_pad count)
        if len(ct_ids_3) != len(legacy_ids_3):
            print(f"    Length mismatch: CT={len(ct_ids_3)} vs Legacy={len(legacy_ids_3)}")
            print(f"    Difference: {len(ct_ids_3) - len(legacy_ids_3)} tokens")
        return False

    print(f"\n  ALL 3 SCENARIOS PASS for {model_name}")
    return True


def main():
    models_to_test = [
        "Qwen/Qwen2.5-VL-3B-Instruct",
    ]

    results = {}
    for model in models_to_test:
        try:
            results[model] = test_qwen_vl(model)
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
