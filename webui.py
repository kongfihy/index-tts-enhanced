import os
import shutil
import sys
from pathlib import Path

# Hugging Face reads its cache configuration during import.  Configure it
# before importing Gradio, Transformers or any other third-party dependency.
from indextts.utils.hf_cache import configure_huggingface_environment

HF_HUB_CACHE = configure_huggingface_environment(
    project_root=Path(__file__).resolve().parent,
)

# Keep other generated runtime caches in writable temp dirs on macOS.
os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/indextts-mpl')
os.environ.setdefault('XDG_CACHE_HOME', '/private/tmp/indextts-cache')
os.environ.setdefault('NUMBA_CACHE_DIR', '/private/tmp/indextts-numba')
Path(os.environ['MPLCONFIGDIR']).mkdir(parents=True, exist_ok=True)
Path(os.environ['XDG_CACHE_HOME']).mkdir(parents=True, exist_ok=True)
Path(os.environ['NUMBA_CACHE_DIR']).mkdir(parents=True, exist_ok=True)
print(f">> Hugging Face primary cache: {HF_HUB_CACHE}")

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import pandas as pd

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, "indextts"))

import argparse
parser = argparse.ArgumentParser(
    description="IndexTTS WebUI",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument("--verbose", action="store_true", default=False, help="Enable verbose mode")
parser.add_argument("--port", type=int, default=7860, help="Port to run the web UI on")
parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to run the web UI on")
parser.add_argument("--model_dir", type=str, default="./checkpoints", help="Model checkpoints directory")
parser.add_argument("--fp16", action="store_true", default=False, help="Use FP16 for inference if available")
parser.add_argument("--use_deepspeed", action="store_true", default=False, help="Use Deepspeed to accelerate if available")
parser.add_argument("--cuda_kernel", action="store_true", default=False, help="Use cuda kernel for inference if available")
parser.add_argument("--gui_seg_tokens", type=int, default=120, help="GUI: Max tokens per generation segment")
cmd_args = parser.parse_args()

if not os.path.exists(cmd_args.model_dir):
    print(f"Model directory {cmd_args.model_dir} does not exist. Please download the model first.")
    sys.exit(1)

for file in [
    "bpe.model",
    "gpt.pth",
    "config.yaml",
    "s2mel.pth",
    "wav2vec2bert_stats.pt"
]:
    file_path = os.path.join(cmd_args.model_dir, file)
    if not os.path.exists(file_path):
        print(f"Required file {file_path} does not exist. Please download it.")
        sys.exit(1)

import gradio as gr
from indextts.infer_v2 import IndexTTS2
from indextts_dubbing.candidates import (
    build_candidate_plan,
    candidate_directory,
    candidate_output_paths,
    read_candidate_manifest,
    write_candidate_manifest,
)
from indextts_dubbing.audio.loudness_matcher import write_loudness_matched_copy
from tools.i18n.i18n import I18nAuto
from indextts_task_center import (
    JobAlreadyRunning,
    JobCancelled,
    JobManager,
    JobProgress,
    TASK_CENTER_PORT,
    start_task_center_server,
    task_center_host_for_webui,
)
from indextts_webui_helpers import (
    GENERATION_STYLE_BALANCED,
    GENERATION_STYLE_CHOICES,
    generation_readiness,
    generation_request_key,
    generation_style_values,
    matched_output_details,
    normalize_advanced_generation_args,
    normalize_choice_index,
    prepare_reference_audio_file,
    reference_audio_quality_message,
    validate_audio_file,
    validate_generation_inputs,
)

# This workspace is used as a local Chinese dubbing tool. Auto locale detection
# can return an unsupported macOS locale and silently fall back to English.
i18n = I18nAuto(language="zh_CN")
job_manager = JobManager(output_root=Path("outputs/tasks"))
start_task_center_server(job_manager, host=task_center_host_for_webui(cmd_args.host))
MODE = 'local'
tts = IndexTTS2(model_dir=cmd_args.model_dir,
                cfg_path=os.path.join(cmd_args.model_dir, "config.yaml"),
                use_fp16=cmd_args.fp16,
                use_deepspeed=cmd_args.use_deepspeed,
                use_cuda_kernel=cmd_args.cuda_kernel,
                )
EMO_CHOICES = [i18n("与音色参考音频相同"),
                i18n("使用情感参考音频"),
                i18n("使用情感向量控制"),
                i18n("使用情感描述文本控制")]
os.makedirs("outputs/tasks",exist_ok=True)
os.makedirs("prompts",exist_ok=True)

def register_job(project_name, text, prompt,
                 emo_control_method, emo_ref_path, emo_weight,
                 vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8,
                 emo_text, emo_random, max_text_tokens_per_segment,
                 seed_value, candidate_count, *advanced_args):
    try:
        clean_text = validate_generation_inputs(prompt, text)
        emotion_method = normalize_choice_index(emo_control_method, EMO_CHOICES)
        vectors = [float(value or 0.0) for value in (vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8)]
        clean_emo_weight = float(emo_weight)
        clean_segment_limit = int(max_text_tokens_per_segment)
        candidate_plan = build_candidate_plan(seed_value, candidate_count)
        clean_advanced_args = normalize_advanced_generation_args(advanced_args)
        if emotion_method == 1:
            emo_ref_path = validate_audio_file(emo_ref_path, "情感参考音频")
        if emotion_method == 2 and sum(vectors) > 1.5:
            raise ValueError(i18n("情感向量之和不能超过1.5，请调整后重试。"))
    except (TypeError, ValueError) as exc:
        raise gr.Error(str(exc))

    request_snapshot = {
        "prompt": str(prompt),
        "text": clean_text,
        "emo_control_method": emotion_method,
        "emo_ref_path": str(emo_ref_path) if emo_ref_path else None,
        "emo_weight": clean_emo_weight,
        "vectors": vectors,
        "emo_text": (emo_text or "").strip() or None,
        "emo_random": bool(emo_random),
        "max_text_tokens_per_segment": clean_segment_limit,
        "seed": candidate_plan.base_seed,
        "candidate_count": candidate_plan.count,
        "candidate_seeds": list(candidate_plan.seeds),
        "advanced_args": clean_advanced_args,
    }
    summary = clean_text.replace("\n", " ")
    job_id = job_manager.create_job(
        project_name=project_name,
        job_type="tts",
        input_summary=summary,
        parameters={
            "prompt_audio": os.path.basename(str(prompt)),
            "emotion_method": emotion_method,
            "request_key": generation_request_key(request_snapshot),
            "seed": candidate_plan.base_seed,
            "candidate_count": candidate_plan.count,
            "loudness_ab": bool(clean_advanced_args[-2]),
            "reference_format_match": bool(clean_advanced_args[-1]),
        },
        seed=candidate_plan.base_seed,
    )
    return (
        job_id,
        request_snapshot,
        gr.update(interactive=False, value="正在排队…"),
        gr.update(interactive=False),
        "任务已加入队列，开始后可以在任务中心查看进度。",
        True,
    )


def apply_generation_style(style):
    top_p_value, top_k_value, temperature_value, num_beams_value = (
        generation_style_values(style)
    )
    return (
        gr.update(value=top_p_value),
        gr.update(value=top_k_value),
        gr.update(value=temperature_value),
        gr.update(value=num_beams_value),
    )


def update_submit_state(prompt, text, is_busy=False):
    if is_busy:
        return gr.update(interactive=False, value="正在生成…"), "当前任务正在生成，可以继续编辑文本准备下一次提交。"
    ready, message = generation_readiness(prompt, text)
    return gr.update(
        interactive=ready,
        value="开始生成" if ready else "请完善输入",
    ), message


def update_reference_state(prompt, text, is_busy=False):
    submit_update, readiness = update_submit_state(prompt, text, is_busy)
    return submit_update, readiness, reference_audio_quality_message(prompt)


def prepare_prompt_audio(prompt):
    try:
        prepared_path, message = prepare_reference_audio_file(prompt)
    except ValueError as exc:
        raise gr.Error(str(exc))
    return gr.update(value=prepared_path), message


def restore_action_buttons(prompt, text):
    ready, _ = generation_readiness(prompt, text)
    return (
        gr.update(interactive=ready, value="开始生成" if ready else "请完善输入"),
        gr.update(interactive=True),
        False,
    )


def reset_ui_state():
    return (
        gr.update(interactive=False, value="请完善输入"),
        "请先上传参考音频，并填写需要生成的文本。",
        "等待生成",
        "输入文本后会在这里预览模型分段。",
        gr.update(value=None, visible=False),
        gr.update(value=None, visible=False),
        gr.update(value=None, visible=False),
        gr.update(value=None, visible=False),
        gr.update(value=None, visible=False),
        gr.update(value=None, visible=False),
        False,
        "上传后会自动检查时长、音量、削波和静音比例。",
    )


def mark_generation_failed():
    return "生成未完成，请根据页面提示检查输入；任务详情可在任务中心查看。"


def candidate_result_updates(results):
    clean_results = []
    for index, result in enumerate(results or [], start=1):
        if isinstance(result, dict):
            path = result.get("path")
            label = result.get("label") or f"生成结果 {index}"
        else:
            path = result
            label = f"生成结果 {index}"
        if path:
            clean_results.append((str(path), str(label)))
    padded = (clean_results + [(None, "生成结果")] * 6)[:6]
    return tuple(
        gr.update(value=path, label=label, visible=bool(path))
        for path, label in padded
    )


def gen_single(job_id, request_snapshot, progress=gr.Progress()):
    output_root = Path("outputs/tasks")
    try:
        should_run, existing_output = job_manager.start_job(job_id)
    except JobCancelled:
        gr.Warning(i18n("任务已取消"))
        return *(gr.update() for _ in range(6)), "任务已取消"
    except JobAlreadyRunning as exc:
        gr.Warning(str(exc))
        return *(gr.update() for _ in range(6)), "相同任务已经在生成，请在任务中心查看进度。"
    except RuntimeError as exc:
        raise gr.Error(str(exc))

    if not should_run:
        restored = read_candidate_manifest(output_root, job_id)
        if not restored and existing_output:
            restored = [{"path": existing_output, "label": "已完成结果"}]
        return (
            *candidate_result_updates(restored),
            f"检测到相同任务已经完成，已恢复 {len(restored) or 1} 个生成结果。",
        )

    candidate_root = candidate_directory(output_root, job_id)
    try:
        if not isinstance(request_snapshot, dict) or not request_snapshot:
            raise ValueError("任务参数快照丢失，请重新提交")

        prompt = request_snapshot["prompt"]
        clean_text = validate_generation_inputs(prompt, request_snapshot["text"])
        emotion_method = normalize_choice_index(request_snapshot["emo_control_method"], EMO_CHOICES)
        emo_ref_path = request_snapshot.get("emo_ref_path")
        emo_weight = float(request_snapshot.get("emo_weight", 1.0))
        vectors = [float(value or 0.0) for value in (request_snapshot.get("vectors") or [])]
        if len(vectors) != 8:
            raise ValueError("情感向量参数不完整，请重新提交")
        emo_text = request_snapshot.get("emo_text")
        emo_random = bool(request_snapshot.get("emo_random", False))
        max_text_tokens_per_segment = int(request_snapshot.get("max_text_tokens_per_segment", 120))
        candidate_plan = build_candidate_plan(
            request_snapshot.get("seed", 0),
            request_snapshot.get("candidate_count", 1),
        )
        saved_candidate_seeds = request_snapshot.get("candidate_seeds") or []
        if len(saved_candidate_seeds) == candidate_plan.count:
            candidate_seeds = tuple(int(seed) for seed in saved_candidate_seeds)
        else:
            candidate_seeds = candidate_plan.seeds
        output_paths = candidate_output_paths(output_root, job_id, candidate_seeds)
        advanced_args = normalize_advanced_generation_args(
            request_snapshot.get("advanced_args") or []
        )

        do_sample, top_p, top_k, temperature, \
            length_penalty, num_beams, repetition_penalty, max_mel_tokens, \
            create_loudness_match, match_reference_format = advanced_args
        kwargs = {
            "do_sample": bool(do_sample),
            "top_p": float(top_p),
            "top_k": int(top_k) if int(top_k) > 0 else None,
            "temperature": float(temperature),
            "length_penalty": float(length_penalty),
            "num_beams": int(num_beams),
            "repetition_penalty": float(repetition_penalty),
            "max_mel_tokens": int(max_mel_tokens),
            # Keep the first result as the untouched dry version. The optional
            # reference-level copy is produced afterwards from this same file,
            # so A/B comparison never requires a second model inference.
            "normalize_output_peak": False,
        }
        if emotion_method == 0:
            emo_ref_path = None
            emo_weight = 1.0
        elif emotion_method == 1:
            emo_ref_path = validate_audio_file(emo_ref_path, "情感参考音频")

        if emotion_method == 2:
            if sum(vectors) > 1.5:
                raise ValueError(i18n("情感向量之和不能超过1.5，请调整后重试。"))
            vec = vectors
        else:
            vec = None

        shutil.rmtree(candidate_root, ignore_errors=True)
        candidate_root.mkdir(parents=True, exist_ok=True)
        generated_results = []
        generated_paths = []
        generated_seeds = []
        generated_labels = []
        generated_variants = []
        match_summaries = []
        for index, (seed, output_path) in enumerate(zip(candidate_seeds, output_paths)):
            candidate_number = index + 1
            tts.gr_progress = JobProgress(
                job_manager,
                job_id,
                progress,
                progress_start=index / candidate_plan.count,
                progress_span=1.0 / candidate_plan.count,
                description_prefix=f"候选 {candidate_number}/{candidate_plan.count} · ",
            )
            dry_output = tts.infer(
                spk_audio_prompt=prompt,
                text=clean_text,
                output_path=str(output_path),
                emo_audio_prompt=emo_ref_path,
                emo_alpha=emo_weight,
                emo_vector=vec,
                use_emo_text=(emotion_method == 3),
                emo_text=emo_text,
                use_random=emo_random,
                verbose=cmd_args.verbose,
                max_text_tokens_per_segment=max_text_tokens_per_segment,
                seed=seed,
                **kwargs,
            )
            dry_label = f"候选 {candidate_number} · 原始干声"
            generated_results.append({"path": dry_output, "label": dry_label})
            generated_paths.append(dry_output)
            generated_seeds.append(seed)
            generated_labels.append(dry_label)
            generated_variants.append("dry")

            if create_loudness_match:
                matched_path, matched_label, matched_variant = matched_output_details(
                    output_path,
                    candidate_number,
                    match_reference_format,
                )
                matched_output, match_stats = write_loudness_matched_copy(
                    dry_output,
                    prompt,
                    matched_path,
                    match_reference_format=match_reference_format,
                )
                generated_results.append({"path": matched_output, "label": matched_label})
                generated_paths.append(matched_output)
                generated_seeds.append(seed)
                generated_labels.append(matched_label)
                generated_variants.append(matched_variant)
                match_summaries.append(f"候选 {candidate_number}：{match_stats.summary()}")

        write_candidate_manifest(
            output_root,
            job_id,
            generated_paths,
            generated_seeds,
            labels=generated_labels,
            variants=generated_variants,
        )
        job_manager.complete_job(job_id, generated_paths)
        seed_text = "、".join(str(seed) for seed in candidate_seeds)
        if match_summaries:
            matched_version_name = "匹配交付版" if match_reference_format else "安全响度匹配版"
            status = (
                f"生成完成 · {candidate_plan.count} 个候选，每个保留原始干声和{matched_version_name} · "
                f"随机种子 {seed_text}\n\n" + "\n\n".join(match_summaries)
            )
        else:
            status = f"生成完成 · {candidate_plan.count} 个原始干声候选 · 随机种子 {seed_text}"
        return (*candidate_result_updates(generated_results), status)
    except Exception as exc:
        job_manager.fail_job(job_id, exc)
        shutil.rmtree(candidate_root, ignore_errors=True)
        if isinstance(exc, JobCancelled):
            raise gr.Error(i18n("任务已取消"))
        raise gr.Error(f"生成失败：{exc}")
    finally:
        tts.gr_progress = None

def empty_segments_frame():
    return pd.DataFrame([], columns=[i18n("序号"), i18n("分句内容"), i18n("Token数")])


def on_input_text_change(text, max_text_tokens_per_segment):
    clean_text = (text or "").strip()
    if not clean_text:
        return gr.update(value=empty_segments_frame()), "输入文本后会在这里预览模型分段。"

    try:
        text_tokens_list = tts.tokenizer.tokenize(clean_text)
        segments = tts.tokenizer.split_segments(
            text_tokens_list,
            max_text_tokens_per_segment=int(max_text_tokens_per_segment),
        )
        data = []
        for index, segment in enumerate(segments, start=1):
            segment_text = "".join(segment)
            data.append([index, segment_text, len(segment)])
        return gr.update(value=data), f"预计分为 {len(data)} 段；这里只做预览，不会修改原文本。"
    except Exception as exc:
        print(f">> segment preview warning: {exc}")
        return (
            gr.update(value=empty_segments_frame()),
            "分句预览暂时不可用，但不会影响文本继续编辑。",
        )


def on_method_select(emo_control_method):
    try:
        emotion_method = normalize_choice_index(emo_control_method, EMO_CHOICES)
    except ValueError:
        emotion_method = 0

    if emotion_method == 1:
        return (
            gr.update(visible=True),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
        )
    if emotion_method == 2:
        return (
            gr.update(visible=False),
            gr.update(visible=True),
            gr.update(visible=True),
            gr.update(visible=False),
        )
    if emotion_method == 3:
        return (
            gr.update(visible=False),
            gr.update(visible=True),
            gr.update(visible=False),
            gr.update(visible=True),
        )
    return (
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(visible=False),
    )


APP_CSS = """
:root {
    color-scheme: light dark;
    --idx-bg: #f6f7fb;
    --idx-surface: rgba(255, 255, 255, 0.92);
    --idx-surface-soft: rgba(255, 255, 255, 0.72);
    --idx-text: #172033;
    --idx-text-muted: #647084;
    --idx-border: rgba(99, 112, 138, 0.22);
    --idx-shadow: 0 18px 48px rgba(15, 23, 42, 0.08);
    --idx-hero-bg:
        radial-gradient(circle at top left, rgba(124, 92, 255, 0.22), transparent 34%),
        linear-gradient(135deg, rgba(255, 255, 255, 0.96), rgba(234, 248, 246, 0.88));
    --idx-accent: #6852f5;
    --idx-accent-soft: rgba(104, 82, 245, 0.12);
    --idx-green-soft: rgba(72, 200, 178, 0.14);
    --idx-output-bg: rgba(248, 250, 252, 0.86);
}

@media (prefers-color-scheme: dark) {
    :root {
        --idx-bg: #0c111d;
        --idx-surface: rgba(21, 28, 43, 0.92);
        --idx-surface-soft: rgba(27, 36, 54, 0.74);
        --idx-text: #eef3ff;
        --idx-text-muted: #a8b3c7;
        --idx-border: rgba(154, 169, 196, 0.2);
        --idx-shadow: 0 20px 58px rgba(0, 0, 0, 0.36);
        --idx-hero-bg:
            radial-gradient(circle at top left, rgba(139, 116, 255, 0.28), transparent 36%),
            linear-gradient(135deg, rgba(25, 33, 52, 0.98), rgba(13, 55, 64, 0.74));
        --idx-accent: #9b8cff;
        --idx-accent-soft: rgba(155, 140, 255, 0.16);
        --idx-green-soft: rgba(82, 211, 189, 0.13);
        --idx-output-bg: rgba(14, 21, 34, 0.76);
    }
}

html.dark,
body.dark,
.dark {
    --idx-bg: #0c111d;
    --idx-surface: rgba(21, 28, 43, 0.92);
    --idx-surface-soft: rgba(27, 36, 54, 0.74);
    --idx-text: #eef3ff;
    --idx-text-muted: #a8b3c7;
    --idx-border: rgba(154, 169, 196, 0.2);
    --idx-shadow: 0 20px 58px rgba(0, 0, 0, 0.36);
    --idx-hero-bg:
        radial-gradient(circle at top left, rgba(139, 116, 255, 0.28), transparent 36%),
        linear-gradient(135deg, rgba(25, 33, 52, 0.98), rgba(13, 55, 64, 0.74));
    --idx-accent: #9b8cff;
    --idx-accent-soft: rgba(155, 140, 255, 0.16);
    --idx-green-soft: rgba(82, 211, 189, 0.13);
    --idx-output-bg: rgba(14, 21, 34, 0.76);
}

body,
.gradio-container {
    background: var(--idx-bg) !important;
}
body {
    overflow-x: hidden;
}
html {
    scrollbar-width: thin;
    scrollbar-color: var(--idx-border) transparent;
}
html::-webkit-scrollbar {
    width: 8px;
}
html::-webkit-scrollbar-track {
    background: transparent;
}
html::-webkit-scrollbar-thumb {
    border: 2px solid transparent;
    border-radius: 999px;
    background: var(--idx-border);
    background-clip: content-box;
}
.gradio-container {
    width: min(1480px, calc(100vw - 32px)) !important;
    max-width: 1480px !important;
    margin: 0 auto !important;
    padding: 20px 0 44px !important;
    color: var(--idx-text) !important;
}
.hero-shell {
    position: relative;
    overflow: hidden;
    padding: 28px 30px;
    margin-bottom: 18px;
    border: 1px solid var(--idx-border);
    border-radius: 20px;
    background: var(--idx-hero-bg);
    box-shadow: var(--idx-shadow);
}
.hero-shell::after {
    content: "";
    position: absolute;
    right: -70px;
    top: -92px;
    width: 230px;
    height: 230px;
    border-radius: 999px;
    background: var(--idx-green-soft);
    filter: blur(4px);
    pointer-events: none;
}
.hero-kicker {
    position: relative;
    z-index: 1;
    margin-bottom: 8px;
    color: var(--idx-accent);
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 0.12em;
}
.hero-shell h1 {
    position: relative;
    z-index: 1;
    margin: 0;
    color: var(--idx-text);
    font-size: clamp(28px, 4vw, 40px);
    line-height: 1.12;
}
.hero-shell p {
    position: relative;
    z-index: 1;
    max-width: 720px;
    margin: 10px 0 0;
    color: var(--idx-text-muted);
    font-size: 15px;
    line-height: 1.7;
}
.workspace-grid {
    display: grid !important;
    grid-template-columns: minmax(280px, 0.95fr) minmax(380px, 1.25fr) minmax(300px, 0.9fr);
    grid-template-areas: "reference content result";
    gap: 18px !important;
    align-items: start !important;
}
.workspace-grid > * {
    min-width: 0 !important;
    width: auto !important;
}
.reference-panel {
    grid-area: reference;
}
.content-panel {
    grid-area: content;
}
.result-panel {
    grid-area: result;
}
.panel-card,
.output-card {
    min-width: 0 !important;
    padding: 20px !important;
    border: 1px solid var(--idx-border) !important;
    border-radius: 18px !important;
    background: var(--idx-surface) !important;
    box-shadow: var(--idx-shadow);
    backdrop-filter: blur(14px);
}
.panel-card h3,
.output-card h3 {
    margin-top: 0;
    color: var(--idx-text);
}
.section-note {
    color: var(--idx-text-muted);
    font-size: 13px;
    line-height: 1.6;
}
.status-copy {
    min-height: 24px;
    color: var(--idx-text-muted);
    font-size: 13px;
}
.gradio-container textarea,
.gradio-container input,
.gradio-container .wrap.default,
.gradio-container .wrap.svelte-1ipelgc,
.gradio-container .block {
    border-color: var(--idx-border) !important;
}
.gradio-container textarea,
.gradio-container input {
    background: var(--idx-surface-soft) !important;
    color: var(--idx-text) !important;
}
#generate-button button {
    min-height: 46px;
    font-weight: 700;
    transition: transform 120ms ease-out, box-shadow 120ms ease-out;
}
#generate-button button:not(:disabled) {
    box-shadow: 0 12px 26px var(--idx-accent-soft);
}
.gradio-container #generate-button button:disabled,
.gradio-container #generate-button button[disabled] {
    background: var(--button-secondary-background-fill) !important;
    border-color: var(--idx-border) !important;
    color: var(--idx-text-muted) !important;
    opacity: 0.78 !important;
    box-shadow: none !important;
}
#generate-button button:active:not(:disabled) {
    transform: scale(0.985);
}
#clear-button button {
    min-height: 46px;
}
.action-row {
    align-items: end;
}
.output-card {
    position: sticky;
    top: 18px;
    align-self: start;
    margin-top: 0;
    background: var(--idx-output-bg) !important;
    overflow: visible !important;
}
.block.generation-status {
    position: relative;
    display: flex;
    width: 100% !important;
    min-height: 84px;
    align-items: center;
    box-sizing: border-box;
    padding: 12px 14px !important;
    border: 1px solid var(--idx-border);
    border-radius: 12px;
    background: var(--idx-surface-soft);
    overflow: visible !important;
}
.block.generation-status .prose.generation-status {
    display: block;
    width: 100%;
    min-height: 0;
    padding: 0 !important;
    border: 0;
    border-radius: 0;
    background: transparent;
}
.block.generation-status .prose.generation-status p {
    margin: 0;
}
.block.generation-status > .wrap.center.full {
    border-radius: inherit;
}
.block.generation-status > .wrap.center.full:not(.hide) {
    inset: 0 !important;
    width: 100% !important;
    height: 100% !important;
    min-height: 84px !important;
    background: var(--idx-surface-soft) !important;
    color: var(--idx-text) !important;
    opacity: 1 !important;
}
.block.generation-status > .wrap.center.full:not(.hide) * {
    color: var(--idx-text) !important;
}
.result-panel audio {
    width: 100%;
}
.output-card > .styler {
    background: transparent !important;
}
.output-card .html-container.padding {
    padding: 0 0 8px !important;
}
.output-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
    margin-bottom: 10px;
}
.output-header h3 {
    margin: 0;
    font-size: 16px;
    line-height: 1.35;
}
.task-center-link {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    min-height: 34px;
    padding: 0 12px;
    border: 1px solid var(--idx-border);
    border-radius: 999px;
    background: var(--idx-surface-soft);
    color: var(--idx-text) !important;
    font-size: 13px;
    font-weight: 650;
    text-decoration: none !important;
}
.task-center-link:hover {
    border-color: var(--idx-accent);
    color: var(--idx-accent) !important;
}
footer {
    display: none !important;
}
@media (max-width: 1120px) {
    .gradio-container {
        width: min(980px, calc(100vw - 28px)) !important;
    }
    .workspace-grid {
        grid-template-columns: minmax(0, 1.35fr) minmax(250px, 0.85fr);
        grid-template-areas:
            "reference result"
            "content result";
    }
}
@media (max-width: 760px) {
    .gradio-container {
        width: calc(100vw - 24px) !important;
        padding: 12px 0 32px !important;
    }
    .hero-shell {
        padding: 20px 18px;
        border-radius: 16px;
    }
    .hero-shell h1 {
        font-size: clamp(26px, 9vw, 34px);
    }
    .workspace-grid {
        grid-template-columns: minmax(0, 1fr);
        grid-template-areas:
            "reference"
            "content"
            "result";
        gap: 14px !important;
    }
    .panel-card,
    .output-card {
        width: 100% !important;
        padding: 16px !important;
        border-radius: 16px !important;
    }
    .action-row {
        flex-direction: column !important;
    }
    .action-row > * {
        width: 100% !important;
        min-width: 0 !important;
    }
    .output-header {
        align-items: flex-start;
    }
}
"""


with gr.Blocks(title="IndexTTS 中文语音生成", css=APP_CSS) as demo:
    gr.HTML(
        """
        <section class="hero-shell">
            <div class="hero-kicker">本地运行 · 隐私优先</div>
            <h1>IndexTTS 中文语音生成</h1>
            <p>选择一段清晰的参考音频，输入需要补录的文本。页面会先检查输入，再进入本地生成队列。</p>
        </section>
        """
    )

    with gr.Row(equal_height=False, elem_classes="workspace-grid"):
        with gr.Column(scale=1, min_width=0, elem_classes=["panel-card", "reference-panel"]):
            gr.Markdown("### 1. 选择参考音频")
            prompt_audio = gr.Audio(
                label="音色参考音频",
                key="prompt_audio",
                sources=["upload", "microphone"],
                type="filepath",
                editable=False,
                show_download_button=False,
            )
            with gr.Row(elem_classes="reference-actions"):
                prepare_reference_button = gr.Button(
                    "整理参考音频",
                    variant="secondary",
                    size="sm",
                )
            reference_quality_note = gr.Markdown(
                "上传后会自动检查时长、音量、削波和静音比例。",
                elem_classes=["section-note", "reference-quality-note"],
            )
            gr.Markdown(
                "“整理参考音频”会在后端去除首尾静音并另存 WAV，原文件不会被覆盖。",
                elem_classes="section-note",
            )

        with gr.Column(scale=1, min_width=0, elem_classes=["panel-card", "content-panel"]):
            gr.Markdown("### 2. 填写生成内容")
            project_name = gr.Textbox(
                label=i18n("项目名称（可选）"),
                value=i18n("默认项目"),
                placeholder=i18n("例如：视频 A 开场补录"),
            )
            input_text_single = gr.TextArea(
                label="需要生成的文本",
                key="input_text_single",
                placeholder=i18n("请输入目标文本"),
                info=f"{i18n('当前模型版本')}{tts.model_version or '1.0'}",
                lines=6,
                max_lines=12,
            )
            readiness_note = gr.Markdown(
                "请先上传参考音频，并填写需要生成的文本。",
                elem_classes="status-copy",
            )
            with gr.Row(elem_classes="action-row"):
                clear_button = gr.ClearButton(
                    None,
                    value="清空本次输入",
                    variant="secondary",
                    size="lg",
                    elem_id="clear-button",
                )
                gen_button = gr.Button(
                    "请完善输入",
                    key="gen_button",
                    interactive=False,
                    variant="primary",
                    size="lg",
                    elem_id="generate-button",
                )

        with gr.Column(scale=1, min_width=0, elem_classes=["output-card", "result-panel"]):
            gr.HTML(
                f"""
                <div class="output-header">
                    <h3>3. 生成结果</h3>
                    <a class="task-center-link" href="http://127.0.0.1:{TASK_CENTER_PORT}/" onclick="this.href='http://'+window.location.hostname+':{TASK_CENTER_PORT}/'" target="_blank" rel="noopener noreferrer">任务中心</a>
                </div>
                """
            )
            generation_status = gr.Markdown(
                "等待生成",
                elem_classes=["status-copy", "generation-status"],
            )
            with gr.Row():
                output_audio = gr.Audio(
                    label="候选 1 · 原始干声",
                    visible=False,
                    key="output_audio",
                    editable=False,
                    interactive=False,
                    show_download_button=True,
                )
                output_audio_2 = gr.Audio(
                    label="候选 1 · 安全响度匹配",
                    visible=False,
                    key="output_audio_2",
                    editable=False,
                    interactive=False,
                    show_download_button=True,
                )
            with gr.Row():
                output_audio_3 = gr.Audio(
                    label="候选 2 · 原始干声",
                    visible=False,
                    key="output_audio_3",
                    editable=False,
                    interactive=False,
                    show_download_button=True,
                )
                output_audio_4 = gr.Audio(
                    label="候选 2 · 安全响度匹配",
                    visible=False,
                    key="output_audio_4",
                    editable=False,
                    interactive=False,
                    show_download_button=True,
                )
            with gr.Row():
                output_audio_5 = gr.Audio(
                    label="候选 3 · 原始干声",
                    visible=False,
                    key="output_audio_5",
                    editable=False,
                    interactive=False,
                    show_download_button=True,
                )
                output_audio_6 = gr.Audio(
                    label="候选 3 · 安全响度匹配",
                    visible=False,
                    key="output_audio_6",
                    editable=False,
                    interactive=False,
                    show_download_button=True,
                )

    with gr.Accordion("声音与情感", open=False):
        gr.Markdown("默认沿用音色参考音频的情感；只有需要精细控制时再展开设置。", elem_classes="section-note")
        emo_control_method = gr.Radio(
            choices=EMO_CHOICES,
            type="index",
            value=EMO_CHOICES[0],
            label=i18n("情感控制方式"),
        )
        with gr.Group(visible=False) as emotion_reference_group:
            emo_upload = gr.Audio(
                label=i18n("上传情感参考音频"),
                type="filepath",
                sources=["upload", "microphone"],
                editable=False,
                show_download_button=False,
            )
            emo_weight = gr.Slider(
                label=i18n("情感权重"),
                minimum=0.0,
                maximum=1.6,
                value=0.8,
                step=0.01,
            )

        emo_random = gr.Checkbox(
            label=i18n("情感随机采样"),
            value=False,
            visible=False,
        )

        with gr.Group(visible=False) as emotion_vector_group:
            with gr.Row():
                with gr.Column():
                    vec1 = gr.Slider(label=i18n("喜"), minimum=0.0, maximum=1.4, value=0.0, step=0.05)
                    vec2 = gr.Slider(label=i18n("怒"), minimum=0.0, maximum=1.4, value=0.0, step=0.05)
                    vec3 = gr.Slider(label=i18n("哀"), minimum=0.0, maximum=1.4, value=0.0, step=0.05)
                    vec4 = gr.Slider(label=i18n("惧"), minimum=0.0, maximum=1.4, value=0.0, step=0.05)
                with gr.Column():
                    vec5 = gr.Slider(label=i18n("厌恶"), minimum=0.0, maximum=1.4, value=0.0, step=0.05)
                    vec6 = gr.Slider(label=i18n("低落"), minimum=0.0, maximum=1.4, value=0.0, step=0.05)
                    vec7 = gr.Slider(label=i18n("惊喜"), minimum=0.0, maximum=1.4, value=0.0, step=0.05)
                    vec8 = gr.Slider(label=i18n("平静"), minimum=0.0, maximum=1.4, value=0.0, step=0.05)

        with gr.Group(visible=False) as emo_text_group:
            emo_text = gr.Textbox(
                label=i18n("情感描述文本"),
                placeholder=i18n("请输入情绪描述（或留空以自动使用目标文本作为情绪描述）"),
                value="",
                info=i18n("例如：高兴，愤怒，悲伤等"),
            )

    with gr.Accordion(i18n("高级生成参数设置"), open=False):
        gr.Markdown("不熟悉这些参数时建议保持默认值。", elem_classes="section-note")
        generation_style = gr.Radio(
            choices=list(GENERATION_STYLE_CHOICES),
            value=GENERATION_STYLE_BALANCED,
            label="生成风格预设",
            info="“表现力增强”会增加语气和节奏变化，也可能让连接词、数字或型号被额外强调；默认平衡模式通常更自然稳定，应用预设后仍可手动调整参数",
        )
        with gr.Row(equal_height=False):
            with gr.Column(scale=1, min_width=340):
                gr.Markdown("#### 采样参数")
                with gr.Row():
                    do_sample = gr.Checkbox(label="启用采样", value=True, info="关闭后生成结果会更加固定")
                    temperature = gr.Slider(label="生成温度", minimum=0.1, maximum=2.0, value=0.8, step=0.1, info="数值越高，声音变化越明显")
                with gr.Row():
                    top_p = gr.Slider(label="核心采样概率", minimum=0.0, maximum=1.0, value=0.8, step=0.01)
                    top_k = gr.Slider(label="候选词数量", minimum=0, maximum=100, value=30, step=1)
                    num_beams = gr.Slider(label="束搜索数量", value=3, minimum=1, maximum=10, step=1)
                with gr.Row():
                    repetition_penalty = gr.Number(label="重复惩罚", precision=None, value=10.0, minimum=0.1, maximum=20.0, step=0.1)
                    length_penalty = gr.Number(label="长度惩罚", precision=None, value=0.0, minimum=-2.0, maximum=2.0, step=0.1)
                max_mel_tokens = gr.Slider(
                    label="最大音频 Token 数",
                    value=1500,
                    minimum=50,
                    maximum=tts.cfg.gpt.max_mel_tokens,
                    step=10,
                    info="数值过小可能导致音频被提前截断",
                    key="max_mel_tokens",
                )
                create_loudness_match = gr.Checkbox(
                    label="同时生成安全响度匹配版（A/B）",
                    value=True,
                    info="保留原始干声，并按参考音频有效人声响度做一次线性匹配；最多提高 6 dB 或降低 12 dB，峰值保护约 -1 dBFS",
                )
                match_reference_format = gr.Checkbox(
                    label="匹配参考音频采样率和单/双声道",
                    value=True,
                    info="只处理 A/B 的第二份交付版；使用高质量重采样并匹配单/双声道，不会凭空增加模型高频细节",
                )
                with gr.Row():
                    seed_value = gr.Number(
                        label="随机种子",
                        value=0,
                        precision=0,
                        info="相同种子可复现；填写 -1 会自动生成新种子",
                    )
                    candidate_count = gr.Slider(
                        label="候选数量",
                        minimum=1,
                        maximum=3,
                        value=1,
                        step=1,
                        info="增加候选更容易选到自然版本，也会相应增加生成时间",
                    )

            with gr.Column(scale=1, min_width=340):
                gr.Markdown("#### 分句设置")
                initial_value = max(20, min(tts.cfg.gpt.max_text_tokens, cmd_args.gui_seg_tokens))
                max_text_tokens_per_segment = gr.Slider(
                    label=i18n("分句最大Token数"),
                    value=initial_value,
                    minimum=20,
                    maximum=tts.cfg.gpt.max_text_tokens,
                    step=2,
                    key="max_text_tokens_per_segment",
                    info=i18n("建议80~200之间，值越大，分句越长；值越小，分句越碎；过小过大都可能导致音频质量不高"),
                )
                segment_note = gr.Markdown(
                    "输入文本后会在这里预览模型分段。",
                    elem_classes="status-copy",
                )
                with gr.Accordion(i18n("预览分句结果"), open=False):
                    segments_preview = gr.Dataframe(
                        value=empty_segments_frame(),
                        headers=[i18n("序号"), i18n("分句内容"), i18n("Token数")],
                        key="segments_preview",
                        wrap=True,
                        interactive=False,
                    )

    advanced_params = [
        do_sample,
        top_p,
        top_k,
        temperature,
        length_penalty,
        num_beams,
        repetition_penalty,
        max_mel_tokens,
        create_loudness_match,
        match_reference_format,
    ]

    generation_style.change(
        fn=apply_generation_style,
        inputs=[generation_style],
        outputs=[top_p, top_k, temperature, num_beams],
        show_progress="hidden",
    )

    clear_button.add([
        prompt_audio,
        input_text_single,
        emo_upload,
        emo_text,
        segments_preview,
    ])
    busy_state = gr.State(False)

    clear_button.click(
        reset_ui_state,
        inputs=None,
        outputs=[
            gen_button,
            readiness_note,
            generation_status,
            segment_note,
            output_audio,
            output_audio_2,
            output_audio_3,
            output_audio_4,
            output_audio_5,
            output_audio_6,
            busy_state,
            reference_quality_note,
        ],
        queue=False,
        show_progress="hidden",
    )

    prompt_audio.change(
        update_reference_state,
        inputs=[prompt_audio, input_text_single, busy_state],
        outputs=[gen_button, readiness_note, reference_quality_note],
        queue=False,
        trigger_mode="always_last",
        show_progress="hidden",
    )
    prepare_reference_button.click(
        prepare_prompt_audio,
        inputs=[prompt_audio],
        outputs=[prompt_audio, reference_quality_note],
        queue=False,
        show_progress="full",
    )
    input_text_single.input(
        update_submit_state,
        inputs=[prompt_audio, input_text_single, busy_state],
        outputs=[gen_button, readiness_note],
        queue=False,
        trigger_mode="always_last",
        show_progress="hidden",
    )
    input_text_single.change(
        on_input_text_change,
        inputs=[input_text_single, max_text_tokens_per_segment],
        outputs=[segments_preview, segment_note],
        queue=False,
        trigger_mode="always_last",
        show_progress="hidden",
    )
    max_text_tokens_per_segment.change(
        on_input_text_change,
        inputs=[input_text_single, max_text_tokens_per_segment],
        outputs=[segments_preview, segment_note],
        queue=False,
        trigger_mode="always_last",
        show_progress="hidden",
    )
    emo_control_method.select(
        on_method_select,
        inputs=[emo_control_method],
        outputs=[emotion_reference_group, emo_random, emotion_vector_group, emo_text_group],
        queue=False,
        show_progress="hidden",
    )

    job_id_state = gr.State("")
    generation_request_state = gr.State({})
    clear_button.add([job_id_state, generation_request_state])
    register_event = gen_button.click(
        register_job,
        inputs=[
            project_name,
            input_text_single,
            prompt_audio,
            emo_control_method,
            emo_upload,
            emo_weight,
            vec1,
            vec2,
            vec3,
            vec4,
            vec5,
            vec6,
            vec7,
            vec8,
            emo_text,
            emo_random,
            max_text_tokens_per_segment,
            seed_value,
            candidate_count,
            *advanced_params,
        ],
        outputs=[
            job_id_state,
            generation_request_state,
            gen_button,
            clear_button,
            generation_status,
            busy_state,
        ],
        queue=False,
        trigger_mode="once",
        show_progress="hidden",
    )
    generation_event = register_event.success(
        gen_single,
        inputs=[job_id_state, generation_request_state],
        outputs=[
            output_audio,
            output_audio_2,
            output_audio_3,
            output_audio_4,
            output_audio_5,
            output_audio_6,
            generation_status,
        ],
        concurrency_limit=1,
        concurrency_id="tts-generation",
        trigger_mode="once",
        show_progress="full",
        # Bind progress to the always-visible status surface. The audio player is
        # hidden before the first result, so targeting it can hide the overlay.
        show_progress_on=[generation_status],
    )
    generation_event.failure(
        mark_generation_failed,
        inputs=None,
        outputs=[generation_status],
        queue=False,
        show_progress="hidden",
    )
    generation_event.then(
        restore_action_buttons,
        inputs=[prompt_audio, input_text_single],
        outputs=[gen_button, clear_button, busy_state],
        queue=False,
        show_progress="hidden",
    )


if __name__ == "__main__":
    demo.queue(max_size=20, default_concurrency_limit=1)
    demo.launch(server_name=cmd_args.host, server_port=cmd_args.port)
