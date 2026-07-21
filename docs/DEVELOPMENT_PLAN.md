# IndexTTS Local Dubbing Development Plan

Updated: 2026-07-21

## Goal

Build a stable local dubbing workflow that preserves original model output, keeps every generated version downloadable, and lets the user compare reproducible candidates before selecting a preferred result.

## Completed foundation

- Chinese-first responsive WebUI and improved trim/reference-audio handling.
- Persistent task history with grouped project versions, progress, cancellation, failure recovery, and downloads.
- Reproducible seed support and up to three independently saved candidates per task.
- Optional safe loudness matching and reference delivery-format matching without replacing the dry model output.
- Balanced generation remains the default; the freer sampling preset is labelled as an experimental expressive mode.
- A preferred output can be selected in the task center and remains selected after refresh.

## Recommended next sequence

### 1. Move multi-candidate controls into the primary workflow

- Move candidate count out of advanced settings and place it near the generate action.
- Provide two clear actions: generate one result quickly, or generate three candidates.
- Keep one candidate as the conservative default and preserve the existing serial inference path.

### 2. Add candidate cards to the generation page

- Show each candidate as a card with its player, seed, output variant, and download action.
- Allow the preferred result to be selected directly without opening the task center.
- Use the same persisted preferred-output state in both the generation page and task center.
- Restore all candidate cards and the preferred selection when a completed task is reopened.

### 3. Establish a real three-candidate serial baseline

With a user-approved reference recording and the default balanced preset, generate three seeds serially and record:

- total wall-clock time and per-candidate latency;
- available MPS allocated-memory and driver-memory peaks;
- observable CPU/GPU utilization;
- same-seed reproducibility;
- voice, emotion, progress, cancellation, recovery, selection, and download correctness;
- whether memory grows across repeated runs.

Do not change the global inference concurrency limit before this baseline exists.

### 4. Evaluate process-level inference concurrency

Do not run multiple inference threads against one shared `IndexTTS2` instance. It contains mutable progress, cache, random-number, and memory-cleanup state.

Evaluate concurrency in this order:

1. one inference process, three candidates serially;
2. two independent inference processes, each with its own model instance, seed state, progress state, and output directory;
3. only if two processes are stable and faster, optionally test three independent processes.

CPU post-processing such as loudness matching, resampling, and archive creation may use a small bounded thread pool independently of model inference.

### 5. Concurrency decision gate

Process concurrency may become an optional setting only when all of the following are true:

- repeated three-candidate runs improve total throughput over the serial baseline;
- same-seed behavior remains reproducible or any difference is understood;
- repeated runs do not show linear memory growth;
- task progress, cancellation, caches, and outputs never cross between jobs;
- the worker count is configurable and can immediately fall back to one worker;
- a failed worker cannot corrupt completed candidates or task history.

If these conditions are not met, keep one inference worker and optimize the serial path instead.

## Validation required before release

- Unit and HTTP tests for candidate restoration and preferred-output persistence.
- Browser validation for one-result and three-candidate interactions at narrow and wide layouts.
- Real inference validation only with explicit approval and non-repository test media.
- Privacy scan confirming that generated media, local paths, databases, logs, and credentials are not tracked.
- Production deployment and service restart remain separate, explicitly approved operations.
