# Performance and Cost Guide

Practical guidance for GPU worker performance and cost estimation.

## Transcription Speed Benchmarks

### Whisper large-v3 Performance

| GPU | Realtime Factor | 1hr Audio | Notes |
|-----|-----------------|-----------|-------|
| CPU (8-core) | 0.5-1x | 60-120 min | Baseline |
| RTX 3090 | 15-20x | 3-4 min | Good value |
| RTX 4090 | 20-25x | 2.5-3 min | Fast |
| A10 | 15-20x | 3-4 min | Cloud standard |
| A40 | 20-25x | 2.5-3 min | More VRAM |
| A100 40GB | 25-35x | 2-2.5 min | Fast |
| A100 80GB | 25-35x | 2-2.5 min | Long files |
| H100 | 35-50x | 1.5-2 min | Maximum |

**Realtime factor**: How many times faster than realtime. 20x means 1 hour of audio processes in 3 minutes.

### Diarization Overhead

Diarization adds significant processing time:

| Audio Length | Diarization Time (A100) | Notes |
|--------------|-------------------------|-------|
| 5 min | +30-60 sec | Minimal overhead |
| 30 min | +3-5 min | Noticeable |
| 1 hour | +8-15 min | Significant |
| 2 hours | +20-40 min | Consider chunking |

**Rule of thumb**: Diarization adds 15-30% to total processing time on GPU, 50-100% on CPU.

## Cost Estimation

### Example: 48 Files × 20 Minutes Each (960 min total audio)

| GPU | Hourly Rate | Processing Time | Total Cost |
|-----|-------------|-----------------|------------|
| CPU | $0 | ~16-32 hours | $0 (but slow) |
| RTX 3090 | $0.40/hr | ~50-65 min | $0.50-0.75 |
| A10 | $0.50/hr | ~50-65 min | $0.50-0.75 |
| A100 | $1.50/hr | ~30-40 min | $0.75-1.00 |

**With diarization enabled**, add 20-40% to processing time and cost.

### Cost Per Hour of Audio

| GPU | Cost per Audio Hour | With Diarization |
|-----|---------------------|------------------|
| RTX 3090 | ~$0.02-0.03 | ~$0.03-0.04 |
| A10 | ~$0.03-0.04 | ~$0.04-0.05 |
| A100 | ~$0.04-0.06 | ~$0.05-0.08 |

## When to Use GPU vs CPU

### Use CPU When:
- Processing < 30 minutes of audio total
- No diarization needed
- Cost is primary concern
- You have time (overnight batch)

### Use GPU When:
- Processing > 1 hour of audio
- Diarization is enabled
- Time is important
- Processing long individual files (> 30 min each)

## Memory Requirements

### Whisper Models

| Model | VRAM Required | Quality |
|-------|---------------|---------|
| tiny | ~1 GB | Basic |
| base | ~1 GB | Good |
| small | ~2 GB | Better |
| medium | ~5 GB | High |
| large-v3 | ~10 GB | Best |

### Diarization

- Base requirement: ~4 GB VRAM
- Per-chunk overhead: ~1-2 GB
- Recommended minimum: 16 GB VRAM for long files

### Total VRAM for Transcription + Diarization

| Scenario | Minimum VRAM |
|----------|--------------|
| Whisper large-v3 only | 12 GB |
| + Short file diarization | 16 GB |
| + Long file diarization | 24 GB |
| + Very long files (2hr+) | 40+ GB |

## Optimization Tips

### 1. Batch Your Jobs

Don't start/stop GPU instances for each file. Upload all files, then run.

```
Bad:  Start GPU → Process 1 file → Stop GPU (repeat 48 times)
Good: Start GPU → Process 48 files → Stop GPU
```

### 2. Use Appropriate GPU Size

- Testing: RTX 3090 or A10 (cheapest)
- Production: A10 or A40 (good balance)
- Long files with diarization: A100 80GB

### 3. Disable Diarization When Not Needed

If you don't need speaker labels, disable diarization to save 20-40% time/cost.

### 4. Use Spot/Preemptible Instances

RunPod Spot instances are 50-70% cheaper. Good for:
- Non-urgent batch jobs
- Jobs that can be restarted if interrupted

### 5. Monitor and Stop Idle Instances

GPU instances charge by the minute/hour even when idle. Stop them when done.

## Real-World Examples

### Legal Deposition (3 hours, needs diarization)

- GPU: A100 80GB
- Time: ~15-20 min
- Cost: ~$0.50-0.75

### Podcast Batch (20 episodes × 1 hour each)

- GPU: A10
- Time: ~60-80 min
- Cost: ~$0.50-0.70

### Meeting Recordings (100 × 30 min, no diarization)

- GPU: RTX 3090
- Time: ~2.5-3 hours
- Cost: ~$1.00-1.20

### Investigation Audio (500 hours, needs diarization)

- GPU: A100 80GB
- Time: ~20-25 hours
- Cost: ~$30-40
- Note: Consider running multiple workers in parallel

## Accuracy Notes

- Whisper large-v3 is the most accurate model
- GPU vs CPU does not affect accuracy (same model)
- Diarization accuracy depends on audio quality and speaker distinctiveness
- Word-level timestamps slightly reduce speed but improve review workflow

## Limitations

- Maximum single file: Limited by VRAM (typically 2-4 hours on 24GB)
- Auto-split handles longer files but may have speaker continuity issues at boundaries
- Very noisy audio may need preprocessing
- Multiple overlapping speakers reduce diarization accuracy
