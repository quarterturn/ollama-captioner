#!/usr/bin/env python3
"""Danbooru 1024 captioning pipeline with character grounding.

Uses structured JSON prompt from ollama-captioner project with character
visual context injection from char_profiles wiki data.

Architecture:
- Phase ENRICH: Build idx_enrich table (char_match + image_category) [DONE]
- Phase CAPTION: Batch-process images through vision model via OpenRouter

Resumable: tracks progress in captions table, safe to restart anytime.

Usage:
  python3 caption_pipeline.py --limit 5               # Test-caption 5 images
  python3 caption_pipeline.py --limit 50 --concurrency 3  # Batch with concurrency
  python3 caption_pipeline.py --status                # Check progress
"""
import argparse
import base64
import concurrent.futures
import json
import os
import re
import sqlite3
import sys
import threading
import time

PROJECT_DIR = "/home/alex/Documents/datasets/danbooru-2026-clean"
DB_PATH = os.path.join(PROJECT_DIR, "highscored_index.db")
PROMPT_PATH = os.path.join(PROJECT_DIR, "caption_prompt.txt")
CAPTIONS_DIR = os.path.join(PROJECT_DIR, "captions")

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
MODEL = "minimax/minimax-m3"  # Cheapest vision model, complete adult content, no truncation
BASE_RATE_DELAY = 0.5  # Cheapest vision model, complete adult content, no truncation  # seconds between requests per worker (permissive)


def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


# ─── Load structured prompt from file ───

with open(PROMPT_PATH, "r") as _f:
    PROMPT_TEMPLATE = _f.read()


def load_char_context(char_name):
    """Get visual tags for a matched character from DB."""
    if not char_name:
        return ""
    try:
        c = sqlite3.connect(DB_PATH)
        row = c.execute(
            "SELECT visual_tags FROM char_profiles WHERE char_name=?", (char_name,)
        ).fetchone()
        c.close()
        if not row:
            # Character matched but no wiki entry — still pass the name
            return f"Known character: {char_name}"
        tags = json.loads(row[0] or "[]")
        if not tags:
            # Has wiki entry but no visual tags — still pass the name
            return f"Known character: {char_name}"
        chars_str = ", ".join(tags[:8])
        return f"Known character [{char_name}] with visual characteristics: {chars_str}"
    except Exception:
        return f"Known character: {char_name}"


# ─── OpenRouter API ───

def call_openrouter(messages, model=None):
    """Call OpenRouter API. Returns (caption_text_or_json_str, usage_dict_or_error)."""
    api_key = OPENROUTER_API_KEY or os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        return None, {"error": "OPENROUTER_API_KEY not found"}

    import urllib.request as ur

    payload = {
        "model": model or MODEL,
        "messages": messages,
        "max_tokens": 4096,
        "temperature": 0.3,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://hermes-agent.nousresearch.com",
        "X-Title": "danbooru-captioner",
    }

    try:
        req = ur.Request(
            "https://openrouter.ai/api/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        with ur.urlopen(req, timeout=120) as resp:
            result = json.loads(resp.read().decode("utf-8"))

            if "error" in result:
                err_msg = result["error"].get("message", str(result["error"]))
                return None, {"error": err_msg}

            output = result["choices"][0]["message"]["content"]
            usage = result.get("usage", {})
            return output.strip(), {
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "model": result.get("model", model or MODEL),
            }

    except Exception as e:
        if hasattr(e, "read"):
            body = e.read().decode("utf-8", errors="replace")
            return None, {"error": f"HTTP {e.code}: {body[:200]}"}
        return None, {"error": str(e)}


# ─── Caption a single image ───

def caption_one_job(args):
    """Worker function for concurrent execution. Returns (filename, caption_or_json, result_dict, elapsed_seconds)."""
    img_path, char_name = args["img_path"], args.get("char_name", "")
    fn = args["fn"]

    start = time.time()

    # Build the prompt with character context injected
    char_ctx = load_char_context(char_name)
    if char_ctx:
        prompt_text = PROMPT_TEMPLATE.replace("{char_context}", char_ctx)
    else:
        prompt_text = PROMPT_TEMPLATE.replace("CHARACTER_CONTEXT: {char_context}\n\n", "")

    # Read image as base64
    with open(img_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode("utf-8")

    mime = "image/webp" if img_path.endswith(".webp") else "image/jpeg"

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt_text},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime};base64,{img_b64}"},
                },
            ],
        }
    ]

    output, result = call_openrouter(messages)
    elapsed = time.time() - start

    # Try to parse JSON from output — strip markdown fences if present
    parsed_json = None
    if output:
        cleaned = output.strip()
        for marker in ["```json", "```"]:
            if cleaned.startswith(marker):
                cleaned = cleaned[len(marker):].strip()
                if cleaned.endswith("```"):
                    cleaned = cleaned[:-3].strip()
        try:
            parsed_json = json.loads(cleaned)
            output = json.dumps(parsed_json, indent=2)
        except json.JSONDecodeError:
            pass

    return fn, output, result, elapsed


def log_result(log_file, log_lock, fn, elapsed, result):
    """Append one line to the results log file. Thread-safe."""
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    if isinstance(result, dict) and "error" in result:
        err = result["error"][:120]
        line = f"{now}  {fn}  FAIL  {elapsed:.1f}s  error=\"{err}\"  tokens=---\n"
    elif isinstance(result, dict):
        pt = result.get("prompt_tokens", 0)
        ct = result.get("completion_tokens", 0)
        line = f"{now}  {fn}  OK  {elapsed:.1f}s  p={pt} c={ct}\n"
    else:
        line = f"{now}  {fn}  FAIL  {elapsed:.1f}s  no result\n"
    
    with log_lock:
        log_file.write(line)
        log_file.flush()


# ─── Caption phase with concurrency ───

def run_caption(limit=10, concurrency=1):
    """Caption images via vision model on OpenRouter."""

    os.makedirs(CAPTIONS_DIR, exist_ok=True)
    conn = get_conn()

    # Ensure captions table exists
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS captions (
            filename TEXT PRIMARY KEY,
            caption TEXT NOT NULL DEFAULT '',
            rating TEXT DEFAULT '',
            char_name TEXT DEFAULT '',
            prompt_tokens INTEGER DEFAULT 0,
            completion_tokens INTEGER DEFAULT 0,
            model TEXT DEFAULT '',
            status TEXT DEFAULT 'complete',
            error_msg TEXT DEFAULT '',
            created_at REAL DEFAULT (strftime('%s','now'))
        )
    """
    )
    conn.commit()

    done_count = (
        conn.execute("SELECT COUNT(*) FROM captions WHERE status='complete'").fetchone()[0]
    )
    total_images = conn.execute("SELECT COUNT(*) FROM idx").fetchone()[0]
    print(f"Already captioned: {done_count:,}/{total_images:,}")

    # Get uncaptioned images — ONLY explicit + questionable ratings — using is_captioned flag for fast lookup
    query = """
        SELECT idx.filename, idx.path, idx.subject_tags, idx.score, idx.fav_count,
               COALESCE(idx_enrich.char_match, ''),
               COALESCE(idx_enrich.image_category, 'regular')
        FROM idx
        LEFT JOIN idx_enrich ON idx.filename = idx_enrich.filename
        WHERE rating IN ('e', 'q')
          AND is_captioned = 0
        ORDER BY CASE WHEN char_match != '' THEN 0 ELSE 1 END, idx.score DESC
        LIMIT ? OFFSET 0
    """

    rows = list(conn.execute(query, (limit,)))
    if not rows:
        print("No uncaptioned images found.")
        conn.close()
        return

    # Pre-resolve image paths and filter missing
    jobs = []
    for row in rows:
        fn, path, tags_str, score, favs, char_name, category = row
        img_path = os.path.join(PROJECT_DIR, path)
        if not os.path.isabs(path):
            # Path stored as 'images-highscored/batch_NNNN/file.ext'
            full = os.path.join(PROJECT_DIR, path)
            if os.path.exists(full):
                img_path = full
            else:
                img_path = None

        if img_path and os.path.exists(img_path):
            jobs.append({
                "fn": fn,
                "img_path": img_path,
                "char_name": char_name or "",
                "tags_str": tags_str,
                "score": score,
                "favs": favs,
                "category": category,
            })

    print(f"\n{'=' * 50}")
    print(f"CAPPING {len(jobs)} images with {MODEL}")
    print(f"Concurrency: {concurrency} workers")
    print(f"Prompt: structured JSON (ollama-captioner style) + character context")
    log_file_path = os.path.join(PROJECT_DIR, "caption_log.txt")
    print(f"Log file: {log_file_path}")
    print(f"{'=' * 50}\n")

    stats = {"ok": 0, "fail": 0, "skip": len(rows) - len(jobs)}
    tok_stats = {"prompt": 0, "completion": 0}
    t0 = time.time()

    # Open log file (append mode for resumability)
    log_file = open(log_file_path, "a")
    log_lock = threading.Lock()

    # Use shared lock for DB writes from concurrent workers
    db_lock = threading.Lock() if concurrency > 1 else None

    def worker_wrapper(job):
        """Run caption job and save result with thread-safe DB access."""
        fn, output, result = caption_one_job(job)
        return fn, output, result, job.get("tags_str", ""), job.get("score", 0), job.get("favs", 0), job.get("char_name", ""), job.get("category", "regular")

    import threading as _th

    # Stream results as they return + save incrementally for crash safety
    import threading as _th

    db_lock = _th.Lock() if concurrency > 1 else None
    batch_tokens_p, batch_tokens_c = 0, 0
    batch_ok, batch_fail = 0, 0
    saved_count = 0  # global save counter for progress/commits

    if concurrency > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
            future_to_job = {executor.submit(caption_one_job, j): j for j in jobs}
            for future in concurrent.futures.as_completed(future_to_job):
                job_info = future_to_job[future]
                fn_out, output, result, elapsed = future.result()

                # Log immediately
                log_result(log_file, log_lock, fn_out, elapsed, result)

                with db_lock:
                    save_result(db_lock, conn, fn_out, output, result,
                        job_info.get("tags_str", ""),
                        job_info.get("score", 0),
                        job_info.get("favs", 0),
                        job_info.get("char_name", ""),
                        job_info.get("category", "regular"),
                    )
                    if isinstance(result, dict) and "error" not in result:
                        batch_tokens_p += result.get("prompt_tokens", 0)
                        batch_tokens_c += result.get("completion_tokens", 0)
                        batch_ok += 1
                    else:
                        batch_fail += 1
                    saved_count += 1

                print_and_accumulate(fn_out, output, result, job_info.get("tags_str"), job_info.get("score", 0), job_info.get("favs", 0), job_info.get("char_name", ""))

                # Commit every 50 + progress report
                if saved_count % 50 == 0:
                    conn.commit()
                    cost_so_far = (batch_tokens_p / 1e6 * 0.30) + (batch_tokens_c / 1e6 * 1.20)
                    elapsed_now = time.time() - t0
                    rate = saved_count / max(elapsed_now, 0.001)
                    remaining_jobs = len(jobs) - saved_count
                    eta_sec = remaining_jobs / max(rate, 0.001)
                    print(f"\n--- PROGRESS: {saved_count:,}/{len(jobs):,} ({saved_count/len(jobs)*100:.1f}%) | OK={batch_ok} FAIL={batch_fail} | Cost=${cost_so_far:.2f} | ETA={eta_sec/60:.0f}min ---\n")

        conn.commit()
    else:
        for job_info in jobs:
            fn_out, output, result, elapsed = caption_one_job(job_info)

            # Log immediately
            log_result(log_file, log_lock, fn_out, elapsed, result)

            save_result(db_lock, conn, fn_out, output, result,
                job_info.get("tags_str", ""),
                job_info.get("score", 0),
                job_info.get("favs", 0),
                job_info.get("char_name", ""),
                job_info.get("category", "regular"),
            )
            if isinstance(result, dict) and "error" not in result:
                batch_tokens_p += result.get("prompt_tokens", 0)
                batch_tokens_c += result.get("completion_tokens", 0)
                batch_ok += 1
            else:
                batch_fail += 1
            saved_count += 1

            print_and_accumulate(fn_out, output, result, job_info.get("tags_str"), job_info.get("score", 0), job_info.get("favs", 0), job_info.get("char_name", ""))

            if saved_count % 50 == 0:
                conn.commit()
                cost_so_far = (batch_tokens_p / 1e6 * 0.30) + (batch_tokens_c / 1e6 * 1.20)
                elapsed_now = time.time() - t0
                rate = saved_count / max(elapsed_now, 0.001)
                remaining_jobs = len(jobs) - saved_count
                eta_sec = remaining_jobs / max(rate, 0.001)
                print(f"\n--- PROGRESS: {saved_count:,}/{len(jobs):,} ({saved_count/len(jobs)*100:.1f}%) | OK={batch_ok} FAIL={batch_fail} | Cost=${cost_so_far:.2f} | ETA={eta_sec/60:.0f}min ---\n")

        conn.commit()

    elapsed = time.time() - t0
    log_file.close()

    # Summary from DB
    done_final = conn.execute("SELECT COUNT(*) FROM captions WHERE status='complete'").fetchone()[0]
    failed_total = conn.execute("SELECT COUNT(*) FROM captions WHERE status='failed'").fetchone()[0]
    row = conn.execute(
        "SELECT SUM(prompt_tokens), SUM(completion_tokens) FROM captions WHERE status='complete'"
    ).fetchone()
    conn.close()

    print(f"\n{'=' * 50}")
    print(f"BATCH DONE")
    print(f"  Newly captioned: {batch_ok}, failed: {batch_fail}")
    print(f"  Tokens this batch: {batch_tokens_p:,}p + {batch_tokens_c:,}c = {batch_tokens_p + batch_tokens_c:,}")
    all_pt, all_ct = row[0] or 0, row[1] or 0
    print(f"  Total in DB: {all_pt:,}p + {all_ct:,}c = {all_pt + all_ct:,}")
    print(f"  Time: {elapsed:.0f}s ({elapsed/max(len(jobs),1):.1f}s/img avg)")

    if elapsed > 0 and len(jobs) > 0:
        remaining = total_images - done_final
        eta_hours = (remaining * elapsed / len(jobs)) / 3600
        print(f"  Progress: {done_final:,}/{total_images:,} ({done_final/max(total_images,1)*100:.1f}%)")
        print(f"  ETA remaining {remaining:,}: ~{eta_hours:.1f}h (~{eta_hours/24:.1f} days)")

        # Cost estimate at current rate (~$0.10/1M prompt + $0.45/1M completion for qwen3-vl-32b)
        total_tokens_per_img = batch_tokens_p / max(batch_ok, 1) + batch_tokens_c / max(batch_ok, 1)
        est_cost_remaining = (remaining * total_tokens_per_img / 1_000_000) * 0.35  # weighted avg price
        print(f"  Est. cost remaining: ~${est_cost_remaining:.2f} at current token rate")


def try_parse_caption_json(text):
    """Parse model output JSON. Handles extra trailing text after closing brace."""
    if not text:
        return None
    clean = re.sub(r'[\u200b\ufeff]', '', text.strip())
    for marker in ["```json", "```"]:
        if clean.startswith(marker):
            clean = clean[len(marker):]
            if clean.endswith("```"):
                clean = clean[:-3].strip()
    # Try direct parse first
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        pass
    # Extra data after JSON? Find the first balanced closing brace.
    start = clean.find('{')
    if start < 0:
        return None
    brace_depth = 0
    in_string = False
    escape_next = False
    for i in range(start, len(clean)):
        ch = clean[i]
        if escape_next:
            escape_next = False
            continue
        if ch == '\\':
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
        elif not in_string:
            if ch == '{':
                brace_depth += 1
            elif ch == '}':
                brace_depth -= 1
                if brace_depth == 0:
                    try:
                        return json.loads(clean[start:i+1])
                    except json.JSONDecodeError:
                        return None
    return None


def save_result(lock, conn, fn, output, result, tags_str, score, favs, char_name, category):
    """Save caption to DB and write .json file beside the image on disk."""
    if isinstance(result, dict) and "error" in result:
        status = "failed"
        error_msg = result["error"][:200]
        caption_text = ""
        pt = ct = model_used = 0
        rating_val = ""
    else:
        # Parse the model output to get clean JSON
        cap_json = try_parse_caption_json(output)
        if cap_json is None:
            status = "failed"
            error_msg = "Model returned non-parseable JSON"
            caption_text = (output or "")[:200]
            pt = ct = 0
            model_used = result.get("model", MODEL) if isinstance(result, dict) else MODEL
            rating_val = ""
        else:
            status = "complete"
            error_msg = ""
            caption_text = json.dumps(cap_json, ensure_ascii=False, indent=2)
            pt = result.get("prompt_tokens", 0) if isinstance(result, dict) else 0
            ct = result.get("completion_tokens", 0) if isinstance(result, dict) else 0
            model_used = result.get("model", MODEL) if isinstance(result, dict) else MODEL
            rating_val = cap_json.get("rating", "")

    # Write caption to DB
    conn.execute(
        "INSERT OR REPLACE INTO captions (filename, caption, rating, char_name, "
        "prompt_tokens, completion_tokens, model, status, error_msg, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, strftime('%s','now'))",
        (fn, caption_text, rating_val, char_name or "", pt, ct, model_used, status, error_msg),
    )

    # Update is_captioned flag on idx
    if status == "complete":
        conn.execute("UPDATE idx SET is_captioned=1 WHERE filename=?", (fn,))

    # Write .json file next to the original image on disk
    # Find which image path maps to this filename
    img_row = conn.execute(
        "SELECT path FROM idx WHERE filename=? LIMIT 1", (fn,)
    ).fetchone()
    if img_row:
        base = os.path.splitext(os.path.join(PROJECT_DIR, img_row[0]))[0]
        json_path = base + ".json"
        with open(json_path, "w") as f:
            json.dump(cap_json if isinstance(cap_json, dict) else {}, f, indent=2, ensure_ascii=False)


def print_and_accumulate(fn_out, output, result, tags_str, score, favs, char_name):
    """Print status line for one image."""
    if isinstance(result, dict) and "error" in result:
        err = result["error"][:80]
        print(f"  [FAIL] {fn_out} s={score}: {err}")
    else:
        pt = result.get("prompt_tokens", 0) if isinstance(result, dict) else 0
        ct = result.get("completion_tokens", 0) if isinstance(result, dict) else 0
        preview = (output or "")[:100].replace("\n", " ")
        cinfo = f"[{char_name}] " if char_name else ""
        print(f"  [OK] {fn_out} {cinfo}s={score} tokens={pt}/{ct}")
        print(f"       {preview}...")


# ─── Status ───

def run_status():
    conn = get_conn()
    total = conn.execute("SELECT COUNT(*) FROM idx").fetchone()[0]
    captioned = conn.execute("SELECT COUNT(*) FROM idx WHERE is_captioned=1").fetchone()[0]
    failed = conn.execute("SELECT COUNT(*) FROM captions WHERE status='failed'").fetchone()[0] or 0

    row = conn.execute(
        "SELECT SUM(prompt_tokens), SUM(completion_tokens) FROM captions WHERE status='complete'"
    ).fetchone()
    pt, ct = row[0] or 0, row[1] or 0

    matched = conn.execute("SELECT COUNT(*) FROM idx_enrich WHERE char_match != ''").fetchone()[0] or 0

    # Rating breakdown for uncaptioned e+q
    eq_remaining = conn.execute(
        "SELECT COUNT(*) FROM idx WHERE rating IN ('e','q') AND is_captioned=0"
    ).fetchone()[0]

    print(f"\n{'=' * 45}")
    print(f"DANBOORU CAPTIONING STATUS")
    print(f"{'=' * 45}")
    print(f"  Total images in idx:     {total:,}")
    print(f"  Captioned (is_captioned): {captioned:,} ({captioned/max(total,1)*100:.1f}%)")
    print(f"  Failed:                  {failed:,}")
    print(f"  e+q remaining:           {eq_remaining:,}")
    print(f"  Char matched:            {matched:,}/{total:,}")
    if pt > 0:
        cost = (pt / 1_000_000) * 0.30 + (ct / 1_000_000) * 1.20
        avg_p = pt / max(captioned, 1)
        avg_c = ct / max(captioned, 1)
        print(f"  Tokens spent:            {pt:,}p + {ct:,}c")
        print(f"  Avg per image:           {avg_p:.0f}p / {avg_c:.0f}c")
        print(f"  Cost so far:             ${cost:.2f}")
    conn.close()


# ─── Main ───

def main():
    parser = argparse.ArgumentParser(description="Danbooru 1024 Captioning Pipeline")
    parser.add_argument("--limit", type=int, default=10, help="Images to caption (default: 10)")
    parser.add_argument("--concurrency", type=int, default=3, help="Parallel API workers (default: 3)")
    parser.add_argument("--status", action="store_true", help="Show progress stats")
    parser.add_argument("--reset", action="store_true", help="Reset all captions and start fresh (clears captions table + is_captioned flags)")
    args = parser.parse_args()

    if args.reset:
        import sys
        print("WARNING: This will delete ALL existing captions and reset progress.")
        reply = input("Type 'YES' to confirm: ")
        if reply != "YES":
            print("Aborted.")
            sys.exit(0)
        conn = sqlite3.connect(DB_PATH)
        affected = conn.execute("SELECT COUNT(*) FROM captions WHERE status='complete'").fetchone()[0]
        conn.execute("DELETE FROM captions")
        conn.execute("UPDATE idx SET is_captioned=0")
        conn.commit()
        # Also remove orphan .json files in images-highscored dirs (optional cleanup)
        import glob, os
        removed = 0
        for jf in glob.glob(os.path.join(PROJECT_DIR, "images-highscored", "**", "*.json"), recursive=True):
            os.remove(jf)
            removed += 1
        conn.close()
        print(f"Reset complete: cleared {affected} captions + removed {removed} .json files. Ready for fresh run.")
        return

    if args.status:
        run_status()
    else:
        run_caption(limit=args.limit, concurrency=args.concurrency)


if __name__ == "__main__":
    main()
