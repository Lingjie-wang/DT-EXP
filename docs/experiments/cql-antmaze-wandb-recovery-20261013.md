# AntMaze live W&B history recovery

On 2026-10-13 the success-boundary campaign continued training locally, while
W&B marked both native runs and the comparison run as `crashed`. The cloud
histories stopped at updates 156742 and 149999. At diagnosis local updates were
341709 and 322342, with six completed evaluations per dataset through 300k.
Best observed uniform scores were 88 (Umaze, 300k) and 51 (Diverse, 250k).

The trainer processes, queue and comparison observer were alive. SDK logs
contained network timeouts. Fresh authenticated GraphQL queries and file-stream
heartbeats succeeded. W&B 0.12.21 sets a Session.timeout attribute but its
file-stream POST does not actually pass a requests timeout. This permits a
request to block despite the intended timeout. The recovery avoids relying on
that transport. No training restart or scientific configuration change is needed.

`scripts/cql_antmaze_success_wandb_recovery/relay.py` is an independent process:

- Reads complete protobuf records from the existing native `.wandb` files.
- Uses exactly the SDK's protobuf-to-dictionary and JSON history conversion.
- Replays history to the same run ID with original zero-based file line offsets.
- Retries partial tail records without skipping them; rejects interior corruption.
- Saves its byte cursor and acknowledged line count only after HTTP success.
  An uncertain request is retried at identical offsets, preventing duplicate rows.
- Uploads the native local summary and a heartbeat with explicit connect/read
  timeouts. These are real local records, not inferred or synthesized scores.
- Includes the existing comparison observer's history and table references,
  retaining the original comparison URL and all official control curves.
- Automatically discovers later queued jobs under their training directories;
  preflight runs are excluded.
- Preserves the native files, frozen plan, official CQL process, queue, configuration,
  rewards and checkpoints. It does not upgrade or monkey-patch the training SDK.

Recovery state is stored separately under the provided `--out` directory. File
identity and a process lock prevent duplicate relays or unnoticed source rotation.
Each run receives bounded batches so backfilling training cannot starve the
comparison. Histories may temporarily be behind the live summary during replay.
Only a terminal queue state plus a known exit code and a fully scanned source
can send a completion marker. The helper is a telemetry recovery process, not a
training supervisor or a replacement for experiment completion verification.

```bash
PROJECT=/home/sckd02/workspace/DT-EXP
ROOT="$PROJECT/results/cql-antmaze-success-rewards-seed21-1m-5090-20261013"
OUT="$PROJECT/results/cql-antmaze-success-wandb-recovery-20261013"
tmux new-session -d -s cql-antmaze-wandb-recovery-20261013 \
  "bash '$PROJECT/scripts/cql_antmaze_success_wandb_recovery/launch.sh' '$ROOT' '$OUT'"
```

Inspect `status.json` and `cursors.json` under `OUT`. Verify uploaded history samples
and comparison tables against native records after replay, and keep any remaining
cloud/local discrepancy explicit.

## Completion recovery on 2026-10-14

Both uniform jobs finished all 1M updates and 20 evaluations, but the old SDK
blocked in its atexit file-stream shutdown. Their best scores are 93 at 550k
(Umaze) and 62 at 850k (Diverse), versus original-reward controls 94 and 55.
The cloud comparison table contained all 80 evaluation rows before intervention.

Recovery evidence is saved separately under
`results/cql-antmaze-finish-recovery-20261014`: the original queue snapshot,
complete evaluations, checkpoint hashes, checkpoint `total_it == 1000000`,
finite actor/critic weights, native exit code 0, process identities and signals.
Two SIGINTs to each completed job's W&B backend stopped its writer gracefully,
flushing the buffered native tail. After verifying writer closure and replay of
all 1M history rows, SIGINT released the completed CLI's atexit wait, and the
remaining backend was terminated. No model training was interrupted or restarted.
The original queue records CLI exit 255 as `failed`; this is a telemetry shutdown
failure, while the independently verified training is complete. Both W&B runs
are `finished`, including the final evaluations. The unchanged queue then admitted
the Shapley jobs under its existing resource limits.

A partial buffered tail must not cause one-second idle heartbeats. The relay now
waits 30 seconds when a scan makes no progress, backs off on network errors, and
does not repost completed runs. Fast replay remains enabled for readable backlog.
This avoids consuming upload quota needed by newly started training runs.
