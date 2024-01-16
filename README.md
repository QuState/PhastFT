# PhastFT

## Profiling

On linux, open
access to performance monitoring and observability operations for processes

```bash
echo -1 | sudo tee /proc/sys/kernel/perf_event_paranoid
```

Add debug to `Cargo.toml` under `profile.release`:

```bash
[profile.release]
debug = true
```

Finally, run:

```bash
cargo flamegraph --bin phastft
```