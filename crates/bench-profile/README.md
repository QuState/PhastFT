# Profiling

From **repo root**:

```bash
# Linux: open access to performance monitoring
echo -1 | sudo tee /proc/sys/kernel/perf_event_paranoid

./crates/bench-profile/profile.sh
```
