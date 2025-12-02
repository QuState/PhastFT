//! Utilities for parallelism

/// Runs the two specified closures in parallel,
/// if and only if `parallel` is set to `true` and the `parallel` feature is enabled
#[allow(unused_variables)] // when `parallel` feature is disabled, the variable is ignored
pub fn run_maybe_in_parallel<A, B, RA, RB>(parallel: bool, oper_a: A, oper_b: B) -> (RA, RB)
where
    A: FnOnce() -> RA + Send,
    B: FnOnce() -> RB + Send,
    RA: Send,
    RB: Send,
{
    #[cfg(feature = "parallel")]
    {
        if parallel {
            chili::Scope::global().join(|_| oper_a(), |_| oper_b())
        } else {
            (oper_a(), oper_b())
        }
    }
    #[cfg(not(feature = "parallel"))]
    {
        (oper_a(), oper_b())
    }
}
