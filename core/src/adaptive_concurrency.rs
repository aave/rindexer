//! Adaptive concurrency control for RPC requests.
//!
//! This module provides rate limiting and adaptive scaling for RPC requests. Each
//! provider owns one controller (see `JsonRpcCachedProvider`), so one network's rate
//! limits never throttle another network's requests. It's used by the RPC layer
//! (layer_extensions.rs) and the indexer (tables.rs, fetch_logs.rs).
//!
//! ## Adaptation granularity
//!
//! Backoff is the only control that adapts *within* a running fetch: it is read live
//! before every request (`current_backoff_ms`, `wait_for_backoff`). Concurrency
//! (`current`) and `batch_size` are read once at the top of each `prefetch_*`
//! invocation into a fixed-size semaphore / chunk size, so any scale-up/down recorded
//! *during* that invocation only takes effect on the *next* one. This is safe because
//! invocations are scoped to a single (naturally small) event batch, so the staleness
//! is bounded to one batch. Do not assume `current` throttles in-flight work in real
//! time — it does not; backoff does.

use crate::metrics::rpc as rpc_metrics;
use once_cell::sync::Lazy;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::time::Instant;
use tracing::{debug, info, warn};

/// Idle period (no rate-limit event) after which time-based recovery begins.
const IDLE_BEFORE_RECOVERY_MS: u64 = 5_000;
/// Minimum gap between successive recovery ticks once idle.
const RECOVERY_INTERVAL_MS: u64 = 2_000;
/// Backoff at or below this (ms) is cleared to zero rather than halved again.
const BACKOFF_CLEAR_FLOOR_MS: u64 = 100;
/// Consecutive batch-fetch successes before concurrency/batch scale up one step.
const SCALE_UP_THRESHOLD: usize = 10;
/// First backoff step applied on the initial rate-limit event.
const INITIAL_BACKOFF_MS: u64 = 500;
/// Ceiling on backoff growth.
const MAX_BACKOFF_MS: u64 = 30_000;
/// Initial RPC batch size (calls per batch).
const INITIAL_BATCH_SIZE: usize = 50;
/// Minimum RPC batch size.
const MIN_BATCH_SIZE: usize = 5;
/// Maximum RPC batch size.
const MAX_BATCH_SIZE: usize = 100;

/// Monotonic epoch for the controller's internal timestamps. `Instant` isn't
/// atomic-storable, so timestamps are kept as millis elapsed since this epoch.
static CONTROLLER_EPOCH: Lazy<Instant> = Lazy::new(Instant::now);

/// Adaptive concurrency controller that scales based on success/failure rates.
/// Scales up when requests succeed, scales down when rate limits are hit.
#[derive(Debug)]
pub struct AdaptiveConcurrency {
    /// Network name used as the `network` label on the adaptive metrics.
    network: String,
    current: AtomicUsize,
    min: usize,
    max: usize,
    /// Count of consecutive successes - used to decide when to scale up
    consecutive_successes: AtomicUsize,
    /// Threshold of consecutive successes before scaling up
    scale_up_threshold: usize,
    /// Current backoff delay in milliseconds (for rate-limited free nodes)
    backoff_ms: AtomicU64,
    /// Maximum backoff delay in milliseconds (30 seconds)
    max_backoff_ms: u64,
    /// Current batch size (number of calls per RPC batch)
    batch_size: AtomicUsize,
    /// Minimum batch size
    min_batch_size: usize,
    /// Maximum batch size
    max_batch_size: usize,
    /// Total rate limit count (for diagnostics)
    rate_limit_count: AtomicU64,
    /// Whether any rate-limit has occurred; gates recovery (a fresh controller has nothing to undo).
    ever_rate_limited: AtomicBool,
    /// Millis (since `CONTROLLER_EPOCH`) of the most recent rate-limit event.
    last_rate_limit_ms: AtomicU64,
    /// Millis (since `CONTROLLER_EPOCH`) of the most recent recovery tick; single-writer guard.
    last_recovery_ms: AtomicU64,
    /// Idle period (no rate-limit) after which time-based recovery begins.
    idle_before_recovery_ms: u64,
    /// Minimum gap between successive recovery ticks once idle.
    recovery_interval_ms: u64,
    /// Test-only clock advance so recovery timing can be driven without sleeps.
    #[cfg(test)]
    test_clock_advance_ms: AtomicU64,
}

impl AdaptiveConcurrency {
    pub fn new(network: impl Into<String>, initial: usize, min: usize, max: usize) -> Self {
        Self {
            network: network.into(),
            current: AtomicUsize::new(initial),
            min,
            max,
            consecutive_successes: AtomicUsize::new(0),
            scale_up_threshold: SCALE_UP_THRESHOLD,
            backoff_ms: AtomicU64::new(0),
            max_backoff_ms: MAX_BACKOFF_MS,
            batch_size: AtomicUsize::new(INITIAL_BATCH_SIZE),
            min_batch_size: MIN_BATCH_SIZE,
            max_batch_size: MAX_BATCH_SIZE,
            rate_limit_count: AtomicU64::new(0),
            ever_rate_limited: AtomicBool::new(false),
            last_rate_limit_ms: AtomicU64::new(0),
            last_recovery_ms: AtomicU64::new(0),
            idle_before_recovery_ms: IDLE_BEFORE_RECOVERY_MS,
            recovery_interval_ms: RECOVERY_INTERVAL_MS,
            #[cfg(test)]
            test_clock_advance_ms: AtomicU64::new(0),
        }
    }

    fn now_millis(&self) -> u64 {
        let now = CONTROLLER_EPOCH.elapsed().as_millis() as u64;
        #[cfg(test)]
        let now = now + self.test_clock_advance_ms.load(Ordering::Relaxed);
        now
    }

    /// Get current concurrency level
    pub fn current(&self) -> usize {
        self.current.load(Ordering::Relaxed)
    }

    /// Get current batch size (calls per RPC batch)
    pub fn current_batch_size(&self) -> usize {
        self.batch_size.load(Ordering::Relaxed)
    }

    /// Get current backoff delay in milliseconds.
    ///
    /// Also drives time-based recovery: this is read once before every RPC request
    /// (see `layer_extensions.rs`), so it is the natural place to let a stale backoff
    /// decay when the provider has stopped rate-limiting us.
    pub fn current_backoff_ms(&self) -> u64 {
        self.recover_if_idle();
        self.backoff_ms.load(Ordering::Relaxed)
    }

    /// Get total rate limit count
    pub fn rate_limit_count(&self) -> u64 {
        self.rate_limit_count.load(Ordering::Relaxed)
    }

    /// Publish current controller state to Prometheus gauges. Called after every
    /// state transition so alerts see backoff/scale-down in near real time.
    fn publish_metrics(&self) {
        rpc_metrics::set_adaptive_state(
            &self.network,
            self.current.load(Ordering::Relaxed),
            self.batch_size.load(Ordering::Relaxed),
            self.backoff_ms.load(Ordering::Relaxed),
        );
        rpc_metrics::set_rate_limit_events(
            &self.network,
            self.rate_limit_count.load(Ordering::Relaxed),
        );
    }

    /// Wait for the backoff delay if one is active.
    /// Call this before making RPC requests to respect rate limits.
    pub async fn wait_for_backoff(&self) {
        self.recover_if_idle();
        let delay_ms = self.backoff_ms.load(Ordering::Relaxed);
        if delay_ms > 0 {
            debug!("Rate limit backoff: waiting {}ms before next RPC request", delay_ms);
            tokio::time::sleep(tokio::time::Duration::from_millis(delay_ms)).await;
        }
    }

    /// Decay backoff by 25% without touching concurrency/batch scale-up.
    ///
    /// Internal helper for `record_success`: only batch-fetch successes are strong
    /// enough evidence to decay backoff. The RPC layer deliberately does NOT call
    /// this — a cheap call (e.g. `eth_blockNumber`) can succeed while a
    /// compute-weight limiter is still throttling heavy calls, so layer-level
    /// successes must not clear backoff (see `layer_extensions.rs`).
    pub fn record_success_backoff_only(&self) {
        let updated = self.backoff_ms.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |cur| {
            (cur > 0).then_some(cur * 3 / 4)
        });
        if let Ok(prev) = updated {
            if prev * 3 / 4 == 0 {
                info!("Adaptive concurrency: backoff cleared after successful requests");
            }
            self.publish_metrics();
        }
    }

    /// Record a successful request - may scale up and reduce backoff
    pub fn record_success(&self) {
        self.record_success_backoff_only();

        let successes = self.consecutive_successes.fetch_add(1, Ordering::Relaxed) + 1;
        if successes >= self.scale_up_threshold {
            self.consecutive_successes.store(0, Ordering::Relaxed);

            // Scale up concurrency
            let current = self.current.load(Ordering::Relaxed);
            if current < self.max {
                // Scale up by 20% or at least 1
                let increase = std::cmp::max(1, current / 5);
                let new_val = std::cmp::min(self.max, current + increase);
                if self
                    .current
                    .compare_exchange(current, new_val, Ordering::Relaxed, Ordering::Relaxed)
                    .is_ok()
                {
                    info!(
                        "Adaptive concurrency: scaling UP from {} to {} (consecutive successes)",
                        current, new_val
                    );
                }
            }

            // Scale up batch size (by 20% or at least 5)
            let current_batch = self.batch_size.load(Ordering::Relaxed);
            if current_batch < self.max_batch_size {
                let increase = std::cmp::max(5, current_batch / 5);
                let new_batch = std::cmp::min(self.max_batch_size, current_batch + increase);
                if self
                    .batch_size
                    .compare_exchange(
                        current_batch,
                        new_batch,
                        Ordering::Relaxed,
                        Ordering::Relaxed,
                    )
                    .is_ok()
                {
                    info!(
                        "Adaptive batch size: scaling UP from {} to {} (consecutive successes)",
                        current_batch, new_batch
                    );
                }
            }
        }

        self.publish_metrics();
    }

    /// Record a rate limit error - scale down aggressively and increase backoff
    pub fn record_rate_limit(&self) {
        self.consecutive_successes.store(0, Ordering::Relaxed);
        let count = self.rate_limit_count.fetch_add(1, Ordering::Relaxed) + 1;
        // Stamp the event so time-based recovery holds off until the provider is quiet.
        self.ever_rate_limited.store(true, Ordering::Relaxed);
        self.last_rate_limit_ms.store(self.now_millis(), Ordering::Relaxed);

        // Increase backoff (double it, starting from 500ms, max 30s). Use fetch_update so
        // concurrent callers can't drop the increase on a lost compare_exchange race.
        let max_backoff_ms = self.max_backoff_ms;
        let new_backoff = self
            .backoff_ms
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |cur| {
                Some(if cur == 0 {
                    INITIAL_BACKOFF_MS
                } else {
                    std::cmp::min(max_backoff_ms, cur.saturating_mul(2))
                })
            })
            .map(|prev| {
                if prev == 0 {
                    INITIAL_BACKOFF_MS
                } else {
                    std::cmp::min(max_backoff_ms, prev.saturating_mul(2))
                }
            })
            .unwrap_or(INITIAL_BACKOFF_MS);

        // Scale down batch size by 50%
        let current_batch = self.batch_size.load(Ordering::Relaxed);
        let new_batch = std::cmp::max(self.min_batch_size, current_batch / 2);
        let batch_changed = self
            .batch_size
            .compare_exchange(current_batch, new_batch, Ordering::Relaxed, Ordering::Relaxed)
            .is_ok();

        // Scale down concurrency by 50%
        let current = self.current.load(Ordering::Relaxed);
        if current > self.min {
            let new_val = std::cmp::max(self.min, current / 2);
            if self
                .current
                .compare_exchange(current, new_val, Ordering::Relaxed, Ordering::Relaxed)
                .is_ok()
            {
                warn!(
                    "Adaptive: concurrency {} -> {}, batch {} -> {}, backoff: {}ms, total_rate_limits: {} (rate limit)",
                    current, new_val, current_batch, new_batch, new_backoff, count
                );
            }
        } else if batch_changed {
            warn!(
                "Adaptive: at min concurrency {}, batch {} -> {}, backoff: {}ms, total_rate_limits: {} (rate limit)",
                self.min, current_batch, new_batch, new_backoff, count
            );
        } else {
            warn!(
                "Adaptive: at minimum (concurrency: {}, batch: {}), backoff: {}ms, total_rate_limits: {}",
                self.min, self.min_batch_size, new_backoff, count
            );
        }

        self.publish_metrics();
    }

    /// Record a general error - scale down slightly
    pub fn record_error(&self) {
        self.consecutive_successes.store(0, Ordering::Relaxed);
        let current = self.current.load(Ordering::Relaxed);
        if current > self.min {
            // Scale down by 10% on general error
            let decrease = std::cmp::max(1, current / 10);
            let new_val = std::cmp::max(self.min, current.saturating_sub(decrease));
            if self
                .current
                .compare_exchange(current, new_val, Ordering::Relaxed, Ordering::Relaxed)
                .is_ok()
            {
                debug!(
                    "Adaptive concurrency: scaling down from {} to {} (error)",
                    current, new_val
                );
            }
        }

        self.publish_metrics();
    }

    /// Time-based recovery safety net.
    ///
    /// The success paths can stall — under CAS contention, or when live streams sit near
    /// the chain tip and stop generating fetch successes — leaving a single transient
    /// rate-limit permanently pinning `backoff_ms`/`current` for this network until the
    /// process restarts.
    ///
    /// This runs from the per-request backoff reads: once no rate-limit has occurred for
    /// `max(IDLE_BEFORE_RECOVERY_MS, 2 × backoff)`, it halves the backoff and steps
    /// concurrency/batch back up at most once per `RECOVERY_INTERVAL_MS`. Recovery is gradual
    /// (not an instant reset) so all streams don't resume full rate at once and immediately
    /// re-trip the limiter.
    fn recover_if_idle(&self) {
        // Never throttled — nothing to undo.
        if !self.ever_rate_limited.load(Ordering::Relaxed) {
            return;
        }

        let backoff = self.backoff_ms.load(Ordering::Relaxed);
        // Already fully recovered.
        if backoff == 0
            && self.current.load(Ordering::Relaxed) >= self.max
            && self.batch_size.load(Ordering::Relaxed) >= self.max_batch_size
        {
            return;
        }

        // While workers sleep out a large backoff no requests complete, so a short quiet
        // period is absence of evidence, not proof of recovery: scale the required idle
        // window with the backoff itself so recovery can't outrun the next wave of
        // (possibly still rate-limited) responses.
        let required_idle_ms =
            std::cmp::max(self.idle_before_recovery_ms, backoff.saturating_mul(2));
        let now = self.now_millis();
        if now.saturating_sub(self.last_rate_limit_ms.load(Ordering::Relaxed)) < required_idle_ms {
            return;
        }

        // Single-writer guard: only the caller that advances `last_recovery_ms` applies this
        // tick, so concurrent readers recover at most once per interval.
        let last_recovery = self.last_recovery_ms.load(Ordering::Relaxed);
        if now.saturating_sub(last_recovery) < self.recovery_interval_ms
            || self
                .last_recovery_ms
                .compare_exchange(last_recovery, now, Ordering::Relaxed, Ordering::Relaxed)
                .is_err()
        {
            return;
        }

        // Re-read state after winning the tick: a rate limit recorded between the pre-checks
        // and here must not have its backoff increase clobbered with stale values.
        if self.now_millis().saturating_sub(self.last_rate_limit_ms.load(Ordering::Relaxed))
            < required_idle_ms
        {
            return;
        }
        // Race-safe recovery: a `record_rate_limit` can still land in the tiny window
        // between the re-check above and these writes. `fetch_update` re-reads the live
        // value on each attempt, so recovery steps *toward* its target from whatever is
        // current instead of clobbering a concurrent scale-down/backoff-increase with a
        // value derived from a stale read.
        let pre_backoff = self.backoff_ms.load(Ordering::Relaxed);
        let backoff_update =
            self.backoff_ms.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |cur| {
                // A concurrent rate-limit that raised backoff past our read must win:
                // don't undo it.
                if cur == 0 || cur > pre_backoff {
                    None
                } else if cur <= BACKOFF_CLEAR_FLOOR_MS {
                    Some(0)
                } else {
                    Some(cur / 2)
                }
            });
        match backoff_update {
            Ok(prev) => {
                if prev <= BACKOFF_CLEAR_FLOOR_MS {
                    info!(
                        "Adaptive: backoff cleared after {}ms idle (recovered)",
                        self.idle_before_recovery_ms
                    );
                }
            }
            // A rate limit landed mid-tick and raised backoff past our read: it must
            // win for all three knobs, so skip the concurrency/batch scale-ups too.
            // (`record_rate_limit` publishes metrics itself.)
            Err(cur) if cur > pre_backoff => return,
            // Backoff already at zero — nothing to decay, but concurrency/batch may
            // still need stepping back up.
            Err(_) => {}
        }

        // Step concurrency/batch up from the live value; no-op once at max. Stepping from
        // `cur` (not a stale read) means a concurrent scale-down is preserved — recovery
        // adds at most one step to whatever the rate-limit path left behind.
        let _ = self.current.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |cur| {
            (cur < self.max).then(|| std::cmp::min(self.max, cur + std::cmp::max(1, cur / 5)))
        });
        let _ = self.batch_size.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |cur| {
            (cur < self.max_batch_size)
                .then(|| std::cmp::min(self.max_batch_size, cur + std::cmp::max(5, cur / 5)))
        });

        self.publish_metrics();
    }
}

/// True if the error means the provider is rate-limiting us *or* is
/// temporarily unavailable (HTTP 429/503, Alchemy `-32001`, etc.). Both warrant
/// backing off rather than hammering the node. Matches specific tokens (e.g.
/// `"http error 503"`, not bare `"503"`) so block numbers don't false-positive.
pub fn is_rate_limited_or_unavailable(error_message: &str) -> bool {
    let msg = error_message.to_lowercase();

    // Classic rate-limit / throttle signals. `429` must stay anchored to an
    // http/status/code token: block numbers and hex ranges embedded in other
    // errors (e.g. "this block range should work: [0x1429680, ...]") contain
    // bare "429" and must not classify as rate limits.
    msg.contains("http error 429")
        || msg.contains("status: 429")
        || msg.contains("status code 429")
        || msg.contains("error code 429")
        || msg.contains("code: 429")
        || msg.contains("rate limit")
        || msg.contains("rate-limit")
        || msg.contains("rate exceeded")
        || msg.contains("too many requests")
        || msg.contains("quota")
        || msg.contains("throttle")
        // Temporary unavailability / overload — back off the same way.
        || msg.contains("http error 503")
        || msg.contains("status: 503")
        || msg.contains("service unavailable")
        || msg.contains("temporarily unavailable")
        || msg.contains("unable to complete request")
        || msg.contains("-32001")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Controller whose recovery tick interval is zero, so ticks fire on every
    /// read once the (backoff-scaled) idle window has been faked past via
    /// [`advance_clock`] — tests need no wall-clock sleeps.
    fn instant_recovery_controller() -> AdaptiveConcurrency {
        let mut c = AdaptiveConcurrency::new("testnet", 20, 2, 200);
        c.idle_before_recovery_ms = 0;
        c.recovery_interval_ms = 0;
        c
    }

    fn advance_clock(c: &AdaptiveConcurrency, ms: u64) {
        c.test_clock_advance_ms.fetch_add(ms, Ordering::Relaxed);
    }

    fn backoff(c: &AdaptiveConcurrency) -> u64 {
        c.backoff_ms.load(Ordering::Relaxed)
    }

    #[test]
    fn record_rate_limit_grows_backoff_and_scales_down() {
        let c = AdaptiveConcurrency::new("testnet", 20, 2, 200);
        c.record_rate_limit();
        assert_eq!(backoff(&c), 500, "first rate limit starts backoff at 500ms");
        c.record_rate_limit();
        assert_eq!(backoff(&c), 1000, "second rate limit doubles backoff");
        assert!(c.current() < 20, "concurrency scales down on rate limit");
    }

    #[test]
    fn recover_is_noop_until_a_rate_limit_has_occurred() {
        // A freshly-started, never-throttled controller must not be nudged by recovery:
        // scale-up stays owned by the success path.
        let c = instant_recovery_controller();
        let before = c.current();
        advance_clock(&c, 3_600_000);
        c.recover_if_idle();
        assert_eq!(backoff(&c), 0);
        assert_eq!(
            c.current(),
            before,
            "recovery does not scale up a controller that never backed off"
        );
    }

    #[test]
    fn backoff_and_concurrency_recover_once_idle() {
        let c = instant_recovery_controller();
        c.record_rate_limit();
        c.record_rate_limit();
        assert_eq!(backoff(&c), 1000);
        assert!(c.current() < 20);

        // Fake a long quiet period, then drive recovery ticks (interval is 0 here).
        advance_clock(&c, 3_600_000);
        for _ in 0..200 {
            c.recover_if_idle();
            if backoff(&c) == 0 && c.current() >= c.max {
                break;
            }
        }

        assert_eq!(backoff(&c), 0, "backoff fully decays once the provider is quiet");
        assert_eq!(c.current(), c.max, "concurrency climbs back to max once recovered");
    }

    #[test]
    fn recovery_holds_off_while_rate_limits_continue() {
        // Default 5s idle window: a just-recorded rate limit must not be undone immediately.
        let c = AdaptiveConcurrency::new("testnet", 20, 2, 200);
        c.record_rate_limit();
        let after_limit = backoff(&c);
        // current_backoff_ms() runs recovery, but the idle window hasn't elapsed.
        assert_eq!(
            c.current_backoff_ms(),
            after_limit,
            "backoff is not decayed within the idle window"
        );
    }

    #[test]
    fn required_idle_window_scales_with_backoff() {
        // With a large backoff all workers are asleep, so a short quiet period
        // proves nothing: the idle requirement is max(5s, 2 × backoff).
        let c = AdaptiveConcurrency::new("testnet", 20, 2, 200);
        for _ in 0..7 {
            c.record_rate_limit();
        }
        assert_eq!(backoff(&c), 30_000, "backoff capped at max");

        // Past the 5s base window but well inside 2 × 30s: recovery must hold.
        advance_clock(&c, 6_000);
        assert_eq!(
            c.current_backoff_ms(),
            30_000,
            "recovery must not fire while sleeping workers could still be rate-limited"
        );

        // Past 60s of true quiet: recovery begins.
        advance_clock(&c, 55_000);
        c.recover_if_idle();
        assert_eq!(backoff(&c), 15_000, "backoff halves once the scaled idle window elapses");
    }

    #[test]
    fn backoff_only_decay_does_not_scale_up() {
        let c = AdaptiveConcurrency::new("testnet", 20, 2, 200);
        c.record_rate_limit();
        c.record_rate_limit();
        assert_eq!(backoff(&c), 1000);
        let throttled_concurrency = c.current();

        for _ in 0..40 {
            c.record_success_backoff_only();
        }

        assert_eq!(backoff(&c), 0, "successful responses decay backoff to zero");
        assert_eq!(
            c.current(),
            throttled_concurrency,
            "decay-only success must not scale concurrency back up"
        );
    }

    #[test]
    fn controllers_are_independent() {
        // The production incident: one network's rate limits must not throttle another's
        // controller.
        let ethereum = AdaptiveConcurrency::new("ethereum", 20, 2, 200);
        let avalanche = AdaptiveConcurrency::new("avalanche", 20, 2, 200);

        for _ in 0..5 {
            ethereum.record_rate_limit();
        }

        assert!(backoff(&ethereum) > 0);
        assert!(ethereum.current() < 20);
        assert_eq!(backoff(&avalanche), 0, "unrelated controller keeps zero backoff");
        assert_eq!(avalanche.current(), 20, "unrelated controller keeps full concurrency");
    }

    #[test]
    fn rate_limit_classifier_requires_anchored_429() {
        // Real 429 shapes: alloy HttpError display, alloy ErrorPayload display,
        // and reqwest-style status text.
        assert!(is_rate_limited_or_unavailable("HTTP error 429 with body: slow down"));
        assert!(is_rate_limited_or_unavailable(
            "server returned an error response: error code 429: capacity exceeded"
        ));
        assert!(is_rate_limited_or_unavailable("429 Too Many Requests"));

        // Block numbers / hex ranges containing "429" must not classify as
        // rate limits — misfiring here backs off and retries the same
        // oversized range instead of shrinking it.
        assert!(!is_rate_limited_or_unavailable(
            "query exceeds max results, this block range should work: [0x1429680, 0x142a000]"
        ));
        assert!(!is_rate_limited_or_unavailable("block range too large: 21429000-21439000"));
    }
}
