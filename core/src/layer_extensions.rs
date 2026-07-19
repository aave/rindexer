use crate::{
    adaptive_concurrency::{is_rate_limited_or_unavailable, AdaptiveConcurrency},
    metrics::rpc as rpc_metrics,
    rindexer_error, rindexer_info,
};
use alloy::{
    rpc::json_rpc::{RequestPacket, ResponsePacket},
    transports::TransportError,
};
use alloy_chains::Chain;
use std::{
    future::Future,
    pin::Pin,
    sync::Arc,
    task::{Context, Poll},
    time::Instant,
};
use tokio::time::Duration;
use tower::{Layer, Service};

#[derive(Clone)]
pub struct RpcLoggingLayer {
    chain_id: u64,
    /// Chain name used as the `network` metric label (matches `provider.rs`).
    network: String,
    rpc_url: String,
    /// This network's adaptive controller — shared with the provider so rate
    /// limits observed here throttle only this network's requests.
    adaptive: Arc<AdaptiveConcurrency>,
}

impl RpcLoggingLayer {
    pub fn new(chain_id: u64, rpc_url: String, adaptive: Arc<AdaptiveConcurrency>) -> Self {
        Self { chain_id, network: Chain::from(chain_id).to_string(), rpc_url, adaptive }
    }
}

impl<S> Layer<S> for RpcLoggingLayer {
    type Service = RpcLoggingService<S>;

    fn layer(&self, inner: S) -> Self::Service {
        RpcLoggingService {
            inner,
            chain_id: self.chain_id,
            network: self.network.clone(),
            rpc_url: self.rpc_url.clone(),
            adaptive: Arc::clone(&self.adaptive),
        }
    }
}

#[derive(Debug, Clone)]
pub struct RpcLoggingService<S> {
    inner: S,
    chain_id: u64,
    network: String,
    rpc_url: String,
    adaptive: Arc<AdaptiveConcurrency>,
}

impl<S> Service<RequestPacket> for RpcLoggingService<S>
where
    S: Service<RequestPacket, Response = ResponsePacket, Error = TransportError>,
    S::Future: Send + 'static,
{
    type Response = S::Response;
    type Error = S::Error;
    type Future = Pin<Box<dyn Future<Output = Result<Self::Response, Self::Error>> + Send>>;

    fn poll_ready(&mut self, cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        self.inner.poll_ready(cx)
    }

    fn call(&mut self, req: RequestPacket) -> Self::Future {
        let start_time = Instant::now();
        let chain_id = self.chain_id;
        let network = self.network.clone();
        let rpc_url = self.rpc_url.clone();
        let adaptive = Arc::clone(&self.adaptive);

        let method_name = match &req {
            RequestPacket::Single(r) => r.method().to_string(),
            RequestPacket::Batch(reqs) => {
                if reqs.is_empty() {
                    "empty_batch".to_string()
                } else if reqs.len() == 1 {
                    reqs[0].method().to_string()
                } else {
                    format!("batch_{}_requests", reqs.len())
                }
            }
        };

        let fut = self.inner.call(req);

        Box::pin(async move {
            // Enforce this network's backoff BEFORE making request (for rate-limited free nodes)
            // Add jitter to prevent thundering herd (all tasks waking at once)
            let backoff_ms = adaptive.current_backoff_ms();
            if backoff_ms > 0 {
                // Add 0-50% random jitter to spread out requests
                let jitter = (backoff_ms as f64 * rand::random::<f64>() * 0.5) as u64;
                tokio::time::sleep(Duration::from_millis(backoff_ms + jitter)).await;
            }

            match fut.await {
                Ok(response) => {
                    let duration = start_time.elapsed();

                    // Backoff decay is intentionally NOT driven from here. A layer-level
                    // success may be a cheap call (e.g. `eth_blockNumber`) that a
                    // compute-weight limiter lets through while heavier calls are still
                    // throttled, so decaying on it would prematurely clear backoff and let
                    // the heavy calls re-trip the limiter. Decay is owned by the batch-fetch
                    // success paths (heavy-call evidence) and the time-based
                    // `recover_if_idle` safety net — which this request already drives via
                    // the `current_backoff_ms()` read above.

                    if duration.as_secs() >= 10 {
                        rpc_metrics::record_slow_call(&network, &method_name);
                        rindexer_info!(
                            "SLOW RPC call - chain_id: {}, method: {}, duration: {:?}, url: {}",
                            chain_id,
                            method_name,
                            duration,
                            rpc_url
                        );
                    }

                    Ok(response)
                }
                Err(err) => {
                    let duration = start_time.elapsed();
                    let error_str = err.to_string();

                    let is_known_error = is_known_retryable_error(&error_str);

                    if !is_known_error {
                        if error_str.contains("timeout") || error_str.contains("timed out") {
                            rpc_metrics::record_rpc_error_kind(
                                &network,
                                &method_name,
                                rpc_metrics::error_kind::TIMEOUT,
                            );
                            rindexer_error!("RPC TIMEOUT (free public nodes do this a lot consider a using a paid node) - chain_id: {}, method: {}, duration: {:?}, url: {}, error: {:?}",
                                           chain_id, method_name, duration, rpc_url, err);
                        } else if is_rate_limited_or_unavailable(&error_str) {
                            // Scale down + grow backoff. The pre-request wait above then
                            // throttles every later call (incl. the retry layer's own retries).
                            adaptive.record_rate_limit();
                            rpc_metrics::record_rpc_error_kind(
                                &network,
                                &method_name,
                                rpc_metrics::error_kind::RATE_LIMITED,
                            );
                            rindexer_info!("RPC RATE LIMITED / UNAVAILABLE (free public nodes do this a lot consider using a paid node) - chain_id: {}, method: {}, duration: {:?}, url: {}, backoff: {}ms, batch_size: {}, rate_limit_count: {}",
                                          chain_id, method_name, duration, rpc_url,
                                          adaptive.current_backoff_ms(),
                                          adaptive.current_batch_size(),
                                          adaptive.rate_limit_count());
                        } else if error_str.contains("connection")
                            || error_str.contains("network")
                            || error_str.contains("sending request")
                        {
                            rpc_metrics::record_rpc_error_kind(
                                &network,
                                &method_name,
                                rpc_metrics::error_kind::CONNECTION,
                            );
                            rindexer_error!("RPC CONNECTION ERROR (free public nodes do this a lot consider a using a paid node) - chain_id: {}, method: {}, duration: {:?}, url: {}, error: {:?}",
                                           chain_id, method_name, duration, rpc_url, err);
                        } else {
                            rpc_metrics::record_rpc_error_kind(
                                &network,
                                &method_name,
                                rpc_metrics::error_kind::OTHER,
                            );
                            rindexer_error!("RPC ERROR (free public nodes do this a lot consider a using a paid node) - chain_id: {}, method: {}, duration: {:?}, url: {}, error: {:?}",
                                           chain_id, method_name, duration, rpc_url, err);
                        }
                    }

                    Err(err)
                }
            }
        })
    }
}

fn is_known_retryable_error(error_message: &str) -> bool {
    // mirror handled logic which is in the `retry_with_block_range`
    error_message.contains("this block range should work")
        || error_message.contains("try with this block range")
        || error_message.contains("block range is too wide")
        || error_message.contains("limited to a")
        || error_message.contains("block range too large")
        || error_message.contains("response is too big")
        || error_message.contains("error decoding response body")
}
