use crate::{
    adaptive_concurrency::{is_rate_limited_or_unavailable, ADAPTIVE_CONCURRENCY},
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
}

impl RpcLoggingLayer {
    pub fn new(chain_id: u64, rpc_url: String) -> Self {
        Self { chain_id, network: Chain::from(chain_id).to_string(), rpc_url }
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
        }
    }
}

#[derive(Debug, Clone)]
pub struct RpcLoggingService<S> {
    inner: S,
    chain_id: u64,
    network: String,
    rpc_url: String,
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
            // Enforce global backoff BEFORE making request (for rate-limited free nodes)
            // Add jitter to prevent thundering herd (all tasks waking at once)
            let backoff_ms = ADAPTIVE_CONCURRENCY.current_backoff_ms();
            if backoff_ms > 0 {
                // Add 0-50% random jitter to spread out requests
                let jitter = (backoff_ms as f64 * rand::random::<f64>() * 0.5) as u64;
                tokio::time::sleep(Duration::from_millis(backoff_ms + jitter)).await;
            }

            match fut.await {
                Ok(response) => {
                    let duration = start_time.elapsed();

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

                    // Guard: a JSON-RPC *error payload* over a successful HTTP call (e.g. Alchemy's
                    // -32001 over HTTP 200) also lands in this Ok branch — that is the outage
                    // signal itself and must not decay the backoff.
                    if !has_throttle_error_payload(&response) {
                        ADAPTIVE_CONCURRENCY.record_success();
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
                            ADAPTIVE_CONCURRENCY.record_rate_limit();
                            rpc_metrics::record_rpc_error_kind(
                                &network,
                                &method_name,
                                rpc_metrics::error_kind::RATE_LIMITED,
                            );
                            rindexer_info!("RPC RATE LIMITED / UNAVAILABLE (free public nodes do this a lot consider using a paid node) - chain_id: {}, method: {}, duration: {:?}, url: {}, backoff: {}ms, batch_size: {}, rate_limit_count: {}",
                                          chain_id, method_name, duration, rpc_url,
                                          ADAPTIVE_CONCURRENCY.current_backoff_ms(),
                                          ADAPTIVE_CONCURRENCY.current_batch_size(),
                                          ADAPTIVE_CONCURRENCY.rate_limit_count());
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

/// True when the response packet carries a JSON-RPC error payload signalling throttling or
/// temporary unavailability (e.g. Alchemy `-32001` delivered over HTTP 200). Such a transport-
/// level "success" is really the outage signal and must not decay the adaptive backoff.
///
/// Matches on code + message only - `data` is deliberately excluded because it can carry
/// arbitrary hex/block numbers that false-positive loose tokens like `"429"`.
fn has_throttle_error_payload(response: &ResponsePacket) -> bool {
    response
        .iter_errors()
        .any(|e| is_rate_limited_or_unavailable(&format!("error code {}: {}", e.code, e.message)))
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

#[cfg(test)]
mod tests {
    use super::*;
    use alloy::rpc::json_rpc::Response;

    fn packet(json: &str) -> ResponsePacket {
        ResponsePacket::Single(serde_json::from_str::<Response>(json).expect("valid response"))
    }

    #[test]
    fn success_payload_is_not_throttle() {
        let p = packet(r#"{"jsonrpc":"2.0","id":1,"result":"0x10"}"#);
        assert!(!has_throttle_error_payload(&p));
    }

    #[test]
    fn alchemy_32001_over_http_200_is_throttle() {
        // The exact shape from the incident: HTTP 200 + JSON-RPC error payload.
        let p = packet(
            r#"{"jsonrpc":"2.0","id":1,"error":{"code":-32001,"message":"Unable to complete request at this time."}}"#,
        );
        assert!(has_throttle_error_payload(&p));
    }

    #[test]
    fn rate_limit_message_is_throttle() {
        let p = packet(
            r#"{"jsonrpc":"2.0","id":1,"error":{"code":-32005,"message":"too many requests"}}"#,
        );
        assert!(has_throttle_error_payload(&p));
    }

    #[test]
    fn block_range_error_is_not_throttle() {
        // Routine range-probing errors must still count as provider-alive (decay allowed).
        let p = packet(
            r#"{"jsonrpc":"2.0","id":1,"error":{"code":-32602,"message":"block range too large"}}"#,
        );
        assert!(!has_throttle_error_payload(&p));
    }

    #[test]
    fn numeric_noise_in_data_does_not_false_positive() {
        // "429" inside the error data must not be mistaken for HTTP 429 — the
        // helper matches on code + message only.
        let p = packet(
            r#"{"jsonrpc":"2.0","id":1,"error":{"code":-32602,"message":"invalid params","data":"0x429abc"}}"#,
        );
        assert!(!has_throttle_error_payload(&p));
    }

    #[test]
    fn batch_with_one_throttle_error_is_throttle() {
        let ok = serde_json::from_str::<Response>(r#"{"jsonrpc":"2.0","id":1,"result":"0x1"}"#)
            .expect("valid response");
        let throttled = serde_json::from_str::<Response>(
            r#"{"jsonrpc":"2.0","id":2,"error":{"code":-32001,"message":"Unable to complete request at this time."}}"#,
        )
        .expect("valid response");
        let p = ResponsePacket::Batch(vec![ok, throttled]);
        assert!(has_throttle_error_payload(&p));
    }
}
