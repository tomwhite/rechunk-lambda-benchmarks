use futures::stream::{StreamExt, TryStreamExt};
use lambda_runtime::tracing::warn;
use lambda_runtime::{Error, LambdaEvent};
use object_store::{path::Path, ObjectStore};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use url::Url;

/// The marker a store URL must contain before this deletes anything: a
/// guard beside IAM's (`s3:DeleteObject` only on `*/_intermediate/*`)
///
/// Keep in step with `rechunk_lambda_benchmarks.rechunk.INTERMEDIATE_PREFIX`.
const INTERMEDIATE_MARKER: &str = "/_intermediate/";

/// Keys per `DeleteObjects` request - S3's own maximum
const DELETE_BATCH: usize = 1_000;

/// Requests in flight.
const DELETE_CONCURRENCY: usize = 2;

/// Deletes everything under an intermediate store URL. The client invokes this
/// asynchronously and does not wait: deletion is idempotent, and a
/// lifecycle rule sweeps up whatever a failure leaves.
#[derive(Deserialize)]
pub(crate) struct Request {
    /// URL of the intermediate array to remove, e.g.
    /// `s3://bucket/prefix/target/_intermediate/stage0`. Must contain
    /// `/_intermediate/`.
    store: String,
}

#[derive(Serialize)]
#[serde(tag = "status", rename_all = "lowercase")]
pub(crate) enum Response {
    /// `failed` is not an error: deletion is best-effort. It is logged, so
    /// the logs can say why intermediates remain.
    Success {
        deleted: usize,
        failed: usize,
    },
    Failure {
        message: String,
    },
}

/// Whether a store URL is somewhere this is allowed to delete.
pub(crate) fn is_intermediate_store(store: &str) -> bool {
    store.contains(INTERMEDIATE_MARKER)
}

pub(crate) async fn function_handler(event: LambdaEvent<Request>) -> Result<Response, Error> {
    let store_url = event.payload.store;

    if !is_intermediate_store(&store_url) {
        // Refused rather than raised: an async invoke retries an error
        // twice, pointlessly for a refusal. Logged, as the body is discarded.
        let message = format!(
            "refusing to delete {store_url}: a store this deletes from must contain \
             {INTERMEDIATE_MARKER}"
        );
        warn!("{message}");
        return Ok(Response::Failure { message });
    }

    let url = Url::parse(&store_url)?;
    let (store, prefix) = object_store::parse_url_opts(&url, std::env::vars())?;
    let store: Arc<dyn ObjectStore> = Arc::from(store);

    let (deleted, failed) = delete_prefix(store, &prefix).await;
    if failed > 0 {
        warn!("{store_url}: deleted {deleted} keys, {failed} left behind (lifecycle rule sweeps)");
    }

    Ok(Response::Success { deleted, failed })
}

/// Delete everything under `prefix`, returning `(deleted, failed)`.
///
/// Never gives up partway: one throttled request must not abandon every
/// key after it, so errors are counted and the walk continues.
async fn delete_prefix(store: Arc<dyn ObjectStore>, prefix: &Path) -> (usize, usize) {
    store
        .list(Some(prefix))
        .map_ok(|meta| meta.location)
        .try_chunks(DELETE_BATCH)
        .map(|chunk| {
            let store = Arc::clone(&store);
            async move {
                match chunk {
                    Ok(locations) => delete_batch(store, locations).await,
                    // the listing itself failed; whatever it had already
                    // gathered is dropped with it
                    Err(e) => {
                        warn!("listing failed: {}", e.1);
                        (0, e.0.len())
                    }
                }
            }
        })
        .buffer_unordered(DELETE_CONCURRENCY)
        .fold((0, 0), |(deleted, failed), (d, f)| async move {
            (deleted + d, failed + f)
        })
        .await
}

/// Delete one batch, returning `(deleted, failed)`.
///
/// A batch is one `DeleteObjects` request, so an error is the whole request
/// failing: throttling that outlasted object_store's ten retries, which
/// can take under ten seconds.
async fn delete_batch(store: Arc<dyn ObjectStore>, locations: Vec<Path>) -> (usize, usize) {
    let total = locations.len();
    let mut deleted = 0;
    let stream = futures::stream::iter(locations.into_iter().map(Ok)).boxed();
    let mut results = store.delete_stream(stream);
    while let Some(result) = results.next().await {
        match result {
            Ok(_) => deleted += 1,
            Err(e) => {
                warn!(
                    "delete request failed, leaving {} keys: {e}",
                    total - deleted
                );
                break;
            }
        }
    }
    (deleted, total - deleted)
}

#[cfg(test)]
mod tests {
    use super::{delete_prefix, is_intermediate_store, DELETE_BATCH};
    use futures::StreamExt;
    use object_store::{local::LocalFileSystem, ObjectStore, ObjectStoreExt};
    use std::sync::Arc;

    async fn store_with(n: usize) -> (tempfile::TempDir, Arc<dyn ObjectStore>) {
        let dir = tempfile::tempdir().unwrap();
        let store: Arc<dyn ObjectStore> =
            Arc::new(LocalFileSystem::new_with_prefix(dir.path()).unwrap());
        for i in 0..n {
            store
                .put(&format!("c/{i}").into(), vec![0u8; 4].into())
                .await
                .unwrap();
        }
        (dir, store)
    }

    #[tokio::test]
    async fn deletes_everything_under_the_prefix() {
        let (_dir, store) = store_with(10).await;
        let (deleted, failed) = delete_prefix(Arc::clone(&store), &"c".into()).await;
        assert_eq!((deleted, failed), (10, 0));
        assert_eq!(store.list(Some(&"c".into())).count().await, 0);
    }

    #[tokio::test]
    async fn deletes_more_keys_than_fit_in_one_request() {
        // the batching loop only runs more than once past DELETE_BATCH,
        // and getting the second batch wrong loses the tail of an
        // intermediate
        let n = DELETE_BATCH + 25;
        let (_dir, store) = store_with(n).await;
        let (deleted, failed) = delete_prefix(Arc::clone(&store), &"c".into()).await;
        assert_eq!((deleted, failed), (n, 0));
        assert_eq!(store.list(Some(&"c".into())).count().await, 0);
    }

    #[tokio::test]
    async fn an_empty_prefix_is_not_an_error() {
        // deletion is idempotent, and is requested without checking
        // whether a previous attempt already succeeded
        let (_dir, store) = store_with(0).await;
        assert_eq!(delete_prefix(store, &"c".into()).await, (0, 0));
    }

    #[tokio::test]
    async fn other_prefixes_are_untouched() {
        let (_dir, store) = store_with(5).await;
        store.put(&"keep/x".into(), vec![1u8].into()).await.unwrap();
        let (deleted, _) = delete_prefix(Arc::clone(&store), &"c".into()).await;
        assert_eq!(deleted, 5);
        assert_eq!(store.list(Some(&"keep".into())).count().await, 1);
    }

    #[test]
    fn intermediate_paths_are_allowed() {
        assert!(is_intermediate_store(
            "s3://bucket/prefix/target/_intermediate/stage0"
        ));
        assert!(is_intermediate_store(
            "file:///tmp/target/_intermediate/stage1"
        ));
    }

    #[test]
    fn anything_else_is_refused() {
        // the paths that would cost real data
        assert!(!is_intermediate_store("s3://bucket/prefix/target"));
        assert!(!is_intermediate_store("s3://bucket/prefix/source"));
        assert!(!is_intermediate_store("s3://bucket"));
        // a near miss: the marker has to be a path segment, not a suffix
        assert!(!is_intermediate_store("s3://bucket/prefix/_intermediate"));
        assert!(!is_intermediate_store("s3://bucket/my_intermediate/data"));
    }
}
