use lambda_runtime::{Error, LambdaEvent};
use serde::Deserialize;

/// Writes chunks of an existing Zarr array with Wang hash values:
/// deterministic, incompressible data whose value at a position never
/// depends on chunking.
#[derive(Deserialize)]
pub(crate) struct Request {
    /// URL of the store containing the array, e.g. `file:///tmp/foo` or
    /// `s3://bucket/prefix`.
    store: String,
    /// Path of the array within the store ("/" for an array at the
    /// store root).
    array_path: String,
    array_spec: common::ArraySpec,
    chunk_coords: Vec<Vec<u64>>,
}

pub(crate) async fn function_handler(event: LambdaEvent<Request>) -> Result<(), Error> {
    let payload = event.payload;

    write_chunks(
        &payload.store,
        &payload.array_path,
        &payload.array_spec,
        &payload.chunk_coords,
    )
    .await?;
    Ok(())
}

async fn write_chunks(
    store_url: &str,
    array_path: &str,
    array_spec: &common::ArraySpec,
    chunk_coords: &[Vec<u64>],
) -> Result<usize, common::Error> {
    let array = common::open_array_from_spec(store_url, array_path, array_spec)?;
    let shape = array.shape().to_vec();

    for coord in chunk_coords {
        let chunk_shape: Vec<u64> = array.chunk_shape(coord)?.iter().map(|n| n.get()).collect();
        let elements = common::hashed_chunk_elements(&shape, &chunk_shape, coord);
        array.async_store_chunk(coord, elements).await?;
    }

    Ok(chunk_coords.len())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_write_chunks() {
        let dir = tempfile::tempdir().unwrap();
        let store_url = format!("file://{}", dir.path().display());

        let (array, array_spec) = common::create_test_array(&store_url, &[2, 6], &[2, 3], "uint32")
            .await
            .unwrap();

        let chunks_written = write_chunks(&store_url, "/", &array_spec, &[vec![0, 0], vec![0, 1]])
            .await
            .unwrap();
        assert_eq!(chunks_written, 2);

        // cross-checked against known values
        let chunk_0_1: Vec<u32> = array.async_retrieve_chunk(&[0, 1]).await.unwrap();
        assert_eq!(
            chunk_0_1,
            vec![2278584254, 3427349084, 3322605197, 454550906, 1415330532, 1798297286]
        );
    }
}
