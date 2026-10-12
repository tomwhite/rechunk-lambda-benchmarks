use lambda_runtime::{Error, LambdaEvent};
use serde::Deserialize;
use zarrs::array::ArraySubset;

/// Validates chunks of a Zarr array against the hash `hashed-integers` writes,
/// which is independent of chunking.
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

    validate_chunks(
        &payload.store,
        &payload.array_path,
        &payload.array_spec,
        &payload.chunk_coords,
    )
    .await?;
    Ok(())
}

/// Returns the number of chunks validated, or an error naming the first mismatch.
async fn validate_chunks(
    store_url: &str,
    array_path: &str,
    array_spec: &common::ArraySpec,
    chunk_coords: &[Vec<u64>],
) -> Result<usize, common::Error> {
    let array = common::open_array_from_spec(store_url, array_path, array_spec)?;
    let shape = array.shape().to_vec();

    for coord in chunk_coords {
        let nominal_chunk_shape: Vec<u64> =
            array.chunk_shape(coord)?.iter().map(|n| n.get()).collect();
        let start: Vec<u64> = coord
            .iter()
            .zip(&nominal_chunk_shape)
            .map(|(c, s)| c * s)
            .collect();
        let subset =
            ArraySubset::new_with_start_shape(start.clone(), nominal_chunk_shape)?.bound(&shape)?;

        let expected = common::hashed_elements(&shape, &start, subset.shape());
        let actual: Vec<u32> = array.async_retrieve_array_subset(&subset).await?;

        if actual.len() != expected.len() {
            return Err(format!(
                "chunk {coord:?}: expected {} elements, got {}",
                expected.len(),
                actual.len()
            )
            .into());
        }

        if let Some(i) = actual.iter().zip(&expected).position(|(a, e)| a != e) {
            return Err(format!(
                "chunk {coord:?} mismatch at local index {i}: expected {}, got {}",
                expected[i], actual[i]
            )
            .into());
        }
    }

    Ok(chunk_coords.len())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_validate_chunks_success() {
        let dir = tempfile::tempdir().unwrap();
        let store_url = format!("file://{}", dir.path().display());
        let (array, array_spec) = common::create_test_array(&store_url, &[2, 6], &[2, 3], "uint32")
            .await
            .unwrap();

        for coord in [vec![0u64, 0], vec![0, 1]] {
            let elements = common::hashed_chunk_elements(&[2, 6], &[2, 3], &coord);
            array.async_store_chunk(&coord, elements).await.unwrap();
        }

        let chunks_validated =
            validate_chunks(&store_url, "/", &array_spec, &[vec![0, 0], vec![0, 1]])
                .await
                .unwrap();
        assert_eq!(chunks_validated, 2);
    }

    #[tokio::test]
    async fn test_validate_chunks_detects_mismatch() {
        let dir = tempfile::tempdir().unwrap();
        let store_url = format!("file://{}", dir.path().display());
        let (array, array_spec) = common::create_test_array(&store_url, &[2, 6], &[2, 3], "uint32")
            .await
            .unwrap();

        // deliberately wrong: zeros instead of the hashed values
        array
            .async_store_chunk(&[0u64, 0], vec![0u32; 6])
            .await
            .unwrap();

        let result = validate_chunks(&store_url, "/", &array_spec, &[vec![0, 0]]).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_validate_chunks_edge_chunk_written_via_array_subset() {
        // (5, 5) chunked (2, 2): edge chunks written as subsets, as copy
        // writes them, must be compared on their in-bounds part only
        let dir = tempfile::tempdir().unwrap();
        let store_url = format!("file://{}", dir.path().display());
        let (array, array_spec) = common::create_test_array(&store_url, &[5, 5], &[2, 2], "uint32")
            .await
            .unwrap();

        for coord in [
            vec![0u64, 0],
            vec![0, 1],
            vec![0, 2],
            vec![1, 0],
            vec![1, 1],
            vec![1, 2],
            vec![2, 0],
            vec![2, 1],
            vec![2, 2],
        ] {
            let nominal_chunk_shape = vec![2u64, 2];
            let start: Vec<u64> = coord
                .iter()
                .zip(&nominal_chunk_shape)
                .map(|(c, s)| c * s)
                .collect();
            let subset = ArraySubset::new_with_start_shape(start.clone(), nominal_chunk_shape)
                .unwrap()
                .bound(&[5, 5])
                .unwrap();
            let elements = common::hashed_elements(&[5, 5], &start, subset.shape());
            array
                .async_store_array_subset(&subset, elements)
                .await
                .unwrap();
        }

        let all_coords: Vec<Vec<u64>> = (0..3)
            .flat_map(|i| (0..3).map(move |j| vec![i, j]))
            .collect();
        let chunks_validated = validate_chunks(&store_url, "/", &array_spec, &all_coords)
            .await
            .unwrap();
        assert_eq!(chunks_validated, 9);
    }
}
