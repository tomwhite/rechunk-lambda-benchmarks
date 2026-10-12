use lambda_runtime::{Error, LambdaEvent};
use serde::Deserialize;
use zarrs::array::{ArrayBytes, ArraySubset};

/// Copies data from an existing (source) Zarr array to a newly-created
/// (target) array with the same shape but different chunking, one
/// "copy chunk" at a time.
#[derive(Deserialize)]
pub(crate) struct Request {
    source_store: String,
    source_array_path: String,
    source_array_spec: common::ArraySpec,
    target_store: String,
    target_array_path: String,
    target_array_spec: common::ArraySpec,
    copy_chunk_shape: Vec<u64>,
    copy_chunk_coord: Vec<u64>,
}

struct ArrayLocation {
    store_url: String,
    array_path: String,
    spec: common::ArraySpec,
}

pub(crate) async fn function_handler(event: LambdaEvent<Request>) -> Result<(), Error> {
    let payload = event.payload;

    let source = ArrayLocation {
        store_url: payload.source_store,
        array_path: payload.source_array_path,
        spec: payload.source_array_spec,
    };
    let target = ArrayLocation {
        store_url: payload.target_store,
        array_path: payload.target_array_path,
        spec: payload.target_array_spec,
    };

    copy_chunk(
        &source,
        &target,
        &payload.copy_chunk_shape,
        &payload.copy_chunk_coord,
    )
    .await?;

    Ok(())
}

async fn copy_chunk(
    source: &ArrayLocation,
    target: &ArrayLocation,
    copy_chunk_shape: &[u64],
    copy_chunk_coord: &[u64],
) -> Result<(), common::Error> {
    let source = common::open_array_from_spec(&source.store_url, &source.array_path, &source.spec)?;
    let target = common::open_array_from_spec(&target.store_url, &target.array_path, &target.spec)?;
    let shape = target.shape().to_vec();

    let start: Vec<u64> = copy_chunk_coord
        .iter()
        .zip(copy_chunk_shape)
        .map(|(c, s)| c * s)
        .collect();
    let subset =
        ArraySubset::new_with_start_shape(start, copy_chunk_shape.to_vec())?.bound(&shape)?;

    let data: ArrayBytes = source.async_retrieve_array_subset(&subset).await?;
    target.async_store_array_subset(&subset, data).await?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_copy_chunk_different_grids() {
        // all three chunkings differ, and a copy chunk spans two target chunks
        let shape = vec![4, 6];
        let source_chunks = vec![2, 3]; // 2x2 source chunk grid
        let target_chunks = vec![2, 2]; // 2x3 target chunk grid
        let copy_chunk_shape = vec![4, 2]; // 1x3 copy chunk grid; each spans 2 target chunks

        let source_dir = tempfile::tempdir().unwrap();
        let source_store_url = format!("file://{}", source_dir.path().display());
        let (source, source_spec) =
            common::create_test_array(&source_store_url, &shape, &source_chunks, "uint32")
                .await
                .unwrap();

        // fill source, keyed by *source's own* chunk grid
        for coord in [vec![0u64, 0], vec![0, 1], vec![1, 0], vec![1, 1]] {
            let elements = common::hashed_chunk_elements(&shape, &source_chunks, &coord);
            source.async_store_chunk(&coord, elements).await.unwrap();
        }

        let target_dir = tempfile::tempdir().unwrap();
        let target_store_url = format!("file://{}", target_dir.path().display());
        let (target, target_spec) =
            common::create_test_array(&target_store_url, &shape, &target_chunks, "uint32")
                .await
                .unwrap();

        let source_location = ArrayLocation {
            store_url: source_store_url,
            array_path: "/".to_string(),
            spec: source_spec,
        };
        let target_location = ArrayLocation {
            store_url: target_store_url,
            array_path: "/".to_string(),
            spec: target_spec,
        };
        // one invocation per copy chunk
        for copy_chunk_coord in [vec![0u64, 0], vec![0, 1], vec![0, 2]] {
            copy_chunk(
                &source_location,
                &target_location,
                &copy_chunk_shape,
                &copy_chunk_coord,
            )
            .await
            .unwrap();
        }

        // the hash depends only on position, so this holds whatever the chunkings
        for coord in [
            vec![0u64, 0],
            vec![0, 1],
            vec![0, 2],
            vec![1, 0],
            vec![1, 1],
            vec![1, 2],
        ] {
            let expected = common::hashed_chunk_elements(&shape, &target_chunks, &coord);
            let actual: Vec<u32> = target.async_retrieve_chunk(&coord).await.unwrap();
            assert_eq!(actual, expected, "mismatch at target chunk {coord:?}");
        }
    }
}
