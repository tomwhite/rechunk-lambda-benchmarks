use serde::Deserialize;
use std::sync::Arc;
use url::Url;
use zarrs::array::{Array, ArrayMetadata, ArrayMetadataV3};
use zarrs::metadata::v3::{FillValueMetadataV3, MetadataV3};
use zarrs::metadata_ext::chunk_grid::regular::RegularChunkGridConfiguration;
use zarrs_object_store::AsyncObjectStore;

pub type Error = Box<dyn std::error::Error + Send + Sync>;

/// An object_store (local for tests and `cargo lambda watch`, S3 in
/// production) scoped to `store_url`'s prefix, for zarrs.
pub type Storage =
    AsyncObjectStore<object_store::prefix::PrefixStore<Box<dyn object_store::ObjectStore>>>;

/// Storage for `store_url`, with no I/O.
fn build_storage(store_url: &str) -> Result<Arc<Storage>, Error> {
    let url = Url::parse(store_url)?;
    // after the environment, so this wins: `parse_url_opts` folds the
    // options in order and the last value for a key is the one kept
    let options = std::env::vars().chain(read_timeout_option());
    let (object_store, prefix) = object_store::parse_url_opts(&url, options)?;
    let object_store = object_store::prefix::PrefixStore::new(object_store, prefix);
    Ok(Arc::new(AsyncObjectStore::new(object_store)))
}

/// How long a request may go without progress before object_store retries it.
const READ_TIMEOUT_ENV: &str = "RECHUNK_LAMBDA_READ_TIMEOUT";
const DEFAULT_READ_TIMEOUT: &str = "5s";

fn read_timeout_option() -> Option<(String, String)> {
    let timeout = std::env::var(READ_TIMEOUT_ENV).unwrap_or_else(|_| DEFAULT_READ_TIMEOUT.into());
    (!timeout.is_empty()).then(|| ("read_timeout".to_string(), timeout))
}

/// The array-to-bytes codec every array here uses, hardcoded; see [`ArraySpec`].
const SERIALIZER_NAME: &str = "bytes";
const SERIALIZER_ENDIAN: &str = "little";

/// A sharded array's codec, with the serializer inside it; the chunk grid
/// then describes the shard and the codec's `chunk_shape` the chunk.
const SHARDING_CODEC_NAME: &str = "sharding_indexed";

/// Hardcoded here and in `rechunker_lambda/spec.py`. The index must agree
/// exactly: written differently, each side would read the other's shards
/// as the wrong bytes rather than fail.
const SHARD_INDEX_LOCATION: &str = "end";
const SHARD_INDEX_CHECKSUM: &str = "crc32c";

/// The allowlisted subset of Zarr v3 array metadata sent with every
/// invocation, so a lambda need not GET zarr.json.
///
/// Every other field must take a default hardcoded identically here and
/// in rechunker_lambda/spec.py - keep the two in sync:
///
/// | field | value |
/// |---|---|
/// | `zarr_format` / `node_type` | `3` / `"array"` |
/// | `chunk_key_encoding` | `default`, separator `"/"` |
/// | `codecs[0]` (serializer) | `bytes`, `endian: "little"` |
/// | sharding `index_codecs` | `bytes` little-endian, then `crc32c` |
/// | sharding `index_location` | `"end"` |
/// | `attributes` | `{}` |
/// | `storage_transformers` | `[]` |
/// | `dimension_names` | absent |
/// | additional fields | none |
///
/// All but the sharding rows are [`ArrayMetadataV3::new`]'s own defaults.
/// `test_spec_matches_zarr_python_fixtures` checks the table against what
/// zarr-python writes.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ArraySpec {
    shape: Vec<u64>,
    chunks: Vec<u64>,
    #[serde(default)]
    shards: Option<Vec<u64>>,
    data_type: MetadataV3,
    fill_value: FillValueMetadataV3,
    #[serde(default)]
    compressors: Vec<MetadataV3>,
}

impl ArraySpec {
    /// Expand the spec into full Zarr v3 array metadata, filling in the
    /// shared defaults. The counterpart of
    /// `rechunker_lambda.spec.metadata_from_spec` on the Python side.
    pub fn to_metadata(&self) -> Result<ArrayMetadataV3, Error> {
        // the grid describes whatever unit becomes one object: the
        // shard if there is one, the chunk otherwise
        let grid_shape = self.shards.as_ref().unwrap_or(&self.chunks);
        let chunk_shape = grid_shape
            .iter()
            .map(|&c| {
                std::num::NonZeroU64::new(c).ok_or_else(|| -> Error {
                    format!("chunk shape {grid_shape:?} has a zero").into()
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let chunk_grid = MetadataV3::new_with_serializable_configuration(
            "regular".to_string(),
            &RegularChunkGridConfiguration { chunk_shape },
        )?;

        let serializer = MetadataV3::new_with_serializable_configuration(
            SERIALIZER_NAME.to_string(),
            &serde_json::json!({ "endian": SERIALIZER_ENDIAN }),
        )?;
        let inner: Vec<MetadataV3> = std::iter::once(serializer)
            .chain(self.compressors.iter().cloned())
            .collect();
        let codecs = if self.shards.is_some() {
            // compression stays inside the shard, where zarr-python
            // puts it, so the sharding codec is the whole of `codecs`
            vec![MetadataV3::new_with_serializable_configuration(
                SHARDING_CODEC_NAME.to_string(),
                &serde_json::json!({
                    "chunk_shape": self.chunks,
                    "codecs": inner,
                    "index_codecs": [
                        { "name": SERIALIZER_NAME,
                          "configuration": { "endian": SERIALIZER_ENDIAN } },
                        { "name": SHARD_INDEX_CHECKSUM },
                    ],
                    "index_location": SHARD_INDEX_LOCATION,
                }),
            )?]
        } else {
            inner
        };

        Ok(ArrayMetadataV3::new(
            self.shape.clone(),
            chunk_grid,
            self.data_type.clone(),
            self.fill_value.clone(),
            codecs,
        ))
    }
}

/// Chunks one invocation reads or writes at a time.
const DEFAULT_CHUNK_CONCURRENCY: usize = 4;

/// Environment variable overriding [`DEFAULT_CHUNK_CONCURRENCY`]
const CHUNK_CONCURRENCY_ENV: &str = "RECHUNK_LAMBDA_CHUNK_CONCURRENCY";

/// Applied once per process: zarrs' configuration is global.
fn configure_zarrs() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        let concurrency = std::env::var(CHUNK_CONCURRENCY_ENV)
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .filter(|value| *value > 0)
            .unwrap_or(DEFAULT_CHUNK_CONCURRENCY);
        zarrs::config::global_config_mut().set_chunk_concurrent_minimum(concurrency);
    });
}

/// The array at `array_path` in `store_url`, built from `spec` with no
/// I/O: the caller created the array, so there is no metadata GET to pay
/// per invocation.
pub fn open_array_from_spec(
    store_url: &str,
    array_path: &str,
    spec: &ArraySpec,
) -> Result<Array<Storage>, Error> {
    configure_zarrs();
    let storage = build_storage(store_url)?;
    Ok(Array::new_with_metadata(
        storage,
        array_path,
        ArrayMetadata::V3(spec.to_metadata()?),
    )?)
}

/// Test support: create an uncompressed array with fill value 0, and
/// return it with its [`ArraySpec`].
pub async fn create_test_array(
    store_url: &str,
    shape: &[u64],
    chunks: &[u64],
    data_type: &str,
) -> Result<(Array<Storage>, ArraySpec), Error> {
    let spec: ArraySpec = serde_json::from_value(serde_json::json!({
        "shape": shape,
        "chunks": chunks,
        "data_type": data_type,
        "fill_value": 0,
        "compressors": [],
    }))?;
    let array = open_array_from_spec(store_url, "/", &spec)?;
    array.async_store_metadata().await?;
    Ok((array, spec))
}

/// Row-major strides for `shape`.
fn strides(shape: &[u64]) -> Vec<u64> {
    let mut strides = vec![1u64; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

/// Wang's hash (<https://burtleburtle.net/bob/hash/integer.html>).
fn wangs_hash(x: u32) -> u32 {
    let x = (x ^ 61) ^ (x >> 16);
    let x = x.wrapping_add(x << 3);
    let x = x ^ (x >> 4);
    let x = x.wrapping_mul(0x27d4_eb2d);
    x ^ (x >> 15)
}

/// The elements of the region at `start` with shape `region_shape`, in
/// row-major order, each a hash of its row-major offset in the whole
/// array - so a position hashes the same however the array is chunked.
pub fn hashed_elements(shape: &[u64], start: &[u64], region_shape: &[u64]) -> Vec<u32> {
    let array_strides = strides(shape);
    let region_strides = strides(region_shape);
    let num_elements = region_shape.iter().product::<u64>();

    (0..num_elements)
        .map(|local_flat| {
            let mut remaining = local_flat;
            let mut global_offset = 0u64;
            for axis in 0..region_shape.len() {
                let local_coord = remaining / region_strides[axis];
                remaining %= region_strides[axis];
                let global_coord = start[axis] + local_coord;
                global_offset += global_coord * array_strides[axis];
            }
            wangs_hash(global_offset as u32)
        })
        .collect()
}

/// `hashed_elements` for the whole nominal chunk `chunk_indices`,
/// unclipped.
pub fn hashed_chunk_elements(
    shape: &[u64],
    chunk_shape: &[u64],
    chunk_indices: &[u64],
) -> Vec<u32> {
    let start: Vec<u64> = chunk_indices
        .iter()
        .zip(chunk_shape)
        .map(|(i, s)| i * s)
        .collect();
    hashed_elements(shape, &start, chunk_shape)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spec_rejects_unknown_fields() {
        // an unknown field means the two sides have drifted
        let result: Result<ArraySpec, _> = serde_json::from_value(serde_json::json!({
            "shape": [4, 4],
            "chunks": [2, 2],
            "data_type": "int32",
            "fill_value": 0,
            "compressors": [],
            "chunk_key_encoding": {"name": "default", "configuration": {"separator": "."}},
        }));
        assert!(result.is_err());
    }

    #[test]
    fn test_hashed_chunk_elements() {
        // cross-checked against known values
        let elements = hashed_chunk_elements(&[2, 6], &[2, 3], &[0, 1]);
        assert_eq!(
            elements,
            vec![2278584254, 3427349084, 3322605197, 454550906, 1415330532, 1798297286]
        );
    }

    #[test]
    fn test_hashed_chunk_elements_independent_of_chunking() {
        let shape = &[4, 6];
        let one_big_chunk = hashed_chunk_elements(shape, &[4, 6], &[0, 0]);
        let small_chunks: Vec<u32> = (0..2)
            .flat_map(|bi| {
                (0..3).flat_map(move |bj| hashed_chunk_elements(shape, &[2, 2], &[bi, bj]))
            })
            .collect::<Vec<_>>();
        // reassemble small_chunks (row-major within each 2x2 block, blocks
        // in row-major order) into the same 4x6 row-major layout and compare
        let mut reassembled = vec![0u32; 24];
        let mut idx = 0;
        for bi in 0..2u64 {
            for bj in 0..3u64 {
                for li in 0..2u64 {
                    for lj in 0..2u64 {
                        let gi = bi * 2 + li;
                        let gj = bj * 2 + lj;
                        reassembled[(gi * 6 + gj) as usize] = small_chunks[idx];
                        idx += 1;
                    }
                }
            }
        }
        assert_eq!(one_big_chunk, reassembled);
    }
}
