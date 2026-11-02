//! map-lambda's dispatcher, built here so this repo's deploy has a binary to ship.

use lambda_runtime::Error;

#[tokio::main]
async fn main() -> Result<(), Error> {
    map_lambda_dispatcher::run().await
}
