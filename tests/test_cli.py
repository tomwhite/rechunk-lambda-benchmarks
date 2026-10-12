import pytest
from click.testing import CliRunner
from map_lambda import LOCAL_ENDPOINT_URL

from rechunk_lambda_benchmarks.cli import cli

runner = CliRunner()


def invoke(*args):
    result = runner.invoke(cli, [*args, "--endpoint-url", LOCAL_ENDPOINT_URL])
    assert result.exit_code == 0, result.output
    return result.output


def test_plan():
    result = runner.invoke(
        cli,
        [
            "plan",
            "--shape", "2480,721,1440",
            "--source-chunks", "31,721,1440",
            "--target-chunks", "2480,10,10",
            "--dtype", "float32",
            "--memory", "1769",
        ],
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    assert "2 copy operation(s)" in result.output


@pytest.mark.lambda_integration
def test_hashed_integers_rechunk_validate_pipeline(tmp_path):
    source = str(tmp_path / "source")
    target = str(tmp_path / "target")

    invoke("hashed-integers", source, "--shape", "12,12", "--chunks", "3,3")
    assert "successfully validated" in invoke("validate-hashed-integers", source)

    output = invoke("rechunk", source, target, "--target-chunks", "2,2")
    assert "dispatching 1 copy chunks" in output
    assert "(3, 3) -> " in output

    assert "successfully validated" in invoke("validate-hashed-integers", target)
