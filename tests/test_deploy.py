from click.testing import CliRunner
from map_lambda.deploy import spec

from rechunk_lambda_benchmarks.cli import cli
from rechunk_lambda_benchmarks.deploy import PROJECT, SPEC, memory_sizes
from rechunk_lambda_benchmarks.hashed_integers import HASHED_INTEGERS_FUNCTION, VALIDATE_FUNCTION
from rechunk_lambda_benchmarks.rechunk import COPY_FUNCTION, DELETE_INTERMEDIATE_FUNCTION


def test_the_project_is_named_in_the_spec():
    assert PROJECT == "rechunk-lambda"


def test_the_spec_deploys_every_worker_the_client_calls():
    names = {f.name for f in spec.load(SPEC).functions}
    assert names == {
        COPY_FUNCTION,
        DELETE_INTERMEDIATE_FUNCTION,
        HASHED_INTEGERS_FUNCTION,
        VALIDATE_FUNCTION,
    }


def test_the_spec_builds_from_this_workspace():
    workspace = spec.load(SPEC).workspace
    assert workspace is not None
    assert (workspace / "Cargo.toml").exists()
    assert set(spec.load(SPEC).binaries.values()) <= {
        p.name for p in workspace.iterdir() if (p / "Cargo.toml").exists()
    }


def test_a_rechunk_refuses_a_memory_size_copy_is_not_deployed_at(tmp_path):
    result = CliRunner().invoke(
        cli,
        [
            "rechunk",
            str(tmp_path / "s"),
            str(tmp_path / "t"),
            "--target-chunks",
            "1,1",
            "--memory",
            "999",
        ],
    )

    assert result.exit_code == 2
    sizes = ", ".join(map(str, memory_sizes(COPY_FUNCTION)))
    assert f"copy is deployed at {sizes} MB" in result.output


def test_the_cli_has_map_lambda_s_deploy_commands():
    result = CliRunner().invoke(cli, ["deploy", "--help"])

    assert result.exit_code == 0, result.output
    assert "--allow-bucket BUCKET" in result.output
