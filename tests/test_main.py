from click.testing import CliRunner

from sbdetection.main import run


def test_main():
    runner = CliRunner()
    result = runner.invoke(run, args=['--modelname', 'ML'])
    assert result.exit_code == 0
