def test_cli_smoke_import():
    import sys
    sys.path.append('src')
    import vlp.eval.runner as runner
    assert hasattr(runner, 'run_suite')
