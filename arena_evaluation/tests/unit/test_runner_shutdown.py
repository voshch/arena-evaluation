import signal
import subprocess
import sys
import textwrap
import time

_SCRIPT = textwrap.dedent(
    """
    import asyncio
    import sys
    import time

    from arena_rclpy_mixins import ArenaMixinNode
    from arena_evaluation.benchmark.runner import BenchmarkRunner


    class Runner(BenchmarkRunner):
        shutdown_deadline_s = 3.0

        def __init__(self):
            ArenaMixinNode.__init__(self, "runner_shutdown_test")
            self._progress = None

        async def setup(self):
            print("running", flush=True)
            await asyncio.sleep(3600)

        async def _shutdown_arena(self):
            if sys.argv[1] == "hang":
                time.sleep(3600)


    Runner.run_main()
    print("returned", flush=True)
    time.sleep(5.0)
    sys.exit(Runner.exit_code)
    """
)


def _start(mode: str) -> subprocess.Popen[str]:
    process = subprocess.Popen([sys.executable, "-c", _SCRIPT, mode], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    assert process.stdout is not None
    assert process.stdout.readline().strip() == "running"
    return process


def test_hung_shutdown_exits_at_the_deadline_with_a_stack_dump() -> None:
    process = _start("hang")
    start = time.monotonic()
    process.send_signal(signal.SIGTERM)
    _, err = process.communicate(timeout=60.0)
    assert process.returncode == 1
    assert time.monotonic() - start < 30.0
    assert "Timeout (0:00:03)!" in err
    assert "_shutdown_arena" in err


def test_clean_shutdown_disarms_the_deadline() -> None:
    process = _start("clean")
    process.send_signal(signal.SIGINT)
    out, err = process.communicate(timeout=60.0)
    assert "returned" in out
    assert "Timeout" not in err
    assert process.returncode == 130
