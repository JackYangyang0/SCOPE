"""Process probes must decode Windows tool output without crashing.

Windows tools write in the OEM code page (GBK on a Chinese system); Python 3.15
enables UTF-8 mode by default.  Decoding strictly therefore fails on the
"no matching tasks" message, and the failure surfaces as a traceback printed from
a subprocess reader *thread* while ``stdout`` arrives as ``None``.
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
import unittest

from SCOPE.utils import harness_util as util


class DecodingTest(unittest.TestCase):
    def test_child_writing_gbk_bytes_does_not_break_run_command(self):
        # 0xd0 0xde is the start of a GBK character ("信" in 信息: ...).
        code = "import sys; sys.stdout.buffer.write(b'\\xd0\\xde\\xd0\\xde'); sys.stdout.flush()"
        result = util.run_command([sys.executable, "-c", code], timeout=60.0)
        self.assertEqual(result["returncode"], 0)
        self.assertIsNotNone(result["stdout"])

    def test_run_text_replaces_undecodable_bytes(self):
        code = "import sys; sys.stdout.buffer.write(b'\\xd0\\xde\\xd0\\xde'); sys.stdout.flush()"
        proc = util.run_text([sys.executable, "-c", code], timeout=60.0)
        self.assertEqual(proc.returncode, 0)
        self.assertIsInstance(proc.stdout, str)
        self.assertNotEqual(proc.stdout, "")


class ProcessProbeTest(unittest.TestCase):
    def test_pid_alive_sees_itself(self):
        self.assertTrue(util.pid_alive(os.getpid()))

    def test_start_token_is_stable_and_present_for_this_process(self):
        first = util.process_start_token(os.getpid())
        second = util.process_start_token(os.getpid())
        self.assertTrue(first, "a live process must have a start token")
        self.assertEqual(first, second)

    def test_dead_pid_is_reported_dead_without_a_thread_exception(self):
        """The exact regression: a thread traceback must not appear for a dead pid."""
        child = subprocess.Popen([sys.executable, "-c", "pass"])
        child.wait()
        time.sleep(0.2)
        caught: list[type[BaseException]] = []
        original = threading.excepthook
        threading.excepthook = lambda args: caught.append(args.exc_type)
        try:
            alive = util.pid_alive(child.pid)
        finally:
            threading.excepthook = original
        self.assertFalse(alive)
        self.assertEqual(caught, [], f"reader thread raised: {caught}")
        self.assertIsNone(util.process_start_token(child.pid))

    def test_impossible_pid_is_not_alive(self):
        self.assertFalse(util.pid_alive(2 ** 31 - 1))


if __name__ == "__main__":
    unittest.main()
