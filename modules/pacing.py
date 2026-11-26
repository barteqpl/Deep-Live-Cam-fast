"""Paced emission for the pipelined live path (jitter buffer).

The pool finishes frames unevenly: a detect frame takes a few ms longer than a
track frame, so emitting each frame the moment it is finalized shows up as
stutter. PacedEmitter is a small jitter buffer: each frame gets a playout slot
on a steady cadence (EMA of completion intervals), kept about ``delay_k`` x
the measured jitter behind the arrivals, so a late frame still meets its slot.
The schedule follows the arrivals with a slow phase-locked correction
(``beta``), so it cannot drift away. A frame is never emitted before it
arrives and never held more than ``max_hold`` deep, so pacing does not lower
throughput and adds only a few ms of latency.

Emission runs on its own thread, so slots are honoured even while the
processing thread is busy with detection or prepare().
"""
from collections import deque
import threading
import time
from typing import Any, Callable, Optional


class PacedEmitter:
    max_hold, alpha, beta, delay_k = 3, 0.1, 0.05, 2.0

    def __init__(self, sink: Callable[[Any], None]):
        self._sink = sink
        self._q: deque = deque()          # (slot, item)
        self._cv = threading.Condition()
        self._ema: Optional[float] = None   # mean completion interval
        self._jit: float = 0.0              # mean |interval - ema|
        self._last_in: Optional[float] = None
        self._last_slot: Optional[float] = None
        self._stop = False
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name="paced-emitter")
        self._thread.start()

    def push(self, item: Any) -> None:
        now = time.perf_counter()
        with self._cv:
            if self._last_in is not None:
                dt = now - self._last_in
                if self._ema is not None and dt > max(4 * self._ema, 0.25):
                    # Gap in arrivals (pool paused, first inference): restart
                    # the estimate instead of pacing on a poisoned EMA.
                    self._ema, self._jit, self._last_slot = None, 0.0, None
                elif self._ema is None:
                    self._ema = dt
                else:
                    self._jit += self.alpha * (abs(dt - self._ema) - self._jit)
                    self._ema += self.alpha * (dt - self._ema)
            self._last_in = now
            target = now + self.delay_k * self._jit
            if self._last_slot is None or self._ema is None:
                slot = target
            else:
                nominal = self._last_slot + self._ema
                slot = nominal + self.beta * (target - nominal)
            # Never before arrival, never further ahead than one interval
            # past the jitter delay.
            slot = min(max(slot, now), target + (self._ema or 0.0))
            self._last_slot = slot
            self._q.append((slot, item))
            self._cv.notify()

    def _run(self) -> None:
        while True:
            with self._cv:
                while not self._q and not self._stop:
                    self._cv.wait()
                if not self._q:
                    return                  # stopped and drained
                slot, item = self._q[0]
                now = time.perf_counter()
                if now < slot and len(self._q) <= self.max_hold:
                    self._cv.wait(timeout=slot - now)
                    continue
                self._q.popleft()
            try:
                self._sink(item)
            except Exception as e:  # a dead emitter would freeze the preview
                print(f"[pacing] sink failed: {e}")

    def close(self, flush: bool = True) -> None:
        """Stop the thread. With flush, held frames are still emitted on schedule."""
        with self._cv:
            self._stop = True
            if not flush:
                self._q.clear()
            self._cv.notify()
        self._thread.join(timeout=2.0)
