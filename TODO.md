# TODO: Dual-session pipelined live swap (branch `feat/dual-session-pipeline`)

> **STATUS 2026-07-10 (po implementacji Fazy 1+2):** Faza 1 zrobiona i zweryfikowana
> (pool w ui.py + bench_live --pool). Zmierzone: serial 14.2 FPS, pool ANE-only
> 14.75 (+4% — GIL ogranicza, patrz 3.2), **pool dual ANE+GPU 18.3-18.9 FPS (+29%)**.
> WAŻNA KOREKTA: raw bench (bench_dual_session, +11-17%) ZANIŻA zysk dualu w
> pipeline — druga sesja GPU absorbuje klatki, gdy CPU-work workera ANE trzyma GIL.
> Dlatego `dual_session` jest default **True** (bramka 2.2 spełniona: 18.3 ≥ 16).
> **2.3 WYKONANE NA KAMERZE — FAIL (2026-07-10):** dual daje wyższe średnie FPS,
> ale subiektywnie obraz "skacze" i aplikacja wydaje się wolna. Root cause:
> WARIANCJA FRAME-TIME, nie throughput — klatka GPU (~150 ms) blokuje emisję
> 2-3 gotowych klatek ANE (~56 ms) w buforze kolejności; potem burst wpada do
> processed_queue (maxsize=2, drop-oldest) i klatki lecą na podłogę → szarpany
> ruch + 200-400 ms latencji. Dlatego dual i pool są z powrotem OPT-IN
> (`--dual-session`); domyślna ścieżka = stabilny serial.
>
> Jak to naprawić (dla przejmującego, w kolejności opłacalności):
> 1. **Paced emission**: emituj z bufora kolejności w stałym takcie
>    (np. EMA frame-time), nie natychmiast po skompletowaniu — kosztem
>    +1 klatki latencji wygładza burst. Wymaga też processed_queue maxsize
>    ~4 dla trybu pool.
> 2. **Fixed-phase GPU**: przydzielaj GPU co K-tą klatkę (K=3, stała faza)
>    zamiast greedy — przewidywalny wzorzec zamiast losowych bąbli.
> 3. **Faza 3.2 (GIL relief)**: przenieś align/color/paste za collect —
>    wtedy ANE-only pool (bez GPU, bez wariancji 3x) może dać realny zysk
>    i problem pacingu znika u źródła. NAJLEPSZA ścieżka długoterminowo.
> Faza 2.4 (soak) dopiero po naprawieniu pacingu.

> Handoff-ready plan. Wszystkie liczby zmierzone na M4 Pro, macOS 26.x, ORT 1.27,
> model `hyperswap_1b_256.onnx`, klip testowy 960x540. Assety testowe robisz tak:
> `ffmpeg -i media/ludwig.gif -frames:v 1 /tmp/source_face.jpg` oraz
> `ffmpeg -i media/live_show.gif -vf scale=960:540 -pix_fmt yuv420p /tmp/target_clip.mp4`.

## Cel

Podnieść live FPS dla hyperswap przez (a) nakładanie detekcji/post-processingu
klatki N+1 na inference klatki N (pipelining etapów) i (b) drugą sesję ONNX na
GPU obok ANE. **Uczciwe oczekiwanie: 14.4 → ~17-19 FPS** (nie teoretyczne 23 —
patrz "Zmierzone fakty").

## Zmierzone fakty (nie podważać, zmierzone w tej sesji)

| Pomiar | Wynik | Narzędzie |
|---|---|---|
| bench_live hyperswap-1b (baseline, sekwencyjny) | 14.4 FPS (infer 55.8 ms) | `benchmarks/bench_live.py` |
| raw inference: single ANE serial | 19.5 swaps/s | `benchmarks/bench_dual_session.py` |
| raw inference: single GPU serial | 6.6-7.1 swaps/s (~150 ms!) | jw. |
| raw inference: ANE+GPU greedy, 2 wątki | 22.9 swaps/s (+17.6%), split 43:17 | jw. |
| SwapperPool przez pełny `swap_face()`: ANE-only | 14.8 FPS swap-stage | smoke test (opis niżej) |
| SwapperPool przez pełny `swap_face()`: ANE+GPU | 17.3 FPS swap-stage (+17%) | jw. |
| Detekcja (fast path) | ~19-23 ms co 3. klatkę; track ~0.6 ms | bench_live |
| PSNR wyjścia ANE vs GPU (ta sama klatka) | 57 dB — miks sesji klatka-po-klatce niewidoczny | bench sesji 2026-07-10 |

Wnioski z liczb:
- GPU jest 2.5x wolniejszy od ANE i trochę kontenduje (unified memory) — druga
  sesja dodaje tylko ~17%, nie podwaja.
- Główny zysk to NAKŁADANIE etapów: dziś detekcja (19-23 ms co 3. klatkę) i post
  blokują wątek między inference'ami. Po pipeliningu FPS ≈ throughput swap-stage.
- Sufit: ANE-only pipeline ~17-18 FPS, ANE+GPU ~19-21 FPS (17.3 swap-stage + detekcja
  nakładana, minus koszt reorderingu).

## Co już jest zrobione (fundamenty na tym branchu)

1. **`modules/swapper_pool.py`** — działający `SwapperPool`:
   - N instancji `HyperSwapSwapper`, każda z WŁASNĄ sesją (ANE + opcjonalnie GPU)
     i własnym lockiem; greedy dispatch ze wspólnej kolejki.
   - `try_submit()` **nieblokujące** (zwraca None gdy pełno) + `collect_ready()`
     emitujące wyniki W KOLEJNOŚCI (min-heap reordering) + `pending()`.
   - Worker woła `swap_face(..., swapper=instancja)` — więc mouth mask, poisson,
     opacity działają identycznie jak w ścieżce sekwencyjnej.
   - Smoke-tested: 40/40 klatek, kolejność zachowana, brak deadlocka.
2. **`face_swapper.py`**: `HyperSwapSwapper` ma per-instance `self.lock` zamiast
   globalnego `THREAD_LOCK` (globalny lock serializowałby obie sesje do zera zysku);
   `swap_face()` przyjmuje opcjonalny parametr `swapper=`.
3. **`benchmarks/bench_dual_session.py`** — walidacja współbieżności ANE+GPU na
   syntetycznych blobach; próg opłacalności wypisuje werdykt.

## GOTCHAS — przeczytaj zanim ruszysz kod

- **DEADLOCK (naprawiony, nie reintrodukuj):** blokujący `submit()` + reorder heap
  = zakleszczenie, gdy szybkie klatki ANE czekają w heapie za wolną klatką GPU,
  semafor wyczerpany, a caller nie dochodzi do `collect_ready()`. Dlatego
  `try_submit()` jest nieblokujące. Integracja w ui.py MUSI drenować
  `collect_ready()` w tej samej pętli, w której submituje.
- **FaceTracker jest stanowy i ściśle sekwencyjny** (optical flow na poprzedniej
  klatce). Detekcja/tracking NIE wchodzi do poola — zostaje w wątku
  przetwarzającym, przed `try_submit`.
- **Interpolacja i sharpening** (`apply_post_processing`) muszą dostawać klatki
  W KOLEJNOŚCI — wołać po `collect_ready()`, nigdy w workerach.
- **GIL:** `swap_face` robi ~6-12 ms pracy CPU (align, color match, ROI paste).
  Dwa workery + wątek główny kontendują. Jeśli zysk niższy niż 17.3 FPS
  swap-stage, sprawdź `cv2.setNumThreads(1)` w workerach i/lub przenieś
  `_color_match` za collect.
- **E5RT "MILCompilerForANE failed" przy ładowaniu sesji hyperswap** — pojawia się
  raz per sesja (także na main branchu!). Model MIMO TO działa (część grafu spada
  z ANE). Osobny wątek do zbadania (Faza 3) — jak się to naprawi, ANE może być szybszy.
- **Pamięć:** druga sesja = +~400 MB (drugi załadowany model 384 MB). Akceptowalne
  na M4 Pro; nie ładuj GPU-sesji gdy `use_gpu_session=False`.
- **Zmiana modelu w UI** (`on_model_change`) musi wołać `pool.stop()` i zbudować
  nowy pool — patrz krok 1.6.
- Model-switch crash spam był już raz naprawiany (commit 5ff56dc) — przetestuj
  przełączanie modeli po integracji.

## Faza 1 — pipelining z pulą (najpierw ANE-only) [~pół dnia]

Cel: przebudować `ui.py:_processing_thread_func` na submit/collect. Gate wejścia:
brak. Oczekiwany wynik: hyperswap-1b **14.4 → ~16-17 FPS**.

- [ ] 1.1 W `_processing_thread_func` dodaj gałąź `use_pool` aktywną, gdy
      `modules.globals.swapper_model.startswith("hyperswap")` i nie `map_faces`
      i nie `many_faces` (pierwsza iteracja: tylko single-face fast path).
      Pool tworzony lazy przy pierwszej klatce:
      `SwapperPool(model_path, use_gpu_session=modules.globals.dual_session)`.
      Dodaj do `modules/globals.py`: `dual_session: bool = False` (Faza 2 włączy).
- [ ] 1.2 Pętla przetwarzania (kształt docelowy):
      ```
      while not stop_event.is_set():
          # 1. drenaż: każda gotowa klatka -> post_processing -> processed_queue
          for idx, frame, bbox in pool.collect_ready():
              out = fs.apply_post_processing(frame, [bbox] if bbox is not None else [])
              ...FPS overlay, processed_queue.put_nowait z drop-oldest...
          # 2. pobierz nową klatkę z capture_queue (timeout 0.005-0.01)
          # 3. mirror, detekcja/track jak dziś (ensure_landmarks gdy mouth_mask)
          # 4. if pool.try_submit(temp_frame, cached_target_face, source_image) is None:
          #        continue  # pool pełny - następna iteracja znowu drenuje
      ```
      Uwaga: gdy brak twarzy (cached_target_face is None) — puść klatkę wprost do
      processed_queue Z ZACHOWANIEM KOLEJNOŚCI względem klatek w poolu (najprościej:
      wyślij do poola "przez" — dodaj do SwapperPool metodę `submit_passthrough(frame)`
      która wrzuca wynik do heapa bez inference; TODO w swapper_pool.py).
- [ ] 1.3 Dodaj `SwapperPool.submit_passthrough(frame)` (idx z sekwencji, wynik
      od razu na heap) — potrzebne do 1.2 i do klatek z enhancerem-bez-swapa.
- [ ] 1.4 Face enhancer w gałęzi pool: wołaj po collect (na gotowej klatce), nie
      przed submit — inaczej enhancujesz klatkę bez twarzy zamienionej.
- [ ] 1.5 FPS licznik: licz na klatkach EMITOWANYCH (collect), nie submitowanych.
- [ ] 1.6 `on_model_change` + stop preview: `pool.stop()` (idempotentne). Pool
      trzymany w zmiennej lokalnej wątku + rejestr globalny do zatrzymania.
- [ ] 1.7 Latencja: `max_in_flight=3` daje +1-2 klatki opóźnienia (~70-140 ms).
      Zweryfikuj wizualnie w podglądzie; jak za dużo — `max_in_flight=2`.
- [ ] 1.8 bench_live: dodaj `--pool` replikujące pętlę 1.2 (submit/collect) tak,
      żeby dało się mierzyć headless bez kamery. Oczekiwane: ≥16 FPS ANE-only.
- [ ] 1.9 Regresja: `--model simswap` (bez poola) FPS bez zmian; mouth mask w
      gałęzi pool działa (landmarki propagowane przez tracker — już zrobione).

## Faza 2 — druga sesja GPU [~1-2h]

Gate: Faza 1 stabilna (brak crashy CoreML przy dłuższym runie ≥5 min, FPS ≥16).

- [ ] 2.1 `modules.globals.dual_session = True` jako default dla hyperswap* +
      CLI `--dual-session/--no-dual-session`.
- [ ] 2.2 bench_live --pool z dual: oczekiwane **~18-20 FPS**. Jeśli <16 → zostaw
      default False, zbadaj kontencję (Instruments / powermetrics: zajętość ANE+GPU).
- [ ] 2.3 Test wizualny miksu sesji: PSNR ANE vs GPU = 57 dB, ale sprawdź na
      żywym podglądzie migotanie (kolor/odcień co ~3. klatka). Jeśli widać:
      przypnij GPU-klatki co stałą fazę albo dodaj temporalny color-lock.
- [ ] 2.4 Dłuższy soak test: 15 min live, obserwuj RSS (leak bufora heapa?) i
      thermal throttling (FPS po 10 min).

## Faza 3 — opcjonalne śruby [gate: Faza 2 shipped]

- [ ] 3.1 Zbadaj E5RT `MILCompilerForANE failed` przy ładowaniu hyperswap —
      `ORT_LOGGING_LEVEL=VERBOSE`, znajdź który subgraph spada z ANE; może
      wystarczy `onnxsim` albo wycięcie problematycznego node'a. Potencjał: ANE
      19.5 → 22+ swaps/s.
- [ ] 3.2 GIL relief: `cv2.setNumThreads(1)` w workerach; przenieś `_color_match`
      i `_paste_back_roi` za collect (kosztem API poola — wtedy pool zwraca
      (bgr_fake, mask, IM) zamiast gotowej klatki).
- [ ] 3.3 Trzecia sesja? NIE — GPU już ledwo dokłada; CPU-sesja to ~1 s/klatkę.
- [ ] 3.4 Rozszerz pool na many_faces (dziś poza zakresem gałęzi pool).

## Kryteria przerwania (abort criteria)

- bench_live --pool ANE-only < 15 FPS po 1.8 → pipelining nie działa jak
  zmierzono, debuguj zanim pójdziesz w Fazę 2.
- Crash CoreML EP przy dwóch sesjach w długim runie, nienaprawialny per-session
  lockami → zostań przy ANE-only pipeline (nadal +2-3 FPS vs baseline).
- Widoczne migotanie ANE/GPU nieusuwalne w 2.3 → dual tylko jako opt-in flaga.

## Weryfikacja końcowa (definicja "done")

```bash
# raw współbieżność (sanity, ~1 min):
venv/bin/python benchmarks/bench_dual_session.py           # oczekiwane: pool > single +15%
# pipeline headless:
venv/bin/python benchmarks/bench_live.py --source /tmp/source_face.jpg \
    --target /tmp/target_clip.mp4 --frames 120 --model hyperswap-1b --pool
# oczekiwane: >=16 (ANE-only) / >=18 (dual) FPS; baseline bez --pool: 14.4
# regresje:
venv/bin/python benchmarks/bench_live.py ... --model simswap          # bez zmian (~28-35)
venv/bin/python benchmarks/bench_live.py ... --model hyperswap-1b --mouth-mask  # działa, koszt ~1.4%
venv/bin/python -m ruff check modules/ benchmarks/
```
Plus test manualny: live preview, przełączanie modeli w dropdownie (2x tam i
z powrotem), włączanie/wyłączanie mouth mask i enhancera w trakcie.

## Mapa plików

- `modules/swapper_pool.py` — pool (gotowy; dodać `submit_passthrough`, punkt 1.3)
- `modules/ui.py:_processing_thread_func` — integracja (Faza 1, główna praca)
- `modules/globals.py` — `dual_session` flag
- `modules/core.py` — `--dual-session` CLI
- `benchmarks/bench_dual_session.py` — walidacja raw współbieżności (gotowy)
- `benchmarks/bench_live.py` — dodać `--pool` (punkt 1.8)
- `modules/processors/frame/face_swapper.py` — `swap_face(swapper=)`, per-instance
  locki (gotowe; nie ruszać THREAD_LOCK dla pozostałych modeli)
