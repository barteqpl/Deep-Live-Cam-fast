# Dokumentacja modeli Face Swap i wydajności CoreML (Apple M4 Pro)

Wprowadziliśmy zaawansowane wsparcie dla modeli 256px oraz zoptymalizowaliśmy generowanie masek erozyjno-blendujących bezpośrednio w wycinku twarzy. Poniżej znajduje się opis techniczny modeli, rekomendacje oraz wyniki testów wydajnościowych.

---

## 1. Dostępne modele i rekomendacje

W projekcie wspieramy cztery główne modele podmieniające twarz. Wyboru modelu dokonuje się za pomocą parametru `--swapper-model` (np. `--swapper-model simswap`).

| Model | Rozdzielczość | Narzut wydajnościowy | Wymaga konwertera | Rekomendowane zastosowanie |
| :--- | :---: | :---: | :---: | :--- |
| **SimSwap 256** | 256x256 | Niski | Tak | **Najlepszy do pracy na żywo (Live Webcam)**. Najszybszy czas wnioskowania, bardzo dobrze zachowuje ekspresję i kąty nachylenia głowy. |
| **HyperSwap 256** | 256x256 | Średni | Nie | **Zalecany jako domyślny wysokojakościowy**. Świetnie przenosi tożsamość źródła, nie potrzebuje dodatkowych konwerterów, generuje bardzo naturalne krawędzie. |
| **HiFiFace 256** | 256x256 | Średni | Tak | **Najlepszy do statycznego renderowania wideo/zdjęć**. Generuje bardzo miękką, wbudowaną maskę wtapiającą twarz, ale mocno trzyma się geometrii twarzy docelowej (możesz "nadal wyglądać jak Ty"). |
| **Inswapper 128** | 128x128 | Średni | Nie | Domyślny model. Niska rozdzielczość powoduje widoczne rozmycie na nowoczesnych kamerach/wideo HD. |

---

## 2. Statystyki wydajności na Apple M4 Pro (CoreML)

Poniższe statystyki reprezentują średni czas operacji face swap (odczyt klatki, wnioskowanie CoreML, blending i nałożenie maski) dla pliku wideo testowego na Apple M4 Pro:

- **SimSwap 256**: **~41.91 ms** na klatkę (**~23.9 FPS**).
- **Inswapper 128**: **~75.57 ms** na klatkę (**~13.2 FPS**).
- **HiFiFace 256**: **~85.87 ms** na klatkę (**~11.7 FPS**).
- **HyperSwap 256**: **~151.59 ms** na klatkę (**~6.6 FPS**).

> [!NOTE]
> Dzięki zaimplementowanemu buforowaniu maski w [face_swapper.py](file:///Users/barteq/repos/ai/Deep-Live-Cam/modules/processors/frame/face_swapper.py#L65) czas postprocessingu spadł z kilkunastu milisekund do **<0.1 ms**, co pozwala na stabilne utrzymanie klatkażu ograniczonego niemal wyłącznie szybkością samej sieci CoreML.

---

## 3. Automatyczne pobieranie modeli

Brakujące modele są automatycznie pobierane przy pierwszym uruchomieniu programu przez funkcję [pre_check](file:///Users/barteq/repos/ai/Deep-Live-Cam/modules/processors/frame/face_swapper.py#L334) z serwerów Hugging Face:
- Dla **HiFiFace**: pobierany jest plik główny `hififace_unofficial_256.onnx` oraz konwerter osadzeń `crossface_hififace.onnx`.
- Dla **SimSwap**: pobierany jest plik główny `simswap_256.onnx` oraz konwerter osadzeń `crossface_simswap.onnx`.
- Pliki zapisywane są bezpośrednio w folderze `models/` projektu.
