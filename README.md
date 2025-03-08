## PCGestureControl

## Autor

- **Alicja Kuszpit**  

## Opis projektu 

PCHandControl to system umożliwiający sterowanie komputerem za pomocą gestów dłoni. Wykorzystuje technologie rozpoznawania gestów z kamery, aby móc wykonać podstawowe
czynności na komputerze, takie jak poruszanie myszką (gest otwartej dłoni z połączonymi palcami), klikanie prawym przyciskiem myszy (gest zaciskania pięści) oraz scrollowanie
do góry (gest kciuka do góry). Dane treningowe zostały zebrane za pomocą programu, który rejestrował obrazy z kamery i zapisywał kluczowe informacje o położeniu dłoni.
System przechwytywał współrzędne punktów charakterystycznych dłoni (landmarków) i przekształcał je w zestawy danych do uczenia modelu.

## Wykorzystywane technologie

- **OpenCV**,
- **MediaPipe Hands**,
- **TensorFlow/Keras**,
- **NumPy**,
- **cv2 (OpenCV)**.
