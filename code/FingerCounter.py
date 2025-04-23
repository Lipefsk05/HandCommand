import cv2
import mediapipe as mp
import subprocess
import time
from collections import deque

# === Configurações Gerais ===
TEMPO = 0  # Tempo necessário para segurar um gesto
MAO_DOMINANTE = "Right"     # Altere para "Left" se quiser usar a outra mão

# === Inicializações ===
video = cv2.VideoCapture(0)
hand = mp.solutions.hands
Hand = hand.Hands(max_num_hands=2, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

estado = {
    "gesto_anterior": None,
    "tempo_ultimo_gesto": 0,
    "inicio_punho": None
}

# === Ações ===
def encerrar_programa():
    print("👊 Mão fechada por tempo - ENCERRANDO programa.")
    video.release()
    cv2.destroyAllWindows()
    exit()

def abrir_github():
    print("BIGODE - ABRINDO github.")
    subprocess.Popen("xdg-open https://github.com/Lipefsk05", shell=True)

def abrir_terminal():
    print("Mindinhos - ABRINDO TERMINAL!")
    subprocess.Popen("gnome-terminal", shell=True)

# === Funções de detecção ===
def detectar_dedos_direita(pontos):
    dedos_ids = [4, 8, 12, 16, 20]
    dedos_levantados = [False] * 5

    if pontos[4][0] < pontos[2][0]:  # Polegar
        dedos_levantados[0] = True

    for i in range(1, 5):
        if pontos[dedos_ids[i]][1] < pontos[dedos_ids[i] - 2][1]:
            dedos_levantados[i] = True

    return tuple(dedos_levantados)

def detectar_dedos_esquerda(pontos):
    dedos_ids = [4, 8, 12, 16, 20]
    dedos_levantados = [False] * 5

    # Polegar (esquerda aponta para direita na imagem espelhada)
    if pontos[4][0] > pontos[2][0]:
        dedos_levantados[0] = True

    # Outros dedos (mesma lógica)
    for i in range(1, 5):
        if pontos[dedos_ids[i]][1] < pontos[dedos_ids[i] - 2][1]:
            dedos_levantados[i] = True

    return tuple(dedos_levantados)


def identificar_maos(results, img, w, h):
    maos_identificadas = {}
    if results.multi_hand_landmarks and results.multi_handedness:
        for idx, classificacao in enumerate(results.multi_handedness):
            label = classificacao.classification[0].label  # 'Left' ou 'Right'
            landmarks = results.multi_hand_landmarks[idx]
            maos_identificadas[label] = landmarks

            # Desenhar landmarks
            mpDraw.draw_landmarks(img, landmarks, hand.HAND_CONNECTIONS)

            # Pegar coordenadas do punho (landmark 0) para exibir o texto
            x = int(landmarks.landmark[0].x * w)
            y = int(landmarks.landmark[0].y * h)

            cv2.putText(img, label, (x - 20, y - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    return maos_identificadas

# === Verificações de gestos ===
# Armazena o tempo de detecção do gesto pela primeira vez
def verificar_mão_fechada(gesto_atual):
    agora = time.time()

    if gesto_atual == (False, False, False, False, False):  # Mão fechada
        if "ultimo_punho" not in estado:
            estado["ultimo_punho"] = agora
        elif agora - estado["ultimo_punho"] >= TEMPO:
            encerrar_programa()
            estado.pop("ultimo_punho")  # Reseta o tempo para esperar um novo gesto
    else:
        estado.pop("ultimo_punho", None)  # Gesto não está mais ativo, reseta

def verificar_paz_e_amor(gesto_atual):
    agora = time.time()

    if gesto_atual == (False, True, True, False, False):  # Paz e Amor
        if "tempo_ultimo_mindinho" not in estado:
            estado["tempo_ultimo_mindinho"] = agora
        elif agora - estado["tempo_ultimo_mindinho"] >= TEMPO:
            abrir_terminal()
            estado.pop("tempo_ultimo_mindinho")
    else:
        estado.pop("tempo_ultimo_mindinho", None)

def verificar_bigode(maos, w, h):
    agora = time.time()

    if "Left" in maos and "Right" in maos:
        pontos_left = [(int(p.x * w), int(p.y * h)) for p in maos["Left"].landmark]
        pontos_right = [(int(p.x * w), int(p.y * h)) for p in maos["Right"].landmark]

        gesto_left = detectar_dedos_esquerda(pontos_left)
        gesto_right = detectar_dedos_direita(pontos_right)

        if gesto_left == (True, False, True, True, True) and gesto_right == (True, False, True, True, True):
            if "ultimo_mindinho_duplo" not in estado:
                estado["ultimo_mindinho_duplo"] = agora
            elif agora - estado["ultimo_mindinho_duplo"] >= TEMPO:
                abrir_github()
                estado.pop("ultimo_mindinho_duplo")
        else:
            estado.pop("ultimo_mindinho_duplo", None)
    else:
        estado.pop("ultimo_mindinho_duplo", None)


# === Loop principal ===
def main():
    global video
    while True:
        check, img = video.read()
        img = cv2.flip(img, 1)  # <<< ESPELHA A IMAGEM
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = Hand.process(imgRGB)
        h, w, _ = img.shape

        maos = identificar_maos(results, img, w, h)

        if MAO_DOMINANTE in maos:
            pontos = []
            hand_landmarks = maos[MAO_DOMINANTE]

            for cord in hand_landmarks.landmark:
                cx, cy = int(cord.x * w), int(cord.y * h)
                pontos.append((cx, cy))

            if pontos:
                gesto_atual = detectar_dedos_direita(pontos)

                agora = time.time()
                if gesto_atual != estado["gesto_anterior"] or (agora - estado["tempo_ultimo_gesto"]) > 2:
                    verificar_mão_fechada(gesto_atual)
                    verificar_paz_e_amor(gesto_atual)
                    verificar_bigode(maos, w, h)

                    estado["gesto_anterior"] = gesto_atual
                    estado["tempo_ultimo_gesto"] = agora

        cv2.imshow("Imagem", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()

# === Início ===
if __name__ == "__main__":
    main()