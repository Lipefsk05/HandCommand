import cv2
import mediapipe as mp
import time

# === Configurações Gerais ===
TEMPO_GESTO_SEGUNDOS = 1  # <<<<< ALTERE ESTE VALOR PARA MUDAR O TEMPO DE TODOS OS GESTOS

# === Inicializações ===
video = cv2.VideoCapture(0)
hand = mp.solutions.hands
Hand = hand.Hands(max_num_hands=1)
mpDraw = mp.solutions.drawing_utils

# === Variáveis de estado ===
estado = {
    "gesto_anterior": None,
    "tempo_ultimo_gesto": 0,
    "inicio_punho": None
}

# === Função genérica para detectar dedos levantados ===
def detectar_dedos_levantados(pontos):
    dedos_ids = [4, 8, 12, 16, 20]
    dedos_levantados = [False] * 5

    if pontos[4][0] < pontos[2][0]:  # Polegar
        dedos_levantados[0] = True

    for i in range(1, 5):  # Indicador até mindinho
        if pontos[dedos_ids[i]][1] < pontos[dedos_ids[i] - 2][1]:
            dedos_levantados[i] = True

    return tuple(dedos_levantados)

# === Função reutilizável para qualquer gesto com tempo ===
def verificar_gesto_por_tempo(gesto_atual, gesto_desejado, chave_tempo, acao):
    agora = time.time()
    if gesto_atual == gesto_desejado:
        if estado[chave_tempo] is None:
            estado[chave_tempo] = agora
        elif agora - estado[chave_tempo] >= TEMPO_GESTO_SEGUNDOS:
            acao()
            estado[chave_tempo] = None
            return True
    else:
        estado[chave_tempo] = None
    return False

# === Ação: encerrar o programa ===
def encerrar_programa():
    print("👊 Mão fechada por tempo - ENCERRANDO programa.")
    video.release()
    cv2.destroyAllWindows()
    exit()

# === Verifica se a mão está fechada por tempo suficiente ===
def verificar_mão_fechada(gesto_atual):
    return verificar_gesto_por_tempo(
        gesto_atual,
        (False, False, False, False, False),
        "inicio_punho",
        encerrar_programa
    )

# === Loop principal ===
def main():
    while True:
        check, img = video.read()
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = Hand.process(imgRGB)
        h, w, _ = img.shape
        pontos = []

        if results.multi_hand_landmarks:
            for points in results.multi_hand_landmarks:
                mpDraw.draw_landmarks(img, points, hand.HAND_CONNECTIONS)
                for cord in points.landmark:
                    cx, cy = int(cord.x * w), int(cord.y * h)
                    pontos.append((cx, cy))

            gesto_atual = detectar_dedos_levantados(pontos)

            agora = time.time()
            if gesto_atual != estado["gesto_anterior"] or (agora - estado["tempo_ultimo_gesto"]) > 2:
                verificar_mão_fechada(gesto_atual)
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
