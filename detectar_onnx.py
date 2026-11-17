"""Roda o modelo YOLO exportado (modelo_final.onnx) no PC, SEM PyTorch (usa onnxruntime + OpenCV).

Instalacao (uma vez, no terminal do PyCharm):   pip install onnxruntime opencv-python numpy

Usos:
  # pre-anotar uma pasta de fotos (grava img_XXXX.txt ao lado das fotos e nao sobrescreve os existentes)
  python detectar_onnx.py modelo_final.onnx --pasta "C:\\...\\fotos_treino" [--conf 0.4 --dedup 0.8]

  # ver a deteccao ao vivo na camera (ESC sai)
  python detectar_onnx.py modelo_final.onnx --camera 0 [--conf 0.5]

  # sem argumentos (botao Run): usa o modelo_final.onnx ao lado do script e abre a camera 0

classes.txt: lido da mesma pasta do .onnx, se existir; senao usa classe_0, classe_1...
"""
import argparse, glob, os, sys, time
import cv2
import numpy as np
import onnxruntime as ort

IMGSZ = 640           # usado so se o modelo nao informar o tamanho de entrada

# ===== CONFIGURACOES DA CAMERA (iguais as do teste_camera.py, para a IA ver a mesma imagem do treino) =====
INDICE_CAMERA = 0     # numero da camera no Windows (0 = primeira camera encontrada)
LARGURA = 2048        # 4:3 de sensor inteiro (mesmo das fotos de treino)
ALTURA = 1536
LARGURA_JANELA = 1280 # so para mostrar na tela
REALCE_INICIAL = 2    # 2 = intensidade 0.2 (as fotos de treino foram salvas COM este realce; a deteccao tambem usa)
FOCO_MAX = 1023
FOCO_INICIAL = 10     # foco manual ao abrir
PASSO_VARREDURA = 20  # tecla A (busca automatica do melhor foco)

PADROES_CAMERA = {
    "Brilho":              (cv2.CAP_PROP_BRIGHTNESS,   6),
    "Contraste":           (cv2.CAP_PROP_CONTRAST,     9),
    "Matiz":               (cv2.CAP_PROP_HUE,          -7),
    "Saturacao":           (cv2.CAP_PROP_SATURATION,   7),
    "Nitidez (driver)":    (cv2.CAP_PROP_SHARPNESS,    7),
    "Gama":                (cv2.CAP_PROP_GAMMA,        5),
    "Comp. luz de fundo":  (cv2.CAP_PROP_BACKLIGHT,    0),
    "Ganho":               (cv2.CAP_PROP_GAIN,         0),
}
BALANCO_BRANCO_AUTO = 1
BALANCO_BRANCO_VALOR = 5045
EXPOSICAO_AUTO = 1
EXPOSICAO_VALOR = -6


def aplicar_padroes_camera(camera):
    for nome, (prop, valor) in PADROES_CAMERA.items():
        camera.set(prop, valor)
    camera.set(cv2.CAP_PROP_AUTO_WB, BALANCO_BRANCO_AUTO)
    if not BALANCO_BRANCO_AUTO:
        camera.set(cv2.CAP_PROP_WB_TEMPERATURE, BALANCO_BRANCO_VALOR)
    camera.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75 if EXPOSICAO_AUTO else 0.25)
    if not EXPOSICAO_AUTO:
        camera.set(cv2.CAP_PROP_EXPOSURE, EXPOSICAO_VALOR)
    print("Padroes da camera aplicados (pedido -> lido do driver):")
    for nome, (prop, valor) in PADROES_CAMERA.items():
        print(f"  {nome}: {valor} -> {camera.get(prop)}")


def definir_foco(camera, valor):
    camera.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    camera.set(cv2.CAP_PROP_FOCUS, valor)


def medir_nitidez(frame):
    return cv2.Laplacian(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()


def nitidez_centro(frame, fracao=0.4):
    h, w = frame.shape[:2]
    dh, dw = int(h * fracao / 2), int(w * fracao / 2)
    return medir_nitidez(frame[h // 2 - dh:h // 2 + dh, w // 2 - dw:w // 2 + dw])


def varredura_foco(camera):
    print("Varrendo o foco... aguarde.")
    melhor_valor, melhor_nitidez = 0, -1
    for valor in range(0, FOCO_MAX + 1, PASSO_VARREDURA):
        definir_foco(camera, valor)
        for _ in range(3):
            camera.read()
        ret, frame = camera.read()
        if not ret:
            continue
        n = nitidez_centro(frame)
        if n > melhor_nitidez:
            melhor_valor, melhor_nitidez = valor, n
    definir_foco(camera, melhor_valor)
    print(f"Melhor foco: {melhor_valor} (nitidez {melhor_nitidez:.1f})")
    return melhor_valor


def aplicar_realce(frame, intensidade):
    if intensidade <= 0:
        return frame
    borrado = cv2.GaussianBlur(frame, (0, 0), 3)
    return cv2.addWeighted(frame, 1 + intensidade, borrado, -intensidade, 0)

def carregar(caminho):
    prov = [p for p in ("CUDAExecutionProvider", "DmlExecutionProvider", "CPUExecutionProvider")
            if p in ort.get_available_providers()]
    return ort.InferenceSession(caminho, providers=prov)

def letterbox(img, tam=IMGSZ):
    h, w = img.shape[:2]
    r = min(tam / h, tam / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    res = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((tam, tam, 3), 114, np.uint8)
    dy, dx = (tam - nh) // 2, (tam - nw) // 2
    canvas[dy:dy + nh, dx:dx + nw] = res
    return canvas, r, dx, dy

def detectar(sess, img_bgr, conf=0.25, iou=0.7):
    """Retorna lista de (classe, conf, x1, y1, x2, y2) em pixels da imagem original."""
    H, W = img_bgr.shape[:2]
    tam = sess.get_inputs()[0].shape[2]                  # tamanho de entrada do proprio modelo (640 no piloto, 1024 no final)
    lb, r, dx, dy = letterbox(img_bgr, tam if isinstance(tam, int) else IMGSZ)
    x = cv2.cvtColor(lb, cv2.COLOR_BGR2RGB).astype(np.float32).transpose(2, 0, 1)[None] / 255.0
    bruto = sess.run(None, {sess.get_inputs()[0].name: x})[0][0]
    if bruto.ndim == 2 and bruto.shape[1] == 6 and bruto.shape[0] > 6:
        # modelo exportado com NMS embutido (end2end): linhas [x1, y1, x2, y2, conf, classe]
        bruto = bruto[bruto[:, 4] >= conf]
        xyxy = bruto[:, :4].astype(np.float64).copy()
        xyxy[:, [0, 2]] = ((xyxy[:, [0, 2]] - dx) / r).clip(0, W)
        xyxy[:, [1, 3]] = ((xyxy[:, [1, 3]] - dy) / r).clip(0, H)
        return [(int(b[5]), float(b[4]), *xyxy[i].tolist()) for i, b in enumerate(bruto)]
    saida = bruto.T                                                      # (8400, 4+nc)
    caixas, scores = saida[:, :4], saida[:, 4:]
    cls = scores.argmax(1); conf_s = scores.max(1)
    m = conf_s >= conf
    caixas, cls, conf_s = caixas[m], cls[m], conf_s[m]
    if len(caixas) == 0:
        return []
    xyxy = np.stack([caixas[:, 0] - caixas[:, 2] / 2, caixas[:, 1] - caixas[:, 3] / 2,
                     caixas[:, 0] + caixas[:, 2] / 2, caixas[:, 1] + caixas[:, 3] / 2], 1)
    xyxy[:, [0, 2]] = (xyxy[:, [0, 2]] - dx) / r
    xyxy[:, [1, 3]] = (xyxy[:, [1, 3]] - dy) / r
    xyxy[:, [0, 2]] = xyxy[:, [0, 2]].clip(0, W); xyxy[:, [1, 3]] = xyxy[:, [1, 3]].clip(0, H)
    wh = np.stack([xyxy[:, 0], xyxy[:, 1], xyxy[:, 2] - xyxy[:, 0], xyxy[:, 3] - xyxy[:, 1]], 1)
    # NMS por classe (instrumentos de classes diferentes podem se sobrepor)
    out = []
    for c in np.unique(cls):
        idx = np.where(cls == c)[0]
        keep = cv2.dnn.NMSBoxes(wh[idx].tolist(), conf_s[idx].tolist(), conf, iou)
        for k in np.array(keep).flatten():
            j = idx[k]
            out.append((int(c), float(conf_s[j]), *xyxy[j].tolist()))
    return out

def ler_imagem(p):
    return cv2.imdecode(np.fromfile(p, np.uint8), cv2.IMREAD_COLOR)

def desenhar(img, dets, nomes):
    for c, s, x1, y1, x2, y2 in dets:
        cor = tuple(int(v) for v in cv2.cvtColor(np.uint8([[[int((c * 0.618) % 1 * 180), 220, 230]]]), cv2.COLOR_HSV2BGR)[0, 0])
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), cor, 3)
        nome = nomes[c] if c < len(nomes) else str(c)
        cv2.putText(img, f"{nome} {s:.2f}", (int(x1), max(int(y1) - 6, 18)), 0, 0.8, cor, 2)
    return img

QUOTA = {c: 1 for c in range(14)}
QUOTA.update({4: 2, 5: 2, 6: 2, 11: 2, 13: 2})   # unidades de cada classe no kit; 14 = desconhecido (sem limite)

def deduplicar(dets, lim=0.8):
    """Remove caixa de menor confianca quando duas caixas (de qualquer classe) cobrem quase a mesma area."""
    dets = sorted(dets, key=lambda d: -d[1]); out = []
    for d in dets:
        ok = True
        for o in out:
            ix = max(0, min(d[4], o[4]) - max(d[2], o[2])); iy = max(0, min(d[5], o[5]) - max(d[3], o[3]))
            inter = ix * iy
            a1 = (d[4]-d[2]) * (d[5]-d[3]); a2 = (o[4]-o[2]) * (o[5]-o[3])
            if inter / max(a1 + a2 - inter, 1e-9) > lim or inter / max(min(a1, a2), 1e-9) > 0.95:
                ok = False; break
        if ok: out.append(d)
    return out

def limitar(dets):
    """Mantem no maximo as unidades do kit por classe (as de maior confianca). Classe 14 sem limite."""
    cont = {}; out = []
    for d in sorted(dets, key=lambda d: -d[1]):
        c = d[0]; cont[c] = cont.get(c, 0) + 1
        if c == 14 or cont[c] <= QUOTA.get(c, 99): out.append(d)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("modelo", nargs="?", default=None,
                    help="caminho do modelo_final.onnx (padrao: modelo_final.onnx ao lado deste script)")
    ap.add_argument("--pasta"); ap.add_argument("--camera", type=int)
    ap.add_argument("--conf", type=float, default=0.5, help="limiar de confianca (0.5 = o mesmo usado na avaliacao do TCC)")
    ap.add_argument("--sobrescrever", action="store_true")
    ap.add_argument("--faixa", help="so fotos de A a B, ex.: 1355-1400 (use junto com --sobrescrever)")
    ap.add_argument("--limite", action="store_true", help="limita cada classe ao numero de unidades do kit")
    ap.add_argument("--dedup", type=float, default=0.0, help="IoU para remover caixas repetidas na mesma peca (0 = desligado, como na avaliacao do TCC; 0.8 ajuda na pre-anotacao)")
    a = ap.parse_args()
    if a.modelo is None:
        a.modelo = os.path.join(os.path.dirname(os.path.abspath(__file__)), "modelo_final.onnx")
    if not os.path.exists(a.modelo):
        print("ERRO: nao achei o modelo:", a.modelo)
        print("Coloque o modelo_final.onnx na mesma pasta deste script, ou passe o caminho como argumento.")
        input("\nPressione Enter para fechar...")
        sys.exit(1)
    if not a.pasta and a.camera is None:
        a.camera = 0          # sem argumentos: abre a camera 0
    sess = carregar(a.modelo)
    print("Provedor:", sess.get_providers()[0])
    cj = os.path.join(os.path.dirname(os.path.abspath(a.modelo)), "classes.txt")
    nomes = [l.strip() for l in open(cj, encoding="utf-8") if l.strip()] if os.path.exists(cj) else []

    if a.pasta:
        fotos = sorted(glob.glob(os.path.join(a.pasta, "*.jpg")) + glob.glob(os.path.join(a.pasta, "*.png")))
        feitas = puladas = caixas = 0
        fa = fb = None
        if a.faixa:
            fa, fb = [int(v) for v in a.faixa.split("-")]
        for f in fotos:
            txt = os.path.splitext(f)[0] + ".txt"
            if fa is not None:
                num = int("".join(ch for ch in os.path.basename(f) if ch.isdigit()) or 0)
                if not (fa <= num <= fb):
                    continue
            if os.path.exists(txt) and not a.sobrescrever:
                puladas += 1; continue
            img = ler_imagem(f); H, W = img.shape[:2]
            dets = detectar(sess, img, a.conf)
            if a.dedup > 0: dets = deduplicar(dets, a.dedup)
            if a.limite: dets = limitar(dets)
            with open(txt, "w", encoding="utf-8") as fh:
                for c, s, x1, y1, x2, y2 in dets:
                    fh.write(f"{c} {(x1+x2)/2/W:.6f} {(y1+y2)/2/H:.6f} {(x2-x1)/W:.6f} {(y2-y1)/H:.6f}\n")
            feitas += 1; caixas += len(dets)
        print(f"Pre-anotadas: {feitas} fotos ({caixas} caixas) | puladas: {puladas}")
        return

    indice = a.camera if a.camera is not None else INDICE_CAMERA
    cam = cv2.VideoCapture(indice, cv2.CAP_DSHOW)
    if not cam.isOpened():
        print("ERRO: nao consegui abrir a camera", indice, "- feche outros programas que a usam (ex.: teste_camera.py) ou tente --camera 1")
        input("\nPressione Enter para fechar...")
        sys.exit(1)
    # MJPG precisa vir ANTES da resolucao, senao a camera cai para 640x480
    cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, LARGURA)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, ALTURA)
    ok, teste = cam.read()
    if ok:
        print(f"Resolucao real: {teste.shape[1]}x{teste.shape[0]}")
        if abs(teste.shape[1] / teste.shape[0] - 4 / 3) > 0.01:
            print("AVISO: proporcao diferente de 4:3 (o driver pode estar cortando o campo de visao).")
    aplicar_padroes_camera(cam)
    definir_foco(cam, FOCO_INICIAL)
    foco_atual, auto_atual, realce_valor = FOCO_INICIAL, 0, REALCE_INICIAL
    print(f"Autofoco: {cam.get(cv2.CAP_PROP_AUTOFOCUS)} | Foco lido: {cam.get(cv2.CAP_PROP_FOCUS)}")
    print("Teclas: ESC = sair | C = config. do driver | R = reaplicar padroes | A = buscar melhor foco")
    print("        + / - = foco (1 por vez) | F = liga/desliga autofoco | [ / ] = realce | S = mostra medidas")

    while True:
        ok, frame = cam.read()
        if not ok:
            print("Erro ao capturar imagem.")
            break
        realce = realce_valor / 10.0
        entrada = aplicar_realce(frame, realce)        # a IA ve a mesma imagem das fotos de treino (com realce)
        t = time.time(); dets = detectar(sess, entrada, a.conf); ms = (time.time() - t) * 1000
        vis = desenhar(entrada.copy(), dets, nomes)
        esc = LARGURA_JANELA / vis.shape[1]
        vis = cv2.resize(vis, None, fx=esc, fy=esc)
        modo = "AUTO" if auto_atual else str(foco_atual)
        cv2.putText(vis, f"{len(dets)} deteccoes | {ms:.0f} ms | Realce: {realce:.1f} | Foco: {modo}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.imshow("detector (ESC sai)", vis)

        tecla = cv2.waitKey(1) & 0xFF
        if tecla == 27:
            break
        elif tecla == ord('s'):
            print(f"Nitidez centro: {nitidez_centro(frame):.1f} | Realce: {realce:.1f} | Foco: {modo}")
        elif tecla == ord('c'):
            cam.set(cv2.CAP_PROP_SETTINGS, 1)          # abre a janela de propriedades do driver
        elif tecla == ord('r'):
            aplicar_padroes_camera(cam)
            if auto_atual == 0:
                definir_foco(cam, foco_atual)
        elif tecla == ord('a'):
            auto_atual = 0
            foco_atual = varredura_foco(cam)
        elif tecla in (ord('+'), ord('=')):
            if auto_atual == 0:
                foco_atual = min(foco_atual + 1, FOCO_MAX)
                cam.set(cv2.CAP_PROP_FOCUS, foco_atual)
        elif tecla == ord('-'):
            if auto_atual == 0:
                foco_atual = max(foco_atual - 1, 0)
                cam.set(cv2.CAP_PROP_FOCUS, foco_atual)
        elif tecla == ord('f'):
            auto_atual = 1 - auto_atual
            cam.set(cv2.CAP_PROP_AUTOFOCUS, auto_atual)
            if auto_atual == 0:
                cam.set(cv2.CAP_PROP_FOCUS, foco_atual)
        elif tecla == ord(']'):
            realce_valor = min(realce_valor + 1, 30)
        elif tecla == ord('['):
            realce_valor = max(realce_valor - 1, 0)
    cam.release(); cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
