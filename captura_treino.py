import os
import re
import ctypes
from ctypes import wintypes
import cv2  # importa a biblioteca OpenCV (captura de câmera, processamento e exibição de imagens)

# ===== CONFIGURAÇÕES =====
INDICE_CAMERA = 0     # número da câmera no Windows (0 = primeira câmera encontrada)
LARGURA = 2048        # largura pedida à câmera, em pixels (4:3 de sensor inteiro; máximo da OV5640 é 2592x1944 a 15fps)
ALTURA = 1536         # altura pedida à câmera, em pixels (mantenha 4:3; 1920x1080 é 16:9 e CORTA o campo de visão)
REALCE_INICIAL = 2    # posição inicial da barra de realce (2 = intensidade 0.2; barra vai de 0 a 30)
LARGURA_JANELA = 1280 # largura usada só para mostrar na tela; a imagem real continua em alta resolução

FOCO_MAX = 1023       # valor máximo da barra de foco (faixa do driver: muitos usam 0-1023, alguns 0-255)
FOCO_INICIAL = 10     # valor de foco aplicado ao abrir o programa (o que você achou ideal na bancada)
PASSO_VARREDURA = 20  # de quanto em quanto o foco anda na busca automática (tecla A)


# ===== PADRÕES DO DRIVER (valores da janela "Propriedades de HD USB Camera") =====
# Aplicados automaticamente toda vez que o programa abre. A tecla C continua abrindo a janela do driver
# para ajuste manual; a tecla R reaplica estes padrões. Para mudar o padrão, edite os números abaixo.
PADROES_CAMERA = {
    "Brilho":              (cv2.CAP_PROP_BRIGHTNESS,   6),
    "Contraste":           (cv2.CAP_PROP_CONTRAST,     9),
    "Matiz":               (cv2.CAP_PROP_HUE,          -7),
    "Saturação":           (cv2.CAP_PROP_SATURATION,   7),
    "Nitidez (driver)":    (cv2.CAP_PROP_SHARPNESS,    7),
    "Gama":                (cv2.CAP_PROP_GAMMA,        5),
    "Comp. luz de fundo":  (cv2.CAP_PROP_BACKLIGHT,    0),
    "Ganho":               (cv2.CAP_PROP_GAIN,         0),
}
BALANCO_BRANCO_AUTO = 1     # 1 = automático (marcado na janela do driver)
BALANCO_BRANCO_VALOR = 5045 # valor mostrado na janela (só vale se o automático estiver desligado)
EXPOSICAO_AUTO = 1          # 1 = automático (marcado na janela do driver)
EXPOSICAO_VALOR = -6        # valor mostrado na janela (só vale se o automático estiver desligado)


def aplicar_padroes_camera(camera):                      # envia todos os padrões do driver para a câmera
    for nome, (prop, valor) in PADROES_CAMERA.items():   # percorre cada parâmetro da tabela acima
        camera.set(prop, valor)                          # envia o valor para o driver
    camera.set(cv2.CAP_PROP_AUTO_WB, BALANCO_BRANCO_AUTO)          # liga/desliga o balanço de branco automático
    if not BALANCO_BRANCO_AUTO:                                    # só no manual o valor é usado
        camera.set(cv2.CAP_PROP_WB_TEMPERATURE, BALANCO_BRANCO_VALOR)
    camera.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75 if EXPOSICAO_AUTO else 0.25)  # 0.75 = auto, 0.25 = manual
    if not EXPOSICAO_AUTO:                                         # só no manual o valor é usado
        camera.set(cv2.CAP_PROP_EXPOSURE, EXPOSICAO_VALOR)
    print("Padrões aplicados (valor pedido -> valor lido do driver):")   # confere o que o driver aceitou
    for nome, (prop, valor) in PADROES_CAMERA.items():
        print(f"  {nome}: {valor} -> {camera.get(prop)}")
    print(f"  Balanço de branco auto: {camera.get(cv2.CAP_PROP_AUTO_WB)} | Exposição: {camera.get(cv2.CAP_PROP_EXPOSURE)}")
    print("  (Rotação 0 e PowerLine 60 Hz não são controláveis pelo OpenCV; ficam como estão no driver)")


# ===== PASTA DAS FOTOS =====
NOME_PASTA_FOTOS = "fotos_treino"   # pasta criada dentro da Área de Trabalho
PREFIXO_FOTO = "img_"               # nomes: img_0001.jpg, img_0002.jpg, ...
QUALIDADE_JPEG = 95                 # qualidade do JPG salvo (0-100)


def pasta_area_de_trabalho():                            # descobre a Área de Trabalho real (inclusive com OneDrive)
    try:
        class GUID(ctypes.Structure):
            _fields_ = [("a", wintypes.DWORD), ("b", wintypes.WORD), ("c", wintypes.WORD), ("d", ctypes.c_ubyte * 8)]
        # FOLDERID_Desktop = {B4BFCC3A-DB2C-424C-B029-7FE99A87C641}
        fid = GUID(0xB4BFCC3A, 0xDB2C, 0x424C, (ctypes.c_ubyte * 8)(0xB0, 0x29, 0x7F, 0xE9, 0x9A, 0x87, 0xC6, 0x41))
        ptr = ctypes.c_wchar_p()
        if ctypes.windll.shell32.SHGetKnownFolderPath(ctypes.byref(fid), 0, None, ctypes.byref(ptr)) == 0:
            caminho = ptr.value
            ctypes.windll.ole32.CoTaskMemFree(ptr)
            return caminho
    except Exception:
        pass
    return os.path.join(os.path.expanduser("~"), "Desktop")   # alternativa caso o método acima falhe


def proximo_numero(pasta):                               # menor número livre: preenche buracos deixados por fotos apagadas
    usados = set()
    for nome in os.listdir(pasta):
        m = re.fullmatch(re.escape(PREFIXO_FOTO) + r"(\d+)\.(jpg|txt)", nome)   # conta .jpg e também .txt (label)
        if m:                                            # (assim não reaproveita um número que ainda tem label antigo)
            usados.add(int(m.group(1)))
    n = 1
    while n in usados:
        n += 1
    return n


def salvar_foto(frame, pasta, numero):                   # salva o JPG (funciona também com acentos no caminho)
    caminho = os.path.join(pasta, f"{PREFIXO_FOTO}{numero:04d}.jpg")
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, QUALIDADE_JPEG])
    if ok:
        buf.tofile(caminho)
    return ok, caminho


def medir_nitidez(frame):                                # calcula um número que indica o quão nítida está a imagem
    cinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)      # converte a imagem colorida (BGR) para tons de cinza
    return cv2.Laplacian(cinza, cv2.CV_64F).var()        # aplica o Laplaciano (detecta bordas) e retorna a variância:
                                                         # muitas bordas definidas = valor alto = imagem nítida


def nitidez_centro(frame, fracao=0.4):                   # mede a nitidez só na região central da imagem
    """Mede a nitidez só no centro da imagem (onde fica o instrumental)."""
    h, w = frame.shape[:2]                               # pega a altura (h) e a largura (w) da imagem
    dh, dw = int(h * fracao / 2), int(w * fracao / 2)    # metade da altura/largura do recorte (40% da imagem no total)
    return medir_nitidez(frame[h//2 - dh:h//2 + dh, w//2 - dw:w//2 + dw])  # recorta o centro e mede a nitidez dele


def aplicar_realce(frame, intensidade):                  # realça as bordas da imagem (técnica "unsharp mask")
    if intensidade <= 0:                                 # se o realce estiver zerado...
        return frame                                     # ...devolve a imagem original sem mexer
    borrado = cv2.GaussianBlur(frame, (0, 0), 3)         # cria uma cópia borrada da imagem (desfoque gaussiano, sigma 3)
    return cv2.addWeighted(frame, 1 + intensidade, borrado, -intensidade, 0)  # original*(1+k) - borrado*k:
                                                         # subtrair o borrado reforça as bordas


def definir_foco(camera, valor):                         # coloca a câmera em foco manual num valor específico
    camera.set(cv2.CAP_PROP_AUTOFOCUS, 0)                # desliga o autofoco (0 = desligado)
    camera.set(cv2.CAP_PROP_FOCUS, valor)                # envia a posição da lente para a câmera


def varredura_foco(camera, janela):                      # procura automaticamente o melhor foco
    """Percorre a faixa de foco e para no valor com maior nitidez no centro."""
    print("Varrendo o foco... aguarde.")                 # avisa no terminal que a busca começou
    melhor_valor, melhor_nitidez = 0, -1                 # guarda o melhor resultado (começa com nitidez -1 para qualquer valor ganhar)
    for valor in range(0, FOCO_MAX + 1, PASSO_VARREDURA):  # testa o foco de 0 até FOCO_MAX, pulando de 20 em 20
        definir_foco(camera, valor)                      # move a lente para o valor atual do teste
        for _ in range(3):                               # lê e descarta 3 frames...
            camera.read()                                # ...enquanto a lente termina de se mover
        ret, frame = camera.read()                       # captura um frame já com a lente parada
        if not ret:                                      # se a leitura falhou...
            continue                                     # ...pula para o próximo valor
        n = nitidez_centro(frame)                        # mede a nitidez do centro nesse foco
        if n > melhor_nitidez:                           # se for a mais nítida até agora...
            melhor_valor, melhor_nitidez = valor, n      # ...guarda esse foco como o melhor
    definir_foco(camera, melhor_valor)                   # ao final, trava a lente no melhor foco encontrado
    print(f"Melhor foco: {melhor_valor} (nitidez {melhor_nitidez:.1f})")  # mostra o resultado no terminal
    return melhor_valor                                  # devolve o valor para o loop principal


# ===== ABERTURA DA CÂMERA =====
camera = cv2.VideoCapture(INDICE_CAMERA, cv2.CAP_DSHOW)  # abre a câmera usando o DirectShow do Windows
                                                         # (necessário para a tecla C abrir a config. do driver)

if not camera.isOpened():                                # se a câmera não abriu (desconectada, em uso por outro programa...)
    print("Erro: não foi possível abrir a câmera.")      # mostra o erro
    exit()                                               # encerra o programa

# MJPG precisa vir ANTES da resolução, senão a câmera cai para 640x480
camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))  # pede o formato comprimido MJPG (o USB aguenta alta resolução)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, LARGURA)            # pede a largura configurada
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, ALTURA)            # pede a altura configurada

ret, teste = camera.read()                               # captura um frame de teste
if ret:                                                  # se a captura funcionou...
    print(f"Resolução real: {teste.shape[1]}x{teste.shape[0]}")  # ...mostra a resolução que a câmera realmente entregou
    if abs(teste.shape[1] / teste.shape[0] - 4 / 3) > 0.01:  # se a proporção não for 4:3...
        print("AVISO: proporção diferente de 4:3 (o driver pode estar cortando o campo de visão).")

# Padrões do driver (brilho, contraste, etc.)
aplicar_padroes_camera(camera)                           # aplica os valores definidos no topo do arquivo

# Foco manual
definir_foco(camera, FOCO_INICIAL)                      # desliga o autofoco e aplica o foco inicial (10)
print(f"Autofoco: {camera.get(cv2.CAP_PROP_AUTOFOCUS)} | Foco lido: {camera.get(cv2.CAP_PROP_FOCUS)}")  # confere o que o driver aceitou

# ===== JANELA E CONTROLES =====
janela = "Camera em tempo real"                          # nome (título) da janela de exibição
cv2.namedWindow(janela, cv2.WINDOW_NORMAL)               # cria a janela redimensionável

print("Teclas: ESC = sair | S = nitidez | C = config. do driver | P = salvar foto")  # lista as teclas no terminal
print("        A = buscar melhor foco | + / - = ajuste fino do foco | F = liga/desliga autofoco")
print("        [ / ] = diminui/aumenta o realce | R = reaplicar padrões do driver")
PASTA_FOTOS = os.path.join(pasta_area_de_trabalho(), NOME_PASTA_FOTOS)   # Área de Trabalho\fotos_treino
os.makedirs(PASTA_FOTOS, exist_ok=True)                  # cria a pasta se ainda não existir
contador_fotos = proximo_numero(PASTA_FOTOS)             # próximo número livre (0001 na primeira vez)
print(f"Fotos serão salvas em: {PASTA_FOTOS} (próxima: {PREFIXO_FOTO}{contador_fotos:04d}.jpg)")
foco_atual = FOCO_INICIAL                                # último foco enviado à câmera (para só reenviar quando mudar)
auto_atual = 0                                           # estado do autofoco (0 = manual, 1 = automático)
realce_valor = REALCE_INICIAL                            # realce atual (0 a 30; a intensidade usada é valor/10)

# ===== LOOP PRINCIPAL =====
while True:                                              # repete até o usuário apertar ESC
    ret, frame = camera.read()                           # captura um frame da câmera
    if not ret:                                          # se a captura falhou...
        print("Erro ao capturar imagem.")                # ...avisa
        break                                            # ...e sai do loop

    realce = realce_valor / 10.0                         # converte o valor do realce (ex.: 2 -> 0.2)
    nitidez = medir_nitidez(frame)                       # mede a nitidez na imagem ORIGINAL (antes do realce)

    # Use 'frame_realcado' na detecção (YOLO)
    frame_realcado = aplicar_realce(frame, realce)       # aplica o realce; esta é a imagem que vai para o YOLO

    # Reduz só para exibir na tela
    escala = LARGURA_JANELA / frame.shape[1]             # calcula o fator de redução (1280 / 2592 ≈ 0.49)
    exibir = cv2.resize(frame_realcado, None, fx=escala, fy=escala)  # cria uma cópia menor só para a tela
    modo = "AUTO" if auto_atual else str(foco_atual)     # texto do foco: "AUTO" ou o número do foco manual
    texto = (f"{frame.shape[1]}x{frame.shape[0]}  Nitidez: {nitidez:.1f}  "   # monta a linha de informações:
             f"Realce: {realce:.1f}  Foco: {modo}")      # resolução, nitidez, realce e foco
    cv2.putText(exibir, texto, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,  # escreve o texto no canto superior esquerdo
                0.7, (0, 0, 0), 2)                       # tamanho 0.7, cor preta (BGR), espessura 2
    cv2.imshow(janela, exibir)                           # mostra a imagem na janela

    tecla = cv2.waitKey(1) & 0xFF                        # espera 1 ms por uma tecla (e atualiza a janela)
    if tecla == 27:                                      # 27 = código da tecla ESC
        break                                            # sai do loop e encerra
    elif tecla == ord('s'):                              # tecla S: mostra as medições no terminal
        print(f"Nitidez: {nitidez:.1f} | Centro: {nitidez_centro(frame):.1f} | "
              f"Realce: {realce:.1f} | Foco: {modo}")
    elif tecla == ord('c'):                              # tecla C: abre a janela de propriedades do driver
        camera.set(cv2.CAP_PROP_SETTINGS, 1)             # (exposição, balanço de branco, ganho, brilho, etc.)
    elif tecla == ord('r'):                              # tecla R: volta aos padrões do driver definidos no topo
        aplicar_padroes_camera(camera)
        if auto_atual == 0:
            definir_foco(camera, foco_atual)
    elif tecla == ord('p'):                            # tecla P: salva uma foto
        contador_fotos = proximo_numero(PASTA_FOTOS)     # confere a pasta a cada foto (preenche buracos)
        ok, caminho = salvar_foto(frame_realcado, PASTA_FOTOS, contador_fotos)   # salva a imagem realçada em resolução cheia
        if ok:
            print(f"Foto salva: {caminho}")
        else:
            print("Erro ao salvar a foto.")
    elif tecla == ord('a'):                              # tecla A: busca automática do melhor foco
        auto_atual = 0                                   # memoriza o modo manual
        foco_atual = varredura_foco(camera, janela)      # executa a varredura e guarda o foco encontrado
    elif tecla in (ord('+'), ord('=')):                  # tecla + (ou =): aumenta o foco
        if auto_atual == 0:
            foco_atual = min(foco_atual + 1, FOCO_MAX)   # +1, sem passar do máximo
            camera.set(cv2.CAP_PROP_FOCUS, foco_atual)
    elif tecla == ord('-'):                              # tecla -: diminui o foco
        if auto_atual == 0:
            foco_atual = max(foco_atual - 1, 0)          # -1, sem ficar abaixo de 0
            camera.set(cv2.CAP_PROP_FOCUS, foco_atual)
    elif tecla == ord('f'):                              # tecla F: liga/desliga o autofoco
        auto_atual = 1 - auto_atual
        camera.set(cv2.CAP_PROP_AUTOFOCUS, auto_atual)
        if auto_atual == 0:                              # voltou para manual: reaplica o último foco manual
            camera.set(cv2.CAP_PROP_FOCUS, foco_atual)
    elif tecla == ord(']'):                              # tecla ]: aumenta o realce
        realce_valor = min(realce_valor + 1, 30)
    elif tecla == ord('['):                              # tecla [: diminui o realce
        realce_valor = max(realce_valor - 1, 0)

# ===== FINALIZAÇÃO =====
camera.release()                                         # libera a câmera para outros programas
cv2.destroyAllWindows()                                  # fecha todas as janelas do OpenCV