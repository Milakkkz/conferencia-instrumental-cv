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
NOME_PASTA_FOTOS = "fotos_teste"    # pasta SEPARADA do treino, criada dentro da Área de Trabalho
PREFIXO_FOTO = "teste_"             # nomes: teste_0001.jpg, teste_0002.jpg, ... (nunca se mistura com as img_ do treino)
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


# ===== PLANILHA DO CONJUNTO DE TESTE (160 fotos, 16 situações x 10) — arquivo .txt, abre no Bloco de Notas =====
ESTADOS = {1: "completa", 2: "faltando", 3: "com_mais", 4: "faltando_com_mais"}   # teclas 1 a 4
# meta de fotos por situação: (estado, desconhecido 0/1, sobreposto 0/1) -> quantidade
META = {
    ("completa", 0, 0): 10, ("completa", 1, 0): 10, ("com_mais", 0, 0): 10, ("com_mais", 1, 0): 10,
    ("faltando", 0, 0): 10, ("faltando", 1, 0): 10, ("faltando_com_mais", 0, 0): 10, ("faltando_com_mais", 1, 0): 10,
    ("completa", 0, 1): 10, ("completa", 1, 1): 10, ("com_mais", 0, 1): 10, ("com_mais", 1, 1): 10,
    ("faltando", 0, 1): 10, ("faltando", 1, 1): 10, ("faltando_com_mais", 0, 1): 10, ("faltando_com_mais", 1, 1): 10,
}
COLUNAS = ["foto", "faltando", "sobrando", "desconhecido", "sobreposto", "quais_faltam", "quais_sobram"]


LARGURAS = [16, 9, 9, 13, 11, 14, 14]                    # largura de cada coluna (so para alinhar no Bloco de Notas)


def _separar(linha):                                     # quebra uma linha em celulas (aceita ; ou TAB)
    sep = "\t" if "\t" in linha else ";"
    cel = [c.strip() for c in linha.rstrip("\r\n").split(sep)]
    return (cel + [""] * len(COLUNAS))[:len(COLUNAS)]


def _ler_arquivo(caminho):                               # le um arquivo de planilha (.txt ou .csv antigo)
    linhas = []
    with open(caminho, encoding="utf-8-sig") as f:
        for linha in f:
            if not linha.strip():
                continue
            cel = _separar(linha)
            if cel[0].lower() == "foto":                 # cabecalho
                continue
            linhas.append(dict(zip(COLUNAS, cel)))
    return linhas


def ler_planilha(caminho):                               # le as linhas ja salvas (para continuar de onde parou)
    if os.path.exists(caminho):
        return _ler_arquivo(caminho)
    antigo = os.path.splitext(caminho)[0] + ".csv"       # se nao existe o .txt mas existe o .csv antigo, aproveita
    if os.path.exists(antigo):
        linhas = _ler_arquivo(antigo)
        print(f"Planilha antiga {os.path.basename(antigo)} importada ({len(linhas)} fotos); daqui para frente usa {os.path.basename(caminho)}.")
        return linhas
    return []


def gravar_planilha(caminho, linhas):                    # grava tudo em .txt (abre no Bloco de Notas, colunas alinhadas)
    try:
        with open(caminho, "w", encoding="utf-8-sig", newline="\r\n") as f:
            f.write(";".join(c.ljust(w) for c, w in zip(COLUNAS, LARGURAS)).rstrip() + "\n")
            for l in linhas:
                f.write(";".join(str(l[c]).ljust(w) for c, w in zip(COLUNAS, LARGURAS)).rstrip() + "\n")
        return True
    except PermissionError:                              # acontece se o arquivo estiver aberto em outro programa
        print("AVISO: feche a planilha no Bloco de Notas; ela sera regravada na proxima foto.")
        return False


def sem_duplicadas(linhas):                              # tira linhas repetidas da mesma foto (fica a primeira)
    vistos, unicas, repetidas = set(), [], []
    for l in linhas:
        if l["foto"] in vistos:
            repetidas.append(l["foto"])
        else:
            vistos.add(l["foto"]); unicas.append(l)
    return unicas, repetidas


def sincronizar(linhas, pasta):                          # tira da planilha as fotos que nao existem mais na pasta
    existem = {n for n in os.listdir(pasta) if re.fullmatch(re.escape(PREFIXO_FOTO) + r"\d+\.jpg", n)}
    mantidas = [l for l in linhas if l["foto"] in existem]
    tiradas = [l["foto"] for l in linhas if l["foto"] not in existem]
    sem_linha = sorted(existem - {l["foto"] for l in linhas})
    return mantidas, tiradas, sem_linha


def registrar(pend, linhas):                             # grava a linha da foto na planilha e mostra o andamento
    nome_estado = pend["estado"]
    linhas.append({"foto": os.path.basename(pend["caminho"]),
                   "faltando": "sim" if "faltando" in nome_estado else "nao",
                   "sobrando": "sim" if "mais" in nome_estado else "nao",
                   "desconhecido": "sim" if pend["desc"] else "nao", "sobreposto": "sim" if pend["sobre"] else "nao",
                   "quais_faltam": pend["quais_faltam"], "quais_sobram": pend["quais_sobram"]})
    gravar_planilha(CAMINHO_PLANILHA, linhas)
    sit = (nome_estado, pend["desc"], pend["sobre"])
    print(f"Foto salva: {os.path.basename(pend['caminho'])} | {nome_situacao(sit)} {contar(linhas, sit)}/{META[sit]} | total {len(linhas)}/{sum(META.values())}")
    return pend["caminho"]


def estado_de(faltando, sobrando):                       # nome do estado a partir das duas colunas sim/nao
    if faltando and sobrando: return "faltando_com_mais"
    if faltando: return "faltando"
    if sobrando: return "com_mais"
    return "completa"


def chave(l):                                            # situacao de uma linha da planilha
    return (estado_de(l["faltando"] == "sim", l["sobrando"] == "sim"),
            1 if l["desconhecido"] == "sim" else 0, 1 if l["sobreposto"] == "sim" else 0)


def contar(linhas, k):                                   # quantas fotos ja existem naquela situacao
    return sum(1 for l in linhas if chave(l) == k)


def nome_situacao(k):                                    # texto legivel da situacao
    return f"{k[0]}{' + desconhecido' if k[1] else ''}{' + sobreposto' if k[2] else ''}"


def tabela_terminal(linhas):                             # tecla T: mostra as 16 situacoes em colunas e o que falta
    print("\n=== SITUACOES ===")
    print("  FALTANDO  SOBRANDO  DESCONHECIDO  SOBREPOSTO   FEITAS/META")
    for k, meta in META.items():
        f = contar(linhas, k)
        falt = "sim" if "faltando" in k[0] else "nao"
        sobr = "sim" if "mais" in k[0] else "nao"
        print(f"  {falt:8s}  {sobr:8s}  {'sim' if k[1] else 'nao':12s}  {'sim' if k[2] else 'nao':10s}   {f}/{meta} {'OK' if f >= meta else ''}")
    print(f"  TOTAL {len(linhas)}/{sum(META.values())}\n")


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

print("Teclas: ESC = sair | S = nitidez | C = config. do driver | P = salvar foto (teste)")  # lista as teclas no terminal
print("        A = buscar melhor foco | + / - = ajuste fino do foco | F = liga/desliga autofoco")
print("        [ / ] = diminui/aumenta o realce | R = reaplicar padrões do driver")
PASTA_FOTOS = os.path.join(pasta_area_de_trabalho(), NOME_PASTA_FOTOS)   # Área de Trabalho\fotos_treino
os.makedirs(PASTA_FOTOS, exist_ok=True)                  # cria a pasta se ainda não existir
contador_fotos = proximo_numero(PASTA_FOTOS)             # próximo número livre (0001 na primeira vez)
CAMINHO_PLANILHA = os.path.join(PASTA_FOTOS, "planilha_teste.txt")
linhas = ler_planilha(CAMINHO_PLANILHA)                  # continua a planilha se ja existir
linhas_lidas = len(linhas)
linhas, repetidas = sem_duplicadas(linhas)               # uma linha por foto
linhas, tiradas, sem_linha = sincronizar(linhas, PASTA_FOTOS)   # acerta a planilha com as fotos que realmente existem
n_jpg = len([n for n in os.listdir(PASTA_FOTOS) if re.fullmatch(re.escape(PREFIXO_FOTO) + r"\d+\.jpg", n)])
print(f"DIAGNOSTICO | pasta: {PASTA_FOTOS}")
print(f"DIAGNOSTICO | linhas lidas: {linhas_lidas} | repetidas: {len(repetidas)} {', '.join(repetidas[:5])} | sem foto: {len(tiradas)} | fotos jpg na pasta: {n_jpg} | ficaram: {len(linhas)}")
if repetidas:
    gravar_planilha(CAMINHO_PLANILHA, linhas)
if tiradas:
    print(f"AVISO: {len(tiradas)} linha(s) da planilha sem foto na pasta foram removidas: {', '.join(tiradas[:5])}{' ...' if len(tiradas) > 5 else ''}")
    gravar_planilha(CAMINHO_PLANILHA, linhas)
if sem_linha:
    print(f"AVISO: {len(sem_linha)} foto(s) na pasta SEM linha na planilha: {', '.join(sem_linha[:5])}{' ...' if len(sem_linha) > 5 else ''}")
print(f"Planilha lida: {len(linhas)} fotos registradas")
estado = 1                                               # 1 completa, 2 faltando, 3 com mais, 4 faltando com mais
flag_desc = 0                                            # 1 = tem instrumento desconhecido (tecla K, fica ligado ate desligar)
flag_sobre = 0                                           # 1 = pecas sobrepostas (tecla O, fica ligado ate desligar)
ultima_foto = None                                       # ultima foto salva (para a tecla Z desfazer)
pend = None                                              # foto salva esperando as classes (digitadas na janela)
print("TESTE: 1=completa 2=faltando 3=sobrando 4=faltando+sobrando | K=desconhecido O=sobreposto (ligam e desligam) | P=salvar Z=desfazer T=tabela L=recarregar planilha")
print("Classes: 0 gancho_pequeno, 1 hohmann_8x236, 2 hohmann_mini_2x8, 3 mini_2x16, 4 senn_mueller, 5 dentes_14cm, 6 dentes_16cm,")
print("         7 verbrugge, 8 pontas_152mm, 9 weitlaner, 10 pontas_20cm, 11 modelador_pequeno, 12 williger, 13 farabeuf")
print(f"Planilha: {CAMINHO_PLANILHA}")
tabela_terminal(linhas)
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
    sit = (ESTADOS[estado], flag_desc, flag_sobre)
    cv2.putText(exibir, f"FALTANDO {'SIM' if 'faltando' in sit[0] else 'nao'} | SOBRANDO {'SIM' if 'mais' in sit[0] else 'nao'} | "
                f"DESCONHECIDO {'SIM' if flag_desc else 'nao'} | SOBREPOSTO {'SIM' if flag_sobre else 'nao'}",
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.putText(exibir, f"esta situacao [{contar(linhas, sit)}/{META[sit]}]   TOTAL {len(linhas)}/{sum(META.values())}",
                (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.putText(exibir, "1 completa 2 faltando 3 sobrando 4 faltando+sobrando | K desconhecido | O sobreposto | P salva | Z desfaz | T tabela | L recarrega",
                (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    if pend is not None:                                 # esperando as classes: mostra o que esta sendo digitado
        campo = "FALTAM" if pend["fases"][0] == "quais_faltam" else "SOBRAM"
        cv2.putText(exibir, f"CLASSES QUE {campo}: {' '.join(pend['lista'])} [{pend['buf']}]", (10, 160),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)
        cv2.putText(exibir, "digite o numero (0-13) | ESPACO = proxima classe | ENTER = confirmar | BACKSPACE = apagar | ESC = cancelar foto",
                    (10, 195), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.imshow(janela, exibir)                           # mostra a imagem na janela

    tecla = cv2.waitKey(1) & 0xFF                        # espera 1 ms por uma tecla (e atualiza a janela)
    if pend is not None:                                 # esta digitando as classes da ultima foto: so estas teclas valem
        if tecla == 27:                                  # ESC: cancela, apaga a foto e nao registra
            try:
                os.remove(pend["caminho"])
            except OSError:
                pass
            print(f"Cancelado: {os.path.basename(pend['caminho'])} apagada.")
            pend = None
        elif ord('0') <= tecla <= ord('9'):
            if len(pend["buf"]) < 2:
                pend["buf"] += chr(tecla)
        elif tecla == 8:                                 # BACKSPACE
            pend["buf"] = pend["buf"][:-1]
        elif tecla in (32, 44, 13):                      # ESPACO/virgula = proxima classe | ENTER = confirma a lista
            if pend["buf"]:
                n = int(pend["buf"])
                if 0 <= n <= 13:
                    pend["lista"].append(str(n))
                else:
                    print(f"Classe {n} invalida (use 0 a 13).")
                pend["buf"] = ""
            if tecla == 13:
                if not pend["lista"]:
                    print("Digite pelo menos uma classe antes de confirmar (ou ESC para cancelar a foto).")
                else:
                    pend[pend["fases"][0]] = " ".join(pend["lista"])
                    pend["fases"].pop(0)
                    pend["lista"] = []
                    if not pend["fases"]:
                        ultima_foto = registrar(pend, linhas)
                        pend = None
        continue
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
    elif tecla in (ord('1'), ord('2'), ord('3'), ord('4')):   # escolhe o estado do kit
        estado = int(chr(tecla))
        print(f"Estado: {ESTADOS[estado]}")
    elif tecla == ord('k'):                              # liga/desliga "tem desconhecido"
        flag_desc = 1 - flag_desc
    elif tecla == ord('o'):                              # liga/desliga "pecas sobrepostas"
        flag_sobre = 1 - flag_sobre
    elif tecla == ord('t'):                              # mostra a tabela das 16 situacoes no terminal
        tabela_terminal(linhas)
    elif tecla == ord('l'):                              # tecla L: recarrega a planilha e acerta com as fotos da pasta
        linhas = ler_planilha(CAMINHO_PLANILHA)          # (use depois de apagar foto/linha na mao)
        linhas, repetidas = sem_duplicadas(linhas)
        linhas, tiradas, sem_linha = sincronizar(linhas, PASTA_FOTOS)
        if tiradas or repetidas:
            gravar_planilha(CAMINHO_PLANILHA, linhas)
        print(f"Planilha recarregada: {len(linhas)} fotos | removidas (sem foto): {len(tiradas)} | fotos sem linha: {len(sem_linha)} {', '.join(sem_linha[:5])}")
    elif tecla == ord('p'):                              # tecla P: salva a foto; se faltar/sobrar peca, pede as classes NA JANELA
        try:
            contador_fotos = proximo_numero(PASTA_FOTOS)     # confere a pasta a cada foto (preenche buracos)
            ok, caminho = salvar_foto(frame_realcado, PASTA_FOTOS, contador_fotos)   # salva a imagem realçada em resolução cheia
            if ok:
                nome_estado = ESTADOS[estado]
                fases = (["quais_faltam"] if "faltando" in nome_estado else []) + (["quais_sobram"] if "mais" in nome_estado else [])
                pend = {"caminho": caminho, "estado": nome_estado, "desc": flag_desc, "sobre": flag_sobre,
                        "fases": fases, "lista": [], "buf": "", "quais_faltam": "", "quais_sobram": ""}
                if not fases:                                # completa: nao precisa perguntar nada
                    ultima_foto = registrar(pend, linhas)
                    pend = None
                else:
                    print(f"Foto {os.path.basename(caminho)} salva. Digite as classes NA JANELA DA CAMERA (nao no terminal).")
            else:
                print("Erro ao salvar a foto.")
        except Exception:
            import traceback; traceback.print_exc()
            print('ERRO ao salvar a foto (o programa continua aberto). Copie o texto acima e me mande.')
    elif tecla == ord('z'):                              # tecla Z: desfaz a ultima foto (apaga arquivo e linha da planilha)
        try:
            if ultima_foto and os.path.exists(ultima_foto) and linhas:
                os.remove(ultima_foto)
                linhas.pop()
                gravar_planilha(CAMINHO_PLANILHA, linhas)
                print(f"Desfeito: {os.path.basename(ultima_foto)} | total {len(linhas)}/{sum(META.values())}")
                ultima_foto = None
        except Exception:
            import traceback; traceback.print_exc()
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