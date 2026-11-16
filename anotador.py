"""
Anotador simples de bounding boxes no formato YOLO, feito so com OpenCV + NumPy.

Uso:
    python anotador.py "C:\\caminho\\fotos_treino"
    python anotador.py "C:\\caminho\\fotos_treino" "C:\\caminho\\classes.txt"

- Sem o segundo argumento, procura o classes.txt na pasta das fotos e depois ao lado do script.
- Sem nenhum argumento, usa a pasta fotos_treino da Area de Trabalho.
- Para cada foto img_0001.jpg, cria/le img_0001.txt na mesma pasta.
- Cada linha do .txt:  classe  x_centro  y_centro  largura  altura  (tudo de 0 a 1)
- Se ja existir um .txt (por exemplo gerado por um pre-modelo), ele e carregado para voce corrigir.
"""
import argparse
import os
import re
import sys

import cv2
import numpy as np

EXTENSOES = (".jpg", ".jpeg", ".png", ".bmp")
LARGURA_PAINEL = 300        # largura do painel de classes (pixels na tela)
LARGURA_MAX = 1400          # largura maxima da area da imagem na tela
MIN_CAIXA = 8               # menor lado aceito para uma caixa (pixels da foto original)
LINHA = 26                  # altura de cada linha da lista de classes
TOPO_LISTA = 52             # onde a lista de classes comeca no painel
JANELA = "Anotador YOLO"

TECLA_ESQ = 2424832         # seta esquerda (Windows)
TECLA_DIR = 2555904         # seta direita (Windows)


def chave_natural(nome):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", nome)]


def ler_classes(caminho):
    with open(caminho, encoding="utf-8") as f:
        return [linha.strip() for linha in f if linha.strip()]


def ler_imagem(caminho):
    """Le a imagem mesmo com acentos no caminho (cv2.imread falha nesses casos no Windows)."""
    dados = np.fromfile(caminho, dtype=np.uint8)
    return cv2.imdecode(dados, cv2.IMREAD_COLOR)


def ler_yolo(caminho_txt, largura, altura):
    """Le um .txt YOLO e devolve caixas [classe, x1, y1, x2, y2] em pixels da foto original."""
    caixas = []
    if not os.path.exists(caminho_txt):
        return caixas
    with open(caminho_txt, encoding="utf-8") as f:
        for linha in f:
            partes = linha.split()
            if len(partes) < 5:
                continue
            try:
                c = int(float(partes[0]))
                cx, cy, w, h = (float(v) for v in partes[1:5])
            except ValueError:
                continue
            x1 = (cx - w / 2) * largura
            y1 = (cy - h / 2) * altura
            x2 = (cx + w / 2) * largura
            y2 = (cy + h / 2) * altura
            caixas.append([c, x1, y1, x2, y2])
    return caixas


def salvar_yolo(caminho_txt, caixas, largura, altura):
    """Grava as caixas no formato YOLO. Sem caixas, apaga o .txt."""
    if not caixas:
        if os.path.exists(caminho_txt):
            os.remove(caminho_txt)
        return
    linhas = []
    for c, x1, y1, x2, y2 in caixas:
        cx = (x1 + x2) / 2 / largura
        cy = (y1 + y2) / 2 / altura
        w = (x2 - x1) / largura
        h = (y2 - y1) / altura
        linhas.append(f"{c} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    with open(caminho_txt, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(linhas) + "\n")


def cor_da_classe(indice):
    matiz = int(((indice * 0.61803398875) % 1.0) * 180)   # espacamento aureo: cores bem diferentes entre classes vizinhas
    hsv = np.uint8([[[matiz, 220, 255]]])
    b, g, r = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return int(b), int(g), int(r)


class Anotador:
    def __init__(self, pasta, classes, altura_max):
        self.pasta = pasta
        self.classes = classes
        self.altura_max = altura_max
        self.arquivos = sorted(
            [f for f in os.listdir(pasta) if f.lower().endswith(EXTENSOES)], key=chave_natural
        )
        if not self.arquivos:
            raise SystemExit(f"Nenhuma foto encontrada em {pasta}")
        self.indice = 0
        self.classe_atual = 0
        self.imagem = None
        self.largura = self.altura = 0
        self.escala = 1.0
        self.caixas = []
        self.sujo = False
        self.arrastando = False
        self.inicio = (0, 0)
        self.mouse = (0, 0)
        self.carregar(0)

    # ---------- arquivos ----------
    def caminho_txt(self, i=None):
        nome = self.arquivos[self.indice if i is None else i]
        return os.path.join(self.pasta, os.path.splitext(nome)[0] + ".txt")

    def salvar(self):
        if self.sujo:
            salvar_yolo(self.caminho_txt(), self.caixas, self.largura, self.altura)
            self.sujo = False

    def carregar(self, i):
        self.salvar()
        self.indice = i % len(self.arquivos)
        caminho = os.path.join(self.pasta, self.arquivos[self.indice])
        self.imagem = ler_imagem(caminho)
        if self.imagem is None:
            raise SystemExit(f"Nao consegui abrir {caminho}")
        self.altura, self.largura = self.imagem.shape[:2]
        self.escala = min(LARGURA_MAX / self.largura, self.altura_max / self.altura, 1.0)
        self.caixas = ler_yolo(self.caminho_txt(), self.largura, self.altura)
        self.sujo = False
        self.arrastando = False

    def proxima_sem_label(self):
        for passo in range(1, len(self.arquivos) + 1):
            j = (self.indice + passo) % len(self.arquivos)
            if not os.path.exists(self.caminho_txt(j)):
                self.carregar(j)
                return
        print("Todas as fotos ja tem label.")

    def contar_com_label(self):
        return sum(os.path.exists(self.caminho_txt(j)) for j in range(len(self.arquivos)))

    # ---------- coordenadas ----------
    def tela_para_foto(self, x, y):
        fx = (x - LARGURA_PAINEL) / self.escala
        fy = y / self.escala
        return min(max(fx, 0), self.largura - 1), min(max(fy, 0), self.altura - 1)

    def caixa_sob_mouse(self):
        fx, fy = self.tela_para_foto(*self.mouse)
        melhor, menor_area = None, None
        for k, (c, x1, y1, x2, y2) in enumerate(self.caixas):
            if x1 <= fx <= x2 and y1 <= fy <= y2:
                area = (x2 - x1) * (y2 - y1)
                if menor_area is None or area < menor_area:
                    melhor, menor_area = k, area
        return melhor

    # ---------- mouse ----------
    def ao_mouse(self, evento, x, y, flags, param):
        self.mouse = (x, y)
        if x < LARGURA_PAINEL:
            if evento == cv2.EVENT_LBUTTONDOWN:
                linha = (y - TOPO_LISTA) // LINHA
                if 0 <= linha < len(self.classes):
                    self.classe_atual = int(linha)
            return
        if evento == cv2.EVENT_LBUTTONDOWN:
            self.arrastando = True
            self.inicio = (x, y)
        elif evento == cv2.EVENT_LBUTTONUP and self.arrastando:
            self.arrastando = False
            x1, y1 = self.tela_para_foto(*self.inicio)
            x2, y2 = self.tela_para_foto(x, y)
            x1, x2 = sorted((x1, x2))
            y1, y2 = sorted((y1, y2))
            if (x2 - x1) >= MIN_CAIXA and (y2 - y1) >= MIN_CAIXA:
                self.caixas.append([self.classe_atual, x1, y1, x2, y2])
                self.sujo = True
        elif evento == cv2.EVENT_RBUTTONDOWN:
            k = self.caixa_sob_mouse()
            if k is not None:
                self.caixas.pop(k)
                self.sujo = True

    # ---------- desenho ----------
    def nome_classe(self, c):
        return self.classes[c] if 0 <= c < len(self.classes) else f"classe_{c}"

    def desenhar(self):
        w_disp = int(self.largura * self.escala)
        h_disp = int(self.altura * self.escala)
        altura_painel = TOPO_LISTA + LINHA * len(self.classes) + 230
        h_tela = max(h_disp, altura_painel)
        tela = np.full((h_tela, LARGURA_PAINEL + w_disp, 3), 30, np.uint8)
        tela[:h_disp, LARGURA_PAINEL:] = cv2.resize(self.imagem, (w_disp, h_disp), interpolation=cv2.INTER_AREA)

        sob = self.caixa_sob_mouse() if self.mouse[0] >= LARGURA_PAINEL else None
        for k, (c, x1, y1, x2, y2) in enumerate(self.caixas):
            cor = cor_da_classe(c)
            p1 = (int(x1 * self.escala) + LARGURA_PAINEL, int(y1 * self.escala))
            p2 = (int(x2 * self.escala) + LARGURA_PAINEL, int(y2 * self.escala))
            cv2.rectangle(tela, p1, p2, cor, 2 if k == sob else 1)
            rotulo = str(c)   # so o numero da classe, pequeno
            (tw, th), _ = cv2.getTextSize(rotulo, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            y_txt = max(p1[1], th + 4)
            cv2.rectangle(tela, (p1[0], y_txt - th - 4), (p1[0] + tw + 4, y_txt), cor, -1)
            cv2.putText(tela, rotulo, (p1[0] + 2, y_txt - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)

        if self.arrastando:
            cv2.rectangle(tela, self.inicio, self.mouse, cor_da_classe(self.classe_atual), 1)

        # painel de classes]
        cv2.putText(tela, "CLASSES (clique para escolher)", (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        for c, nome in enumerate(self.classes):
            y = TOPO_LISTA + c * LINHA
            if c == self.classe_atual:
                cv2.rectangle(tela, (4, y), (LARGURA_PAINEL - 4, y + LINHA - 2), (90, 90, 90), -1)
            cv2.rectangle(tela, (10, y + 5), (24, y + 19), cor_da_classe(c), -1)
            cv2.putText(tela, f"{c}  {nome}", (32, y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        y0 = TOPO_LISTA + len(self.classes) * LINHA + 24
        linhas = [
            f"Foto {self.indice + 1}/{len(self.arquivos)}: {self.arquivos[self.indice]}",
            f"Caixas nesta foto: {len(self.caixas)}",
            f"Fotos com label: {self.contar_com_label()}/{len(self.arquivos)}",
            "",
            "Mouse esq.: arrastar = nova caixa",
            "Mouse dir.: apagar caixa sob o mouse",
            "W / S: classe anterior / proxima",
            "E: troca a classe da caixa sob o mouse",
            "D ou seta dir.: proxima foto",
            "A ou seta esq.: foto anterior",
            "N: proxima foto SEM label",
            "U: desfazer ultima caixa",
            "X: apagar todas as caixas",
            "ESC: salvar e sair",
        ]
        for k, texto in enumerate(linhas):
            cv2.putText(tela, texto, (10, y0 + k * 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (210, 210, 210), 1, cv2.LINE_AA)
        return tela

    # ---------- teclado ----------
    def ao_teclar(self, tecla):
        """Devolve False quando o usuario pede para sair."""
        if tecla == -1:
            return True
        letra = chr(tecla & 0xFF).lower() if (tecla & 0xFF) < 128 and tecla < 256 else ""
        if tecla == 27:
            return False
        if letra == "d" or tecla == TECLA_DIR:
            self.carregar(self.indice + 1)
        elif letra == "a" or tecla == TECLA_ESQ:
            self.carregar(self.indice - 1)
        elif letra == "w":
            self.classe_atual = (self.classe_atual - 1) % len(self.classes)
        elif letra == "s":
            self.classe_atual = (self.classe_atual + 1) % len(self.classes)
        elif letra == "u" and self.caixas:
            self.caixas.pop()
            self.sujo = True
        elif letra == "x" and self.caixas:
            self.caixas = []
            self.sujo = True
        elif letra == "e":
            k = self.caixa_sob_mouse()
            if k is not None:
                self.caixas[k][0] = self.classe_atual
                self.sujo = True
        elif letra == "n":
            self.proxima_sem_label()
        return True


def pasta_area_de_trabalho():
    """Area de Trabalho real do usuario (funciona mesmo com OneDrive)."""
    try:
        import ctypes
        from ctypes import wintypes
        guid = ctypes.c_byte * 16
        # FOLDERID_Desktop = B4BFCC3A-DB2C-424C-B029-7FE99A87C641
        fid = (ctypes.c_ubyte * 16)(0x3A,0xCC,0xBF,0xB4,0x2C,0xDB,0x4C,0x42,0xB0,0x29,0x7F,0xE9,0x9A,0x87,0xC6,0x41)
        p = ctypes.c_wchar_p()
        if ctypes.windll.shell32.SHGetKnownFolderPath(ctypes.byref(fid), 0, None, ctypes.byref(p)) == 0:
            return p.value
    except Exception:
        pass
    return os.path.join(os.path.expanduser("~"), "Desktop")


def main():
    ap = argparse.ArgumentParser(description="Anotador de bounding boxes no formato YOLO")
    ap.add_argument("pasta", nargs="?", default=None,
                    help="pasta com as fotos (padrao: fotos_treino na Area de Trabalho)")
    ap.add_argument("classes", nargs="?", help="arquivo classes.txt (padrao: classes.txt dentro da pasta)")
    ap.add_argument("--altura", type=int, default=860, help="altura maxima da janela em pixels (padrao 860)")
    args = ap.parse_args()

    if args.pasta is None:
        args.pasta = os.path.join(pasta_area_de_trabalho(), "fotos_treino")
    print("Pasta de fotos:", args.pasta)
    if not os.path.isdir(args.pasta):
        sys.exit(f"Pasta nao encontrada: {args.pasta}")
    caminho_classes = args.classes
    if not caminho_classes:
        for cand in (os.path.join(args.pasta, "classes.txt"),
                     os.path.join(os.path.dirname(os.path.abspath(__file__)), "classes.txt")):
            if os.path.exists(cand):
                caminho_classes = cand
                break
        else:
            caminho_classes = os.path.join(args.pasta, "classes.txt")
    if not os.path.exists(caminho_classes):
        sys.exit(f"Nao encontrei o arquivo de classes: {caminho_classes}")
    classes = ler_classes(caminho_classes)
    print(f"{len(classes)} classes carregadas de {caminho_classes}")

    app = Anotador(args.pasta, classes, args.altura)
    cv2.namedWindow(JANELA, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(JANELA, app.ao_mouse)

    while True:
        cv2.imshow(JANELA, app.desenhar())
        if not app.ao_teclar(cv2.waitKeyEx(30)):
            break
        if cv2.getWindowProperty(JANELA, cv2.WND_PROP_VISIBLE) < 1:
            break
    app.salvar()
    cv2.destroyAllWindows()
    print(f"Pronto. Fotos com label: {app.contar_com_label()}/{len(app.arquivos)}")


if __name__ == "__main__":
    main()