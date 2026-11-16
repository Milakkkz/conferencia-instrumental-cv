"""
conferir_rotulos.py - confere os .txt (formato YOLO) de uma pasta de fotos e lista o que revisar.

Uso (deixe este arquivo na mesma pasta do detectar_onnx.py):
  python conferir_rotulos.py "C:\\...\\lote2"
  python conferir_rotulos.py "C:\\...\\lote2" --modelo "C:\\...\\piloto2.onnx"
  python conferir_rotulos.py "C:\\...\\fotos_teste" --modelo "C:\\...\\piloto2.onnx"     (usa planilha_teste.txt se existir)
  python conferir_rotulos.py "C:\\...\\lote2" --de 1355 --ate 1400 --contagem

O que confere (sem modelo):
  - linha invalida, classe fora de 0-14, caixa fora da foto
  - caixa duplicada (duas caixas quase iguais, de qualquer classe)
  - tamanho estranho comparado com a mesma classe nas outras fotos
  - contagem por classe: com planilha_teste.txt na pasta, compara com o que voce anotou (faltando / sobrando /
    desconhecido); sem planilha, so compara com o kit completo se usar --contagem
O que confere a mais (com --modelo): peca vista pelo modelo sem caixa, classe diferente da do modelo
(so quando a classe sugerida ainda tem vaga na foto) e caixa bem diferente da do modelo (curta demais ou larga demais).

Nada e alterado nos seus .txt. O resultado vai para o terminal e para conferir_relatorio.txt dentro da pasta.
Niveis: A = quase certeza | B = provavel, olhe a peca | C = sinal fraco, pode estar certo.
"""
import argparse, csv, glob, math, os, re, sys
from collections import Counter, defaultdict

import cv2
import numpy as np

NOMES = ["gancho_pequeno", "afastador_hohmann_8x236", "afastador_hohmann_mini_2x8", "afastador_mini_2x16",
         "retrator_senn_mueller", "pinca_reducao_dentes_14cm", "pinca_reducao_dentes_16cm", "pinca_verbrugge",
         "pinca_reducao_pontas_152mm", "retrator_weitlaner", "pinca_reducao_pontas_20cm", "modelador_pequeno",
         "descolador_williger", "retrator_farabeuf", "instrumento_desconhecido"]
KIT = {c: 1 for c in range(14)}
KIT.update({4: 2, 5: 2, 6: 2, 11: 2, 13: 2})        # unidades de cada classe no kit
ARTICULADAS = {5, 6, 7}                             # a caixa varia mais de tamanho conforme a abertura


def numero(caminho):
    m = re.findall(r"\d+", os.path.basename(caminho))
    return int(m[-1]) if m else -1


def posicao(cx, cy):
    v = "topo" if cy < 1 / 3 else ("meio" if cy < 2 / 3 else "base")
    h = "esquerda" if cx < 1 / 3 else ("centro" if cx < 2 / 3 else "direita")
    return f"{v}-{h}"


def iou_cont(a, b):
    """a, b = (x1, y1, x2, y2). Retorna (IoU, parte da menor caixa que esta dentro da outra)."""
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    aa = (a[2] - a[0]) * (a[3] - a[1]); ab = (b[2] - b[0]) * (b[3] - b[1])
    return inter / max(aa + ab - inter, 1e-12), inter / max(min(aa, ab), 1e-12)


def ler_rotulos(txt):
    """Retorna (caixas, erros). caixa = (classe, x1, y1, x2, y2) normalizada 0-1."""
    caixas, erros = [], []
    with open(txt, encoding="utf-8", errors="replace") as f:
        for i, linha in enumerate(f, 1):
            p = linha.split()
            if not p:
                continue
            try:
                c = int(p[0]); cx, cy, w, h = (float(v) for v in p[1:5])
                if len(p) != 5:
                    raise ValueError
            except ValueError:
                erros.append(f"linha {i} invalida: '{linha.strip()[:40]}'"); continue
            if not 0 <= c <= 14:
                erros.append(f"linha {i}: classe {c} fora de 0-14")
            if w <= 0 or h <= 0 or cx - w / 2 < -0.002 or cy - h / 2 < -0.002 or cx + w / 2 > 1.002 or cy + h / 2 > 1.002:
                erros.append(f"linha {i}: caixa da classe {c} invalida ou fora da foto")
            caixas.append((c, cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2))
    return caixas, erros


def ler_planilha(pasta):
    """Le planilha_teste.txt (ou .csv antigo): foto;faltando;sobrando;desconhecido;sobreposto;quais_faltam;quais_sobram."""
    cam = os.path.join(pasta, "planilha_teste.txt")
    if not os.path.exists(cam):
        cam = os.path.join(pasta, "planilha_teste.csv")
    if not os.path.exists(cam):
        return {}
    cols = ["foto", "faltando", "sobrando", "desconhecido", "sobreposto", "quais_faltam", "quais_sobram"]
    out = {}
    with open(cam, encoding="utf-8-sig") as f:
        for linha in f:
            if not linha.strip():
                continue
            sep = "\t" if "\t" in linha else ";"
            cel = [x.strip() for x in linha.rstrip("\r\n").split(sep)]
            cel = (cel + [""] * len(cols))[:len(cols)]
            if cel[0].lower() == "foto":
                continue
            l = dict(zip(cols, cel))
            nome = os.path.splitext(l["foto"])[0]
            esp = dict(KIT)
            for tok in l.get("quais_faltam", "").replace(",", " ").split():
                if tok.isdigit() and int(tok) in esp: esp[int(tok)] -= 1
            for tok in l.get("quais_sobram", "").replace(",", " ").split():
                if tok.isdigit() and int(tok) in esp: esp[int(tok)] += 1
            out[nome] = dict(esp=esp, desc=l.get("desconhecido") == "sim", sobre=l.get("sobreposto") == "sim")
    return out


def main():
    ap = argparse.ArgumentParser(description="Confere rotulos YOLO de uma pasta.")
    ap.add_argument("pasta")
    ap.add_argument("--modelo", help="arquivo .onnx para cruzar com os rotulos")
    ap.add_argument("--de", type=int, default=0); ap.add_argument("--ate", type=int, default=10 ** 9)
    ap.add_argument("--contagem", action="store_true", help="compara a contagem com o kit completo (sem planilha)")
    ap.add_argument("--conf-modelo", type=float, default=0.5, help="confianca minima do modelo para apontar (padrao 0.5)")
    a = ap.parse_args()

    fotos = sorted((f for f in glob.glob(os.path.join(a.pasta, "*")) if f.lower().endswith((".jpg", ".jpeg", ".png"))), key=numero)
    fotos = [f for f in fotos if a.de <= numero(f) <= a.ate]
    if not fotos:
        print("Nenhuma foto encontrada em", a.pasta); return

    dados = {}                                       # foto -> (caixas, erros)
    sem_txt = []
    for f in fotos:
        txt = os.path.splitext(f)[0] + ".txt"
        if os.path.exists(txt):
            dados[f] = ler_rotulos(txt)
        else:
            sem_txt.append(f)

    # tamanho tipico de cada classe (diagonal normalizada) nas proprias fotos da pasta
    diag = defaultdict(list)
    for caixas, _ in dados.values():
        for c, x1, y1, x2, y2 in caixas:
            diag[c].append(math.hypot(x2 - x1, y2 - y1))
    mediana = {c: float(np.median(v)) for c, v in diag.items() if len(v) >= 8}

    planilha = ler_planilha(a.pasta)
    detectar = None
    if a.modelo:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        try:
            import detectar_onnx as d
            sess = d.carregar(a.modelo); detectar = lambda img, conf: d.detectar(sess, img, conf)
            ler_img = d.ler_imagem
        except Exception as e:
            print("Nao consegui usar o modelo:", e, "\nDeixe conferir_rotulos.py na mesma pasta do detectar_onnx.py."); return

    achados = defaultdict(list)                      # numero -> [(nivel, texto)]
    for f, (caixas, erros) in dados.items():
        n = numero(f); nome = os.path.splitext(os.path.basename(f))[0]
        add = lambda nv, t: achados[n].append((nv, t))
        for e in erros:
            add("A", e)
        if not caixas:
            add("A", "arquivo .txt vazio (foto sem nenhuma caixa)")
        # duplicadas
        for i in range(len(caixas)):
            for j in range(i + 1, len(caixas)):
                ci, cj = caixas[i], caixas[j]
                io, ct = iou_cont(ci[1:], cj[1:])
                if io > 0.85 or (ct > 0.97 and io > 0.55):
                    cx = (ci[1] + ci[3]) / 2; cy = (ci[2] + ci[4]) / 2
                    nv = "A" if io > 0.85 else "B"
                    if ci[0] == cj[0]:
                        add(nv, f"2 caixas de classe {ci[0]} na mesma peca ({posicao(cx, cy)}): apague uma")
                    else:
                        add(nv, f"2 caixas sobre a mesma peca ({posicao(cx, cy)}): classes {ci[0]} e {cj[0]}; deixe so a certa")
        # tamanho
        for c, x1, y1, x2, y2 in caixas:
            if c in mediana:
                r = math.hypot(x2 - x1, y2 - y1) / mediana[c]
                lim = (0.55, 1.5) if c in ARTICULADAS else (0.65, 1.35)
                if r < lim[0] or r > lim[1]:
                    add("C", f"caixa classe {c} ({posicao((x1 + x2) / 2, (y1 + y2) / 2)}) com {r * 100:.0f}% do tamanho normal: confira se pega a peca inteira")
        # contagem
        cnt = Counter(c for c, *_ in caixas)
        if nome in planilha:
            p = planilha[nome]
            nv = "B" if p["sobre"] else "A"
            for c in range(14):
                if cnt[c] < p["esp"][c]:
                    add(nv, f"planilha diz {p['esp'][c]}x classe {c}, mas ha {cnt[c]} caixa(s): falta caixa" + (" (pode estar escondida na pilha)" if p["sobre"] else ""))
                elif cnt[c] > p["esp"][c]:
                    add(nv, f"planilha diz {p['esp'][c]}x classe {c}, mas ha {cnt[c]} caixas: sobra caixa ou classe errada")
            if p["desc"] and cnt[14] == 0:
                add("A", "planilha diz que tem desconhecido, mas nao ha caixa classe 14")
            if not p["desc"] and cnt[14] > 0:
                add("A", "planilha diz que NAO tem desconhecido, mas ha caixa classe 14")
        elif a.contagem:
            falta = [c for c in range(14) if cnt[c] < KIT[c]]; sobra = [c for c in range(14) if cnt[c] > KIT[c]]
            if sobra:
                add("B", "sobra caixa nas classes " + ", ".join(f"{c} (+{cnt[c] - KIT[c]})" for c in sobra))
            if falta:
                add("C", "faltam as classes " + ", ".join(str(c) for c in falta) + " (ok se a peca nao esta na foto)")
        # modelo
        if detectar:
            img = ler_img(f); H, W = img.shape[:2]
            dets = [(c, s, x1 / W, y1 / H, x2 / W, y2 / H) for c, s, x1, y1, x2, y2 in detectar(img, 0.25)]
            fortes = [d_ for d_ in dets if d_[1] >= a.conf_modelo]
            usados = set()
            for dc, ds, *db in fortes:
                melhor, mi = 0.0, None
                for k, (c, *b) in enumerate(caixas):
                    io, _ = iou_cont(db, b)
                    if io > melhor: melhor, mi = io, k
                cx, cy = (db[0] + db[2]) / 2, (db[1] + db[3]) / 2
                if mi is None or melhor < 0.3:
                    if dc != 14:
                        add("B", f"o modelo ve uma peca classe {dc} ({posicao(cx, cy)}, conf {ds:.2f}) sem caixa")
                    continue
                c = caixas[mi][0]
                if c != dc:
                    outros = [b for k, b in enumerate(caixas) if k != mi]
                    cheio = sum(1 for b in outros if b[0] == dc) >= KIT.get(dc, 99)
                    if not cheio and dc != 14 and c != 14 and ds >= 0.7:
                        add("B", f"caixa classe {c} ({posicao(cx, cy)}): o modelo acha que e {dc} (conf {ds:.2f})")
                elif melhor < 0.7:
                    ac = (caixas[mi][3] - caixas[mi][1]) * (caixas[mi][4] - caixas[mi][2]); am = (db[2] - db[0]) * (db[3] - db[1])
                    add("C", f"caixa classe {c} ({posicao(cx, cy)}) difere da do modelo (IoU {melhor:.2f}): sua caixa esta "
                             f"{'menor' if ac < am else 'maior'} que a dele; confira se cobre a peca inteira")
            for c, *b in caixas:
                if c == 14: continue
                if not any(dc == c and iou_cont(b, db)[0] > 0.3 for dc, ds, *db in dets):
                    add("C", f"caixa classe {c} ({posicao((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)}): o modelo nao confirma essa peca")

    ordem = {"A": 0, "B": 1, "C": 2}
    nA = sum(1 for v in achados.values() if any(t == "A" for t, _ in v))
    nB = sum(1 for v in achados.values() if not any(t == "A" for t, _ in v) and any(t == "B" for t, _ in v))
    nC = len(achados) - nA - nB
    linhas = [f"Pasta: {a.pasta}", f"Fotos conferidas: {len(dados)} | sem .txt: {len(sem_txt)} | modelo: {'sim' if detectar else 'nao'} | planilha: {'sim' if planilha else 'nao'}",
              f"Fotos com A: {nA} | so com B: {nB} | so com C: {nC} | sem nada: {len(dados) - len(achados)}", ""]
    if nA: linhas.append("FOTOS COM A: " + ", ".join(str(n) for n in sorted(achados) if any(t == "A" for t, _ in achados[n])))
    if nB: linhas.append("FOTOS SO COM B: " + ", ".join(str(n) for n in sorted(achados) if not any(t == "A" for t, _ in achados[n]) and any(t == "B" for t, _ in achados[n])))
    if sem_txt: linhas.append("FOTOS SEM .txt: " + ", ".join(str(numero(f)) for f in sem_txt[:60]) + (" ..." if len(sem_txt) > 60 else ""))
    linhas.append("")
    for n in sorted(achados):
        linhas.append(f"img_{n:04d}")
        for nv, t in sorted(achados[n], key=lambda z: ordem[z[0]]):
            linhas.append(f"  [{nv}] {t}")
    texto = "\n".join(linhas)
    print(texto)
    with open(os.path.join(a.pasta, "conferir_relatorio.txt"), "w", encoding="utf-8") as fh:
        fh.write(texto + "\n")
    print("\nRelatorio salvo em", os.path.join(a.pasta, "conferir_relatorio.txt"))


if __name__ == "__main__":
    main()
