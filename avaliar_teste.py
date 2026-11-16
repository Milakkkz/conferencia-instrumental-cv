"""Avalia o modelo final no conjunto de teste (pasta fotos_teste). Roda no PC, sem PyTorch.

Precisa:
  - detectar_onnx.py na MESMA pasta deste script
  - modelo_final.onnx (e o classes.txt ao lado dele)
  - planilha_teste.txt dentro da pasta das fotos (gerada pelo camera_teste.py)
  - opcional: .txt de rotulo (caixas) ao lado das fotos -> calcula tambem precisao, revocacao, mAP50 e mAP50-95

Uso (terminal do PyCharm):
  python avaliar_teste.py "C:\\...\\modelo_final.onnx"
  python avaliar_teste.py "C:\\...\\modelo_final.onnx" --pasta "C:\\...\\fotos_teste" --conf 0.5

Sem argumentos (botao Run): usa modelo_final.onnx ao lado do script e a pasta fotos_teste da Area de Trabalho.

Saidas, dentro da pasta das fotos, em resultado_teste\\ :
  relatorio_teste.txt   -> todos os numeros para o artigo
  resultado_por_foto.txt -> uma linha por foto (abre no Bloco de Notas)
  exemplos\\            -> 1 foto desenhada por situacao (para figuras do artigo)
  erros\\               -> fotos em que o diagnostico do sistema errou, desenhadas
"""
import argparse, ctypes, datetime, glob, math, os, platform, re, sys, time
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import detectar_onnx as D          # reaproveita carregar(), detectar(), ler_imagem() e desenhar()

KIT = {c: 1 for c in range(14)}
KIT.update({4: 2, 5: 2, 6: 2, 11: 2, 13: 2})     # unidades de cada classe no kit (19 pecas)
DESC = 14                                         # classe instrumento_desconhecido
NOMES_PADRAO = ["gancho_pequeno", "afastador_hohmann_8x236", "afastador_hohmann_mini_2x8", "afastador_mini_2x16",
                "retrator_senn_mueller", "pinca_reducao_dentes_14cm", "pinca_reducao_dentes_16cm", "pinca_verbrugge",
                "pinca_reducao_pontas_152mm", "retrator_weitlaner", "pinca_reducao_pontas_20cm", "modelador_pequeno",
                "descolador_williger", "retrator_farabeuf", "instrumento_desconhecido"]
ORDEM_SIT = [(e, d, s) for s in (0, 1) for e in ("completa", "faltando", "com_mais", "faltando_com_mais") for d in (0, 1)]


# ---------------------------------------------------------------- utilitarios
def pasta_area_de_trabalho():
    try:
        from ctypes import wintypes
        class GUID(ctypes.Structure):
            _fields_ = [("a", wintypes.DWORD), ("b", wintypes.WORD), ("c", wintypes.WORD), ("d", ctypes.c_ubyte * 8)]
        fid = GUID(0xB4BFCC3A, 0xDB2C, 0x424C, (ctypes.c_ubyte * 8)(0xB0, 0x29, 0x7F, 0xE9, 0x9A, 0x87, 0xC6, 0x41))
        ptr = ctypes.c_wchar_p()
        if ctypes.windll.shell32.SHGetKnownFolderPath(ctypes.byref(fid), 0, None, ctypes.byref(ptr)) == 0:
            caminho = ptr.value
            ctypes.windll.ole32.CoTaskMemFree(ptr)
            return caminho
    except Exception:
        pass
    return os.path.join(os.path.expanduser("~"), "Desktop")


def wilson(k, n, z=1.96):
    """Intervalo de confianca de Wilson (95%) para uma proporcao k/n."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    centro = (p + z * z / (2 * n)) / den
    meia = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, centro - meia), min(1.0, centro + meia))


def pct(k, n):
    if n == 0:
        return "  -  "
    a, b = wilson(k, n)
    return f"{k}/{n} = {100 * k / n:5.1f}%  (IC95% {100 * a:4.1f}-{100 * b:5.1f}%)"


def nome_sit(k):
    e, d, s = k
    return f"{e}{' + desconhecido' if d else ''}{' + sobreposto' if s else ''}"


def ler_planilha(pasta):
    cam = os.path.join(pasta, "planilha_teste.txt")
    if not os.path.exists(cam):
        cam = os.path.join(pasta, "planilha_teste.csv")
    if not os.path.exists(cam):
        sys.exit(f"ERRO: nao achei planilha_teste.txt em {pasta}")
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
            esp = dict(KIT)
            for t in l["quais_faltam"].replace(",", " ").split():
                if t.isdigit() and int(t) in esp:
                    esp[int(t)] -= 1
            for t in l["quais_sobram"].replace(",", " ").split():
                if t.isdigit() and int(t) in esp:
                    esp[int(t)] += 1
            falt, sobr = l["faltando"] == "sim", l["sobrando"] == "sim"
            estado = "faltando_com_mais" if falt and sobr else "faltando" if falt else "com_mais" if sobr else "completa"
            desc, sobre = l["desconhecido"] == "sim", l["sobreposto"] == "sim"
            out[os.path.splitext(l["foto"])[0]] = dict(esp=esp, desc=desc, sobre=sobre, estado=estado,
                                                       sit=(estado, int(desc), int(sobre)),
                                                       qf=l["quais_faltam"], qs=l["quais_sobram"])
    return out


def ler_rotulos(txt, W, H):
    caixas = []
    with open(txt, encoding="utf-8") as f:
        for linha in f:
            p = linha.split()
            if len(p) < 5:
                continue
            c, x, y, w, h = int(float(p[0])), *map(float, p[1:5])
            caixas.append((c, (x - w / 2) * W, (y - h / 2) * H, (x + w / 2) * W, (y + h / 2) * H))
    return caixas


def iou_matriz(a, b):
    """a: (N,4) b: (M,4) em xyxy -> (N,M)."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    a = np.asarray(a, float)[:, None, :]
    b = np.asarray(b, float)[None, :, :]
    ix = np.clip(np.minimum(a[..., 2], b[..., 2]) - np.maximum(a[..., 0], b[..., 0]), 0, None)
    iy = np.clip(np.minimum(a[..., 3], b[..., 3]) - np.maximum(a[..., 1], b[..., 1]), 0, None)
    inter = ix * iy
    area_a = (a[..., 2] - a[..., 0]) * (a[..., 3] - a[..., 1])
    area_b = (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])
    return inter / np.maximum(area_a + area_b - inter, 1e-9)


def dedup_iou(dets, lim):
    """Remove a caixa de menor confianca quando duas caixas (qualquer classe) tem IoU > lim."""
    dets = sorted(dets, key=lambda d: -d[1])
    out = []
    for d in dets:
        if all(iou_matriz([d[2:6]], [o[2:6]])[0, 0] <= lim for o in out):
            out.append(d)
    return out


def ap_interp(rec, prec):
    """AP com interpolacao em 101 pontos (mesmo metodo do Ultralytics/COCO)."""
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([1.0], prec, [0.0]))
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    x = np.linspace(0, 1, 101)
    trap = getattr(np, "trapezoid", None) or np.trapz
    return float(trap(np.interp(x, mrec, mpre), x))


def casar(pred, gt, limiar):
    """Casamento guloso por confianca, mesma classe. pred: [(conf, box)], gt: [box]. Retorna lista TP (bool) por pred."""
    tp = []
    usado = [False] * len(gt)
    if len(gt):
        M = iou_matriz([p[1] for p in pred], gt) if pred else np.zeros((0, len(gt)))
    for i, _ in enumerate(pred):
        j_melhor, v_melhor = -1, limiar
        for j in range(len(gt)):
            if not usado[j] and M[i, j] >= v_melhor:
                j_melhor, v_melhor = j, M[i, j]
        if j_melhor >= 0:
            usado[j_melhor] = True
            tp.append(True)
        else:
            tp.append(False)
    return tp


def metricas_deteccao(preds, gts, nc, conf_limiar):
    """preds/gts: dict foto -> lista. preds: (c, conf, x1,y1,x2,y2); gts: (c, x1,y1,x2,y2).
    Retorna AP50 e AP50-95 por classe, P e R no limiar de confianca (IoU 0.5) por classe."""
    limiares = np.linspace(0.5, 0.95, 10)
    res = {}
    for c in range(nc):
        n_gt = sum(sum(1 for g in gts[f] if g[0] == c) for f in gts)
        if n_gt == 0:
            continue
        lista = []                          # (conf, foto, box)
        for f in gts:
            for p in preds.get(f, []):
                if p[0] == c:
                    lista.append((p[1], f, p[2:6]))
        lista.sort(key=lambda t: -t[0])
        aps = []
        for lim in limiares:
            tp_total = np.zeros(len(lista), bool)
            por_foto = {}
            for k, (cf, f, b) in enumerate(lista):
                por_foto.setdefault(f, []).append((k, cf, b))
            for f, itens in por_foto.items():
                g = [x[1:5] for x in gts[f] if x[0] == c]
                tp = casar([(cf, b) for _, cf, b in itens], g, lim)
                for (k, _, _), t in zip(itens, tp):
                    tp_total[k] = t
            if len(lista) == 0:
                aps.append(0.0)
                continue
            tpc = np.cumsum(tp_total)
            fpc = np.cumsum(~tp_total)
            rec = tpc / n_gt
            prec = tpc / np.maximum(tpc + fpc, 1e-9)
            aps.append(ap_interp(rec, prec))
            if lim == 0.5:
                conf_arr = np.array([t[0] for t in lista])
                m = conf_arr >= conf_limiar
                tp_l = int(tp_total[m].sum())
                fp_l = int(m.sum() - tp_l)
                res_c = dict(tp=tp_l, fp=fp_l, fn=n_gt - tp_l, n_gt=n_gt)
        if len(lista) == 0:
            res_c = dict(tp=0, fp=0, fn=n_gt, n_gt=n_gt)
        res_c.update(ap50=aps[0], ap5095=float(np.mean(aps)))
        res[c] = res_c
    return res


def confusao(preds, gts, nc, conf_limiar):
    """Matriz [classe real][classe prevista] com casamento sem olhar a classe (IoU >= 0.5).
    Ultima coluna = nao detectado; ultima linha = deteccao sem peca real (falso positivo)."""
    M = np.zeros((nc + 1, nc + 1), int)
    for f in gts:
        g = gts[f]
        p = sorted([x for x in preds.get(f, []) if x[1] >= conf_limiar], key=lambda t: -t[1])
        usado_g = [False] * len(g)
        I = iou_matriz([x[2:6] for x in p], [x[1:5] for x in g]) if p and g else None
        for i, pi in enumerate(p):
            j_m, v_m = -1, 0.5
            for j in range(len(g)):
                if not usado_g[j] and I[i, j] >= v_m:
                    j_m, v_m = j, I[i, j]
            if j_m >= 0:
                usado_g[j_m] = True
                M[g[j_m][0], pi[0]] += 1
            else:
                M[nc, pi[0]] += 1
        for j, gj in enumerate(g):
            if not usado_g[j]:
                M[gj[0], nc] += 1
    return M


# ---------------------------------------------------------------- principal
def main():
    pasta_script = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description="Avalia o modelo final nas fotos de teste.")
    ap.add_argument("modelo", nargs="?", default=os.path.join(pasta_script, "modelo_final.onnx"))
    ap.add_argument("--pasta", default=os.path.join(pasta_area_de_trabalho(), "fotos_teste"))
    ap.add_argument("--conf", type=float, default=0.5, help="limiar de confianca para contar uma peca (padrao 0.5)")
    ap.add_argument("--dedup", type=float, default=0.0,
                    help="IoU para juntar caixas repetidas da mesma peca antes de contar (0 = desligado, padrao)")
    ap.add_argument("--sem-imagens", action="store_true", help="nao salva as fotos desenhadas (mais rapido)")
    a = ap.parse_args()

    if not os.path.exists(a.modelo):
        sys.exit(f"ERRO: nao achei o modelo: {a.modelo}")
    if not os.path.isdir(a.pasta):
        sys.exit(f"ERRO: nao achei a pasta: {a.pasta}")

    planilha = ler_planilha(a.pasta)
    impossiveis = [f"{f} (classe {c}: {v['esp'][c]} unidades)" for f, v in sorted(planilha.items())
                   for c in range(14) if v["esp"][c] < 0]
    if impossiveis:
        print("ATENCAO: a planilha tem contagem impossivel (faltam mais pecas do que o kit tem). Corrija e rode de novo:")
        for x in impossiveis:
            print("  -", x)
        sys.exit(1)
    fotos = sorted(glob.glob(os.path.join(a.pasta, "*.jpg")))
    fotos = [f for f in fotos if os.path.splitext(os.path.basename(f))[0] in planilha]
    if not fotos:
        sys.exit("ERRO: nenhuma foto da planilha foi encontrada na pasta.")

    sess = D.carregar(a.modelo)
    provedor = sess.get_providers()[0]
    tam = sess.get_inputs()[0].shape[2]
    cj = os.path.join(os.path.dirname(os.path.abspath(a.modelo)), "classes.txt")
    nomes = [l.strip() for l in open(cj, encoding="utf-8") if l.strip()] if os.path.exists(cj) else NOMES_PADRAO
    nc = len(nomes)

    saida = os.path.join(a.pasta, "resultado_teste")
    pasta_ex = os.path.join(saida, "exemplos")
    pasta_er = os.path.join(saida, "erros")
    for p in (saida, pasta_ex, pasta_er):
        os.makedirs(p, exist_ok=True)
    for p in (pasta_ex, pasta_er):                       # limpa desenhos de uma rodada anterior
        for f in glob.glob(os.path.join(p, "*.jpg")):
            os.remove(f)

    # aquecimento (a primeira inferencia e sempre mais lenta; nao entra na media)
    D.detectar(sess, D.ler_imagem(fotos[0]), 0.001)

    linhas, preds_all, gts_all, tempos_inf, tempos_tot = [], {}, {}, [], []
    feitos_exemplo = set()
    print(f"Avaliando {len(fotos)} fotos com {os.path.basename(a.modelo)} (entrada {tam}, {provedor}) ...")
    for i, f in enumerate(fotos, 1):
        nome = os.path.splitext(os.path.basename(f))[0]
        info = planilha[nome]
        t0 = time.perf_counter()
        img = D.ler_imagem(f)
        H, W = img.shape[:2]
        t1 = time.perf_counter()
        dets_todas = D.detectar(sess, img, 0.001)            # todas as deteccoes (para o mAP)
        t2 = time.perf_counter()
        dets = [d for d in dets_todas if d[1] >= a.conf]     # as que o sistema conta
        if a.dedup > 0:
            dets = dedup_iou(dets, a.dedup)
        prev = {c: 0 for c in range(nc)}
        for d in dets:
            prev[d[0]] = prev.get(d[0], 0) + 1
        esp = info["esp"]
        ok_cont = all(prev.get(c, 0) == esp[c] for c in range(14)) and ((prev.get(DESC, 0) > 0) == info["desc"])
        esp_conf = info["estado"] == "completa" and not info["desc"]
        prev_conf = all(prev.get(c, 0) == KIT[c] for c in range(14)) and prev.get(DESC, 0) == 0
        t3 = time.perf_counter()
        tempos_inf.append((t2 - t1) * 1000)
        tempos_tot.append((t3 - t0) * 1000)

        faltam_prev = " ".join(str(c) for c in range(14) for _ in range(max(0, KIT[c] - prev.get(c, 0))))
        sobram_prev = " ".join(str(c) for c in range(14) for _ in range(max(0, prev.get(c, 0) - KIT[c])))
        sub = [c for c in range(14) if prev.get(c, 0) < esp[c]]
        sup = [c for c in range(14) if prev.get(c, 0) > esp[c]]
        linhas.append(dict(foto=nome, sit=info["sit"], esp_conf=esp_conf, prev_conf=prev_conf, ok_ver=esp_conf == prev_conf,
                           ok_diag=ok_cont, desc=info["desc"], desc_prev=prev.get(DESC, 0), sub=sub, sup=sup,
                           faltam_prev=faltam_prev, sobram_prev=sobram_prev, qf=info["qf"], qs=info["qs"],
                           ms=tempos_inf[-1], n=len(dets)))

        txt = os.path.splitext(f)[0] + ".txt"
        if os.path.exists(txt):
            gts_all[nome] = ler_rotulos(txt, W, H)
            preds_all[nome] = dets_todas

        if not a.sem_imagens:
            precisa_ex = info["sit"] not in feitos_exemplo
            if precisa_ex or not ok_cont:
                vis = D.desenhar(img.copy(), dets, nomes)
                msg1 = f"{nome} | {nome_sit(info['sit'])} | esperado: faltam [{info['qf']}] sobram [{info['qs']}]"
                msg2 = f"sistema: faltam [{faltam_prev}] sobram [{sobram_prev}] desconhecido {prev.get(DESC, 0)} | " \
                       f"{'CONFORME' if prev_conf else 'NAO CONFORME'} | {'ACERTOU' if ok_cont else 'ERROU'}"
                for y, m in ((45, msg1), (95, msg2)):
                    cv2.putText(vis, m, (15, y), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 6)
                    cv2.putText(vis, m, (15, y), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)
                if precisa_ex:
                    e, d, s = info["sit"]
                    ok_, buf = cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 90])
                    buf.tofile(os.path.join(pasta_ex, f"{e}_desc{d}_sobre{s}_{nome}.jpg"))
                    feitos_exemplo.add(info["sit"])
                if not ok_cont:
                    ok_, buf = cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 90])
                    buf.tofile(os.path.join(pasta_er, f"{nome}.jpg"))
        if i % 20 == 0 or i == len(fotos):
            print(f"  {i}/{len(fotos)}")

    # ------------------------------------------------------------ relatorio
    R = []
    add = R.append
    n = len(linhas)
    add("RELATORIO DO TESTE - conferencia automatica do Instrumental de Apoio Pequenos Fragmentos")
    add(f"Data: {datetime.datetime.now():%d/%m/%Y %H:%M}")
    add(f"Modelo: {os.path.basename(a.modelo)} | entrada {tam}x{tam} | provedor {provedor}")
    add(f"Limiar de confianca para contar: {a.conf} | juntar caixas repetidas: {'IoU > ' + str(a.dedup) if a.dedup > 0 else 'nao'}")
    add(f"Computador: {platform.processor() or platform.machine()} | {os.cpu_count()} nucleos logicos | {platform.system()} {platform.release()}")
    add(f"Python {platform.python_version()} | onnxruntime {D.ort.__version__} | OpenCV {cv2.__version__}")
    add(f"Fotos avaliadas: {n} | com rotulo de caixas: {len(gts_all)}")
    add("")
    add("1) CONFORMIDADE DA CAIXA (todas as fotos; gabarito = planilha_teste.txt)")
    add("   Veredito correto = o sistema disse CONFORME/NAO CONFORME igual ao gabarito.")
    add("   Diagnostico correto = acertou a quantidade de TODAS as 14 classes e a presenca de desconhecido.")
    k_ver = sum(l["ok_ver"] for l in linhas)
    k_diag = sum(l["ok_diag"] for l in linhas)
    nao_conf = [l for l in linhas if not l["esp_conf"]]
    conf_ = [l for l in linhas if l["esp_conf"]]
    fc = sum(1 for l in nao_conf if l["prev_conf"])
    fnc = sum(1 for l in conf_ if not l["prev_conf"])
    add(f"   Veredito correto ............ {pct(k_ver, n)}")
    add(f"   Diagnostico correto ......... {pct(k_diag, n)}")
    add(f"   FALSO CONFORME .............. {pct(fc, len(nao_conf))}   <- caixa errada aprovada (erro critico)")
    add(f"   Falso nao conforme .......... {pct(fnc, len(conf_))}   <- caixa certa reprovada (retrabalho)")
    add("")
    add("2) POR GRUPO (diagnostico correto)")
    for titulo, chave in (("Sobreposicao", lambda l: "sim" if l["sit"][2] else "nao"),
                          ("Desconhecido", lambda l: "sim" if l["sit"][1] else "nao"),
                          ("Estado", lambda l: l["sit"][0])):
        grupos = {}
        for l in linhas:
            grupos.setdefault(chave(l), []).append(l)
        for g in sorted(grupos):
            ls = grupos[g]
            add(f"   {titulo:13s} {g:18s} diagnostico {pct(sum(x['ok_diag'] for x in ls), len(ls))}"
                f" | veredito {sum(x['ok_ver'] for x in ls)}/{len(ls)}")
    add("")
    add("3) POR SITUACAO (16 situacoes)")
    add(f"   {'situacao':48s} {'n':>3s} {'veredito':>9s} {'diagnost.':>10s}")
    for k in ORDEM_SIT:
        ls = [l for l in linhas if l["sit"] == k]
        if ls:
            add(f"   {nome_sit(k):48s} {len(ls):3d} {sum(x['ok_ver'] for x in ls):5d}/{len(ls):<3d} "
                f"{sum(x['ok_diag'] for x in ls):6d}/{len(ls):<3d}")
    add("")
    add("4) ERROS DE CONTAGEM POR CLASSE (numero de fotos; ordem de Pareto)")
    add("   sub = contou MENOS que o real (peca nao vista) | super = contou MAIS (peca inventada ou classe trocada)")
    err = []
    for c in range(14):
        s_ = sum(c in l["sub"] for l in linhas)
        p_ = sum(c in l["sup"] for l in linhas)
        err.append((s_ + p_, c, s_, p_))
    tot_err = sum(e[0] for e in err)
    acum = 0
    for t, c, s_, p_ in sorted(err, reverse=True):
        acum += t
        add(f"   {c:2d} {nomes[c] if c < len(nomes) else c:28s} sub {s_:3d} | super {p_:3d} | total {t:3d}"
            f" | acumulado {100 * acum / tot_err if tot_err else 0:5.1f}%")
    com_desc = [l for l in linhas if l["desc"]]
    sem_desc = [l for l in linhas if not l["desc"]]
    add("")
    add("5) INSTRUMENTO DESCONHECIDO")
    add(f"   Fotos com desconhecido em que ele foi detectado .... {pct(sum(l['desc_prev'] > 0 for l in com_desc), len(com_desc))}")
    add(f"   Fotos sem desconhecido com desconhecido falso ...... {pct(sum(l['desc_prev'] > 0 for l in sem_desc), len(sem_desc))}")
    add("")

    if gts_all:
        add(f"6) DETECCAO NAS {len(gts_all)} FOTOS COM CAIXAS (gabarito = .txt de rotulo)")
        # coerencia gabarito: caixas x planilha
        incoer = []
        for f, g in gts_all.items():
            cont = {}
            for x in g:
                cont[x[0]] = cont.get(x[0], 0) + 1
            info = planilha[f]
            dif = [c for c in range(14) if cont.get(c, 0) != info["esp"][c]]
            if dif or (cont.get(DESC, 0) > 0) != info["desc"]:
                incoer.append(f"{f} (classes {' '.join(map(str, dif)) or '-'}; desconhecido caixas {cont.get(DESC, 0)} / planilha {'sim' if info['desc'] else 'nao'})")
        if incoer:
            add(f"   ATENCAO: {len(incoer)} foto(s) com caixas que NAO batem com a planilha (corrija antes de usar os numeros):")
            for x in incoer:
                add(f"     - {x}")
        met = metricas_deteccao(preds_all, gts_all, nc, a.conf)
        TP = sum(m["tp"] for m in met.values())
        FP = sum(m["fp"] for m in met.values())
        FN = sum(m["fn"] for m in met.values())
        add(f"   Precisao (conf >= {a.conf}, IoU 0,5) .... {TP}/{TP + FP} = {100 * TP / max(TP + FP, 1):.1f}%")
        add(f"   Revocacao (conf >= {a.conf}, IoU 0,5) ... {TP}/{TP + FN} = {100 * TP / max(TP + FN, 1):.1f}%")
        add(f"   mAP50 ................................. {100 * np.mean([m['ap50'] for m in met.values()]):.1f}%")
        add(f"   mAP50-95 .............................. {100 * np.mean([m['ap5095'] for m in met.values()]):.1f}%")
        add(f"   {'classe':32s} {'inst':>5s} {'P':>6s} {'R':>6s} {'AP50':>6s} {'AP50-95':>8s}")
        for c, m in met.items():
            P = m["tp"] / max(m["tp"] + m["fp"], 1)
            Rr = m["tp"] / max(m["n_gt"], 1)
            add(f"   {c:2d} {nomes[c] if c < len(nomes) else c:29s} {m['n_gt']:5d} {100 * P:6.1f} {100 * Rr:6.1f} "
                f"{100 * m['ap50']:6.1f} {100 * m['ap5095']:8.1f}")
        M = confusao(preds_all, gts_all, nc, a.conf)
        add("")
        add("   Matriz de confusao (linha = classe real, coluna = prevista; ND = nao detectado; FP = deteccao sem peca)")
        cab = "      " + "".join(f"{c:>4d}" for c in range(nc)) + "  ND"
        add(cab)
        for r in range(nc + 1):
            rot = f"{r:4d}" if r < nc else "  FP"
            add(f"   {rot} " + "".join(f"{M[r, c]:>4d}" for c in range(nc + 1)))
        trocas = [(M[r, c], r, c) for r in range(nc) for c in range(nc) if r != c and M[r, c] > 0]
        if trocas:
            add("   Trocas de classe mais frequentes:")
            for v, r, c in sorted(trocas, reverse=True)[:8]:
                add(f"     {v}x  {nomes[r]} -> detectado como {nomes[c]}")
        add("")

    ti = np.array(tempos_inf)
    tt = np.array(tempos_tot)
    add("7) TEMPO POR FOTO (sem a primeira, de aquecimento)")
    add(f"   Inferencia (pre + rede + pos) .. media {ti.mean():.0f} ms | desvio {ti.std(ddof=1) if len(ti) > 1 else 0:.0f} ms | maximo {ti.max():.0f} ms")
    add(f"   Total (ler foto + inferencia + comparacao com o checklist) .. media {tt.mean():.0f} ms | maximo {tt.max():.0f} ms")
    add(f"   Criterio do artigo (< 5 s por imagem): {'ATENDIDO' if tt.max() < 5000 else 'NAO ATENDIDO'}")
    add("")
    add("Arquivos: resultado_por_foto.txt | exemplos\\ (1 foto por situacao) | erros\\ (fotos com diagnostico errado)")

    with open(os.path.join(saida, "relatorio_teste.txt"), "w", encoding="utf-8-sig", newline="\r\n") as f:
        f.write("\n".join(R) + "\n")

    cols = [("foto", 12), ("situacao", 46), ("esperado", 13), ("sistema", 13), ("veredito", 9), ("diagnostico", 12),
            ("faltam_real", 14), ("faltam_sist", 14), ("sobram_real", 12), ("sobram_sist", 12), ("desc_sist", 10), ("ms", 6)]
    with open(os.path.join(saida, "resultado_por_foto.txt"), "w", encoding="utf-8-sig", newline="\r\n") as f:
        f.write(";".join(c.ljust(w) for c, w in cols).rstrip() + "\n")
        for l in linhas:
            v = [l["foto"], nome_sit(l["sit"]), "CONFORME" if l["esp_conf"] else "NAO_CONFORME",
                 "CONFORME" if l["prev_conf"] else "NAO_CONFORME", "ok" if l["ok_ver"] else "ERRO",
                 "ok" if l["ok_diag"] else "ERRO", l["qf"], l["faltam_prev"], l["qs"], l["sobram_prev"],
                 str(l["desc_prev"]), f"{l['ms']:.0f}"]
            f.write(";".join(str(x).ljust(w) for x, (_, w) in zip(v, cols)).rstrip() + "\n")

    print("\n".join(R))
    print(f"\nSalvo em: {saida}")


if __name__ == "__main__":
    main()
