import argparse, os, re, shutil, ctypes
from ctypes import wintypes

PREFIXO = "teste_"          # nome das fotos: teste_0001.jpg, teste_0001.txt (rotulo, se existir)

# ===== VALORES PADRAO (usados quando voce clica em Run, sem argumentos) =====
DE = 50       # primeira foto a mover
ATE = 75      # ultima foto a mover
SOMA = 1      # quanto somar (50 vira 51, 75 vira 76)


def pasta_area_de_trabalho():                            # descobre a Area de Trabalho real (inclusive com OneDrive)
    try:
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


def main():
    ap = argparse.ArgumentParser(description="Renumera fotos teste_NNNN (jpg, txt de rotulo e linhas da planilha).")
    ap.add_argument("--pasta", default=os.path.join(pasta_area_de_trabalho(), "fotos_teste"))
    ap.add_argument("--de", type=int, default=DE, help="primeiro numero a mover")
    ap.add_argument("--ate", type=int, default=ATE, help="ultimo numero a mover")
    ap.add_argument("--soma", type=int, default=SOMA, help="quanto somar (1 = 50 vira 51); negativo diminui")
    ap.add_argument("--simular", action="store_true", help="so mostra o que faria, sem mexer em nada")
    a = ap.parse_args()

    pasta = a.pasta
    if not os.path.isdir(pasta):
        print("ERRO: pasta nao encontrada:", pasta); return
    arq = {}
    for nome in os.listdir(pasta):
        m = re.fullmatch(re.escape(PREFIXO) + r"(\d+)\.(jpg|txt)", nome)
        if m:
            arq[(int(m.group(1)), m.group(2))] = nome

    mover = sorted([k for k in arq if a.de <= k[0] <= a.ate], reverse=a.soma > 0)   # do maior para o menor evita sobrescrever
    if not mover:
        print("Nenhuma foto entre", a.de, "e", a.ate); return
    destinos = {(n + a.soma, ext) for n, ext in mover}
    conflito = [k for k in destinos if k in arq and not (a.de <= k[0] <= a.ate)]
    if conflito or any(n + a.soma < 1 for n, _ in mover):
        print("ERRO: o destino ja existe ou ficaria menor que 1:", sorted(conflito)[:5], "- nada foi alterado.")
        return

    print(f"Pasta: {pasta}")
    print(f"Vai mover {len([k for k in mover if k[1] == 'jpg'])} foto(s): {a.de}..{a.ate}  ->  {a.de + a.soma}..{a.ate + a.soma}")
    if a.simular:
        print("(simulacao: nada foi alterado)"); return

    plan = os.path.join(pasta, "planilha_teste.txt")
    if os.path.exists(plan):
        shutil.copy(plan, os.path.join(pasta, "planilha_teste_backup.txt"))   # copia de seguranca
    for n, ext in mover:
        os.rename(os.path.join(pasta, arq[(n, ext)]), os.path.join(pasta, f"{PREFIXO}{n + a.soma:04d}.{ext}"))

    if os.path.exists(plan):                             # acerta os nomes na planilha, mantendo o alinhamento
        with open(plan, encoding="utf-8-sig") as f:
            linhas = f.read().splitlines()
        def troca(l):
            m = re.match(re.escape(PREFIXO) + r"(\d+)(\.jpg)", l)
            if m and a.de <= int(m.group(1)) <= a.ate:
                return f"{PREFIXO}{int(m.group(1)) + a.soma:04d}{m.group(2)}" + l[m.end():]
            return l
        linhas = [troca(l) for l in linhas]
        cab, dados = linhas[:1], linhas[1:]
        dados.sort(key=lambda l: (int(re.match(re.escape(PREFIXO) + r"(\d+)", l).group(1)) if re.match(re.escape(PREFIXO) + r"\d+", l) else 10 ** 9))
        with open(plan, "w", encoding="utf-8-sig", newline="\r\n") as f:
            f.write("\n".join(cab + dados) + "\n")
    print("Pronto. Backup da planilha: planilha_teste_backup.txt")
    print(f"O numero {a.de} ficou livre: a proxima foto que a camera tirar vai ser {PREFIXO}{a.de:04d}.jpg")


if __name__ == "__main__":
    main()
