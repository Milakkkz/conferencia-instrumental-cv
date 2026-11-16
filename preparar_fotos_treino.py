r"""Reduz as fotos de treino para 1024x768 (JPEG 80), copia os rotulos .txt e gera o zip que vai para o Colab.

Uso: python preparar_fotos_treino.py            (usa Area de Trabalho\fotos_treino)
     python preparar_fotos_treino.py "C:\caminho\fotos_treino"
Saida: fotos_treino_leve\ e fotos_treino_leve.zip ao lado da pasta de origem.
"""
import ctypes, os, shutil, sys
from pathlib import Path
import cv2, numpy as np


def area_de_trabalho():
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


origem = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(area_de_trabalho()) / "fotos_treino"
destino = origem.parent / "fotos_treino_leve"
if destino.exists():
    shutil.rmtree(destino)                       # evita sobrar foto antiga de uma rodada anterior
destino.mkdir()

n = 0
for p in sorted(origem.glob("*.jpg")):
    img = cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_COLOR)
    img = cv2.resize(img, (1024, 768), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
    buf.tofile(str(destino / p.name))
    t = p.with_suffix(".txt")                    # rotulo YOLO (coordenadas normalizadas: nao muda com o tamanho)
    if t.exists():
        (destino / t.name).write_bytes(t.read_bytes())
    n += 1
shutil.make_archive(str(destino), "zip", destino)
print(f"pronto: {n} fotos em {destino} e {destino}.zip")
