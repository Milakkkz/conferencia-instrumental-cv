# Conferência de instrumental cirúrgico por visão computacional

Protótipo desenvolvido no Trabalho de Conclusão de Curso de Engenharia Mecatrônica (UniSATC, 2026).

Uma câmera fotografa a caixa cirúrgica montada sobre a bancada. Um modelo **YOLO26s** detecta cada
instrumento, e o sistema compara o resultado com o checklist da caixa. A saída aponta peças **faltantes**,
peças **excedentes** e **instrumentos desconhecidos**, e declara a caixa **conforme** ou **não conforme**.
É uma ferramenta de **verificação assistiva**: a decisão de liberar a caixa continua com o técnico da
Central de Material e Esterilização (CME).

- **Autor:** Nicolas Milak de Bitencourt
- **Orientadores:** João Mota Neto e Fernando Guessi Placido

![Exemplos de detecção](figuras/figura_10_exemplos_deteccao.jpg)

## Resultados no conjunto de teste

O conjunto de teste tem 160 imagens inéditas, distribuídas em 16 situações: caixa completa, com peças
faltantes, com peças excedentes ou com faltantes e excedentes, com ou sem instrumento desconhecido, e com
peças espalhadas ou sobrepostas.

| Métrica | Resultado |
|---|---|
| Caixas não conformes aprovadas (falso conforme) | **0 de 140** |
| Veredito correto (conforme / não conforme) | 158 de 160 (98,8%) |
| Diagnóstico completo correto | 150 de 160 (93,8%) |
| Diagnóstico com peças espalhadas / sobrepostas | 100% / 87,5% |
| Precisão / revocação (IoU 0,5) | 99,7% / 99,8% |
| mAP50 / mAP50-95 | 99,5% / 96,4% |
| Tempo por imagem (Ryzen 7 5700X, sem GPU) | 0,18 s (máximo 0,21 s) |

Todos os erros de diagnóstico ocorreram com peças sobrepostas. Os detalhes estão em
[`resultados/teste/relatorio_teste.txt`](resultados/teste/relatorio_teste.txt), e o resultado de cada
imagem está em [`resultados/teste/resultado_por_foto.txt`](resultados/teste/resultado_por_foto.txt).

## Conjunto instrumental

Instrumental de Apoio Pequenos Fragmentos: 19 peças de 14 tipos, sendo 9 tipos com 1 unidade e 5 tipos
com 2 unidades. Há também uma classe para instrumentos que não pertencem à caixa.

| Classe | Nome | Unidades na caixa |
|---:|---|:---:|
| 0 | gancho_pequeno | 1 |
| 1 | afastador_hohmann_8x236 | 1 |
| 2 | afastador_hohmann_mini_2x8 | 1 |
| 3 | afastador_mini_2x16 | 1 |
| 4 | retrator_senn_mueller | 2 |
| 5 | pinca_reducao_dentes_14cm | 2 |
| 6 | pinca_reducao_dentes_16cm | 2 |
| 7 | pinca_verbrugge | 1 |
| 8 | pinca_reducao_pontas_152mm | 1 |
| 9 | retrator_weitlaner | 1 |
| 10 | pinca_reducao_pontas_20cm | 1 |
| 11 | modelador_pequeno | 2 |
| 12 | descolador_williger | 1 |
| 13 | retrator_farabeuf | 2 |
| 14 | instrumento_desconhecido | 0 (qualquer um é divergência) |

## Arquivos, na ordem de uso

| Etapa | Arquivo | Função |
|---|---|---|
| 1. Captura (treino) | `captura_treino.py` | Fotografa o conjunto com a câmera configurada: foco manual, parâmetros fixos do driver e realce de bordas. As fotos vão para `fotos_treino` na Área de Trabalho. |
| 2. Anotação | `anotador.py` | Desenha as caixas e grava os rótulos no formato YOLO, um `.txt` por foto. |
| 3. Pré-anotação | `detectar_onnx.py --pasta` | Gera caixas preliminares com um modelo já treinado, para depois corrigir no anotador. |
| 4. Auditoria | `conferir_rotulos.py` | Aponta classes inválidas, caixas duplicadas, tamanhos atípicos e contagens que não batem com o checklist. |
| 5. Preparação | `preparar_fotos_treino.py` | Reduz as fotos para 1024×768, junta os rótulos e gera o zip para o Colab. |
| 6. Treinamento | `treino_colab.ipynb` | Treina no Google Colab (GPU T4) e exporta para ONNX. |
| 7. Captura (teste) | `captura_teste.py` | Fotografa o conjunto de teste e registra, em `planilha_teste.txt`, a situação e as peças faltantes e excedentes de cada foto. |
| 8. Utilitário | `renumerar_fotos.py` | Renumera uma faixa de fotos de teste e ajusta a planilha. |
| 9. Avaliação | `avaliar_teste.py` | Mede veredito, diagnóstico, falso conforme, precisão, revocação, mAP e tempo. |
| Uso | `detectar_onnx.py --camera 0` | Detecção ao vivo na câmera. |

Os scripts rodam no PC (Windows) com Python, OpenCV, NumPy e ONNX Runtime:
```
pip install -r requirements.txt
python avaliar_teste.py modelo_final.onnx --pasta "<pasta fotos_teste>"
```

## Treinamento

O modelo é um YOLO26s pré-treinado no COCO, com entrada de 1024 × 1024 e batch 8. O treino rodou 100
épocas, com paciência de 20 e otimizador AdamW (lr 0,000526, escolhido automaticamente). Usou espelhamento
horizontal e vertical e semente 0. Foram 1.500 imagens: 1.350 de treino e 150 de validação, divididas por
blocos de 10 imagens consecutivas. O ambiente foi Ultralytics 8.4.214 no Google Colab (Tesla T4), e o treino
levou 2,04 h. Os melhores pesos são da época 90, com mAP50-95 de 94,1% na validação. Curvas, matriz de
confusão e parâmetros estão em [`resultados/treino/`](resultados/treino/).

## Dados e pesos

As fotos (1.500 de treino e 160 de teste) não estão no repositório, por causa do tamanho. Os pesos
(`modelo_final.onnx` e `best.pt`) ficam anexados na Release `v1.0`. A planilha de gabarito do teste está em
`resultados/teste/planilha_teste.txt`.

## Licença

GNU AGPL-3.0, a mesma do framework Ultralytics usado no treinamento.

## Como citar

BITENCOURT, N. M. de. *Conferência de instrumental cirúrgico por visão computacional*. GitHub, 2026.
Disponível em: <https://github.com/Milakkkz/NOME-DO-REPOSITORIO>.
