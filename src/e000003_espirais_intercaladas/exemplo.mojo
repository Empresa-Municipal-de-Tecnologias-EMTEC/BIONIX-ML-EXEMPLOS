import src.camadas.mlp as mlp_pkg
import src.dados as dados_pkg
import src.graficos as graficos_pkg
import src.conjuntos as conjuntos_pkg
import src.nucleo.Tensor as tensor_defs
import src.uteis as uteis
import os


fn _gerar_bmp_variado_em(var caminho_bmp: String, var idx: Int):
    var bytes_bmp = graficos_pkg.gerar_bmp_espirais_intercaladas_variacao_bytes(
        192,
        192,
        idx * 3 + 1,
        idx * 5 + 2,
        (idx % 5) - 2,
    )
    _ = dados_pkg.gravar_arquivo_binario(caminho_bmp, bytes_bmp)


fn _dataset_valido(caminhos: List[String]) -> Bool:
    for c in caminhos:
        try:
            if not dados_pkg.diagnosticar_bmp(c):
                return False
        except _:
            return False
    return True


fn _garantir_dataset_bmp(var dir_dataset: String, var caminho_ok: String):
    var dir_treino = os.path.join(dir_dataset, "treino")
    var dir_teste = os.path.join(dir_dataset, "teste")
    try:
        os.makedirs(dir_treino, exist_ok=True)
        os.makedirs(dir_teste, exist_ok=True)
    except _:
        pass

    var arquivos_treino = List[String]()
    var arquivos_teste = List[String]()
    for i in range(8):
        arquivos_treino.append(os.path.join(dir_treino, "spiral_train_" + String(i) + ".bmp"))
    for i in range(4):
        arquivos_teste.append(os.path.join(dir_teste, "spiral_test_" + String(i) + ".bmp"))

    var marcador = uteis.ler_texto_seguro(caminho_ok).strip()
    if marcador == "ok" and _dataset_valido(arquivos_treino) and _dataset_valido(arquivos_teste):
        print("Dataset BMP já existe; reutilizando:", dir_dataset)
        return

    print("Gerando dataset BMP de espirais intercaladas...")
    for i in range(len(arquivos_treino)):
        _gerar_bmp_variado_em(arquivos_treino[i], i)
    for i in range(len(arquivos_teste)):
        _gerar_bmp_variado_em(arquivos_teste[i], 100 + i)

    _ = uteis.gravar_texto_seguro(caminho_ok, "ok")
    print("Dataset gerado em:", dir_dataset)


fn _acuracia_binaria(pred: tensor_defs.Tensor, alvos: tensor_defs.Tensor) -> Float32:
    if len(pred.dados) == 0:
        return 0.0
    var acertos = 0
    for i in range(len(pred.dados)):
        var p = 1.0 if pred.dados[i] >= 0.5 else Float32(0.0)
        if p == alvos.dados[i]:
            acertos = acertos + 1
    return Float32(acertos) / Float32(len(pred.dados))


def executar_exemplo():
    print("--- Exemplo e000003: espirais intercaladas (BMP + autograd + ativações + MLP) ---")

    var tipo_computacao = "cpu"
    var dir_dataset = "exemplos/e000003_espirais_intercaladas/dataset"
    var caminho_ok = "exemplos/e000003_espirais_intercaladas/dataset/dataset.ok"
    var dir_treino = os.path.join(dir_dataset, "treino")
    var dir_teste = os.path.join(dir_dataset, "teste")

    # 1) Geração condicional do dataset em BMP
    _garantir_dataset_bmp(dir_dataset, caminho_ok)

    # 2) Carregamento dos conjuntos de treino e teste a partir dos diretórios
    var conjunto_treino = conjuntos_pkg.carregar_bitmap_supervisionado(dir_treino, tipo_computacao, 2, 0.05, 0.6)
    var conjunto_teste = conjuntos_pkg.carregar_bitmap_supervisionado(dir_teste, tipo_computacao, 2, 0.05, 0.6)
    var entradas = conjunto_treino.entradas.copy()
    var alvos = conjunto_treino.alvos.copy()

    if len(entradas.dados) == 0:
        print("Falha ao carregar dataset de treino a partir do BMP.")
        return
    if len(conjunto_teste.entradas.dados) == 0:
        print("Falha ao carregar dataset de teste a partir do BMP.")
        return

    print("Amostras treino:", entradas.formato[0], "| Features:", entradas.formato[1])
    print("Amostras teste:", conjunto_teste.entradas.formato[0])

    var epocas = 50
    var prep_lotes = conjuntos_pkg.preparar_treino_validacao_em_lotes(conjunto_treino, 128, epocas, 0.0)
    var lotes_teste = conjuntos_pkg.quebrar_em_lotes(conjunto_teste, 128)
    print("Lotes de treino (epocas x lotes):", len(prep_lotes.treino_por_epoca), "| Lotes de teste:", len(lotes_teste))
    print("Epocas de treino:", epocas)

    # 3) Treino do bloco MLP por lotes (autograd + funções de ativação), validando no conjunto de teste
    var topologia = List[Int]()
    topologia.append(2)
    topologia.append(16)
    topologia.append(16)
    topologia.append(1)
    var mlp = mlp_pkg.BlocoMLP(topologia^, tipo_computacao)
    var loss_final = mlp_pkg.treinar_por_lotes(mlp, prep_lotes.treino_por_epoca, lotes_teste, 0.03, 1)
    print("Loss final:", loss_final)

    # 4) Métricas de acurácia para treino e teste
    var pred_treino = mlp_pkg.inferir(mlp, entradas)
    var pred_teste = mlp_pkg.inferir(mlp, conjunto_teste.entradas)
    var acuracia_treino = _acuracia_binaria(pred_treino, alvos)
    var acuracia_teste = _acuracia_binaria(pred_teste, conjunto_teste.alvos)
    print("Acurácia treino:", acuracia_treino)
    print("Acurácia teste:", acuracia_teste)
    print("Acurácia de predição entre A e B (treino):", acuracia_treino)
    print("Acurácia de predição entre A e B (teste):", acuracia_teste)

    print("--- Fim do exemplo e000003 ---")
