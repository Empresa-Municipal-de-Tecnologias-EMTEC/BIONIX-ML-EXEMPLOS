import src.camadas.linear as linear_pkg
import src.dados as dados_pkg
import src.conjuntos as conjuntos_pkg
import src.computacao as computacao_pkg
import src.nucleo.Tensor as tensor_defs

fn _criar_tensor_uma_linha(var valores: List[Float32], var tipo_computacao: String) -> tensor_defs.Tensor:
    var formato = List[Int]()
    formato.append(1)
    formato.append(len(valores))
    var t = tensor_defs.Tensor(formato^, tipo_computacao)
    for i in range(len(valores)):
        t.dados[i] = valores[i]
    return t^


def executar_exemplo():
    print("--- Exemplo e000002: modelo linear com Tensor parametrizável ---")

    var tipo_computacao = computacao_pkg.backend_nome_de_id(computacao_pkg.backend_cpu_id())
    var caminho_csv = "exemplos/e000002_modelo_linear/dados.csv"
    var caminho_pesos = "exemplos/e000002_modelo_linear/pesos_linear.txt"
    var caminho_norm = "exemplos/e000002_modelo_linear/normalizacao_linear.txt"
    var normalizacao_entradas_id = dados_pkg.normalizacao_zscore_id()
    var normalizacao_alvo_id = dados_pkg.normalizacao_zscore_id()

    var conjunto = conjuntos_pkg.carregar_csv_supervisionado(
        caminho_csv,
        -1,
        ",",
        True,
        tipo_computacao,
        normalizacao_entradas_id,
        normalizacao_alvo_id,
    )
    if len(conjunto.entradas.dados) == 0 or conjunto.entradas.formato[1] == 0:
        print("Falha ao carregar conjunto supervisionado do CSV:", caminho_csv)
        return

    print("Tipo de computação do tensor:", conjunto.entradas.tipo_computacao)
    print("Amostras:", conjunto.entradas.formato[0], "| Features:", conjunto.entradas.formato[1])
    print("Normalização das entradas:", dados_pkg.normalizacao_nome_de_id(conjunto.tipo_normalizacao_entradas_id))
    print("Normalização do alvo:", dados_pkg.normalizacao_nome_de_id(conjunto.tipo_normalizacao_alvo_id))

    print("[debug] Criando camada linear...")
    var camada = linear_pkg.CamadaLinear(conjunto.entradas.formato[1], tipo_computacao)
    print("[debug] Camada criada. Iniciando treinamento...")

    print("Treinando modelo linear...")
    var loss_final = linear_pkg.treinar(camada, conjunto.entradas, conjunto.alvos, 0.0001, 4000, 500)
    print("Loss final:", loss_final)

    linear_pkg.salvar_pesos(camada, caminho_pesos)
    print("Pesos salvos em:", caminho_pesos)
    var norm_treino = dados_pkg.criar_normalizacao_persistida(
        conjunto.tipo_normalizacao_entradas_id,
        conjunto.media_entradas.copy(),
        conjunto.desvio_entradas.copy(),
        conjunto.tipo_normalizacao_alvo_id,
        conjunto.media_alvo,
        conjunto.desvio_alvo,
    )
    dados_pkg.salvar_normalizacao_persistida(norm_treino, caminho_norm)
    print("Normalização salva em:", caminho_norm)

    var modelo_carregado = linear_pkg.carregar_pesos(caminho_pesos, tipo_computacao)
    var norm_carregada = dados_pkg.carregar_normalizacao_persistida(caminho_norm)
    print("Modelo recarregado. Bias:", modelo_carregado.bias.dados[0])
    print(
        "Normalização recarregada. Tipo entradas:",
        dados_pkg.normalizacao_nome_de_id(norm_carregada.tipo_entradas_id),
        "| Tipo alvo:",
        dados_pkg.normalizacao_nome_de_id(norm_carregada.tipo_alvo_id),
    )

    var amostra = List[Float32]()
    amostra.append(78.0)
    amostra.append(3.0)
    amostra.append(2.0)
    var amostra_norm = dados_pkg.normalizar_amostra_entradas(norm_carregada, amostra)
    var entrada_nova = _criar_tensor_uma_linha(amostra_norm^, tipo_computacao)
    var pred = linear_pkg.inferir(modelo_carregado, entrada_nova)
    var pred_escala_original = dados_pkg.desnormalizar_valor_alvo(norm_carregada, pred.dados[0])

    print("Inferência normalizada para [area=78, quartos=3, idade=2]:", pred.dados[0])
    print("Inferência em escala original (preço):", pred_escala_original)
    print("--- Fim do exemplo e000002 ---")