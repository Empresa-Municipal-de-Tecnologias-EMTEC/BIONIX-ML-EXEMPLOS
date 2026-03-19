import src.camadas.mlp as mlp_pkg
import src.autograd as autograd
import src.computacao.dispatcher_gradiente as dispatcher_gradiente
import src.dados as dados_pkg
import src.nucleo.Tensor as tensor_defs
import src.uteis as uteis
import os


fn _lista_arquivos_bmp(var dir_raiz: String) -> List[String]:
    var out = List[String]()
    var classes = List[String]()
    try:
        classes = os.listdir(dir_raiz)
    except _:
        return out^

    for c in classes:
        var dir_classe = os.path.join(dir_raiz, c)
        if not os.path.isdir(dir_classe):
            continue
        var arquivos = List[String]()
        try:
            arquivos = os.listdir(dir_classe)
        except _:
            continue
        for nome in arquivos:
            if nome.endswith(".bmp"):
                out.append(os.path.join(dir_classe, nome))
    return out^


fn _parse_label_de_caminho(var caminho: String) -> Int:
    var normalizado = caminho.replace("\\", "/")
    var partes = normalizado.split("/")
    if len(partes) < 2:
        return 0
    var ultima_pasta = String(partes[len(partes) - 2])
    try:
        return Int(ultima_pasta)
    except _:
        return 0


fn _nome_arquivo_de_caminho(var caminho: String) -> String:
    var normalizado = caminho.replace("\\", "/")
    var partes = normalizado.split("/")
    if len(partes) <= 0:
        return caminho
    return String(partes[len(partes) - 1])


fn _int_list_para_csv(valores: List[Int]) -> String:
    var out = ""
    for i in range(len(valores)):
        out = out + String(valores[i])
        if i < len(valores) - 1:
            out = out + ","
    return out


fn _csv_para_int_list(var csv: String) -> List[Int]:
    var out = List[Int]()
    var itens = uteis.split_csv_simples(csv)
    for it in itens:
        var s = String(it.strip())
        if len(s) > 0:
            out.append(Int(uteis.parse_float_ascii(s)))
    return out^


fn _csv_para_float_list(var csv: String) -> List[Float32]:
    var out = List[Float32]()
    var itens = uteis.split_csv_simples(csv)
    for it in itens:
        var s = String(it.strip())
        if len(s) > 0:
            out.append(uteis.parse_float_ascii(s))
    return out^


fn _salvar_checkpoint_mlp(bloco: mlp_pkg.BlocoMLP, var caminho_checkpoint: String, var epoca: Int):
    var chaves = List[String]()
    var valores = List[String]()

    chaves.append("tipo")
    valores.append(bloco.tipo_computacao)
    chaves.append("epoca")
    valores.append(String(epoca))
    chaves.append("topologia")
    valores.append(_int_list_para_csv(bloco.topologia.copy()))
    chaves.append("num_camadas")
    valores.append(String(len(bloco.pesos)))

    for camada in range(len(bloco.pesos)):
        chaves.append("w_" + String(camada))
        valores.append(uteis.float_list_para_csv(bloco.pesos[camada].dados.copy()))
        chaves.append("b_" + String(camada))
        valores.append(uteis.float_list_para_csv(bloco.biases[camada].dados.copy()))

    _ = uteis.salvar_kv_arquivo_seguro(caminho_checkpoint, chaves, valores)


fn _carregar_checkpoint_mlp(
    var caminho_checkpoint: String,
    topologia_padrao: List[Int],
    var tipo_computacao_padrao: String,
    var ativacao_saida_id: Int,
    var perda_id: Int,
) -> mlp_pkg.BlocoMLP:
    if not os.path.isfile(caminho_checkpoint):
        return mlp_pkg.BlocoMLP(topologia_padrao.copy(), tipo_computacao_padrao, ativacao_saida_id, perda_id)

    var kv = uteis.carregar_kv_arquivo_seguro(caminho_checkpoint)
    if len(kv.chaves) == 0:
        return mlp_pkg.BlocoMLP(topologia_padrao.copy(), tipo_computacao_padrao, ativacao_saida_id, perda_id)

    var tipo = uteis.obter_valor_ou_padrao(kv, "tipo", tipo_computacao_padrao)
    var topologia_csv = uteis.obter_valor_ou_padrao(kv, "topologia", _int_list_para_csv(topologia_padrao))
    var topologia_lida = _csv_para_int_list(topologia_csv)
    if len(topologia_lida) < 2:
        topologia_lida = topologia_padrao.copy()

    var bloco = mlp_pkg.BlocoMLP(topologia_lida.copy(), tipo, ativacao_saida_id, perda_id)

    for camada in range(len(bloco.pesos)):
        var w_key = "w_" + String(camada)
        var b_key = "b_" + String(camada)

        var w_csv = uteis.obter_valor_ou_padrao(kv, w_key, "")
        var b_csv = uteis.obter_valor_ou_padrao(kv, b_key, "")

        var w_lidos = _csv_para_float_list(w_csv)
        var b_lidos = _csv_para_float_list(b_csv)

        var total_w = len(w_lidos)
        if len(bloco.pesos[camada].dados) < total_w:
            total_w = len(bloco.pesos[camada].dados)
        for i in range(total_w):
            bloco.pesos[camada].dados[i] = w_lidos[i]

        var total_b = len(b_lidos)
        if len(bloco.biases[camada].dados) < total_b:
            total_b = len(bloco.biases[camada].dados)
        for i in range(total_b):
            bloco.biases[camada].dados[i] = b_lidos[i]

    return bloco^


fn _selecionar_uma_imagem_por_classe(caminhos: List[String]) -> List[String]:
    var out = List[String]()
    var achou = List[Bool]()
    for _ in range(10):
        achou.append(False)

    for caminho in caminhos:
        var cls = _parse_label_de_caminho(caminho)
        if cls < 0 or cls > 9:
            continue
        if not achou[cls]:
            out.append(caminho)
            achou[cls] = True

    return out^


fn _log_inferencia_10_classes(
    bloco: mlp_pkg.BlocoMLP,
    caminhos_exemplo: List[String],
    var tipo_computacao: String,
    var altura_alvo: Int,
    var largura_alvo: Int,
):
    if len(caminhos_exemplo) == 0:
        print("Inferência: nenhuma imagem disponível.")
        return

    var dados = _carregar_dataset_digitos_de_arquivos(caminhos_exemplo, tipo_computacao, altura_alvo, largura_alvo)
    var x = dados[0].copy()
    if x.formato[0] == 0:
        print("Inferência: falha ao carregar imagens de exemplo.")
        return

    var pred = x.copy()
    _ = pred
    try:
        pred = mlp_pkg.inferir(bloco, x)
    except _:
        print("Inferência: falha ao inferir no modelo carregado.")
        return
    print("--- Inferência (uma imagem por classe) ---")
    for i in range(len(caminhos_exemplo)):
        var classe_real = _parse_label_de_caminho(caminhos_exemplo[i].copy())
        var classe_prevista = _argmax_linha(pred, i)
        var nome_arquivo = _nome_arquivo_de_caminho(caminhos_exemplo[i].copy())
        print("Classe", classe_real, "| previsão", classe_prevista, "| nome do arquivo de imagem:", nome_arquivo)


fn _carregar_dataset_digitos_de_arquivos(caminhos: List[String], var tipo_computacao: String) -> List[tensor_defs.Tensor]:
    return _carregar_dataset_digitos_de_arquivos(caminhos, tipo_computacao, -1, -1)


fn _carregar_dataset_digitos_de_arquivos(
    caminhos: List[String],
    var tipo_computacao: String,
    var altura_alvo: Int,
    var largura_alvo: Int,
) -> List[tensor_defs.Tensor]:
    if len(caminhos) == 0:
        var fx = List[Int]()
        fx.append(0)
        fx.append(0)
        var fy = List[Int]()
        fy.append(0)
        fy.append(10)
        var vazio_x = tensor_defs.Tensor(fx^, tipo_computacao)
        var vazio_y = tensor_defs.Tensor(fy^, tipo_computacao)
        var out_vazio = List[tensor_defs.Tensor]()
        out_vazio.append(vazio_x.copy())
        out_vazio.append(vazio_y.copy())
        return out_vazio^

    var primeira = List[List[Float32]]()
    _ = primeira
    try:
        var caminho0 = caminhos[0].copy()
        if altura_alvo > 0 and largura_alvo > 0:
            primeira = dados_pkg.carregar_bmp_grayscale_matriz(caminho0, altura_alvo, largura_alvo)
        else:
            primeira = dados_pkg.carregar_bmp_grayscale_matriz(caminho0)
    except _:
        var fx_erro = List[Int]()
        fx_erro.append(0)
        fx_erro.append(0)
        var fy_erro = List[Int]()
        fy_erro.append(0)
        fy_erro.append(10)
        var vazio_x_erro = tensor_defs.Tensor(fx_erro^, tipo_computacao)
        var vazio_y_erro = tensor_defs.Tensor(fy_erro^, tipo_computacao)
        var out_vazio_erro = List[tensor_defs.Tensor]()
        out_vazio_erro.append(vazio_x_erro.copy())
        out_vazio_erro.append(vazio_y_erro.copy())
        return out_vazio_erro^

    var altura = len(primeira)
    var largura = len(primeira[0]) if altura > 0 else 0
    var features = largura * altura
    var amostras = len(caminhos)
    print("Carregando", amostras, "imagens...")

    var formato_x = List[Int]()
    formato_x.append(amostras)
    formato_x.append(features)
    var formato_y = List[Int]()
    formato_y.append(amostras)
    formato_y.append(10)

    var x_t = tensor_defs.Tensor(formato_x^, tipo_computacao)
    var y_t = tensor_defs.Tensor(formato_y^, tipo_computacao)

    for i in range(amostras):
        var m = List[List[Float32]]()
        _ = m
        try:
            var caminho_i = caminhos[i].copy()
            if altura_alvo > 0 and largura_alvo > 0:
                m = dados_pkg.carregar_bmp_grayscale_matriz(caminho_i, altura_alvo, largura_alvo)
            else:
                m = dados_pkg.carregar_bmp_grayscale_matriz(caminho_i)
        except _:
            m = primeira.copy()

        var label = _parse_label_de_caminho(caminhos[i].copy())

        var k = 0
        for yy in range(altura):
            for xx in range(largura):
                # Usar normalização de pixel não negativa com ReLU
                x_t.dados[i * features + k] = m[yy][xx]
                k = k + 1

        for c in range(10):
            y_t.dados[i * 10 + c] = 1.0 if c == label else Float32(0.0)

        if i > 0 and i % 500 == 0:
            print("  progresso:", i, "/", amostras)

    var out = List[tensor_defs.Tensor]()
    out.append(x_t.copy())
    out.append(y_t.copy())
    return out^


fn _dataset_tem_classes_0_a_9(var dir_dataset: String) -> Bool:
    for classe in range(10):
        var dir_classe = os.path.join(dir_dataset, String(classe))
        if not os.path.isdir(dir_classe):
            return False
    return True


fn _dividir_arquivos_treino_valid_teste(var dir_dataset: String) -> List[List[String]]:
    var treino = List[String]()
    var valid = List[String]()
    var teste = List[String]()

    # Split estratificado determinístico por classe: 70% treino, 15% validação, 15% teste
    for classe in range(10):
        var dir_classe = os.path.join(dir_dataset, String(classe))
        if not os.path.isdir(dir_classe):
            continue

        var arquivos = List[String]()
        _ = arquivos
        try:
            arquivos = os.listdir(dir_classe)
        except _:
            continue
        var idx = 0
        var usados_classe = 0
        var max_por_classe = 320
        for nome in arquivos:
            if not nome.endswith(".bmp"):
                continue
            if max_por_classe > 0 and usados_classe >= max_por_classe:
                break

            var caminho = os.path.join(dir_classe, nome)
            var bucket = idx % 20
            if bucket < 14:
                treino.append(caminho)
            elif bucket < 17:
                valid.append(caminho)
            else:
                teste.append(caminho)
            idx = idx + 1
            usados_classe = usados_classe + 1

    var out = List[List[String]]()
    out.append(treino^)
    out.append(valid^)
    out.append(teste^)
    return out^


fn _fatiar_2d(t: tensor_defs.Tensor, var inicio: Int, var fim: Int) -> tensor_defs.Tensor:
    var linhas = t.formato[0]
    var colunas = t.formato[1]
    if inicio < 0:
        inicio = 0
    if fim > linhas:
        fim = linhas
    if fim < inicio:
        fim = inicio

    var n = fim - inicio
    var f = List[Int]()
    f.append(n)
    f.append(colunas)
    var out = tensor_defs.Tensor(f^, t.tipo_computacao)

    for i in range(n):
        var src_i = inicio + i
        for j in range(colunas):
            out.dados[i * colunas + j] = t.dados[src_i * colunas + j]

    return out^


fn _fatiar_2d_por_indices(t: tensor_defs.Tensor, indices: List[Int], var inicio: Int, var fim: Int) -> tensor_defs.Tensor:
    var total_idx = len(indices)
    var colunas = t.formato[1]
    if inicio < 0:
        inicio = 0
    if fim > total_idx:
        fim = total_idx
    if fim < inicio:
        fim = inicio

    var n = fim - inicio
    var f = List[Int]()
    f.append(n)
    f.append(colunas)
    var out = tensor_defs.Tensor(f^, t.tipo_computacao)

    for i in range(n):
        var src_i = indices[inicio + i]
        for j in range(colunas):
            out.dados[i * colunas + j] = t.dados[src_i * colunas + j]

    return out^


fn _permutacao_deterministica(var n: Int, var epoca: Int) -> List[Int]:
    var perm = List[Int]()
    for i in range(n):
        perm.append(i)

    if n <= 1:
        return perm^

    var seed = (epoca + 1) * 1103515245 + 12345
    if seed < 0:
        seed = -seed

    for i in range(n - 1, 0, -1):
        seed = (seed * 1664525 + 1013904223) % 2147483647
        var j = seed % (i + 1)
        var tmp = perm[i]
        perm[i] = perm[j]
        perm[j] = tmp

    return perm^


fn _argmax_linha(t: tensor_defs.Tensor, var linha: Int) -> Int:
    var colunas = t.formato[1]
    var melhor = 0
    var melhor_v = t.dados[linha * colunas]
    for c in range(1, colunas):
        var v = t.dados[linha * colunas + c]
        if v > melhor_v:
            melhor_v = v
            melhor = c
    return melhor


fn _acuracia_multiclasse(pred: tensor_defs.Tensor, alvos_one_hot: tensor_defs.Tensor) -> Float32:
    var n = pred.formato[0]
    if n <= 0:
        return 0.0
    var acertos = 0
    for i in range(n):
        var p = _argmax_linha(pred, i)
        var y = _argmax_linha(alvos_one_hot, i)
        if p == y:
            acertos = acertos + 1
    return Float32(acertos) / Float32(n)


fn _treinar_por_lotes_multiclasse(
    mut bloco: mlp_pkg.BlocoMLP,
    x_treino: tensor_defs.Tensor,
    y_treino: tensor_defs.Tensor,
    x_valid: tensor_defs.Tensor,
    y_valid: tensor_defs.Tensor,
    var epocas: Int,
    var tamanho_lote: Int,
    var taxa_aprendizado: Float32,
    var usar_reduce_on_plateau: Bool = False,
    var fator_reducao: Float32 = 0.5,
    var paciencia_reducao: Int = 3,
    var lr_min: Float32 = 0.00005,
    var limiar_saida_inercia: Float32 = 0.12,
    var tolerancia_recuo: Float32 = 0.001,
    var salvar_checkpoint_a_cada_epoca: Bool = False,
    var caminho_checkpoint: String = "",
) raises:
    var total = x_treino.formato[0]
    if total <= 0:
        return

    if tamanho_lote <= 0:
        tamanho_lote = total

    var lr_atual = taxa_aprendizado
    var melhor_acc_val: Float32 = -1.0
    var epocas_recuando = 0
    var saiu_inercia = False
    var workspace_cuda = dispatcher_gradiente.criar_workspace_gradiente_cuda()

    for epoca in range(epocas):
        var soma_loss: Float32 = 0.0
        var lotes = 0
        var perm = _permutacao_deterministica(total, epoca)

        var inicio = 0
        while inicio < total:
            var fim = inicio + tamanho_lote
            if fim > total:
                fim = total

            var xb = _fatiar_2d_por_indices(x_treino, perm, inicio, fim)
            var yb = _fatiar_2d_por_indices(y_treino, perm, inicio, fim)

            try:
                var ctx = autograd.construir_contexto_mlp(xb, yb, bloco.pesos, bloco.biases, bloco.ativacao_saida_id, bloco.perda_id)

                var manter_gradientes_na_ram_principal = bloco.tipo_computacao == "cpu"
                var grads: autograd.MLPGradientes
                if bloco.tipo_computacao == "cuda":
                    grads = dispatcher_gradiente.calcular_gradientes_mlp_com_workspace_cuda(
                        ctx,
                        bloco.pesos,
                        workspace_cuda,
                        manter_gradientes_na_ram_principal,
                    )
                else:
                    grads = dispatcher_gradiente.calcular_gradientes_mlp(ctx, bloco.pesos, manter_gradientes_na_ram_principal)

                soma_loss = soma_loss + grads.loss
                lotes = lotes + 1

                for camada in range(len(bloco.pesos)):
                    for i in range(len(bloco.pesos[camada].dados)):
                        bloco.pesos[camada].dados[i] = bloco.pesos[camada].dados[i] - lr_atual * grads.grad_ws[camada].dados[i]
                    for j in range(len(bloco.biases[camada].dados)):
                        bloco.biases[camada].dados[j] = bloco.biases[camada].dados[j] - lr_atual * grads.grad_bs[camada].dados[j]

            except _:
                print("Falha ao construir contexto MLP durante batch, pulando lote.")
                inicio = fim
                continue
            inicio = fim

        var pred_val = x_valid.copy()
        _ = pred_val
        try:
            pred_val = mlp_pkg.inferir(bloco, x_valid)
        except _:
            print("Falha na inferência de validação durante o treino.")
            return
        var loss_val = autograd.calcular_loss_mlp(pred_val, y_valid, bloco.perda_id)
        var acc_val = _acuracia_multiclasse(pred_val, y_valid)

        if usar_reduce_on_plateau:
            if acc_val > melhor_acc_val:
                melhor_acc_val = acc_val
                epocas_recuando = 0
            else:
                if melhor_acc_val >= limiar_saida_inercia:
                    saiu_inercia = True
                if saiu_inercia and acc_val < melhor_acc_val - tolerancia_recuo:
                    epocas_recuando = epocas_recuando + 1
                else:
                    epocas_recuando = 0

            if saiu_inercia and epocas_recuando >= paciencia_reducao and lr_atual > lr_min:
                var nova_lr = lr_atual * fator_reducao
                if nova_lr < lr_min:
                    nova_lr = lr_min
                if nova_lr < lr_atual:
                    print("[ReduceLROnPlateau] acc recuou após inércia. LR:", lr_atual, "->", nova_lr)
                    lr_atual = nova_lr
                epocas_recuando = 0

        var loss_treino_medio = soma_loss / Float32(lotes) if lotes > 0 else 0.0
        print("Época", epoca, "| Loss treino médio:", loss_treino_medio, "| Loss validação:", loss_val, "| Acc validação:", acc_val, "| LR:", lr_atual)

        #if salvar_checkpoint_a_cada_epoca and len(caminho_checkpoint.strip()) > 0:
        #    _salvar_checkpoint_mlp(bloco, caminho_checkpoint, epoca)


def executar_exemplo_configuravel(
    var tipo_computacao: String,
    var caminho_checkpoint: String,
    var id_exemplo: String,
):
    print("--- Exemplo " + id_exemplo + ": reconhecimento de dígitos (0-9) com MLP ---")

    var dir_dataset = "exemplos/e000004_reconhecimento_digitos/dataset"
    if not _dataset_tem_classes_0_a_9(dir_dataset):
        var candidatos_dataset = List[String]()
        candidatos_dataset.append("../exemplos/e000004_reconhecimento_digitos/dataset")
        candidatos_dataset.append("../../exemplos/e000004_reconhecimento_digitos/dataset")
        candidatos_dataset.append("../../../exemplos/e000004_reconhecimento_digitos/dataset")
        for c in candidatos_dataset:
            if _dataset_tem_classes_0_a_9(c):
                dir_dataset = c
                break

    if not os.path.isfile(caminho_checkpoint):
        var candidatos_checkpoint = List[String]()
        candidatos_checkpoint.append("../" + caminho_checkpoint)
        candidatos_checkpoint.append("../../" + caminho_checkpoint)
        candidatos_checkpoint.append("../../../" + caminho_checkpoint)
        for ckp in candidatos_checkpoint:
            if os.path.isfile(ckp):
                caminho_checkpoint = ckp
                break

    if not _dataset_tem_classes_0_a_9(dir_dataset):
        print("Dataset inválido: esperado", dir_dataset, "com subpastas 0..9 contendo BMPs.")
        return

    var arquivos_split = _dividir_arquivos_treino_valid_teste(dir_dataset)
    var arquivos_treino = arquivos_split[0].copy()
    var arquivos_valid = arquivos_split[1].copy()
    var arquivos_teste = arquivos_split[2].copy()
    print("Arquivos split | treino:", len(arquivos_treino), "| valid:", len(arquivos_valid), "| teste:", len(arquivos_teste))
    var altura_alvo = 32
    var largura_alvo = 32
    print("Redimensionamento alvo:", largura_alvo, "x", altura_alvo)

    var treino = _carregar_dataset_digitos_de_arquivos(arquivos_treino, tipo_computacao, altura_alvo, largura_alvo)
    var valid = _carregar_dataset_digitos_de_arquivos(arquivos_valid, tipo_computacao, altura_alvo, largura_alvo)
    var teste = _carregar_dataset_digitos_de_arquivos(arquivos_teste, tipo_computacao, altura_alvo, largura_alvo)

    var x_treino = treino[0].copy()
    var y_treino = treino[1].copy()
    var x_valid = valid[0].copy()
    var y_valid = valid[1].copy()
    var x_teste = teste[0].copy()
    var y_teste = teste[1].copy()

    if x_treino.formato[0] == 0 or x_valid.formato[0] == 0 or x_teste.formato[0] == 0:
        print("Falha ao carregar dataset de dígitos.")
        return

    print("Amostras treino:", x_treino.formato[0], "| Features:", x_treino.formato[1])
    print("Amostras validação:", x_valid.formato[0], "| Amostras teste:", x_teste.formato[0], "| Classes:", y_treino.formato[1])

    var topologia = List[Int]()
    topologia.append(x_treino.formato[1])
    topologia.append(1024)
    topologia.append(64)
    topologia.append(10)

    var usar_pesos_salvos = False
    var inferencia_somente = False
    var salvar_checkpoint_a_cada_epoca = False

    var mlp = mlp_pkg.BlocoMLP(
        topologia.copy(),
        tipo_computacao,
        mlp_pkg.ativacao_saida_softmax_id(),
        mlp_pkg.perda_cross_entropy_id(),
    )

    if usar_pesos_salvos and os.path.isfile(caminho_checkpoint):
        print("Carregando pesos salvos de:", caminho_checkpoint)
        mlp = _carregar_checkpoint_mlp(
            caminho_checkpoint,
            topologia,
            tipo_computacao,
            mlp_pkg.ativacao_saida_softmax_id(),
            mlp_pkg.perda_cross_entropy_id(),
        )

    # Imagens 128x128 possuem alta dimensionalidade; configuração mais estável para esse cenário
    var epocas = 1500
    var tamanho_lote = 64
    var taxa_aprendizado: Float32 = 0.005
    var usar_reduce_on_plateau = True
    var fator_reducao_lr: Float32 = 0.5
    var paciencia_reducao_lr = 5
    var lr_min_reduce_on_plateau: Float32 = 0.0001
    print(
        "Configuração MLP | Ativação saída:",
        mlp_pkg.ativacao_saida_nome_de_id(mlp.ativacao_saida_id),
        "| Perda:",
        mlp_pkg.perda_nome_de_id(mlp.perda_id),
    )
    print(
        "Epocas:", epocas,
        "| Lote:", tamanho_lote,
        "| LR:", taxa_aprendizado,
        "| ReduceOnPlateau:", usar_reduce_on_plateau,
        "| FatorReduceLR:", fator_reducao_lr,
        "| PacienciaReduceLR:", paciencia_reducao_lr,
        "| LRMinReduce:", lr_min_reduce_on_plateau,
        "| InferenciaSomente:", inferencia_somente,
        "| SalvarCheckpoint:", salvar_checkpoint_a_cada_epoca,
    )

    if inferencia_somente:
        if not os.path.isfile(caminho_checkpoint):
            print("Inferência solicitada, mas checkpoint não encontrado em:", caminho_checkpoint)
            return
    else:
        _treinar_por_lotes_multiclasse(
            mlp,
            x_treino,
            y_treino,
            x_valid,
            y_valid,
            epocas,
            tamanho_lote,
            taxa_aprendizado,
            usar_reduce_on_plateau,
            fator_reducao_lr,
            paciencia_reducao_lr,
            lr_min_reduce_on_plateau,
            0.12,
            0.001,
            salvar_checkpoint_a_cada_epoca,
            caminho_checkpoint,
        )

        #if salvar_checkpoint_a_cada_epoca and len(caminho_checkpoint.strip()) > 0:
        #    _salvar_checkpoint_mlp(mlp, caminho_checkpoint, epocas)

    var pred_treino = mlp_pkg.inferir(mlp, x_treino)
    var pred_valid = mlp_pkg.inferir(mlp, x_valid)
    var pred_teste = mlp_pkg.inferir(mlp, x_teste)
    var acc_treino = _acuracia_multiclasse(pred_treino, y_treino)
    var acc_valid = _acuracia_multiclasse(pred_valid, y_valid)
    var acc_teste = _acuracia_multiclasse(pred_teste, y_teste)

    print("Acurácia treino (0-9):", acc_treino)
    print("Acurácia validação (0-9):", acc_valid)
    print("Acurácia teste (0-9):", acc_teste)

    var arquivos_todos = List[String]()
    for arq in arquivos_treino:
        arquivos_todos.append(arq)
    for arq in arquivos_valid:
        arquivos_todos.append(arq)
    for arq in arquivos_teste:
        arquivos_todos.append(arq)
    var amostras_inferencia = _selecionar_uma_imagem_por_classe(arquivos_todos)
    _log_inferencia_10_classes(mlp, amostras_inferencia, tipo_computacao, altura_alvo, largura_alvo)

    print("--- Fim do exemplo " + id_exemplo + " ---")


def executar_exemplo():
    try:
        executar_exemplo_configuravel(
            "cpu",
            "exemplos/e000004_reconhecimento_digitos/pesos_mlp_digits.txt",
            "e000004",
        )
    except _:
        print("Erro no exemplo e000004: exceção não tratada, pulando exemplo.")
        return
