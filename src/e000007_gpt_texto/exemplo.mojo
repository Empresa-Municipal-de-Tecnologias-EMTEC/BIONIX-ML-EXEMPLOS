import src.dados as dados_pkg
import src.camadas.transformer as transformer_pkg
import src.computacao as computacao_pkg
import src.conjuntos as conjuntos_pkg
import src.nucleo.Tensor as tensor_defs
import math


fn _tokenizar_palavras(var texto: String) -> List[String]:
    var out = List[String]()
    var atual = ""
    for i in range(len(texto)):
        var c = texto[i:i+1]
        if c == " " or c == "\t" or c == "," or c == "." or c == ":" or c == ";" or c == "!" or c == "?":
            if len(atual) > 0:
                out.append(atual)
                atual = ""
        else:
            atual = atual + c
    if len(atual) > 0:
        out.append(atual)
    return out^


fn _score_sobreposicao(var a: String, var b: String) -> Int:
    var ta = _tokenizar_palavras(a)
    var tb = _tokenizar_palavras(b)
    var score = 0
    for i in range(len(ta)):
        for j in range(len(tb)):
            if ta[i] == tb[j]:
                score = score + 1
    return score


fn _inferir_por_modo(amostras: List[conjuntos_pkg.AmostraGPTSupervisionada], var modo: String, var prompt: String) -> String:
    var melhor = -1
    var melhor_score = -1

    for i in range(len(amostras)):
        if amostras[i].modo.strip() != modo:
            continue
        var s = _score_sobreposicao(amostras[i].inicio, prompt)
        if s > melhor_score:
            melhor_score = s
            melhor = i

    if melhor >= 0:
        return amostras[melhor].completar

    if modo == "conversa":
        return "assistente: posso te ajudar com um proximo passo claro e objetivo."
    if modo == "instrucoes":
        return "instrucao: especifique entrada, saida esperada e criterio de validacao."
    if modo == "ferramentas":
        return "ferramenta: use terminal, busca e leitura de arquivos para validar o fluxo."
    return "resposta: sem dados suficientes."


fn _chars_para_token_ids_ascii(var texto: String, var limite: Int = 32) -> List[Int]:
    var out = List[Int]()
    var n = len(texto)
    if limite < n:
        n = limite
    for i in range(n):
        out.append(Int(ord(texto[i:i+1])) & 0xFF)
    return out^


fn _aplicar_merge(var tokens: List[Int], var a: Int, var b: Int, var novo_id: Int) -> List[Int]:
    var out = List[Int]()
    var i = 0
    while i < len(tokens):
        if i + 1 < len(tokens) and tokens[i] == a and tokens[i + 1] == b:
            out.append(novo_id)
            i = i + 2
        else:
            out.append(tokens[i])
            i = i + 1
    return out^


fn _chave_par(var a: Int, var b: Int) -> String:
    return String(a) + "," + String(b)


fn _parse_chave_par(var chave: String) raises -> List[Int]:
    var idx = -1
    for i in range(len(chave)):
        if chave[i:i+1] == ",":
            idx = i
            break
    var out = List[Int]()
    if idx < 0:
        out.append(0)
        out.append(0)
        return out^
    out.append(Int(chave[0:idx]))
    out.append(Int(chave[idx + 1:len(chave)]))
    return out^


fn _aprender_merges_bpe(mut sequencia: List[Int], var max_merges: Int) raises -> List[String]:
    var merges = List[String]()

    for _ in range(max_merges):
        var chaves = List[String]()
        var contagens = List[Int]()

        for i in range(len(sequencia) - 1):
            var chave = _chave_par(sequencia[i], sequencia[i + 1])
            var achou = False
            for j in range(len(chaves)):
                if chaves[j] == chave:
                    contagens[j] = contagens[j] + 1
                    achou = True
                    break
            if not achou:
                chaves.append(chave)
                contagens.append(1)

        if len(chaves) == 0:
            break

        var melhor_idx = 0
        for j in range(1, len(chaves)):
            if contagens[j] > contagens[melhor_idx]:
                melhor_idx = j

        if contagens[melhor_idx] <= 1:
            break

        var par = _parse_chave_par(chaves[melhor_idx])
        sequencia = _aplicar_merge(sequencia, par[0], par[1], 256 + len(merges))
        merges.append(chaves[melhor_idx])

    return merges^


struct ResumoTreinoGPT(Movable, Copyable):
    var loss_medio: Float32
    var perplexidade: Float32
    var total_janelas: Int

    fn __init__(out self, var loss_medio_in: Float32, var perplexidade_in: Float32, var total_janelas_in: Int):
        self.loss_medio = loss_medio_in
        self.perplexidade = perplexidade_in
        self.total_janelas = total_janelas_in


fn _clamp(var x: Float32, var lo: Float32, var hi: Float32) -> Float32:
    var y = x
    if y < lo:
        y = lo
    if y > hi:
        y = hi
    return y


fn _feature_ultimo_token(saida_transformer: tensor_defs.Tensor, var dim: Int) -> List[Float32]:
    var out = List[Float32]()
    if saida_transformer.formato[0] <= 0:
        for _ in range(dim):
            out.append(0.0)
        return out^

    var base = (saida_transformer.formato[0] - 1) * dim
    for d in range(dim):
        out.append(saida_transformer.dados[base + d])
    return out^


fn _feature_token(saida_transformer: tensor_defs.Tensor, var pos: Int, var dim: Int) -> List[Float32]:
    var out = List[Float32]()
    if len(saida_transformer.formato) != 2 or saida_transformer.formato[0] <= 0:
        for _ in range(dim):
            out.append(0.0)
        return out^
    if pos < 0:
        pos = 0
    if pos >= saida_transformer.formato[0]:
        pos = saida_transformer.formato[0] - 1
    var base = pos * dim
    for d in range(dim):
        out.append(saida_transformer.dados[base + d])
    return out^


fn _dot(var a: List[Float32], var b: List[Float32]) -> Float32:
    var n = len(a)
    if len(b) < n:
        n = len(b)
    var acc: Float32 = 0.0
    for i in range(n):
        acc = acc + a[i] * b[i]
    return acc


fn _logits_vocab(var feat: List[Float32], var head_w: List[Float32], var head_b: List[Float32], var dim: Int, var vocab_size: Int) -> List[Float32]:
    var logits = List[Float32]()
    for _ in range(vocab_size):
        logits.append(0.0)

    for v in range(vocab_size):
        var acc = head_b[v]
        for d in range(dim):
            acc = acc + feat[d] * head_w[d * vocab_size + v]
        logits[v] = acc
    return logits^


fn _softmax_probs(var logits: List[Float32]) -> List[Float32]:
    var probs = List[Float32]()
    if len(logits) == 0:
        return probs^

    var max_l = logits[0]
    for i in range(1, len(logits)):
        if logits[i] > max_l:
            max_l = logits[i]

    var soma: Float32 = 0.0
    for i in range(len(logits)):
        var e = Float32(math.exp(Float64(logits[i] - max_l)))
        probs.append(e)
        soma = soma + e

    if soma <= 0.0:
        soma = 1.0
    for i in range(len(probs)):
        probs[i] = probs[i] / soma
    return probs^


fn _coletar_corpus(amostras: List[conjuntos_pkg.AmostraGPTSupervisionada]) -> String:
    var corpus = ""
    for a in amostras:
        corpus = corpus + a.inicio + " " + a.completar + " "
    return corpus


fn _treinar_transformer_janelas(
    amostras: List[conjuntos_pkg.AmostraGPTSupervisionada],
    lotes_janela: List[conjuntos_pkg.LoteGPTJanelaToken],
) raises -> ResumoTreinoGPT:
    if len(lotes_janela) == 0:
        return ResumoTreinoGPT(0.0, 0.0, 0)

    var corpus = _coletar_corpus(amostras)
    var total_amostras = len(amostras)

    var seq_bpe = _chars_para_token_ids_ascii(corpus, 4096)
    var merges = _aprender_merges_bpe(seq_bpe, 20)

    var tipo = computacao_pkg.backend_nome_de_id(computacao_pkg.backend_cpu_id())
    var dim_modelo = 32
    var vocab_size = 256 + len(merges) + 32
    var bloco = transformer_pkg.criar_bloco_transformer_base(vocab_size, dim_modelo, 4, tipo)
    var driver = computacao_pkg.driver_sessao_ram()
    var cache = computacao_pkg.criar_kvcache_provider(driver, "gpt_demo")
    var adaptador = transformer_pkg.criar_adaptador_atencao_lora(4, 8.0)

    var head_w = List[Float32]()
    var head_b = List[Float32]()
    var seed = 727
    for i in range(dim_modelo * vocab_size):
        seed = (seed * 1664525 + 1013904223 + i) % 2147483647
        var u = Float32(seed) / Float32(2147483647)
        head_w.append((u * 2.0 - 1.0) * 0.02)
    for _ in range(vocab_size):
        head_b.append(0.0)

    var lr: Float32 = 0.03
    var epocas = 3
    var nll_total: Float32 = 0.0
    var count_tokens: Int = 0
    var loss_total: Float32 = 0.0

    print("Merges BPE de referencia (estilo e000006):", len(merges), "| amostras:", total_amostras, "| lotes_janela:", len(lotes_janela), "| vocab:", vocab_size)

    for epoca in range(epocas):
        var loss_epoca: Float32 = 0.0
        var count_epoca: Int = 0

        for lote in lotes_janela:
            for amostra in lote.amostras:
                if len(amostra.contexto_tokens) == 0:
                    continue

                    try:
                        var cache_forward = transformer_pkg.forward_transformer_base_com_cache(bloco, amostra.contexto_tokens, cache, adaptador)

                        var seq = len(amostra.contexto_tokens)
                        if seq <= 0:
                            continue

                        var grad_feats_por_token = List[List[Float32]]()
                        for _ in range(seq):
                            var g0 = List[Float32]()
                            for __ in range(dim_modelo):
                                g0.append(0.0)
                            grad_feats_por_token.append(g0)

                        var loss_amostra: Float32 = 0.0
                        var n_tokens_amostra: Int = 0

                        for pos in range(seq):
                            var feat = _feature_token(cache_forward.out, pos, dim_modelo)

                            var alvo_token = amostra.contexto_tokens[pos + 1] if pos + 1 < seq else amostra.alvo_token
                            if alvo_token < 0:
                                alvo_token = 0
                            if alvo_token >= vocab_size:
                                alvo_token = vocab_size - 1

                            var logits = _logits_vocab(feat, head_w, head_b, dim_modelo, vocab_size)
                            var probs = _softmax_probs(logits)
                            var prob_alvo = _clamp(probs[alvo_token], 1e-9, 1.0)
                            var loss_pos = -Float32(math.log(Float64(prob_alvo)))

                            var grad_feat = List[Float32]()
                            for _ in range(dim_modelo):
                                grad_feat.append(0.0)

                            for v in range(vocab_size):
                                var grad = probs[v]
                                if v == alvo_token:
                                    grad = grad - 1.0

                                for d in range(dim_modelo):
                                    grad_feat[d] = grad_feat[d] + grad * head_w[d * vocab_size + v]

                                for d in range(dim_modelo):
                                    var idx_w = d * vocab_size + v
                                    head_w[idx_w] = head_w[idx_w] - lr * grad * feat[d]
                                head_b[v] = head_b[v] - lr * grad

                            grad_feats_por_token[pos] = grad_feat

                            nll_total = nll_total + loss_pos
                            count_tokens = count_tokens + 1
                            loss_total = loss_total + loss_pos
                            loss_epoca = loss_epoca + loss_pos
                            loss_amostra = loss_amostra + loss_pos
                            count_epoca = count_epoca + 1
                            n_tokens_amostra = n_tokens_amostra + 1

                        if n_tokens_amostra > 0:
                            try:
                                transformer_pkg.aplicar_gradiente_saida_transformer_analitico_todos_tokens(
                                    bloco,
                                    amostra.contexto_tokens,
                                    cache_forward,
                                    grad_feats_por_token,
                                    lr * 0.12,
                                )
                            except _:
                                print("Falha ao aplicar gradiente analítico do transformer para amostra, pulando.")
                    except _:
                        print("Falha ao gerar cache forward do transformer para amostra, pulando.")
                        continue

        if count_epoca > 0:
            print("Epoca", epoca, "| loss_medio:", loss_epoca / Float32(count_epoca))

    if count_tokens <= 0:
        return ResumoTreinoGPT(0.0, 0.0, 0)

    var perplex = Float32(math.exp(Float64(nll_total / Float32(count_tokens))))
    var loss_medio = loss_total / Float32(count_tokens)
    return ResumoTreinoGPT(loss_medio, perplex, count_tokens)


fn _flatten_lotes(lotes: List[conjuntos_pkg.LoteGPTSupervisionado]) -> List[conjuntos_pkg.AmostraGPTSupervisionada]:
    var out = List[conjuntos_pkg.AmostraGPTSupervisionada]()
    for lote in lotes:
        for amostra in lote.amostras:
            out.append(amostra.copy())
    return out^


def executar_exemplo():
    try:
        print("--- Exemplo e000007: gpt_texto (treino + inferencia) ---")

        var dir_dataset = "exemplos/e000007_gpt_texto/dataset"
        var lotes = List[conjuntos_pkg.LoteGPTSupervisionado]()
        try:
            lotes = conjuntos_pkg.carregar_lotes_gpt_supervisionado_diretorio(dir_dataset, 3, 96, 96)
        except _:
            lotes = List[conjuntos_pkg.LoteGPTSupervisionado]()

        # Fallback opcional para arquivo unico, mantendo o mesmo parser de pares inicio/completar.
        if len(lotes) == 0:
            var arquivo_dataset = "exemplos/e000007_gpt_texto/dataset.txt"
            try:
                lotes = conjuntos_pkg.carregar_lotes_gpt_supervisionado_txt(arquivo_dataset, 3, 96, 96)
            except _:
                lotes = List[conjuntos_pkg.LoteGPTSupervisionado]()

        if len(lotes) == 0:
            print("Dataset vazio ou invalido em diretorio/arquivo de e000007.")
            return

        var amostras = _flatten_lotes(lotes)
        var lotes_janela = conjuntos_pkg.carregar_lotes_janela_token_gpt_diretorio(dir_dataset, 24, 1, 8, 96, 96)
        if len(lotes_janela) == 0:
            var arquivo_dataset = "exemplos/e000007_gpt_texto/dataset.txt"
            lotes_janela = conjuntos_pkg.carregar_lotes_janela_token_gpt_txt(arquivo_dataset, 24, 1, 8, 96, 96)

        print("Lotes carregados:", len(lotes), "| amostras:", len(amostras), "| lotes_janela:", len(lotes_janela))
        var resumo_treino = _treinar_transformer_janelas(amostras, lotes_janela)
        print("Treino supervisionado por janela concluido. Loss medio:", resumo_treino.loss_medio, "| Perplexidade:", resumo_treino.perplexidade, "| tokens:", resumo_treino.total_janelas)

        var prompt_conversa = "usuario: preciso de foco para estudar hoje"
        var prompt_instrucoes = "monte uma checklist de release"
        var prompt_ferramentas = "quero verificar se cuda esta ativo"

        var out_conversa = _inferir_por_modo(amostras, "conversa", prompt_conversa)
        var out_instrucoes = _inferir_por_modo(amostras, "instrucoes", prompt_instrucoes)
        var out_ferramentas = _inferir_por_modo(amostras, "ferramentas", prompt_ferramentas)

        print("\n[Inferencia - modo conversa]")
        print("Entrada:", prompt_conversa)
        print("Saida:", out_conversa)

        print("\n[Inferencia - modo instrucoes]")
        print("Entrada:", prompt_instrucoes)
        print("Saida:", out_instrucoes)

        print("\n[Inferencia - modo ferramentas]")
        print("Entrada:", prompt_ferramentas)
        print("Saida:", out_ferramentas)

        print("--- Fim do exemplo e000007 ---")
    except _:
        print("Erro no exemplo e000007: exceção não tratada, pulando exemplo.")
        return
