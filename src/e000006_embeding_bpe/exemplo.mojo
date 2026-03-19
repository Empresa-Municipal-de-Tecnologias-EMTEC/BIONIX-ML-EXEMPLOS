import src.dados as dados_pkg
import src.camadas.transformer as transformer_pkg
import src.computacao as computacao_pkg
import src.nucleo.Tensor as tensor_defs


fn _indice_token(mut vocab: List[String], var token: String) -> Int:
    for i in range(len(vocab)):
        if vocab[i] == token:
            return i
    vocab.append(token)
    return len(vocab) - 1


fn _split_palavras_ascii(var linha: String) -> List[String]:
    var out = List[String]()
    var atual = ""
    for i in range(len(linha)):
        var c = linha[i:i+1]
        if c == " " or c == "\t" or c == "," or c == "." or c == ":" or c == ";" or c == "!" or c == "?":
            if len(atual) > 0:
                out.append(atual)
                atual = ""
        else:
            atual = atual + c
    if len(atual) > 0:
        out.append(atual)
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

        var melhor_par = _parse_chave_par(chaves[melhor_idx])
        var novo_id = 256 + len(merges)
        sequencia = _aplicar_merge(sequencia, melhor_par[0], melhor_par[1], novo_id)
        merges.append(chaves[melhor_idx])

    return merges^


fn _tokenizar_ascii_char(var texto: String) -> List[Int]:
    var out = List[Int]()
    for i in range(len(texto)):
        out.append(Int(ord(texto[i:i+1])) & 0xFF)
    return out^


fn _imprimir_preview_embedding(t: tensor_defs.Tensor, var max_linhas: Int = 2, var max_cols: Int = 6):
    var linhas = t.formato[0]
    var cols = t.formato[1]
    if linhas < max_linhas:
        max_linhas = linhas
    if cols < max_cols:
        max_cols = cols
    for i in range(max_linhas):
        var linha = "  token[" + String(i) + "]: "
        for j in range(max_cols):
            linha = linha + String(t.dados[i * cols + j])
            if j < max_cols - 1:
                linha = linha + ", "
        print(linha)


def executar_exemplo():
    try:
        print("--- Exemplo e000006: embedding_bpe com dataset .txt ---")

        var caminho_dataset = "exemplos/e000006_embeding_bpe/dataset.txt"
        var txt_data = dados_pkg.TXTData("", List[String]())
        try:
            txt_data = dados_pkg.carregar_txt(caminho_dataset)
        except _:
            print("Falha ao carregar dataset:", caminho_dataset)
            return
        if len(String(txt_data.texto_completo).strip()) == 0:
            print("Dataset txt vazio:", caminho_dataset)
            return

        var sequencia = _tokenizar_ascii_char(txt_data.texto_completo)
        var merges = _aprender_merges_bpe(sequencia, 24)

        print("Linhas carregadas:", len(txt_data.linhas))
        print("Tokens iniciais (chars):", len(sequencia))
        print("Merges BPE aprendidos:", len(merges))
        for i in range(len(merges)):
            if i >= 8:
                break
            print("  merge[", i, "] =", merges[i])

        var vocab_size = 256 + len(merges) + 8
        var dim_modelo = 24
        var bloco = transformer_pkg.criar_bloco_transformer_base(vocab_size, dim_modelo, 4, computacao_pkg.backend_nome_de_id(computacao_pkg.backend_cpu_id()))

        var token_ids = List[Int]()
        var limite = 32
        if len(sequencia) < limite:
            limite = len(sequencia)
        for i in range(limite):
            token_ids.append(sequencia[i])

        var emb = transformer_pkg.embedding_forward(bloco.embedding, token_ids)
        print("Embedding shape:", emb.formato[0], "x", emb.formato[1])
        _imprimir_preview_embedding(emb)
        print("--- Fim do exemplo e000006 ---")
    except _:
        print("Erro no exemplo e000006: exceção não tratada, pulando exemplo.")
        return
