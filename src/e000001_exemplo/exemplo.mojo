import src.dados as dados_pkg
import src.dados.print_helpers as print_helpers
import src.nucleo.Tensor as tensor_defs
import src.uteis as uteis

fn _normalizar_imagem_min_max_global(var matriz: List[List[Float32]]) -> List[List[Float32]]:
    if len(matriz) == 0:
        return List[List[Float32]]()

    var first_found = False
    var min_v: Float32 = 0.0
    var max_v: Float32 = 0.0

    for row in matriz:
        for v in row:
            if not first_found:
                min_v = v
                max_v = v
                first_found = True
            else:
                if v < min_v:
                    min_v = v
                if v > max_v:
                    max_v = v

    if not first_found:
        return List[List[Float32]]()

    var denom = max_v - min_v
    var out = List[List[Float32]]()
    for row in matriz:
        var row_out = List[Float32]()
        for v in row:
            if denom == 0.0:
                row_out.append(0.0)
            else:
                row_out.append((v - min_v) / denom)
        out.append(row_out^)
    return out^

def executar_exemplo():
    print("--- Exemplo e000001: leitura e normalização de dados ---")

    # Tenta carregar CSV do diretório do exemplo; se falhar, usa CSV embutido
    var caminho_csv = "exemplos/e000001_exemplo/dados.csv"
    var parsed = dados_pkg.carregar_csv(caminho_csv, ",", True)
    var usado_arquivo = True
    if len(parsed.linhas) == 0:
        usado_arquivo = False
        var csv_text = "x,y\n1.0,2.0\n2.0,4.1\n3.0,6.0\n4.0,8.1\n5.0,10.2\n"
        parsed = dados_pkg.carregar_csv_de_texto(csv_text, ",", True)

    print("Fonte CSV:", ("arquivo: " + caminho_csv) if usado_arquivo else "embutido")
    print("Cabeçalho detectado:")
    print_helpers.imprimir_cabecalho(parsed.cabecalho.copy())

    print("Linhas (raw):")
    print_helpers.imprimir_linhas_raw(parsed.linhas.copy(), 50)

    # Converter colunas para Float32 (assume todas as colunas numéricas neste exemplo)
    var dados_numericos = List[List[Float32]]()
    for r in parsed.linhas:
        var linha_numerica = True
        for j in range(len(r)):
            var campo_check = r[j].strip().replace(",", ".")
            if campo_check == "":
                linha_numerica = False
                break
            var i: Int = 0
            var zero = "0"[0:1]
            var nine = "9"[0:1]
            var dot = "."[0:1]
            var minus = "-"[0:1]
            var plus = "+"[0:1]
            var e_l = "e"[0:1]
            var e_U = "E"[0:1]
            while i < len(campo_check):
                var ch = campo_check[i:i+1]
                if not (ch >= zero and ch <= nine) and ch != dot and ch != minus and ch != plus and ch != e_l and ch != e_U:
                    linha_numerica = False
                    break
                i = i + 1
            if not linha_numerica:
                break
        if not linha_numerica:
            continue

        var linha = List[Float32](len(r))
        for j in range(len(r)):
            var campo = r[j].strip()
            var campo_clean = campo.replace(",", ".")
            linha[j] = uteis.parse_float_ascii(campo_clean)
        dados_numericos.append(linha.copy())

    print("Matriz numérica (primeiras linhas):")
    print_helpers.imprimir_matriz_float(dados_numericos.copy(), 50)

    # Normalização Min-Max
    var mm = dados_pkg.normalizar_min_max(dados_numericos.copy())
    print("\n")
    print_helpers.imprimir_min_max(mm.copy())

    # Normalização Z-Score
    var zs = dados_pkg.normalizar_zscore(dados_numericos.copy())
    print("\n")
    print_helpers.imprimir_zscore(zs.copy())

    # --- Exemplo de imagem ---
    print("\nExemplo de processamento de imagem:")
    # Tenta carregar BMP a partir do pacote `dados`
    var caminho_bmp = "exemplos/e000001_exemplo/dados.bmp"
    var bmp_diag_ok = dados_pkg.diagnosticar_bmp(caminho_bmp)
    var grayscale_matriz = dados_pkg.carregar_bmp_grayscale_matriz(caminho_bmp)

    print("\nExemplo de processamento de imagem (GRAYSCALE):")
    var img_mm_matriz = List[List[Float32]]()
    if bmp_diag_ok and len(grayscale_matriz) > 0:
        print("Arquivo BMP encontrado:", caminho_bmp, "h=", len(grayscale_matriz), "w=", len(grayscale_matriz[0]))
        img_mm_matriz = _normalizar_imagem_min_max_global(grayscale_matriz^)
    else:
        print("Arquivo BMP não encontrado ou inválido — usando imagem simulada")
        var row1 = List[Float32]()
        row1.append(0.0)
        row1.append(128.0)
        var row2 = List[Float32]()
        row2.append(255.0)
        row2.append(64.0)
        var fallback_img = List[List[Float32]]()
        fallback_img.append(row1^)
        fallback_img.append(row2^)
        img_mm_matriz = _normalizar_imagem_min_max_global(fallback_img^)

    print("Imagem normalizada (Min-Max):")
    print_helpers.imprimir_matriz_float(img_mm_matriz.copy())

    print("\nExemplo de processamento de imagem (RGB):")
    var bmp_rgb = dados_pkg.carregar_bmp_rgb(caminho_bmp)
    if bmp_rgb.width > 0 and bmp_rgb.height > 0 and len(bmp_rgb.pixels) > 0 and len(bmp_rgb.pixels[0]) > 0 and len(bmp_rgb.pixels[0][0]) >= 3:
        print("RGB carregado: w=", bmp_rgb.width, "h=", bmp_rgb.height)
        var p0 = bmp_rgb.pixels[0][0]
        print("Primeiro pixel RGB (normalizado): R=", p0[0], " G=", p0[1], " B=", p0[2])
    else:
        print("BMP RGB não encontrado ou inválido")

    print("\nExemplo de processamento de imagem (Preto e Branco):")
    var bmp_pb = dados_pkg.carregar_bmp_preto_branco(caminho_bmp)
    if bmp_pb.width > 0 and bmp_pb.height > 0 and len(bmp_pb.preto_branco) > 0 and len(bmp_pb.preto_branco[0]) > 0:
        print("PB carregado: w=", bmp_pb.width, "h=", bmp_pb.height)
        print("Primeiro pixel PB:", bmp_pb.preto_branco[0][0])
        var ativos_primeira_linha = 0
        for v in bmp_pb.preto_branco[0]:
            if v >= 0.5:
                ativos_primeira_linha = ativos_primeira_linha + 1
        print("Pixels ativos na 1a linha:", ativos_primeira_linha, "/", len(bmp_pb.preto_branco[0]))
    else:
        print("BMP preto e branco não encontrado ou inválido")

    # --- Exemplo de áudio ---
    print("\nExemplo de processamento de áudio:")
    var caminho_wav = "exemplos/e000001_exemplo/dados.wav"
    var wav_diag_ok = dados_pkg.diagnosticar_wav(caminho_wav)
    var wav_info = dados_pkg.carregar_wav(caminho_wav)

    var audio_para_normalizar = List[List[Float32]]()
    if wav_diag_ok and wav_info.sample_rate > 0 and wav_info.num_channels > 0 and len(wav_info.samples) > 0:
        print("Arquivo WAV encontrado:", caminho_wav, "sr=", wav_info.sample_rate, "ch=", wav_info.num_channels, "bps=", wav_info.bits_per_sample)
        for i in range(len(wav_info.samples)):
            var frame = List[Float32]()
            if len(wav_info.samples[i]) > 0:
                frame.append(wav_info.samples[i][0])
            else:
                frame.append(0.0)
            audio_para_normalizar.append(frame^)
    else:
        print("Arquivo WAV não encontrado ou inválido — usando áudio simulado")
        var frame0 = List[Float32]()
        frame0.append(0.1)
        frame0.append(-0.2)
        frame0.append(0.3)
        audio_para_normalizar.append(frame0^)
        var frame1 = List[Float32]()
        frame1.append(-0.1)
        frame1.append(0.2)
        frame1.append(-0.05)
        audio_para_normalizar.append(frame1^)

    var audio_zs = dados_pkg.normalizar_zscore(audio_para_normalizar.copy())
    print("Áudio normalizado (Z-Score):")
    print_helpers.imprimir_matriz_float(audio_zs.dados_normalizados.copy())

    print("\n--- Fim do exemplo e000001 ---")

    # --- Exemplo do núcleo (operação tensorial) ---
    print("\n--- Exemplo do núcleo Bionix (operações tensoriais) ---")
    var formato_a = List[Int]()
    formato_a.append(2)
    formato_a.append(2)
    var formato_b = List[Int]()
    formato_b.append(2)
    formato_b.append(2)
    var a = tensor_defs.Tensor(formato_a^)
    var b = tensor_defs.Tensor(formato_b^)
    a.dados[0] = 1.0
    a.dados[1] = 2.0
    a.dados[2] = 3.0
    a.dados[3] = 4.0
    b.dados[0] = 0.5
    b.dados[1] = 1.5
    b.dados[2] = -1.0
    b.dados[3] = 2.0

    var soma = tensor_defs.somar_elemento_a_elemento(a, b)
    print("resultado da soma no exemplo:")
    for i in range(len(soma.dados)):
        print("  ", soma.dados[i])

    print("--- Fim do exemplo ---")
