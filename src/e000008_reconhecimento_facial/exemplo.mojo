import src.camadas.cnn as cnn_pkg
import src.computacao as computacao_pkg
import src.dados as dados_pkg
import src.graficos as graficos_pkg
import src.nucleo.Tensor as tensor_defs
import src.uteis as uteis
import os
import math


fn _resize_if_wider(var img: List[List[Float32]], var max_w: Int) -> (List[List[Float32]], Float32):
    # Se a imagem for mais larga que max_w, redimensiona mantendo proporção
    var h = len(img)
    if h == 0:
        return img, 1.0
    var w = len(img[0])
    if w <= max_w or w == 0:
        return img, 1.0
    var new_w = max_w
    var new_h = max(1, Int(Float32(h) * Float32(new_w) / Float32(w)))
    var resized = graficos_pkg.redimensionar_matriz_grayscale_nearest(img, new_h, new_w)
    var scale: Float32 = Float32(new_w) / Float32(w)
    return resized, scale


# Parser de bounding boxes: cada linha com quatro números (x1 y1 x2 y2)
fn _ler_bboxes(var caminho_txt: String, var scale: Float32 = 1.0) -> List[List[Int]]:
    var out = List[List[Int]]()
    var linhas = List[String]()
    try:
        linhas = dados_pkg.carregar_txt_linhas(caminho_txt)
    except _:
        return out^

    for l in linhas:
        var campos = uteis.split_csv_simples(l.replace(" ", ","))
        if len(campos) < 4:
            continue
        var b = List[Int]()
        for i in range(4):
            try:
                var v = Float32(uteis.parse_float_ascii(String(campos[i].strip())))
                var sv = Int(v * scale)
                b.append(sv)
            except _:
                b.append(0)
        out.append(b^)
    return out^


fn _crop_matrix(var m: List[List[Float32]], var x1: Int, var y1: Int, var x2: Int, var y2: Int) -> List[List[Float32]]:
    var h = len(m)
    if h == 0:
        return List[List[Float32]]()
    var w = len(m[0])
    if x1 < 0:
        x1 = 0
    if y1 < 0:
        y1 = 0
    if x2 >= w:
        x2 = w - 1
    if y2 >= h:
        y2 = h - 1
    if x2 < x1 or y2 < y1:
        return List[List[Float32]]()
    var out = List[List[Float32]]()
    for yy in range(y1, y2 + 1):
        var row = List[Float32]()
        for xx in range(x1, x2 + 1):
            row.append(m[yy][xx])
        out.append(row^)
    return out^


fn _matriz_para_tensor_flat(var m: List[List[Float32]], var tipo: String) -> tensor_defs.Tensor:
    var h = len(m)
    var w = 0
    if h > 0:
        w = len(m[0])
    var f = List[Int]()
    f.append(1)
    f.append(h * w)
    var t = tensor_defs.Tensor(f^, tipo)
    var k = 0
    for y in range(h):
        for x in range(w):
            t.dados[k] = m[y][x]
            k = k + 1
    return t^


fn _iou(boxA: List[Int], boxB: List[Int]) -> Float32:
    var xA = max(boxA[0], boxB[0])
    var yA = max(boxA[1], boxB[1])
    var xB = min(boxA[2], boxB[2])
    var yB = min(boxA[3], boxB[3])
    var interW = xB - xA + 1
    var interH = yB - yA + 1
    if interW <= 0 or interH <= 0:
        return 0.0
    var interArea = Float32(interW * interH)
    var boxAArea = Float32((boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1))
    var boxBArea = Float32((boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1))
    return interArea / (boxAArea + boxBArea - interArea)


fn _carregar_dataset_detector(var dir_treino: String, var altura_alvo: Int, var largura_alvo: Int, var tipo: String) -> List[tensor_defs.Tensor]:
    var positivos = List[List[List[Float32]]]()
    var negativos = List[List[List[Float32]]]()

    var classes = List[String]()
    try:
        classes = os.listdir(dir_treino)
    except _:
        return List[tensor_defs.Tensor]()

    for c in classes:
        var dirc = os.path.join(dir_treino, c)
        if not os.path.isdir(dirc):
            continue
        var arquivos = List[String]()
        try:
            arquivos = os.listdir(dirc)
        except _:
            continue
        for nome in arquivos:
            if not nome.endswith(".bmp"):
                continue
            var caminho_bmp = os.path.join(dirc, nome)
            var caminho_txt = caminho_bmp.replace('.bmp', '.txt')
            var img = List[List[Float32]]()
            try:
                img = dados_pkg.carregar_bmp_grayscale_matriz(caminho_bmp)
            except _:
                continue
            # redimensionar se muito larga e ajustar escala para bboxes
            var img_scale: Float32 = 1.0
            try:
                img, img_scale = _resize_if_wider(img, 128)
            except _:
                img_scale = 1.0

            var h = len(img)
            var w = 0
            if h > 0:
                w = len(img[0])

            var bboxes = _ler_bboxes(caminho_txt, img_scale)
            for b in bboxes:
                var crop = _crop_matrix(img, b[0], b[1], b[2], b[3])
                if len(crop) == 0:
                    continue
                var resized = graficos_pkg.redimensionar_matriz_grayscale_nearest(crop, altura_alvo, largura_alvo)
                positivos.append(resized^)

            # gerar alguns negativos por imagem (amostras aleatórias que não sobrepõem bboxes)
            var tentativas = 0
            var criados = 0
            while tentativas < 50 and criados < max(1, len(bboxes)):
                tentativas = tentativas + 1
                var rw = largura_alvo
                var rh = altura_alvo
                var max_x = max(1, w - rw)
                var max_y = max(1, h - rh)
                var rx = 0
                var ry = 0
                if max_x > 0:
                    rx = (tentativas * 13) % max_x
                if max_y > 0:
                    ry = (tentativas * 17) % max_y
                var box = List[Int]()
                box.append(rx)
                box.append(ry)
                box.append(rx + rw - 1)
                box.append(ry + rh - 1)
                var ok = True
                for b in bboxes:
                    if _iou(box, b) > 0.1:
                        ok = False
                        break
                if not ok:
                    continue
                var cropn = _crop_matrix(img, box[0], box[1], box[2], box[3])
                if len(cropn) == 0:
                    continue
                var resizedn = graficos_pkg.redimensionar_matriz_grayscale_nearest(cropn, altura_alvo, largura_alvo)
                negativos.append(resizedn^)
                criados = criados + 1

    var total = len(positivos) + len(negativos)
    if total == 0:
        var fx = List[Int]()
        fx.append(0)
        fx.append(0)
        var fy = List[Int]()
        fy.append(0)
        fy.append(1)
        var vazio_x = tensor_defs.Tensor(fx^, tipo)
        var vazio_y = tensor_defs.Tensor(fy^, tipo)
        var out = List[tensor_defs.Tensor]()
        out.append(vazio_x.copy())
        out.append(vazio_y.copy())
        return out^

    var amostras = total
    var features = altura_alvo * largura_alvo
    var formato_x = List[Int]()
    formato_x.append(amostras)
    formato_x.append(features)
    var formato_y = List[Int]()
    formato_y.append(amostras)
    formato_y.append(1)

    var x_t = tensor_defs.Tensor(formato_x^, tipo)
    var y_t = tensor_defs.Tensor(formato_y^, tipo)

    var idx = 0
    for p in positivos:
        for yy in range(altura_alvo):
            for xx in range(largura_alvo):
                x_t.dados[idx * features + yy * largura_alvo + xx] = p[yy][xx]
        y_t.dados[idx] = 1.0
        idx = idx + 1
    for n in negativos:
        for yy in range(altura_alvo):
            for xx in range(largura_alvo):
                x_t.dados[idx * features + yy * largura_alvo + xx] = n[yy][xx]
        y_t.dados[idx] = 0.0
        idx = idx + 1

    var out = List[tensor_defs.Tensor]()
    out.append(x_t.copy())
    out.append(y_t.copy())
    return out^


fn _construir_centroides(var dir_treino: String, mut bloco: cnn_pkg.BlocoCNN, var altura_alvo: Int, var largura_alvo: Int, var tipo: String) raises -> (List[List[Float32]], List[String]):
    var classes = List[String]()
    try:
        classes = os.listdir(dir_treino)
    except _:
        return List[List[Float32]](), List[String]()

    var amostras_por_classe = List[List[List[Float32]]]()
    var representantes = List[String]()

    for c in classes:
        var dirc = os.path.join(dir_treino, c)
        if not os.path.isdir(dirc):
            continue
        var id = 0
        try:
            id = Int(c)
        except _:
            id = len(amostras_por_classe)

        while id >= len(amostras_por_classe):
            amostras_por_classe.append(List[List[Float32]]())

        var arquivos = List[String]()
        try:
            arquivos = os.listdir(dirc)
        except _:
            continue

        for nome in arquivos:
            if not nome.endswith(".bmp"):
                continue
            var caminho_bmp = os.path.join(dirc, nome)
            var caminho_txt = caminho_bmp.replace('.bmp', '.txt')
            var img = List[List[Float32]]()
            try:
                img = dados_pkg.carregar_bmp_grayscale_matriz(caminho_bmp)
            except _:
                continue
            # ajustar escala se imagem for larga
            var img_scale: Float32 = 1.0
            try:
                img, img_scale = _resize_if_wider(img, 128)
            except _:
                img_scale = 1.0
            var bboxes = _ler_bboxes(caminho_txt, img_scale)
            if len(bboxes) == 0:
                continue
            var b = bboxes[0]
            var crop = _crop_matrix(img, b[0], b[1], b[2], b[3])
            if len(crop) == 0:
                continue
            var resized = graficos_pkg.redimensionar_matriz_grayscale_nearest(crop, altura_alvo, largura_alvo)
            # extrair feature via CNN (usar batch 1)
            var t = _matriz_para_tensor_flat(resized, tipo)
            var feats = cnn_pkg.extrair_features(bloco, t)
            var fvec = List[Float32]()
            for i in range(len(feats.dados)):
                fvec.append(feats.dados[i])

            amostras_por_classe[id].append(fvec^)
            # registrar primeiro arquivo como representante da classe (se ainda nao existe)
            while id >= len(representantes):
                representantes.append("")
            if representantes[id] == "":
                representantes[id] = caminho_bmp

    var centroides = List[List[Float32]]()
    for cls in amostras_por_classe:
        if len(cls) == 0:
            centroides.append(List[Float32]())
            continue
        var dim = len(cls[0])
        var c = List[Float32]()
        for _ in range(dim):
            c.append(0.0)
        for s in cls:
            for i in range(dim):
                c[i] = c[i] + s[i]
        for i in range(dim):
            c[i] = c[i] / Float32(len(cls))
        centroides.append(c^)

    return centroides^, representantes^


fn _argmin_distancia(var vec: List[Float32], var centroides: List[List[Float32]]) -> (Int, Float32):
    var best = -1
    var bestd: Float32 = 1e38
    for i in range(len(centroides)):
        var c = centroides[i]
        if len(c) == 0 or len(c) != len(vec):
            continue
        var acc: Float32 = 0.0
        for j in range(len(vec)):
            var d = vec[j] - c[j]
            acc = acc + d * d
        if acc < bestd:
            bestd = acc
            best = i
    return best, bestd


fn executar_exemplo() raises:
    print("--- Exemplo e000008: reconhecimento_facial (detector + reconhecedor PoC) ---")

    var dir_dataset = "exemplos/e000008_reconhecimento_facial/dataset"
    # fallback: se a estrutura atual usa o dataset ao lado do código, use o dataset local
    if not os.path.isdir(dir_dataset):
        # fallback simples: usar caminho relativo ao diretório de execução (./e000008_reconhecimento_facial/dataset)
        dir_dataset = "e000008_reconhecimento_facial/dataset"
    var dir_treino = os.path.join(dir_dataset, "treino")
    var dir_teste = os.path.join(dir_dataset, "teste")

    var tipo = computacao_pkg.backend_nome_de_id(computacao_pkg.backend_cpu_id())

    var altura = 100
    var largura = 64

    print("Construindo dataset detector (crops", largura, "x", altura, ")...")
    var det = _carregar_dataset_detector(dir_treino, altura, largura, tipo)
    if len(det) < 2 or det[0].formato[0] == 0:
        print("Falha ao construir dataset detector. Verifique arquivos e .txt de bboxes.")
        return

    var x_det = det[0].copy()
    var y_det = det[1].copy()
    print("Amostras detector:", x_det.formato[0], "| Features:", x_det.formato[1])

    var bloco = cnn_pkg.BlocoCNN(altura, largura, 6, 3, 3, tipo)
    print("Treinando detector (PoC) com reduce-on-plateau...")
    # Treino em blocos para suportar redução de taxa quando estagnar
    var taxa: Float32 = 0.05
    var total_epocas: Int = 1000
    var bloco_epocas: Int = 100
    var paciencia: Int = 3
    var fator_reducao: Float32 = 0.5
    var lr_min: Float32 = 1e-4
    var melhor_loss: Float32 = 1e38
    var sem_melhora: Int = 0
    var epocas_treinado: Int = 0
    try:
        while epocas_treinado < total_epocas and taxa >= lr_min:
            var atual = min(bloco_epocas, total_epocas - epocas_treinado)
            var loss = cnn_pkg.treinar(bloco, x_det, y_det, taxa, atual, 20)
            print("Loss apos +", atual, "epocas (lr=", taxa, "):", loss)
            epocas_treinado = epocas_treinado + atual
            if loss + 1e-6 < melhor_loss:
                melhor_loss = loss
                sem_melhora = 0
            else:
                sem_melhora = sem_melhora + 1
            if sem_melhora >= paciencia:
                taxa = taxa * fator_reducao
                sem_melhora = 0
                print("Reduzindo taxa de aprendizado para:", taxa)
                if taxa < lr_min:
                    print("Taxa minima atingida, parando reducoes.")
                    break
        print("Treino finalizado. Epocas treinadas:", epocas_treinado, "loss final:", melhor_loss)
    except _:
        print("Erro durante o treino do detector.")

    print("Construindo centroides de reconhecimento usando o mesmo extractor...")
    var centroides = List[List[Float32]]()
    var representantes = List[String]()
    try:
        centroides, representantes = _construir_centroides(dir_treino, bloco, altura, largura, tipo)
    except _:
        print("Falha ao construir centroides")
        centroides = List[List[Float32]]()
        representantes = List[String]()
    print("Centroides calculados para", len(centroides), "ids (posições de array).")

    # salvar centroides e representantes para uso em produção leve
    var caminho_centroides = os.path.join(dir_dataset, "centroides.txt")
    var ok_save = _salvar_centroides(caminho_centroides, centroides)
    if ok_save:
        print("Centroides salvos em:", caminho_centroides)
    else:
        print("Falha ao salvar centroides em:", caminho_centroides)

    var caminho_representantes = os.path.join(dir_dataset, "representantes.txt")
    var ok_rep = _salvar_representantes(caminho_representantes, representantes)
    if ok_rep:
        print("Representantes salvos em:", caminho_representantes)
    else:
        print("Falha ao salvar representantes em:", caminho_representantes)

    print("\n--- Inferência em imagens de teste (detector + reconhecedor) ---")
    var arquivos_teste = List[String]()
    try:
        arquivos_teste = os.listdir(dir_teste)
    except _:
        print("Nenhum arquivo de teste encontrado em:", dir_teste)
        return

    for nome in arquivos_teste:
        if not nome.endswith('.bmp'):
            continue
        var caminho = os.path.join(dir_teste, nome)
        var img = List[List[Float32]]()
        try:
            img = dados_pkg.carregar_bmp_grayscale_matriz(caminho)
        except _:
            print("Falha ao carregar imagem de teste:", caminho, "pulando")
            continue
        var id_best: Int = -1
        var dist: Float32 = 0.0
        var box: List[Int] = List[Int]()
        var prob: Float32 = 0.0
        try:
            id_best, dist, box, prob = inferencia_imagem(caminho, bloco, centroides, altura, largura, tipo, 0.5)
        except _:
            print("Erro em inferencia para:", caminho)
            continue

        if id_best < 0:
            print("Imagem", nome, "| deteccao: nao_rosto ou abaixo do limiar | prob:", prob)
            continue

        print("Imagem", nome, "| deteccao: rosto | coords:", box[0], box[1], box[2], box[3], "| prob:", prob)
        # recalcular features na caixa original para medir similaridade com centroide
        var crop_orig = _crop_matrix(img, box[0], box[1], box[2], box[3])
        var t_feat = _matriz_para_tensor_flat(graficos_pkg.redimensionar_matriz_grayscale_nearest(crop_orig, altura, largura), tipo)
        var f = cnn_pkg.extrair_features(bloco, t_feat)
        var vec = List[Float32]()
        for i in range(len(f.dados)):
            vec.append(f.dados[i])
        var sim: Float32 = 0.0
        if id_best >= 0 and id_best < len(centroides) and len(centroides[id_best]) > 0:
            var nvec = _l2_normalize(vec)
            var ncent = _l2_normalize(centroides[id_best])
            sim = _cosine_similarity(nvec, ncent)
        var mesma = sim >= 0.7
        var repr = ""
        if id_best >= 0 and id_best < len(representantes):
            repr = representantes[id_best]
        print("Imagem", nome, "| reconhecimento: label=", id_best, "| distancia=", dist, "| similaridade=", sim, "| mesma_pessoa=", mesma, "| representante=", repr)

        # imprimir GT bboxes e overlap imediatamente para visibilidade
        var caminho_txt = caminho.replace('.bmp', '.txt')
        try:
            var gtb = _ler_bboxes(caminho_txt, 1.0)
            if len(gtb) == 0:
                print("GT: nenhum bbox encontrado em:", caminho_txt)
            else:
                for gi in range(len(gtb)):
                    var gt = gtb[gi]
                    var inter = _intersection_area(box, gt)
                    var gt_area = (gt[2] - gt[0] + 1) * (gt[3] - gt[1] + 1)
                    var pct: Float32 = -1.0
                    if gt_area > 0:
                        pct = Float32(inter) * 100.0 / Float32(gt_area)
                    print("GT[", gi, "]:", gt[0], gt[1], gt[2], gt[3], "| Intersect:", inter, "| Overlap%:", pct)
        except _:
            print("GT: falha ao ler bboxes de:", caminho_txt)

        # calcular overlap com bbox(s) do .txt de ground-truth (percentual sobre area do gt)
        caminho_txt = caminho.replace('.bmp', '.txt')
        var gt_over_percent: Float32 = -1.0
        try:
            var gtb = _ler_bboxes(caminho_txt, 1.0)
            if len(gtb) > 0:
                var best_inter: Int = 0
                var best_gt_idx: Int = -1
                for gi in range(len(gtb)):
                    var inter = _intersection_area(box, gtb[gi])
                    if inter > best_inter:
                        best_inter = inter
                        best_gt_idx = gi
                if best_gt_idx >= 0:
                    var gt = gtb[best_gt_idx]
                    var gt_area = (gt[2] - gt[0] + 1) * (gt[3] - gt[1] + 1)
                    if gt_area > 0:
                        gt_over_percent = Float32(best_inter) * 100.0 / Float32(gt_area)
        except _:
            gt_over_percent = -1.0

        # comparacoes adicionais: comparar com representante da mesma label e com representante de outra label
        if id_best >= 0 and id_best < len(representantes) and representantes[id_best] != "":
            var ok_same, score_same = comparar_duas_fotos(caminho, representantes[id_best], bloco, centroides, altura, largura, tipo, 0.7)
            print("Comparacao com representante mesma label:", representantes[id_best], "mesma=", ok_same, "score=", score_same)
        # comparar com representante de outro label (proximo modulo)
        if len(representantes) > 0:
            var other_idx = (id_best + 1) % max(1, len(representantes))
            if representantes[other_idx] != "":
                var ok_diff, score_diff = comparar_duas_fotos(caminho, representantes[other_idx], bloco, centroides, altura, largura, tipo, 0.7)
                print("Comparacao com representante outra label:", representantes[other_idx], "mesma=", ok_diff, "score=", score_diff)
        if gt_over_percent >= 0.0:
            print("Overlap com GT (% da area do GT):", gt_over_percent)

    print("--- Fim do exemplo e000008 ---")


fn _l2_normalize(var v: List[Float32]) -> List[Float32]:
    var s: Float32 = 0.0
    for x in v:
        s = s + x * x
    var norm = Float32(math.sqrt(Float64(s)))
    if norm <= 0.0:
        norm = 1.0
    var out = List[Float32]()
    for x in v:
        out.append(x / norm)
    return out^


fn _cosine_similarity(var a: List[Float32], var b: List[Float32]) -> Float32:
    if len(a) != len(b) or len(a) == 0:
        return 0.0
    var dot: Float32 = 0.0
    for i in range(len(a)):
        dot = dot + a[i] * b[i]
    return dot


fn _intersection_area(var boxA: List[Int], var boxB: List[Int]) -> Int:
    var xA = max(boxA[0], boxB[0])
    var yA = max(boxA[1], boxB[1])
    var xB = min(boxA[2], boxB[2])
    var yB = min(boxA[3], boxB[3])
    var interW = xB - xA + 1
    var interH = yB - yA + 1
    if interW <= 0 or interH <= 0:
        return 0
    return interW * interH


fn _detectar_melhor_caixa(var img: List[List[Float32]], mut bloco: cnn_pkg.BlocoCNN, var altura: Int, var largura: Int, var tipo: String) raises -> (List[Int], Float32):
    # Multi-scale sliding window + NMS
    var h = len(img)
    var w = 0
    if h > 0:
        w = len(img[0])

    var scales = List[Float32]()
    scales.append(1.0)
    scales.append(1.5)
    scales.append(0.75)

    # tentar múltiplas razões de aspecto (width/height) para suportar rostos retangulares
    var aspect_ratios = List[Float32]()
    aspect_ratios.append(0.75) # mais alto que largo
    aspect_ratios.append(1.0)  # quadrado
    aspect_ratios.append(1.25) # mais largo que alto
    aspect_ratios.append(1.5)

    var boxes = List[List[Int]]()
    var scores = List[Float32]()

    for s in scales:
        for ar in aspect_ratios:
            var rw = Int(Float32(largura) * s * ar)
            var rh = Int(Float32(altura) * s)
            if rw < 8 or rh < 8:
                continue
            var stride = max(4, Int(min(rw, rh) // 8))
            for y in range(0, max(1, h - rh + 1), stride):
                for x in range(0, max(1, w - rw + 1), stride):
                    var crop = _crop_matrix(img, x, y, x + rw - 1, y + rh - 1)
                    if len(crop) == 0:
                        continue
                    var resized = graficos_pkg.redimensionar_matriz_grayscale_nearest(crop, altura, largura)
                    var t = _matriz_para_tensor_flat(resized, tipo)
                    var p = cnn_pkg.inferir(bloco, t)
                    var prob = p.dados[0]
                    if prob <= 0.0:
                        continue
                    var box = List[Int]()
                    box.append(x)
                    box.append(y)
                    box.append(x + rw - 1)
                    box.append(y + rh - 1)
                    boxes.append(box^)
                    scores.append(prob)

    if len(boxes) == 0:
        return List[Int](), 0.0

    var keep_idx = _nms(boxes, scores, 0.3)
    if len(keep_idx) == 0:
        return List[Int](), 0.0

    # pick highest score among kept
    var best = -1
    var best_score: Float32 = 0.0
    for idx in keep_idx:
        if scores[idx] > best_score:
            best_score = scores[idx]
            best = idx

    if best < 0:
        return List[Int](), 0.0

    var best_box = boxes[best].copy()
    return best_box, best_score


fn _nms(var boxes: List[List[Int]], var scores: List[Float32], var iou_thresh: Float32) -> List[Int]:
    var idxs = List[Int]()
    for i in range(len(scores)):
        idxs.append(i)
    # sort by score desc (simple selection sort since no builtin sort)
    for i in range(len(idxs)):
        var best_i = i
        for j in range(i + 1, len(idxs)):
            if scores[idxs[j]] > scores[idxs[best_i]]:
                best_i = j
        if best_i != i:
            var tmp = idxs[i]
            idxs[i] = idxs[best_i]
            idxs[best_i] = tmp

    var keep = List[Int]()
    for i in idxs:
        var keep_it = True
        for k in keep:
            var iou = _iou(boxes[i], boxes[k])
            if iou > iou_thresh:
                keep_it = False
                break
        if keep_it:
            keep.append(i)
    return keep^


fn comparar_duas_fotos(
    var caminho1: String,
    var caminho2: String,
    mut bloco: cnn_pkg.BlocoCNN,
    var centroides: List[List[Float32]],
    var altura: Int,
    var largura: Int,
    var tipo: String,
    var threshold: Float32 = 0.7,
) raises -> (Bool, Float32):
    # Carregar imagens
    var img1 = List[List[Float32]]()
    var img2 = List[List[Float32]]()
    try:
        img1 = dados_pkg.carregar_bmp_grayscale_matriz(caminho1)
    except _:
        print("Falha ao carregar:", caminho1)
        return False, 0.0
    try:
        img2 = dados_pkg.carregar_bmp_grayscale_matriz(caminho2)
    except _:
        print("Falha ao carregar:", caminho2)
        return False, 0.0

    # Detectar faces (usar inferencia_imagem que já faz resize/mapeamento)
    var id1_tmp: Int = -1
    var dist1_tmp: Float32 = 0.0
    var box1 = List[Int]()
    var prob1: Float32 = 0.0
    try:
        id1_tmp, dist1_tmp, box1, prob1 = inferencia_imagem(caminho1, bloco, centroides, altura, largura, tipo, 0.0)
    except _:
        box1 = List[Int]()

    var id2_tmp: Int = -1
    var dist2_tmp: Float32 = 0.0
    var box2 = List[Int]()
    var prob2: Float32 = 0.0
    try:
        id2_tmp, dist2_tmp, box2, prob2 = inferencia_imagem(caminho2, bloco, centroides, altura, largura, tipo, 0.0)
    except _:
        box2 = List[Int]()

    if len(box1) == 0 or len(box2) == 0:
        print("Deteccao falhou em uma das imagens. probs:", prob1, prob2)
        return False, 0.0

    print("comparar_duas_fotos: caixa1=", box1[0], box1[1], box1[2], box1[3], "prob=", prob1)
    print("comparar_duas_fotos: caixa2=", box2[0], box2[1], box2[2], box2[3], "prob=", prob2)

    # Extrair features
    var crop1 = _crop_matrix(img1, box1[0], box1[1], box1[2], box1[3])
    var crop2 = _crop_matrix(img2, box2[0], box2[1], box2[2], box2[3])
    var t1 = _matriz_para_tensor_flat(graficos_pkg.redimensionar_matriz_grayscale_nearest(crop1, altura, largura), tipo)
    var t2 = _matriz_para_tensor_flat(graficos_pkg.redimensionar_matriz_grayscale_nearest(crop2, altura, largura), tipo)
    var f1 = cnn_pkg.extrair_features(bloco, t1)
    var f2 = cnn_pkg.extrair_features(bloco, t2)
    var v1 = List[Float32]()
    var v2 = List[Float32]()
    for i in range(len(f1.dados)):
        v1.append(f1.dados[i])
    for i in range(len(f2.dados)):
        v2.append(f2.dados[i])

    # Normalizar e comparar
    var n1 = _l2_normalize(v1)
    var n2 = _l2_normalize(v2)
    var score = _cosine_similarity(n1, n2)
    var same = score >= threshold
    print("comparar_duas_fotos: score=", score, "same=", same, "threshold=", threshold)
    return same, score


fn _salvar_centroides(var caminho: String, var centroides: List[List[Float32]]) -> Bool:
    var linhas = List[String]()
    for c in centroides:
        if len(c) == 0:
            linhas.append("")
            continue
        var linha = ""
        for i in range(len(c)):
            linha = linha + String(c[i])
            if i < len(c) - 1:
                linha = linha + ","
        linhas.append(linha)
    var conteudo = ""
    for i in range(len(linhas)):
        conteudo = conteudo + linhas[i]
        if i < len(linhas) - 1:
            conteudo = conteudo + "\n"
    return uteis.gravar_texto_seguro(caminho, conteudo)


fn _salvar_representantes(var caminho: String, var representantes: List[String]) -> Bool:
    var linhas = List[String]()
    for r in representantes:
        linhas.append(r)
    var conteudo = ""
    for i in range(len(linhas)):
        conteudo = conteudo + linhas[i]
        if i < len(linhas) - 1:
            conteudo = conteudo + "\n"
    return uteis.gravar_texto_seguro(caminho, conteudo)


fn _carregar_centroides(var caminho: String) -> List[List[Float32]]:
    var out = List[List[Float32]]()
    var linhas = dados_pkg.carregar_txt_linhas(caminho)
    if len(linhas) == 0:
        return out^
    for l in linhas:
        var campos = uteis.split_csv_simples(l.replace(" ", ","))
        if len(campos) == 0:
            out.append(List[Float32]()^)
            continue
        var c = List[Float32]()
        for s in campos:
            try:
                c.append(Float32(uteis.parse_float_ascii(s)))
            except _:
                c.append(0.0)
        out.append(c^)
    return out^


fn inferencia_imagem(var caminho: String, mut bloco: cnn_pkg.BlocoCNN, var centroides: List[List[Float32]], var altura: Int, var largura: Int, var tipo: String, var det_threshold: Float32 = 0.5) raises -> (Int, Float32, List[Int], Float32):
    # carregar imagem original
    var img_orig = List[List[Float32]]()
    img_orig = dados_pkg.carregar_bmp_grayscale_matriz(caminho)
    # redimensionar se muito larga para largura max 128 e detectar na imagem redimensionada
    var img = img_orig
    var scale: Float32 = 1.0
    try:
        img, scale = _resize_if_wider(img_orig, 128)
    except _:
        img = img_orig
        scale = 1.0

    var box_res, prob = _detectar_melhor_caixa(img, bloco, altura, largura, tipo)
    if len(box_res) == 0 or prob < det_threshold:
        return -1, 0.0, List[Int]() , prob

    # mapear caixa detectada de volta para coordenadas da imagem original
    var box = List[Int]()
    for c in box_res:
        if scale <= 0.0:
            box.append(c)
        else:
            box.append(Int(Float32(c) / scale))

    var crop = _crop_matrix(img_orig, box[0], box[1], box[2], box[3])
    var t = _matriz_para_tensor_flat(graficos_pkg.redimensionar_matriz_grayscale_nearest(crop, altura, largura), tipo)
    var f = cnn_pkg.extrair_features(bloco, t)
    var vec = List[Float32]()
    for i in range(len(f.dados)):
        vec.append(f.dados[i])
    var id_best, dist = _argmin_distancia(vec, centroides)
    return id_best, dist, box, prob
