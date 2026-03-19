import src.e000001_exemplo.exemplo as exemplo
import src.e000002_modelo_linear.exemplo as exemplo_linear
import src.e000003_espirais_intercaladas.exemplo as exemplo_espirais
import src.e000004_reconhecimento_digitos.exemplo as exemplo_digitos
import src.e000005_reconhecimento_digitos_cuda.exemplo as exemplo_digitos_cuda
import src.e000006_embeding_bpe.exemplo as exemplo_embedding_bpe
import src.e000007_gpt_texto.exemplo as exemplo_gpt_texto
import src.e000008_reconhecimento_facial.exemplo as exemplo_facial_cnn

def main():
    print("\n" + "="*60)
    print("EXECUTANDO EXEMPLOS DO BIONIX")
    print("="*60)

    print("\n[1/8] Exemplo 1: Testes do Núcleo (exemplo)...")
    try:
        print("")
        #exemplo.executar_exemplo()
    except _:
        print("Exemplo 1 falhou; continuando com os próximos exemplos.")

    print("\n[2/8] Exemplo 2: Modelo Linear com CSV e persistência...")
    try:
        print("")
        #exemplo_linear.executar_exemplo()
    except _:
        print("Exemplo 2 falhou; continuando com os próximos exemplos.")

    print("\n[3/8] Exemplo 3: Espirais intercaladas com BMP + autograd + MLP...")
    try:
        print("")
        #exemplo_espirais.executar_exemplo()
    except _:
        print("Exemplo 3 falhou; continuando com os próximos exemplos.")

    print("\n[4/8] Exemplo 4: Reconhecimento de dígitos 0-9 com MLP...")
    try:
        print("")
        #exemplo_digitos.executar_exemplo()
    except _:
        print("Exemplo 4 falhou; continuando com os próximos exemplos.")

    print("\n[5/8] Exemplo 5: Reconhecimento de dígitos 0-9 com MLP (CUDA)...")
    try:
        print("")
        #exemplo_digitos_cuda.executar_exemplo()
    except _:
        print("Exemplo 5 falhou; continuando com os próximos exemplos.")

    print("\n[6/8] Exemplo 6: Embedding com BPE a partir de .txt...")
    try:
        print("")
        #exemplo_embedding_bpe.executar_exemplo()
    except _:
        print("Exemplo 6 falhou; continuando com os próximos exemplos.")

    print("\n[7/8] Exemplo 7: GPT texto (treino + inferência em conversa/instruções/ferramentas)...")
    try:
        print("")
        #exemplo_gpt_texto.executar_exemplo()
    except _:
        print("Exemplo 7 falhou; continuando com os próximos exemplos.")

    print("\n[8/8] Exemplo 8: Reconhecimento facial com bloco CNN...")
    try:
        exemplo_facial_cnn.executar_exemplo()
    except _:
        print("Exemplo 8 falhou; finalizando execução de exemplos.")

    print("\n" + "="*60)
    print("CONCLUÍDO: EXEMPLOS")
    print("="*60 + "\n")
