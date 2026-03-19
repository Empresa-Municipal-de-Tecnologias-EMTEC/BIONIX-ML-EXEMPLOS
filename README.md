# BIONIX-ML-EXEMPLOS

<p align="center">
    <img src="ICONE.png" alt="Ícone do BIONIX-ML-EXEMPLOS" width="160">
</p>

Repositório contendo exemplos e demos de uso do framework **BIONIX-ML** em Mojo.

Importante: para executar os exemplos este repositório deve estar no mesmo nível que o repositório principal `BIONIX-ML` (clonado como um diretório irmão). Exemplo de layout esperado:

```
parent/
├─ BIONIX-ML/
└─ BIONIX-ML-EXEMPLOS/
```

Visite o repositório principal do framework: https://github.com/Empresa-Municipal-de-Tecnologias-EMTEC/BIONIX-ML

## O que contém

Este repositório agrupa exemplos práticos que demonstram funcionalidades do `BIONIX-ML`:

- `e000001_exemplo` — exemplos básicos de uso do núcleo e I/O.
- `e000002_modelo_linear` — exemplo de treinamento e persistência de um modelo linear usando CSV.
- `e000003_espirais_intercaladas` — exemplo de classificação com MLP em dataset sintético (espirais).
- `e000004_reconhecimento_digitos` — exemplo de reconhecimento de dígitos com MLP.
- `e000005_reconhecimento_digitos_cuda` — mesma ideia com backend CUDA (exige suporte CUDA e dependências).
- `e000006_embeding_bpe` — pipeline simples de embedding com BPE.
- `e000007_gpt_texto` — exemplo de componentes para treino/inferência de modelo de texto.
- `e000008_reconhecimento_facial` — prova de conceito de detector + reconhecedor facial (usa datasets locais).

## Como executar

1. Garanta que o repositório `BIONIX-ML` esteja clonado no mesmo diretório-pai.
2. Abra uma sessão Linux/WSL e navegue até `BIONIX-ML-EXEMPLOS/src`.
3. Execute os exemplos com `pixi`/`mojo` (exemplo):

```bash
cd BIONIX-ML-EXEMPLOS/src
pixi run mojo principal.mojo
```

Observação: alguns exemplos dependem de módulos do `BIONIX-ML/src` e de datasets locais; verifique cada pasta de exemplo para instruções específicas.

## Licença

Consulte a licença no repositório principal `BIONIX-ML`.
