# GINOmoto

NPC de Minecraft cujo cérebro é um **cross-encoder NLI local** — Qwen3.5-4B ajustado como modelo "jev". Ele não gera
texto em momento nenhum: a única primitiva é a probabilidade de entailment entre uma premissa e uma hipótese, e **toda**
decisão sai dela.

Três usos da mesma primitiva:

* **planejar** — o scaffold da árvore de receitas pergunta ao modelo "O jogador tem pelo menos 2 gravetos?" contra a
  negação; a primeira exigência julgada falsa vira o próximo sub-objetivo, e o mineflayer executa a skill;
* **entender o chat** — cada mensagem do jogador vira premissa, cada intenção é uma frase sobre o que ele quer
  ("The player wants the bot to follow him around."), e o argmax do entailment é a ordem obedecida;
* **ver** — quando o scaffold fica sem nada visível para fazer, o bot tira um frame da própria visão (prismarine-viewer
  em Chrome headless), o modelo responde entailment sobre frases do tipo "Tem um tronco de árvore na frente do bot", e a
  resposta guia o passo.

## Índice

* [Como funciona](#como-funciona)
* [Requisitos](#requisitos)
* [Instalação](#instalação)
* [Rodar](#rodar)
* [Ordens de chat](#ordens-de-chat)
* [Servir o cérebro de outra máquina](#servir-o-cérebro-de-outra-máquina)
* [Medições](#medições)
* [Estrutura](#estrutura)
* [Limitações conhecidas](#limitações-conhecidas)
* [Créditos e licença](#créditos-e-licença)

## Como funciona

O modelo (`AlexWortega/openjev`, checkpoint `qwen3.5-4b-nli-v2`) é um `Qwen3_5ForSequenceClassification`: o backbone
Qwen3.5-4B com uma cabeça linear de 3 classes no hidden state do último token, `contradiction / entailment / neutral`.
O tower de visão do Qwen3.5 continua no checkpoint e é usado para premissas com imagem.

```
probs("Premise: ...\nHypothesis: ...") -> [p_contradiction, p_entailment, p_neutral]
```

Nada de "prompt engineering" escondido: as hipóteses de cada camada estão em `jev/planner.py` (predicados do jogo),
`jev/chat.py` (intenções) e `jev/vision.py` (frases sobre o frame). Trocar o comportamento do NPC = trocar hipóteses.

### Por que llama.cpp e não `transformers`

Qwen3.5 é **híbrido**: 24 das 32 camadas são Gated DeltaNet (atenção linear), não softmax. Sem o pacote
`flash-linear-attention` (que exige GPU), o `transformers` cai na implementação de referência do delta rule em PyTorch
puro — **medido: 2,4 s por token** (`jev/bench2.py`). Inviável para um NPC.

Então o backbone roda no **llama.cpp** (kernels próprios de CPU/Vulkan/CUDA, com a Gated DeltaNet implementada de
verdade) e a cabeça de 3 classes fica em numpy: o servidor sobe com `--pooling none`, devolve os hidden states por
token, e a decisão é `softmax(W @ h_último)` com `W` (3 × hidden) extraído do próprio checkpoint.

## Requisitos

* **Node.js ≥ 20** (mineflayer 4.39, prismarine-viewer, puppeteer-core)
* **Python ≥ 3.11** (cliente do modelo, conversão, scripts de medição)
* **llama.cpp** compilado — de preferência com `-DGGML_CUDA=ON` (NVIDIA) ou `-DGGML_VULKAN=ON` (Intel/AMD); CPU puro
  funciona, é só mais lento
* **Google Chrome / Chromium** — só para o serviço de frames (visão)
* **Java 21+** e um Paper para o servidor local de testes (opcional; dá para apontar para qualquer servidor 1.21.4)
* Disco: ~12 GB (checkpoint HF 9 GB + GGUF f16 8 GB) ou ~3,5 GB se você for direto para o Q4_K_M + mmproj
* GPU ajuda muito: o lote de intenções do chat e o encoder de imagem são o gargalo

## Instalação

```sh
git clone https://github.com/ezequias2d/GINOmoto.git && cd GINOmoto
npm i && scripts/shim-canvas.sh          # o shim explica-se no próprio script
```

`prismarine-viewer` exige o pacote nativo `canvas` (cairo + pango). Se você tem os headers, `npm i canvas` resolve;
sem pango, `scripts/shim-canvas.sh` aponta `canvas` para `@napi-rs/canvas`, que é pré-compilado.

Python (cliente do modelo + conversão):

```sh
uv venv venv
VIRTUAL_ENV=$PWD/venv uv pip install --index-url https://download.pytorch.org/whl/cpu torch
VIRTUAL_ENV=$PWD/venv uv pip install transformers accelerate pillow numpy gguf safetensors sentencepiece
```

llama.cpp (o mesmo checkout serve para o `llama-server`, o `llama-quantize` e o conversor):

```sh
scripts/build-llama.sh      # LLAMA_CPP=$HOME/llama.cpp VULKAN_SDK=/opt/vulkan-sdk/x86_64 por padrão
```

Modelo e conversão para GGUF:

```sh
hf download AlexWortega/openjev --include "qwen3.5-4b-nli-v2/*" --local-dir models/openjev
hf download Qwen/Qwen3.5-4B --include "preprocessor_config.json" "video_preprocessor_config.json" --local-dir models/qwen35-4b-proc

python tools/convert_jev.py --text   --out models/jev/jev-qwen35-4b-f16.gguf        --outtype f16
python tools/convert_jev.py --mmproj --out models/jev/mmproj-jev-qwen35-4b-f16.gguf --outtype f16
$LLAMA_CPP/build/bin/llama-quantize models/jev/jev-qwen35-4b-f16.gguf models/jev/jev-qwen35-4b-Q4_K_M.gguf Q4_K_M
```

`tools/convert_jev.py` existe por dois motivos concretos, os dois verificados com `tools/check_gguf.py`:

1. o `AutoConfig` do `transformers` lê esse checkpoint de *sequence classification* como uma config composta do Qwen3.5 e
   preenche os campos de texto com os defaults dele (**hidden 4096, ffn 12288, ctx 32768**), que não batem com os pesos
   (**2560 / 9216 / 262144**) — o script passa `hparams` lido direto do `config.json`;
2. o checkpoint não tem tensores de MTP, mas o conversor declara o bloco nextn e o `block_count` fica 33 — usa-se
   `--no-mtp` (32 blocos).

Ele também extrai a cabeça NLI (`score.weight` → `models/jev-head.npz`) e monta o diretório de conversão com symlink para
os pesos, sem duplicar 9 GB.

## Rodar

```sh
# 1) cérebro (llama.cpp): pooling none + mmproj = hidden states + visão
scripts/run-jev-server.sh                 # 127.0.0.1:8099; NGL=99 é o default (GPU), NGL=0 força CPU

# 2) mãos (mineflayer): estado do jogo + skills + viewer
node bot/skill_server.js --host 127.0.0.1 --port 25565 --version 1.21.4 --name GINOmoto --op \
                         --http 3010 --viewer 3007

# 3) olhos (opcional, para a visão)
node bot/shot.js --viewer 3007 --port 3008

# 4) o NPC
scripts/run-agent.sh --chat-only          # só obedece o chat
scripts/run-agent.sh --goal iron_pickaxe  # e, quando ninguém manda nada, persegue a meta do zero
```

Flags que importam no agente: `--goal` (item alvo), `--after` (meta seguinte ao concluir), `--vision`, `--full-intents`
(pontua as 19 intenções em vez do shortlist), `--ignore` / `--ignore-prefix` (outros bots que ecoam o chat),
`--api-key` e `--server-url` (cérebro remoto), `--dashboard` (painel com estado, chat, decisões e último frame).

Testes e medições, todos contra o servidor rodando:

| comando | o que verifica |
| --- | --- |
| `tools/text_check.py` | os 3 controles NLI (entailment / contradiction / neutral) e o custo por chamada |
| `tools/bench_server.py` | custo de 1, 6 e 19 sequências no lote (é o número que decide o UX do chat) |
| `tools/check_gguf.py` | metadados do GGUF contra os tensores do checkpoint (blocos, hidden, ffn, ctx) |
| `tools/test_api_key.py` | cliente com e sem bearer token contra um `llama-server --api-key` |
| `tools/vision_sanity.py` | o frame realmente influencia a decisão (duas imagens bem diferentes) |
| `jev/chat_test.py` | acerto das intenções em pt-BR e inglês (29 casos) |
| `jev/smoke.py` | carga do checkpoint + controles de texto e imagem (backend `torch` ou `llama`) |
| `jev/bench2.py` | por que o `transformers` não serve aqui (ms/token no delta rule de referência) |
| `bot/probe.js` | conexão: versão, dimensão, quem está online, se o servidor é online-mode |

## Ordens de chat

Ele escuta o chat do jogo e responde. Interpretação pelo modelo, com fallback por palavra-chave quando o entailment não
fica confiante (o log diz qual dos dois decidiu).

| você digita | intenção | o que ele faz |
| --- | --- | --- |
| `pega madeira pra mim` | `mine_logs` | mina troncos |
| `minera pedra` / `pega ferro` | `mine_stone` / `mine_iron` | mina pedra / minério de ferro |
| `faz um picareta de ferro` | `goal` | troca a meta e o scaffold assume o item inteiro |
| `me segue` | `follow` | segue o jogador que falou |
| `vem aqui` | `come` | anda até ele |
| `vai pra 120 -400` | `goto` | anda até as coordenadas |
| `para` | `stop` | interrompe na hora (a skill `stop` fura a fila, mesmo no meio de outra) |
| `mata esses zumbi` | `attack` | caça mobs próximos |
| `faz uma torre pra eu subir` | `tower` | sobe uma torre de blocos |
| `cava pra baixo` | `dig_down` | escava descendo |
| `me dá 3 madeira` | `give` | larga itens no chão |
| `olha pra mim` | `look` | vira a cabeça para o jogador |
| `o que você tem no inventario?` | `status` | responde o inventário no chat |
| `explora aí` | `explore` | sai andando (e, com `--vision`, usa o frame para escolher a direção) |
| `cala a boca` | `quiet` | para de falar (as ordens continuam funcionando) |

## Servir o cérebro de outra máquina

O contrato são duas rotas, nada mais:

* `GET /props` → `media_marker` (o `llama-server` randomiza esse marcador por processo)
* `POST /embedding {"content": ["premissa\nhipótese", ...]}` → hidden states por token (o cliente usa o último)
* com visão: `{"prompt_string": "... <media_marker> ...", "multimodal_data": ["<png em base64>"]}`

Quem serve precisa do mesmo GGUF + mmproj e sobe com `--embeddings --pooling none -b/--ub` grande o bastante para um
prompt com imagem e sem esquecer `--api-key`. Do lado do agente:

```sh
SERVER_URL=http://outra-maquina:8099 API_KEY=segredo scripts/run-agent.sh
```

## Medições

Máquina de referência: Intel Meteor Lake (iGPU Iris Xe, sem dGPU), 14 threads, sem acelerador dedicado.

* `transformers` (PyTorch, CPU): **2,4 s por token** no delta rule de referência — 16 tokens = 38 s. É o motivo de o
  caminho llama.cpp existir.
* llama.cpp CPU, GGUF f16: **3,6 s por sequência** de ~55 tokens; lote de 19 intenções = **74,7 s**.
* llama.cpp CPU, **Q4_K_M**: ~**2,5 s por sequência**; rerank de 3 opções = 7,4 s. A quantização **não** deslocou o
  classificador: os três controles continuam iguais (entailment 0,93 / contradiction 1,00 / neutral 1,00).
* Por isso o chat usa *shortlist*: a palavra-chave levanta 3–4 candidatos e **o modelo decide entre eles** — uma ordem
  ao vivo fica em **~10–15 s**. `--full-intents` pontua as 19, exato e lento.
* Visão: caminho completo funciona, mas **~40–60 s por frame** com o encoder na CPU. Fica atrás de `--vision` e só
  dispara quando o scaffold não tem nada visível para fazer.
* Primeira requisição depois de subir o `llama-server` é bem mais lenta (warmup) — mantenha o servidor quente.

## Estrutura

```
bot/skill_server.js   mineflayer: estado + skills + fila de chat + prismarine-viewer
bot/shot.js           Chrome headless no viewer, congela a página entre frames
bot/fake_player.js    jogador falso, para testar o chat sem abrir o cliente
bot/probe.js          teste de conexão/versão/modo online
jev/model.py          checkpoint HF em PyTorch (correto; lento nesta classe de CPU)
jev/model_llama.py    o mesmo modelo via llama.cpp + cabeça numpy   ← é o que roda
jev/backend.py        escolhe o backend
jev/planner.py        árvore de receitas, backward chaining, render do estado em frases
jev/chat.py           intenções por entailment, alvo por rerank, fallback por palavra-chave
jev/vision.py         frases sobre o frame + política de direção
jev/agent.py          o NPC: laço principal, fila de skills, interrupção, dashboard
jev/mc.py             cliente do skill server (+ serviço de frame)
tools/convert_jev.py  checkpoint HF → GGUF + mmproj + cabeça npz
tools/                checagens e medições (ver tabela acima)
scripts/              build-llama.sh, run-jev-server.sh, run-agent.sh, shim-canvas.sh, tunnel.sh
```

## Limitações conhecidas

* Sem `op` no servidor, o bot não usa `/clear` nem `/spreadplayers`; `reset` responde erro e o agente ignora.
* A visão é caríssima em CPU; em GPU deve cair para poucos segundos por frame (mmproj no Vulkan ainda não foi validado).
* O planejador usa **1 frase por predicado** por padrão; `--phrasings 3` reproduz a configuração do upstream (mais
  robusto, ~3x mais caro).
* A primeira requisição após subir o servidor é lenta (warmup do llama.cpp).
* O shortlist do chat é deliberado: 19 intenções por mensagem custa ~75 s em CPU.

## Créditos e licença

* Modelo e abordagem do "jev": **AlexWortega/openjev** (MIT) — Qwen3.5-4B virado cross-encoder NLI, com o scaffold de
  Minecraft (`code/minecraft.py`, `code/mc_bot.js`) que serviu de base para `jev/planner.py` e `bot/skill_server.js`.
  As mudanças aqui: host/versão/`--op` parametrizados, skills novas (`say`, `follow`, `come`, `goto`, `stop`, `attack`,
  `look`, `tower`, `digdown`, `toss`), fila de chat com interrupção, camada de visão e o caminho llama.cpp.
* [mineflayer](https://github.com/PrismarineJS/mineflayer) e [prismarine-viewer](https://github.com/PrismarineJS/prismarine-viewer) (MIT)
* [llama.cpp](https://github.com/ggml-org/llama.cpp) (MIT)

Este projeto é MIT — ver [LICENSE](LICENSE).
