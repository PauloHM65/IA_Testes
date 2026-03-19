# DRAG com Qwen 2.5 14B + LangChain + Redis

Sistema de **Document Retrieval-Augmented Generation (DRAG)** com reranking via cross-encoder.

## Tecnologias

| Tecnologia | Função |
|---|---|
| **Qwen 2.5 14B** | LLM (via Ollama, local) |
| **intfloat/multilingual-e5-base** | Embedding (texto → vetor 768d) |
| **cross-encoder/ms-marco-MiniLM-L-6-v2** | Reranker |
| **Redis Stack** | Banco vetorial (Docker) |
| **LangChain (LCEL)** | Orquestração do pipeline |

## Arquitetura

```
Documentos (.pdf/.txt)
        │
        ▼
   ┌──────────┐     ┌────────────────────┐
   │ Ingestão  │────▶│ Redis (vetores)    │
   │ Chunks +  │     │ index: rag_docs    │
   │ Embeddings│     └────────┬───────────┘
   └──────────┘              │
                              │
   Pergunta ──▶ Embedding ──▶ Similaridade (k=20)
                                     │
                                     ▼
                              Rerank (cross-encoder)
                                     │
                                     ▼
                              Top 5 chunks
                                     │
                                     ▼
                              Prompt + Qwen 2.5 14B
                                     │
                                     ▼
                                  Resposta
```

## Pre-requisitos

1. **Docker** (para o Redis Stack)
2. **Ollama** (para rodar o Qwen 2.5 localmente)
3. **Python 3.10+**

## Setup

### 1. Subir o Redis

```bash
docker compose up -d
```

Redis disponivel em `localhost:6379`. UI em `http://localhost:8001`.

### 2. Instalar o Ollama e baixar o modelo

```bash
# Instalar Ollama (https://ollama.com)
curl -fsSL https://ollama.com/install.sh | sh

# Baixar o Qwen 2.5 14B
ollama pull qwen2.5:14b
```

### 3. Instalar dependencias Python

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 4. Configurar variaveis de ambiente

```bash
cp .env.example .env
```

Variaveis disponiveis:

| Variavel | Default | Descricao |
|---|---|---|
| `REDIS_URL` | `redis://localhost:6379` | URL do Redis |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | URL do Ollama |
| `LLM_MODEL` | `qwen2.5:14b` | Modelo LLM |
| `EMBEDDING_MODEL` | `intfloat/multilingual-e5-base` | Modelo de embedding |
| `RERANK_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Modelo de reranking |
| `RERANK_TOP_N` | `5` | Chunks apos rerank |
| `CHUNK_SIZE` | `1500` | Tamanho dos chunks |
| `CHUNK_OVERLAP` | `300` | Overlap entre chunks |

## Uso

### Modo rapido (recomendado)

```bash
./run.sh
```

Sobe o Redis, faz a ingestao e inicia o chat automaticamente.

### Comandos individuais

```bash
# Indexar documentos da pasta fontes/
python main.py ingest

# Indexar de outro diretorio
python main.py ingest --docs-dir /caminho/para/documentos

# Pergunta unica
python main.py ask "Qual e o tema principal do documento?"

# Chat interativo
python main.py chat
```

## Estrutura do Projeto

```
rag-llama-redis/
├── fontes/                # Documentos fonte (.pdf, .txt)
├── logs/                  # Logs detalhados de cada sessao
├── src/
│   ├── __init__.py
│   ├── ingest.py          # Carrega, divide e indexa documentos
│   ├── rag_chain.py       # Pipeline DRAG (Retriever → Rerank → LLM)
│   └── logger.py          # Logging detalhado de cada etapa
├── main.py                # CLI (ingest / ask / chat)
├── run.sh                 # Script de execucao completa
├── docker-compose.yml     # Redis Stack
├── fluxo.excalidraw       # Diagrama do fluxo
├── FLUXO.md               # Descricao do fluxo
├── requirements.txt
├── .env.example
└── README.md
```

## Pipeline DRAG

```
INPUT → Embedding → Busca Redis (k=20) → Rerank (top 5) → Prompt + LLM → OUTPUT
```

1. **Embedding**: converte a pergunta em vetor 768d
2. **Retriever**: busca os 20 chunks mais similares no Redis
3. **Rerank**: cross-encoder reordena e seleciona os 5 melhores
4. **LLM**: Qwen 2.5 14B gera a resposta usando os chunks como contexto