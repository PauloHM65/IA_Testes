# DRAG + LangChain + Redis

Sistema de **Document Retrieval-Augmented Generation (DRAG)** com MultiQuery, reranking via cross-encoder e expansao de vizinhos.

## Tecnologias

| Tecnologia | Funcao |
|---|---|
| **Qwen 2.5 14B** (configuravel) | LLM (via Ollama, local) |
| **intfloat/multilingual-e5-base** | Embedding (texto -> vetor) |
| **cross-encoder/ms-marco-MiniLM-L-6-v2** | Reranker |
| **Redis Stack** | Banco vetorial (Docker) |
| **LangChain** | Orquestracao do pipeline |

## Arquitetura

```
Documentos (.pdf/.txt)
        |
        v
   +------------+     +--------------------+
   | Ingestao   |---->| Redis (vetores)    |
   | Chunks +   |     | index: rag_docs    |
   | Embeddings |     | + rag_chunks_map   |
   +------------+     +--------+-----------+
                               |
   Pergunta                    |
       |                       |
       v                       |
   MultiQuery (LLM gera       |
   N variacoes)                |
       |                       |
       v                       v
   Embedding -------> Similaridade (k=20 por query)
                               |
                               v
                      Vizinhos (+-2 adjacentes)
                               |
                               v
                      Rerank (cross-encoder)
                               |
                               v
                      Top 5 chunks
                               |
                               v
                      Prompt + LLM
                               |
                               v
                      latex_to_unicode
                               |
                               v
                            Resposta
```

## Pre-requisitos

1. **Docker** (para o Redis Stack)
2. **Ollama** (para rodar o LLM localmente)
3. **Python 3.10+**

## Setup

### 1. Subir o Redis

```bash
docker compose up -d
```

Redis disponivel em `localhost:63790`. UI em `http://localhost:63791`.

### 2. Instalar o Ollama e baixar o modelo

```bash
# Instalar Ollama (https://ollama.com)
curl -fsSL https://ollama.com/install.sh | sh

# Baixar o modelo (exemplo com Qwen 2.5 14B)
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
| `RERANK_TOP_N` | `5` | Chunks mantidos apos rerank |
| `RETRIEVER_K` | `20` | Chunks candidatos por query |
| `MULTI_QUERY_N` | `3` | Variacoes geradas pelo MultiQuery |
| `NEIGHBOR_WINDOW` | `2` | Vizinhos adjacentes (+-N) |
| `CHUNK_SIZE` | `1500` | Tamanho maximo dos chunks |
| `CHUNK_OVERLAP` | `300` | Sobreposicao entre chunks |

## Uso

### Modo rapido (recomendado)

```bash
./run.sh
```

Sobe o Redis, faz a ingestao (incremental) e inicia o chat automaticamente.

### Comandos individuais

```bash
# Indexar documentos da pasta fontes/
python main.py ingest

# Forcar reingestao mesmo sem mudancas
python main.py ingest --force

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
│   ├── config.py          # Configuracao centralizada (lazy, le .env)
│   ├── ingest.py          # Carrega, divide e indexa documentos
│   ├── rag_chain.py       # DragPipeline (MultiQuery -> Vizinhos -> Rerank -> LLM)
│   └── logger.py          # Logging estruturado de cada etapa
├── main.py                # CLI (ingest / ask / chat)
├── run.sh                 # Script de execucao completa
├── docker-compose.yml     # Redis Stack
├── requirements.txt
├── .env.example
└── README.md
```

## Pipeline DRAG

```
Pergunta -> MultiQuery (N variacoes) -> Embedding -> Redis KNN (k=20)
    -> Vizinhos (+-2) -> Rerank (top 5) -> Prompt + LLM -> latex_to_unicode -> Resposta
```

1. **MultiQuery**: o LLM gera N variacoes da pergunta para ampliar a cobertura de busca
2. **Retriever**: busca os k chunks mais similares no Redis para cada variacao
3. **Vizinhos**: expande cada chunk com +-2 adjacentes (contexto mais amplo)
4. **Rerank**: cross-encoder reordena todos os chunks expandidos e seleciona os melhores
5. **LLM**: gera a resposta usando os chunks como contexto
6. **Pos-processamento**: converte LaTeX residual para Unicode (terminal-friendly)

### Ingestao incremental

A ingestao calcula um hash SHA-256 dos documentos (nome + tamanho + primeiros 8KB do conteudo). Se os documentos nao mudaram desde a ultima execucao, a ingestao e pulada automaticamente. Use `--force` para forcar.

### Modelo E5

Modelos da familia `intfloat/e5-*` exigem prefixos especificos:
- Queries: `"query: <texto>"`
- Documentos: `"passage: <texto>"`

Isso e tratado automaticamente pela classe `E5Embeddings` em `src/ingest.py`.
