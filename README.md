# RAG com Llama 3.1 8B + LangChain + Redis

Sistema de Retrieval-Augmented Generation (RAG) usando:
- **LLM**: Llama 3.1 8B (via Ollama)
- **Framework**: LangChain
- **Banco Vetorial**: Redis Stack (RediSearch)
- **Embeddings**: sentence-transformers/all-MiniLM-L6-v2

## Arquitetura

```
Documentos (.pdf/.txt)
        │
        ▼
   ┌─────────┐     ┌────────────────┐
   │ Ingestão │────▶│ Redis (vetores)│
   └─────────┘     └───────┬────────┘
                           │
   Pergunta ──▶ Retriever ─┘
                    │
                    ▼
              ┌───────────┐
              │  Prompt +  │
              │ Llama 3.1  │
              └─────┬─────┘
                    │
                    ▼
                Resposta
```

## Pré-requisitos

1. **Docker** (para o Redis Stack)
2. **Ollama** (para rodar o Llama 3.1 localmente)
3. **Python 3.11+**

## Setup

### 1. Subir o Redis

```bash
docker compose up -d
```

Redis disponível em `localhost:6379`. UI em `http://localhost:8001`.

### 2. Instalar o Ollama e baixar o modelo

```bash
# Instalar Ollama (https://ollama.com)
curl -fsSL https://ollama.com/install.sh | sh

# Baixar o Llama 3.1 8B
ollama pull llama3.1:8b
```

### 3. Instalar dependências Python

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 4. Configurar variáveis de ambiente

```bash
cp .env.example .env
# Edite o .env se necessário
```

## Uso

### Indexar documentos

Coloque seus arquivos `.pdf` ou `.txt` na pasta `docs/`, depois:

```bash
python main.py ingest
# ou com diretório customizado:
python main.py ingest --docs-dir /caminho/para/documentos
```

### Fazer uma pergunta

```bash
python main.py ask "Qual é o tema principal do documento?"
```

### Chat interativo

```bash
python main.py chat
```

## Estrutura do Projeto

```
rag-llama-redis/
├── docs/                  # Seus documentos (.pdf, .txt)
├── src/
│   ├── __init__.py
│   ├── ingest.py          # Carrega, divide e indexa documentos
│   └── rag_chain.py       # Chain RAG (Retriever → Prompt → LLM)
├── main.py                # CLI (ingest / ask / chat)
├── docker-compose.yml     # Redis Stack
├── requirements.txt
├── .env.example
└── README.md
```
