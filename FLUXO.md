# Fluxo Completo: `./run.sh` → Resposta no Terminal

## FASE 0: Shell Script (`run.sh`)

```
./run.sh
   │
   ├─ 1. cd "$(dirname "$0")"
   │      Garante que o script rode no diretório do projeto, independente
   │      de onde você chamou.
   │
   ├─ 2. trap cleanup EXIT
   │      Registra a função cleanup() para rodar quando o script terminar
   │      (normal, Ctrl+C, erro). Ela faz `docker compose down` para parar o Redis.
   │
   ├─ 3. docker compose up -d
   │      Sobe o container Docker com Redis Stack (imagem redis/redis-stack).
   │      • -d = detached (background)
   │      • Porta 63790 do host → 6379 do container (Redis)
   │      • Porta 63791 do host → 8001 do container (RedisInsight UI)
   │      • Redis Stack inclui o módulo RediSearch (busca vetorial)
   │
   ├─ 4. until docker exec redis-rag redis-cli ping | grep -q PONG
   │      Loop que espera o Redis responder PONG ao PING.
   │      Garante que o banco está pronto antes de continuar.
   │
   ├─ 5. source .venv/bin/activate
   │      Ativa o virtualenv Python com todas as dependências instaladas.
   │
   ├─ 6. python main.py ingest    ←── FASE 1
   │
   └─ 7. python main.py chat      ←── FASE 2
```

---

## FASE 1: INGESTÃO (`python main.py ingest`)

```
main.py
   │
   ├─ argparse interpreta "ingest" → chama cmd_ingest()
   │
   ├─ load_dotenv()
   │    Carrega o .env para variáveis de ambiente do processo.
   │    Tecnologia: python-dotenv
   │    Lê: REDIS_URL, OLLAMA_BASE_URL, LLM_MODEL, EMBEDDING_MODEL,
   │        CHUNK_SIZE, CHUNK_OVERLAP
   │
   ├─ warnings.filterwarnings("ignore")
   │    Suprime warnings do LangChain no terminal (ficam só no log).
   │
   ├─ Logger registra: cabeçalho, redis, variáveis de ambiente
   │
   ├─ Thread com spinner animado (⣾⣽⣻⢿⡿⣟⣯⣷)
   │    threading.Thread + threading.Event para rodar animação
   │    enquanto a ingestão processa em paralelo.
   │
   └─ run_ingest("fontes")  ← src/ingest.py
```

### Passo 1: `get_embeddings()`

```python
HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
    device="cpu"
)
```

- Baixa (1ª vez) o modelo de embedding do HuggingFace Hub (~80MB)
- Carrega o modelo na memória (CPU)
- Esse modelo transforma qualquer texto em um vetor de 384 números decimais (dimensões)
- Textos similares geram vetores próximos no espaço vetorial (similaridade por cosseno)

**Tecnologias:**

- `langchain-huggingface` — wrapper do LangChain
- `sentence-transformers` — framework de embeddings
- `PyTorch (CPU)` — motor de cálculo do modelo

### Passo 2: `load_documents("fontes")`

```
DirectoryLoader + PyPDFLoader  →  **/*.pdf
DirectoryLoader + TextLoader   →  **/*.txt
```

- `PyPDFLoader` abre cada PDF, extrai o texto de cada página e cria um `Document` por página
- Cada `Document` tem:
  - `page_content`: texto da página
  - `metadata`: `{source, page, total_pages}`
- Resultado: 59 Documents (33 pgs guia_01 + 26 pgs guia_02)

**Tecnologias:**

- `langchain-community` — loaders de documentos
- `pypdf` — extração de texto de PDFs

### Passo 3: `split_documents(documents)`

```python
RecursiveCharacterTextSplitter(
    chunk_size=500,     # max 500 chars por chunk
    chunk_overlap=75    # 75 chars sobrepostos
)
```

- Pega cada Document e divide em pedaços menores
- Tenta quebrar por: `\n\n` → `\n` → `" "` → `""`
  (prioriza manter parágrafos inteiros)
- `overlap=75` garante que contexto não se perca entre dois chunks adjacentes

**Por que dividir?**

- Embeddings funcionam melhor com textos curtos
- Busca vetorial é mais precisa com chunks focados
- O prompt do LLM tem limite de contexto

**Normalização com regex:**

- Colapsa tabs/espaços múltiplos → 1 espaço
- Remove espaços ao redor de `\n`
- Colapsa 3+ linhas vazias → 2

Resultado: **190 chunks** com metadata preservado

**Tecnologia:** `langchain-text-splitters`

### Passo 4: `index_documents(chunks, embeddings)`

```python
RedisVectorStore.from_documents(
    documents=chunks,
    embedding=embeddings,
    redis_url="redis://localhost:63790",
    index_name="rag_docs"
)
```

Para cada chunk, internamente:

1. Pega `chunk.page_content` (texto)
2. Passa pelo modelo de embedding → vetor `[0.023, -0.156, ..., 0.089]` (384 dims)
3. Salva no Redis:
   - chave: `rag_docs:{ulid}`
   - campos: `content`, `content_vector`, `source`, `page`, etc.
4. Cria índice RediSearch `"rag_docs"` com:
   - VECTOR field (FLAT, FLOAT32, 384 dims, COSINE)
   - TEXT fields (content, source, etc.)

**Tecnologias:**

- `langchain-redis` — integração LangChain + Redis
- `Redis Stack` com `RediSearch` — busca vetorial
- Algoritmo `FLAT` — busca exata (força bruta)
- Distância `COSINE` — mede ângulo entre vetores

**Terminal mostra:** `Ingestão concluída com sucesso!`

---

## FASE 2: CHAT (`python main.py chat`)

```
main.py
   │
   ├─ argparse interpreta "chat" → chama cmd_chat()
   │
   └─ build_rag_chain()  ← src/rag_chain.py
```

### Passo 5: `get_vectorstore()`

```python
RedisVectorStore(
    redis_url, index_name="rag_docs",
    embeddings=embeddings
)
```

Conecta ao índice `"rag_docs"` que já existe no Redis (criado na ingestão). Não reindexa nada, apenas se conecta.

### Passo 6: `as_retriever(k=6)`

Cria um Retriever que, dada uma pergunta:

- Converte a pergunta em vetor (384 dims)
- Busca no Redis os 6 vetores mais próximos (similaridade por cosseno)
- Retorna os 6 Documents correspondentes

### Passo 7: `get_llm()`

```python
ChatOllama(
    model="llama3.1:8b",
    base_url="http://localhost:11434",
    temperature=0.1
)
```

- **Ollama** é um servidor local que roda LLMs
- **Llama 3.1 8B** é o modelo da Meta (~4.7GB)
- `temperature=0.1` = respostas mais determinísticas
- Comunicação via API REST HTTP (localhost:11434)

**Tecnologias:**

- `langchain-ollama` — wrapper LangChain para Ollama
- `Ollama` — runtime local de LLMs
- `Llama 3.1 8B` — modelo de linguagem da Meta

### Passo 8: Monta a Chain (LCEL)

```python
chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)
```

**LCEL (LangChain Expression Language):**

- O pipe `|` conecta componentes em sequência
- O dict `{}` roda componentes em paralelo
- `RunnablePassthrough()` apenas passa o input adiante sem modificar

Resultado: um pipeline executável que recebe uma string (pergunta) e retorna uma string (resposta).

**Terminal mostra:** `Chat RAG iniciado. Digite 'sair' para encerrar.`

---

## FASE 3: CADA PERGUNTA (loop do chat)

O usuário digita uma pergunta e o fluxo abaixo é executado:

### Etapa A: Embedding da Pergunta

```
"Como converter 0,625 de decimal para base 2?"
       │
       ▼  HuggingFace all-MiniLM-L6-v2
[0.045, -0.123, 0.067, ..., 0.091]  (384 floats)
```

O **mesmo modelo** usado na ingestão. Isso garante que perguntas e chunks estão no mesmo espaço vetorial.

### Etapa B: Busca Vetorial (Retriever)

Redis recebe o vetor da pergunta e executa:

```
FT.SEARCH rag_docs "*=>[KNN 6 @content_vector $vec]"
```

Para cada um dos 190 chunks armazenados:

- Calcula distância cosseno entre vetor da pergunta e vetor do chunk
- Ordena do mais similar ao menos similar
- Retorna os top 6

**Similaridade cosseno:**

- `1.0` = idênticos (vetores apontam mesma direção)
- `0.0` = sem relação (perpendiculares)
- Funciona porque o modelo de embedding aprendeu a mapear conceitos similares para regiões próximas

### Etapa C: `format_docs`

Pega os 6 Documents retornados e formata:

```
[Fonte: fontes/2026-1_arq1_guia_02.pdf]
Para converter a parte fracionária de um valor
decimal para binário, usar multiplicações sucessivas...

---

[Fonte: fontes/2026-1_arq1_guia_01.pdf]
Para converter um valor decimal (base=10) para
binário (base=2), usar divisões sucessivas por 2...
```

(e assim para os 6 chunks)

### Etapa D: Prompt Template

`ChatPromptTemplate` monta 2 mensagens:

**SYSTEM:**

```
Você é um assistente inteligente. Use SOMENTE o contexto abaixo para responder.
Se a resposta não estiver no contexto, diga que não tem informação suficiente.
Quando citar informações, mencione a fonte correta.

Contexto:
[Fonte: ...] texto do chunk 1
---
[Fonte: ...] texto do chunk 2
... (6 chunks)
```

**HUMAN:**

```
Como converter 0,625 de decimal para base 2?
```

- `{context}` → substituído pelo texto formatado
- `{question}` → substituído pela pergunta do usuário

### Etapa E: LLM (Llama 3.1 8B via Ollama)

LangChain envia `POST http://localhost:11434/api/chat` com as mensagens system + human.

Ollama recebe e:

1. **Tokeniza** as mensagens (texto → tokens numéricos)
2. **Processa** pelos 32 layers do transformer
3. **Gera tokens** de saída um a um (autoregressive)
4. **Decodifica** tokens → texto

O modelo **NÃO "sabe" nada** sobre seus PDFs. Ele apenas lê o contexto injetado no prompt e formula uma resposta baseada naquele texto. Isso é o **"R" de RAG: Retrieval-Augmented**.

`temperature=0.1` → pouca aleatoriedade na escolha dos próximos tokens, respostas mais consistentes.

### Etapa F: `StrOutputParser`

O LLM retorna um objeto `AIMessage`. O `StrOutputParser()` extrai apenas a string `.content` e retorna texto puro.

**Terminal mostra:**

```
IA: Para converter 0,625 de decimal para base 2,
    use multiplicações sucessivas por 2...
```

(em cor laranja)

---

## Arquitetura Visual

```
┌─────────┐    ┌──────────┐    ┌─────────────┐    ┌───────────┐
│  PDFs   │───▶│  Loader  │───▶│  Splitter   │───▶│ Embedding │
│ fontes/ │    │ (pypdf)  │    │ (500 chars) │    │(MiniLM-L6)│
└─────────┘    └──────────┘    └─────────────┘    └─────┬─────┘
                                                        │
                                                        ▼
                                                 ┌─────────────┐
                                                 │ Redis Stack │
                                                 │ (RediSearch)│
                                                 │ 190 vetores │
                                                 └──────┬──────┘
                                                        │
          ┌─────────────────────────────────────────────┘
          │  busca top-6 por similaridade cosseno
          ▼
┌──────────┐    ┌───────────┐    ┌──────────┐    ┌──────────┐
│ Pergunta │───▶│ Retriever │───▶│  Prompt  │───▶│Llama 3.1 │──▶ Resposta
│(usuário) │    │  (Redis)  │    │(sys+user)│    │(Ollama)  │
└──────────┘    └───────────┘    └──────────┘    └──────────┘
```

---

## Tecnologias e Seus Papéis

| Tecnologia | Papel |
|---|---|
| **Docker Compose** | Orquestra o container do Redis Stack |
| **Redis Stack** | Banco de dados vetorial (armazena e busca vetores) |
| **RediSearch** | Módulo do Redis que permite busca vetorial (KNN) |
| **python-dotenv** | Carrega configurações do `.env` |
| **LangChain Core** | Framework que conecta todos os componentes (LCEL) |
| **langchain-community** | Loaders de documentos (PDF, TXT) |
| **langchain-text-splitters** | Divide documentos em chunks |
| **langchain-huggingface** | Wrapper para modelos de embedding |
| **langchain-redis** | Integração LangChain + Redis como vector store |
| **langchain-ollama** | Wrapper para se comunicar com Ollama |
| **sentence-transformers** | Framework que roda o modelo de embedding |
| **all-MiniLM-L6-v2** | Modelo que converte texto → vetor 384D |
| **PyTorch (CPU)** | Motor de cálculo para o modelo de embedding |
| **pypdf** | Extração de texto de arquivos PDF |
| **Ollama** | Servidor local que roda o LLM |
| **Llama 3.1 8B** | Modelo de linguagem que gera as respostas |
| **threading** | Animação do spinner enquanto processa |
| **argparse** | Parse dos comandos CLI (ingest/chat/ask) |

## Arquivos do Projeto

| Arquivo | Responsabilidade |
|---|---|
| `main.py` | CLI, interface com usuário, cores, animação |
| `src/ingest.py` | Carrega, divide e indexa documentos no Redis |
| `src/rag_chain.py` | Monta a chain RAG (retriever + prompt + LLM) |
| `src/logger.py` | Logging estruturado de todas as etapas |
| `docker-compose.yml` | Define o container Redis Stack |
| `.env` | Configurações (URLs, modelo, chunk size) |
| `run.sh` | Script que sobe tudo e inicia o chat |
| `fontes/` | Diretório com os documentos fonte (PDFs/TXTs) |
| `logs/` | Logs de cada sessão (inicialização + conversas) |
