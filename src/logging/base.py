"""Interfaces de logging segregadas (ISP)."""

from __future__ import annotations

from abc import ABC, abstractmethod


class IngestLogger(ABC):
    """Interface de logging para a fase de ingestao."""

    @abstractmethod
    def config_carregado(self, variaveis: dict): ...

    @abstractmethod
    def documentos_carregados(self, num_docs: int, fontes: list): ...

    @abstractmethod
    def chunks_gerados(self, chunks: list, chunk_size: int, chunk_overlap: int): ...

    @abstractmethod
    def embeddings_carregado(self, model_name: str): ...

    @abstractmethod
    def indexacao_redis(self, num_chunks: int, redis_url: str, index_name: str): ...

    @abstractmethod
    def alerta(self, mensagem: str): ...


class SessionLogger(ABC):
    """Interface de logging para controle de sessao."""

    @abstractmethod
    def cabecalho(self, service_name: str = ""): ...

    @abstractmethod
    def section(self, titulo: str): ...

    @abstractmethod
    def step(self, descricao: str): ...

    @abstractmethod
    def pipeline_montado(self, steps: list[str]): ...

    @abstractmethod
    def pronto(self): ...

    @abstractmethod
    def encerramento(self, motivo: str): ...


class ChatLogger(ABC):
    """Interface de logging para interacoes de chat."""

    @abstractmethod
    def separador_conversas(self): ...

    @abstractmethod
    def interacao(self, numero: int, pergunta: str, chunks: list, contexto: str,
                  resposta: str, generated_queries: list[str], raw_count: int,
                  rerank_count: int): ...


class BaseLogger(IngestLogger, SessionLogger, ChatLogger, ABC):
    """Interface completa que combina todas as interfaces de logging.

    Use esta classe quando precisar de todas as capacidades.
    Use as interfaces individuais quando precisar de apenas uma fatia.
    """
    pass
