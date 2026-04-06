"""Classifica a pergunta: exercicio ou teoria, materia, documento, palavras-chave."""

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from src.steps import register
from src.steps.base import BaseStep, PipelineData


_CLASSIFICAR_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Voce e um classificador academico. Dada a pergunta do aluno, identifique:\n"
     "1. Se e um pedido de RESOLUCAO DE EXERCICIO ou uma PERGUNTA TEORICA\n"
     "2. Qual materia/assunto trata\n"
     "3. Palavras-chave para buscar a teoria\n"
     "4. Se o aluno mencionou um documento/guia/lista especifico (ex: guia_02, lista_3)\n\n"
     "Responda APENAS neste formato (sem mais nada):\n"
     "TIPO: exercicio ou teoria\n"
     "MATERIA: <nome da materia/assunto>\n"
     "DOCUMENTO: <nome do documento mencionado, ou vazio se nao mencionou>\n"
     "BUSCA: <palavras-chave para buscar teoria, separadas por virgula>"),
    ("human", "{question}"),
])


@register
class ClassificarStep(BaseStep):
    name = "classificar"
    label = "Identificando materia e exercicio"

    def execute(self, data: PipelineData) -> PipelineData:
        chain = _CLASSIFICAR_PROMPT | self.ctx.llm | StrOutputParser()
        resultado = chain.invoke({"question": data.pergunta})

        tipo = "teoria"
        materia = ""
        documento = ""
        busca = ""
        for linha in resultado.strip().split("\n"):
            upper = linha.upper()
            if upper.startswith("TIPO:"):
                tipo = linha.split(":", 1)[1].strip().lower()
            elif upper.startswith("MATERIA:"):
                materia = linha.split(":", 1)[1].strip()
            elif upper.startswith("DOCUMENTO:"):
                documento = linha.split(":", 1)[1].strip()
            elif upper.startswith("BUSCA:"):
                busca = linha.split(":", 1)[1].strip()

        data.materia_identificada = materia
        data.documento_filtro = documento
        if busca:
            data.generated_queries = [busca]

        data.exercicio_texto = "" if tipo == "teoria" else "__EXERCICIO__"

        return data
