
from enum import Enum
from pydantic import BaseModel, ConfigDict, Field

class RouteTarget(str, Enum):
    RETRIEVER = "retriever"
    FALLBACK = "fallback"
    GENERATOR = "generator"
    REWRITER = "rewriter"

class GuardrailDecision(BaseModel):
    """Decisão estruturada do porteiro clínico."""

    model_config = ConfigDict(extra="ignore")

    is_safe: bool = Field(
        description="True quando a pergunta está dentro do escopo clínico/saúde/enfermagem."
    )
    is_ambiguous: bool = Field(
        description=(
            "True APENAS quando a entrada for um substantivo ou expressão genérica demais sem objeto "
            "clínico definido (ex: 'medidas', 'cuidados', 'valores', 'materiais', 'procedimentos'). "
            "False quando houver um procedimento, material, medicamento, sinal clínico ou sigla identificável "
            "(ex: 'insulina', 'sonda', 'curativo', 'flebite', 'SVD')."
        )
    )
    reason: str = Field(
        description="Justificativa curta da classificação."
    )

class EvaluationDecision(BaseModel):
    """Avaliação binária da relevância do contexto recuperado."""

    model_config = ConfigDict(extra="ignore")

    is_approved: bool = Field(
        description="True APENAS se os documentos contiverem a resposta específica para a pergunta; False se apenas citarem o tema sem responder."
    )
    reason: str = Field(
        description="Motivo objetivo da decisão."
    )

class RewriteDecision(BaseModel):
    """Consulta reformulada para nova busca no Elasticsearch."""

    model_config = ConfigDict(extra="ignore")

    standalone_question: str = Field(
        min_length=3,
        description="Consulta limpa e otimizada para busca sem prefixos conversacionais."
    )

class CondenseDecision(BaseModel):
    model_config = ConfigDict(extra="ignore")

    is_dependent: bool = Field(
        description=(
            "True apenas se a nova mensagem exigir o histórico anterior para fazer sentido "
            "(ex: pronomes como 'isso', 'dela', ou continuação direta). "
            "False se for uma mudança de assunto, novo termo ou pergunta autônoma."
        )
    )
    standalone_question: str = Field(
        min_length=1,
        description=(
            "Se is_dependent for True, a pergunta reescrita integrando o contexto anterior. "
            "Se is_dependent for False, exatamente a mensagem original do usuário sem alterações."
        ),
    )

