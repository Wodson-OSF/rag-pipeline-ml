"""
API REST para o RAG Pipeline - Sistema de Consulta Inteligente
Autor: Wodson-OSF
Data: 28/06/2026
Versão: 1.0.0
"""

import sys
import os
from typing import List, Optional, Dict, Any

# Adiciona a pasta atual ao path do Python
sys.path.insert(0, os.path.dirname(__file__))

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Importa o RAG Pipeline
from rag_pipeline import RAGPipeline

# Carrega variáveis de ambiente
load_dotenv()

# ============================================================
# 1. DEFINIÇÃO DOS MODELOS DE DADOS (Pydantic)
# ============================================================

class QueryRequest(BaseModel):
    """
    Modelo da requisição para consulta ao RAG Pipeline.
    """
    question: str = Field(
        ...,
        description="Pergunta do usuário",
        example="Como resetar minha senha?",
        min_length=1,
        max_length=500
    )
    top_k: Optional[int] = Field(
        3,
        description="Número de documentos a recuperar",
        ge=1,
        le=10
    )

class DocumentResponse(BaseModel):
    """Modelo de um documento recuperado."""
    id: str
    text: str
    score: float
    metadata: Optional[Dict[str, Any]] = {}

class QueryResponse(BaseModel):
    """Modelo da resposta da API."""
    question: str
    answer: str
    context: List[DocumentResponse]
    error: bool = False

class HealthResponse(BaseModel):
    """Modelo de resposta do health check."""
    status: str
    version: str
    total_documents: int
    model: str

# ============================================================
# 2. INICIALIZAÇÃO DO PIPELINE
# ============================================================

# Instancia o pipeline uma única vez
print("🔄 Inicializando RAG Pipeline...")
pipeline = RAGPipeline()

# Indexa documentos de exemplo ao iniciar a API
print("📂 Indexando documentos de exemplo...")
sample_docs = [
    {"id": "doc1", "text": "Resetar senha: Acesse 'Esqueci minha senha' na tela de login do aplicativo."},
    {"id": "doc2", "text": "Limpar cache: Vá em Configurações > Armazenamento > Limpar cache para melhorar o desempenho."},
    {"id": "doc3", "text": "Emitir segunda via do boleto: Acesse o portal do cliente, seção 'Boletos', clique em 'Emitir segunda via'."},
    {"id": "doc4", "text": "Alterar dados cadastrais: Acesse 'Meu Perfil' no aplicativo e clique em 'Editar dados'."},
    {"id": "doc5", "text": "Recuperar acesso ao sistema: Caso não consiga resetar a senha, entre em contato com o suporte via chat."},
]
pipeline.batch_index(sample_docs)
print("✅ Documentos indexados com sucesso!")

# ============================================================
# 3. CRIAÇÃO DA APLICAÇÃO FASTAPI
# ============================================================

app = FastAPI(
    title="RAG Pipeline API",
    description="API para consulta inteligente usando RAG (Retrieval-Augmented Generation)",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    contact={
        "name": "Wodson-OSF",
        "email": "WOTH_O@HOTMAIL.COM",
    },
    license_info={
        "name": "MIT",
        "url": "https://opensource.org/licenses/MIT",
    }
)

# ============================================================
# 4. ENDPOINTS
# ============================================================

@app.get(
    "/",
    response_model=HealthResponse,
    summary="Health Check",
    description="Verifica se a API está funcionando."
)
async def health_check():
    """Endpoint de saúde."""
    stats = pipeline.get_stats()
    return HealthResponse(
        status="online",
        version="1.0.0",
        total_documents=stats.get("total_documents", 0),
        model=stats.get("embedding_model", "all-MiniLM-L6-v2")
    )

@app.post(
    "/ask",
    response_model=QueryResponse,
    summary="Fazer uma pergunta",
    description="Envie uma pergunta e receba uma resposta baseada nos documentos indexados."
)
async def ask_question(request: QueryRequest):
    """Processa uma pergunta usando o RAG Pipeline."""
    try:
        if not request.question or len(request.question.strip()) == 0:
            raise HTTPException(
                status_code=400,
                detail="A pergunta não pode estar vazia."
            )

        result = pipeline.ask(request.question)

        if result.get("error", False):
            raise HTTPException(
                status_code=404,
                detail="Nenhum documento relevante encontrado."
            )

        context_docs = [
            DocumentResponse(
                id=doc.get("id", ""),
                text=doc.get("text", ""),
                score=doc.get("score", 0.0),
                metadata=doc.get("metadata", {})
            )
            for doc in result.get("context", [])
        ]

        return QueryResponse(
            question=result.get("query", ""),
            answer=result.get("response", "Não foi possível gerar uma resposta."),
            context=context_docs,
            error=False
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erro interno: {str(e)}"
        )

@app.get(
    "/stats",
    summary="Estatísticas do Pipeline",
    description="Retorna estatísticas detalhadas do pipeline."
)
async def get_stats():
    """Retorna estatísticas do pipeline."""
    stats = pipeline.get_stats()
    return JSONResponse(content=stats)

@app.get(
    "/documents",
    summary="Listar Documentos",
    description="Retorna a lista de todos os documentos indexados."
)
async def list_documents(
    limit: int = Query(10, ge=1, le=50, description="Número máximo de documentos")
):
    """Lista os documentos indexados."""
    docs = pipeline.documents[:limit]
    return JSONResponse(content=[
        {
            "id": doc["id"],
            "text": doc["text"][:100] + "..." if len(doc["text"]) > 100 else doc["text"],
            "metadata": doc.get("metadata", {})
        }
        for doc in docs
    ])

# ============================================================
# 5. PONTO DE ENTRADA
# ============================================================

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("API_PORT", 8000))
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=port,
        reload=True,
        log_level="info"
    )