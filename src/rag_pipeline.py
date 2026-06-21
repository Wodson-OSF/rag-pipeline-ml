"""
RAG Pipeline - Sistema de Recuperação e Geração Aumentada
Autor: Wodson
Data: 21/06/2026
Versão: 2.0.0 (com embeddings reais)
"""

from sentence_transformers import SentenceTransformer
import hashlib
from typing import Dict, List, Optional

from data_cleaner import DataCleaner


class RAGPipeline:
    def __init__(self) -> None:
        self.cleaner = DataCleaner()
        self.documents: List[Dict] = []
        self.total_docs: int = 0
        # Modelo de embedding real (baixa automaticamente na primeira execução)
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')

    def _get_embedding(self, text: str) -> List[float]:
        """Gera embedding real usando sentence-transformers."""
        return self.embedder.encode(text).tolist()

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calcula similaridade por cosseno entre dois vetores."""
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = sum(a * a for a in vec1) ** 0.5
        norm2 = sum(b * b for b in vec2) ** 0.5
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot_product / (norm1 * norm2)

    def index_document(self, doc_id: str, text: str, metadata: Optional[Dict] = None) -> None:
        """Indexa um documento na base."""
        cleaned_text = self.cleaner.clean_text(text)
        if cleaned_text is None:
            print(f"⚠️ Documento '{doc_id}' ignorado")
            return
        embedding = self._get_embedding(cleaned_text)
        self.documents.append({
            "id": doc_id,
            "text": cleaned_text,
            "embedding": embedding,
            "metadata": metadata or {}
        })
        self.total_docs += 1
        print(f"✅ Documento '{doc_id}' indexado")

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict]:
        """Recupera os documentos mais relevantes."""
        cleaned_query = self.cleaner.clean_text(query)
        if cleaned_query is None:
            return []
        
        # Extrai palavras-chave
        keywords = cleaned_query.split()
        query_embedding = self._get_embedding(cleaned_query)
        
        scores = []
        for idx, doc in enumerate(self.documents):
            # Similaridade por cosseno (embedding real)
            embedding_score = self._cosine_similarity(query_embedding, doc["embedding"])
            
            # Bônus por palavras-chave
            keyword_score = 0
            doc_text = doc["text"].lower()
            for keyword in keywords:
                if keyword in doc_text:
                    keyword_score += 0.15
            
            # Score final (70% embedding, 30% palavras-chave)
            final_score = (embedding_score * 0.7) + (keyword_score * 0.3)
            scores.append((idx, final_score))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        
        results = []
        for idx, score in scores[:top_k]:
            doc = self.documents[idx].copy()
            doc["score"] = score
            results.append(doc)
        
        return results

    def generate_prompt(self, query: str, context_docs: List[Dict]) -> str:
        """Constrói o prompt com contexto."""
        context_parts = []
        for i, doc in enumerate(context_docs, 1):
            context_parts.append(f"Fonte {i}: {doc['text']}")
        context = "\n".join(context_parts)
        return f"""Instrução: Responda à pergunta usando APENAS as informações do CONTEXTO abaixo.

CONTEXTO:
{context}

PERGUNTA: {query}

RESPOSTA:"""

    def generate_response(self, prompt: str) -> str:
        """Gera resposta extraindo o contexto do prompt."""
        try:
            start = prompt.find("CONTEXTO:") + len("CONTEXTO:")
            end = prompt.find("PERGUNTA:")
            context_text = prompt[start:end].strip()
            
            lines = context_text.split('\n')
            context_lines = []
            for line in lines:
                line = line.strip()
                if line and not line.startswith('Instrução'):
                    if 'Fonte' in line and ':' in line:
                        parts = line.split(':', 1)
                        if len(parts) > 1:
                            line = parts[1].strip()
                    if line:
                        context_lines.append(line)
            
            if not context_lines:
                return "Não encontrei essa informação no meu banco de dados."
            
            response = "📌 **Com base nas informações disponíveis:**\n\n"
            for i, line in enumerate(context_lines[:3], 1):
                response += f"**{i}.** {line}\n"
            
            response += "\n✅ _Esta resposta foi gerada com base nos documentos disponíveis._"
            return response
            
        except Exception as e:
            return f"Erro ao processar: {str(e)}"

    def ask(self, query: str) -> Dict:
        """Pipeline completo: query → retrieval → response."""
        print("\n" + "=" * 60)
        print("🔍 Processando consulta...")
        print("=" * 60)
        
        context_docs = self.retrieve(query, top_k=3)
        
        if not context_docs:
            return {
                "query": query,
                "response": "Nenhum documento relevante encontrado.",
                "context": [],
                "error": True
            }
        
        print(f"✅ Encontrados {len(context_docs)} documentos relevantes")
        
        prompt = self.generate_prompt(query, context_docs)
        response = self.generate_response(prompt)
        
        return {
            "query": query,
            "response": response,
            "context": context_docs,
            "prompt": prompt,
            "error": False
        }

    def batch_index(self, documents: List[Dict]) -> None:
        """Indexa múltiplos documentos."""
        print(f"\n📂 Indexando {len(documents)} documentos...")
        print("-" * 50)
        for doc in documents:
            self.index_document(
                doc_id=doc.get("id", f"doc_{self.total_docs + 1}"),
                text=doc.get("text", ""),
                metadata=doc.get("metadata", {})
            )
        print("-" * 50)
        print(f"✅ Total: {self.total_docs} documentos")

    def get_stats(self) -> Dict:
        """Retorna estatísticas do pipeline."""
        return {
            "total_documents": self.total_docs,
            "total_errors": len(self.cleaner.errors),
            "embedding_dimension": 384,  # Dimensão do all-MiniLM-L6-v2
            "embedding_model": "all-MiniLM-L6-v2"
        }


def main():
    """Função principal para demonstração."""
    print("=" * 60)
    print("🚀 RAG PIPELINE - Sistema de Consulta Inteligente")
    print("📐 Usando embeddings reais: all-MiniLM-L6-v2")
    print("=" * 60)
    
    pipeline = RAGPipeline()
    
    # Documentos de exemplo
    sample_docs = [
        {"id": "doc1", "text": "Resetar senha: Acesse 'Esqueci minha senha' na tela de login do aplicativo."},
        {"id": "doc2", "text": "Limpar cache: Vá em Configurações > Armazenamento > Limpar cache para melhorar o desempenho."},
        {"id": "doc3", "text": "Emitir segunda via do boleto: Acesse o portal do cliente, seção 'Boletos', clique em 'Emitir segunda via'."},
        {"id": "doc4", "text": "Alterar dados cadastrais: Acesse 'Meu Perfil' no aplicativo e clique em 'Editar dados'."},
        {"id": "doc5", "text": "Recuperar acesso ao sistema: Caso não consiga resetar a senha, entre em contato com o suporte via chat."},
    ]
    
    pipeline.batch_index(sample_docs)
    
    queries = [
        "Como resetar minha senha?",
        "Como emitir um boleto?",
        "Perdi acesso ao sistema, e agora?",
        "Como limpar o cache?",
    ]
    
    for query in queries:
        print("\n" + "-" * 50)
        result = pipeline.ask(query)
        print(f"\n📝 Pergunta: {result['query']}")
        print(f"\n💬 Resposta:\n{result['response']}")
        
        if result.get('context'):
            print(f"\n📚 Documentos utilizados (relevância):")
            for i, doc in enumerate(result['context'], 1):
                score = doc.get('score', 0)
                print(f"   {i}. {doc['text'][:50]}... ({score:.2%})")
    
    print("\n" + "=" * 60)
    print("📊 ESTATÍSTICAS")
    print("=" * 60)
    stats = pipeline.get_stats()
    print(f"📝 Documentos: {stats['total_documents']}")
    print(f"⚠️  Erros: {stats['total_errors']}")
    print(f"📐 Dimensão embedding: {stats['embedding_dimension']}")
    print(f"🔤 Modelo: {stats['embedding_model']}")


if __name__ == "__main__":
    main()