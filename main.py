from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, List, Dict, Any
from pydantic import BaseModel
import os
import uuid
import tempfile
from supabase import create_client
from openai import OpenAI
import pypdf
from dotenv import load_dotenv

# Carrega variáveis de ambiente
load_dotenv()

# Configurações
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# Inicializa clientes
openai_client = OpenAI(api_key=OPENAI_API_KEY)
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# Modelos de dados
class Question(BaseModel):
    text: str

class FunnelAnalysisRequest(BaseModel):
    description: str

class EmailRequest(BaseModel):
    offer: str
    audience: str
    objective: str

class DocumentResponse(BaseModel):
    success: bool
    document_count: Optional[int] = None
    chunk_count: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

# Prompts do sistema
SYSTEM_PROMPT = """
Você é o Funnel Mastermind AI, um especialista em funis de vendas, copywriting e marketing digital.
Você foi criado por Glauco, um especialista em funis de vendas que está se posicionando como autoridade em funis perpétuos.
Seja específico, estratégico e utilize os princípios de Brevidade Inteligente: comunicação clara, direta e valiosa.
Use um tom consultivo profissional, direto e preciso, evitando linguagem genérica ou "coachzística".
"""

KNOWLEDGE_PROMPT = """
Você é o Funnel Mastermind AI, um especialista em funis de vendas, copywriting e marketing digital.
Responda à pergunta do usuário usando apenas as informações fornecidas no CONTEXTO abaixo. 
Se a resposta não estiver contida no CONTEXTO, diga que você não tem essa informação específica na sua base de conhecimento.
CONTEXTO:
{context}
PERGUNTA:
{question}
"""

# Inicializa a API
app = FastAPI(
    title="Funnel Mastermind AI API",
    description="Backend API para o assistente de funis de vendas",
    version="2.0.0"
)

# Configuração de CORS para permitir acesso do frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Para desenvolvimento. Em produção, especifique a origem exata
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Funções auxiliares
def extract_text_from_pdf(file_content):
    """Extrai o texto de um arquivo PDF"""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(file_content)
        temp_file_path = temp_file.name
    
    try:
        reader = pypdf.PdfReader(temp_file_path)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        
        return text
    finally:
        os.unlink(temp_file_path)

def split_text(text, chunk_size=1000, overlap=200):
    """Divide o texto em chunks menores"""
    chunks = []
    start = 0
    
    while start < len(text):
        end = min(start + chunk_size, len(text))
        
        # Ajusta o final para não cortar palavras
        if end < len(text):
            # Procura o próximo espaço após o tamanho do chunk
            next_space = text.find(' ', end)
            if next_space != -1:
                end = next_space
        
        chunks.append(text[start:end])
        start = end - overlap  # Sobreposição para manter contexto
    
    return chunks

def create_embedding(text):
    """Cria um embedding para o texto usando OpenAI"""
    response = openai_client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    return response.data[0].embedding

def store_document_chunks(chunks, metadata):
    """Armazena chunks de documento com embeddings no Supabase"""
    chunk_count = 0
    
    for i, chunk in enumerate(chunks):
        # Cria embedding
        embedding = create_embedding(chunk)
        
        # Prepara metadados
        doc_metadata = metadata.copy()
        doc_metadata["chunk_index"] = i
        
        # Armazena no Supabase
        supabase.table("funnel_documents").insert({
            "id": str(uuid.uuid4()),
            "content": chunk,
            "embedding": embedding,
            "metadata": doc_metadata
        }).execute()
        
        chunk_count += 1
    
    return chunk_count

def search_similar_documents(query, limit=5):
    """Busca documentos similares à query"""
    # Cria embedding para a query
    query_embedding = create_embedding(query)
    
    # Busca documentos similares
    result = supabase.rpc(
        "match_documents",
        {"query_embedding": query_embedding, "match_count": limit}
    ).execute()
    
    return result.data if result.data else []

def ask_openai(question, contexts=None):
    """Consulta o modelo OpenAI com ou sem contexto"""
    if contexts and len(contexts) > 0:
        # Prepara o contexto combinado
        context_text = "\n\n---\n\n".join([c["content"] for c in contexts])
        prompt = KNOWLEDGE_PROMPT.format(context=context_text, question=question)
        messages = [{"role": "system", "content": prompt}]
    else:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question}
        ]
    
    # Consulta o modelo
    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        temperature=0.7,
    )
    
    # Extrai e formata a resposta
    answer = response.choices[0].message.content
    
    # Adiciona fontes se existirem
    if contexts and len(contexts) > 0:
        sources = []
        for context in contexts:
            if "metadata" in context and "title" in context["metadata"]:
                source = context["metadata"]["title"]
                if source not in sources:
                    sources.append(source)
        
        if sources:
            answer += "\n\n**Fontes:**\n"
            for source in sources:
                answer += f"- {source}\n"
    
    return answer

# Rotas da API
@app.get("/")
async def root():
    return {"message": "Bem-vindo à API do Funnel Mastermind AI"}

@app.post("/documents/upload", response_model=DocumentResponse)
async def upload_document(
    file: UploadFile = File(...),
    title: str = Form(...),
    author: Optional[str] = Form(None),
    category: Optional[str] = Form(None)
):
    """Processa um documento PDF e adiciona à base de conhecimento"""
    try:
        # Valida formato do arquivo
        if not file.filename.endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Apenas arquivos PDF são aceitos")
        
        # Lê o conteúdo do arquivo
        content = await file.read()
        
        # Extrai texto do PDF
        text = extract_text_from_pdf(content)
        
        if not text or len(text.strip()) == 0:
            raise HTTPException(status_code=400, detail="Não foi possível extrair texto do documento")
        
        # Divide o texto em chunks
        chunks = split_text(text)
        
        # Prepara metadados
        metadata = {
            "title": title,
            "filename": file.filename
        }
        
        if author:
            metadata["author"] = author
        
        if category:
            metadata["category"] = category
        
        # Armazena os chunks no Supabase
        chunk_count = store_document_chunks(chunks, metadata)
        
        return {
            "success": True,
            "document_count": 1,
            "chunk_count": chunk_count,
            "metadata": metadata
        }
    
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

@app.post("/query")
async def query(question: Question):
    """Responde a uma pergunta com base na base de conhecimento"""
    try:
        # Busca documentos relevantes
        similar_docs = search_similar_documents(question.text)
        
        # Obtém resposta do modelo
        response = ask_openai(question.text, similar_docs)
        
        # Formata fontes para resposta
        sources = []
        for doc in similar_docs:
            if "metadata" in doc and doc["metadata"]:
                meta = doc["metadata"]
                if "title" in meta and meta["title"] not in [s.get("title") for s in sources]:
                    sources.append({"title": meta["title"]})
        
        return {
            "answer": response,
            "sources": sources
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze-funnel")
async def analyze_funnel(request: FunnelAnalysisRequest):
    """Analisa um funil de vendas"""
    try:
        # Prepara prompt específico para análise de funil
        prompt = f"Analise o seguinte funil de vendas em detalhes, avaliando estrutura, pontos fortes, oportunidades, e métricas a serem monitoradas:\n\n{request.description}"
        
        # Busca documentos relevantes
        similar_docs = search_similar_documents(prompt)
        
        # Obtém resposta do modelo
        response = ask_openai(prompt, similar_docs)
        
        # Formata fontes para resposta
        sources = []
        for doc in similar_docs:
            if "metadata" in doc and doc["metadata"]:
                meta = doc["metadata"]
                if "title" in meta and meta["title"] not in [s.get("title") for s in sources]:
                    sources.append({"title": meta["title"]})
        
        return {
            "answer": response,
            "sources": sources
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/create-email")
async def create_email(request: EmailRequest):
    """Cria um e-mail com base no framework F4"""
    try:
        # Prepara prompt específico para criação de e-mail
        prompt = f"""
        Crie um e-mail seguindo o framework F4 (Seinfeld + Brevidade Inteligente) com:
        1. Assunto magnetizante que gera curiosidade
        2. Abertura com gancho ou história que prende atenção
        3. Ponte para o conteúdo principal
        4. Conteúdo relevante e com valor prático
        5. Call-to-action claro e persuasivo

        O e-mail deve parecer pessoal, criar conexão, ter elementos de storytelling e seguir o princípio da Brevidade Inteligente.

        Produto/Oferta: {request.offer}
        Público-alvo: {request.audience}
        Objetivo do e-mail: {request.objective}
        """
        
        # Busca documentos relevantes
        similar_docs = search_similar_documents(prompt)
        
        # Obtém resposta do modelo
        response = ask_openai(prompt, similar_docs)
        
        # Formata fontes para resposta
        sources = []
        for doc in similar_docs:
            if "metadata" in doc and doc["metadata"]:
                meta = doc["metadata"]
                if "title" in meta and meta["title"] not in [s.get("title") for s in sources]:
                    sources.append({"title": meta["title"]})
        
        return {
            "answer": response,
            "sources": sources
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/documents")
async def list_documents():
    """Lista todos os documentos únicos na base de conhecimento"""
    try:
        # Busca metadados de todos os documentos
        result = supabase.table("funnel_documents").select("metadata").execute()
        
        if not result.data:
            return {"documents": []}
        
        # Organiza documentos únicos por título
        documents = {}
        for item in result.data:
            if "metadata" in item and "title" in item["metadata"]:
                title = item["metadata"]["title"]
                if title not in documents:
                    documents[title] = item["metadata"]
        
        # Converte para lista
        document_list = [
            {
                "title": metadata.get("title", ""),
                "author": metadata.get("author", ""),
                "category": metadata.get("category", ""),
                "filename": metadata.get("filename", "")
            }
            for metadata in documents.values()
        ]
        
        return {"documents": document_list}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
