import google.generativeai as genai
from neo4j import GraphDatabase
import os

# --- CONFIGURACIÓN ---
# Pega aquí tu clave de API de Google AI Studio
# Es más seguro guardarla como una variable de entorno
GEMINI_API_KEY = "AIzaSyDrH5l2UmRn7zJudvxrHsNRyuIRs4DY-kc" 

NEO4J_URI = "bolt://127.0.0.1:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "root1234"
NEO4J_DATABASE = "neo4j"

class RAG_Neo4j_Gemini:
    def __init__(self, gemini_key, neo4j_uri, neo4j_user, neo4j_pass, neo4j_db):
        # Configurar Gemini
        genai.configure(api_key=gemini_key)
        self.generation_model = genai.GenerativeModel('gemini-flash-lite-latest')
        self.embedding_model = 'text-embedding-004'

        # Configurar Neo4j
        self.driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_pass))
        self.database = neo4j_db

    def close(self):
        self.driver.close()

    def embed_text(self, text):
        """Convierte un texto en un vector usando el modelo de embedding."""
        result = genai.embed_content(model=self.embedding_model, content=text)
        return result['embedding']

    def semantic_search(self, query_text, limit=1):
        """Realiza una búsqueda vectorial en Neo4j."""
        query_embedding = self.embed_text(query_text)
        
        # Esta es la consulta Cypher para la búsqueda vectorial
        cypher_query = """
        CALL db.index.vector.queryNodes('chunk_embeddings', $limit, $embedding) YIELD node AS chunk, score
        RETURN chunk.text AS text, score
        """
        
        with self.driver.session(database=self.database) as session:
            result = session.run(cypher_query, limit=limit, embedding=query_embedding)
            # Devolvemos el texto del chunk con el mejor puntaje
            best_result = result.single()
            return best_result['text'] if best_result else None

    def get_source_for_chunk(self, chunk_text):
        """Encuentra el artículo de origen para un chunk de texto dado."""
        # Esta consulta encuentra el chunk por su texto y viaja "hacia atrás"
        # por la relación HAS_CHUNK para encontrar el artículo.
        cypher_query = """
        MATCH (c:Chunk {text: $chunk_text})<-[:HAS_CHUNK]-(a:Article)
        RETURN a.title AS title, a.link AS link
        """
        with self.driver.session(database=self.database) as session:
            result = session.run(cypher_query, chunk_text=chunk_text).single()
            return result

    def generate_response(self, question, context, source):
        """Genera una respuesta usando Gemini, con el contexto y la fuente."""
        prompt = f"""
        Eres un asistente experto en biociencia espacial de la NASA. Tu tarea es responder a la pregunta del usuario basándote ÚNICA Y EXCLUSIVAMENTE en el siguiente contexto extraído de un artículo científico.

        Contexto del artículo:
        ---
        {context}
        ---

        Pregunta del usuario:
        "{question}"

        Instrucciones de respuesta:
        1. Responde a la pregunta de forma clara y concisa.
        2. Basa tu respuesta solo en la información del contexto proporcionado. No añadas conocimiento externo.
        3. Al final de tu respuesta, cita tu fuente de la siguiente manera:
           "Fuente: {source['title']}. Disponible en: {source['link']}"
        """
        
        response = self.generation_model.generate_content(prompt)
        return response.text

    def ask(self, question):
        """Orquesta el proceso completo de RAG."""
        print(f"Pregunta recibida: '{question}'")
        
        print("1. Buscando el contexto más relevante en Neo4j...")
        context_chunk = self.semantic_search(question)
        
        if not context_chunk:
            return "Lo siento, no pude encontrar información relevante en la base de datos para responder a tu pregunta."

        print("2. Rastreando la fuente del artículo...")
        source = self.get_source_for_chunk(context_chunk)
        
        if not source:
             return "Encontré información relevante, pero no pude rastrear el artículo de origen."

        print("3. Generando respuesta con Gemini...")
        answer = self.generate_response(question, context_chunk, source)
        
        return answer


# --- EJECUCIÓN DEL EJEMPLO ---
if __name__ == "__main__":
    rag_system = RAG_Neo4j_Gemini(GEMINI_API_KEY, NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, NEO4J_DATABASE)
    
    # Puedes cambiar la pregunta aquí
    pregunta_ejemplo = "¿Qué efectos tiene la microgravedad en los ratones?"

    try:
        respuesta_final = rag_system.ask(pregunta_ejemplo)
        print("\n--- Respuesta de Gemini ---\n")
        print(respuesta_final)
    except Exception as e:
        print(f"\nOcurrió un error: {e}")
    finally:
        rag_system.close()
        print("\nConexión con Neo4j cerrada.")