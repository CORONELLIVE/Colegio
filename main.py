import os
import sys
import re
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import edge_tts
import uvicorn
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from openai import OpenAI
from pydantic import BaseModel, Field

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))
from cursos import CATALOGO_CURSOS

def cargar_env():
    env_file = BASE / ".env"
    if env_file.exists():
        try:
            with open(env_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        k = k.strip()
                        v = v.strip().strip("'\"")
                        if k and k not in os.environ:
                            os.environ[k] = v
        except Exception:
            pass

cargar_env()

API_KEY = os.getenv("GROQ_API_KEY")
MODELOS = [m.strip() for m in os.getenv("MODELOS_LLM", "openai/gpt-oss-20b,openai/gpt-oss-120b").split(",")]
STT_MODEL = "whisper-large-v3-turbo"
VOZ = "es-PE-CamilaNeural"
MAX_INTENTOS = 3

client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=API_KEY, timeout=30)
app = FastAPI(title="Aula Clara")
_CACHE_CLASES: Dict[str, List[Dict[str, str]]] = {}

def json_defensivo(texto: str) -> Optional[Dict[str, Any]]:
    limpio = re.sub(r"```(?:json)?", "", texto or "").strip()
    m = re.search(r"\{.*\}", limpio, re.DOTALL)
    for candidato in (limpio, m.group(0) if m else ""):
        try:
            return json.loads(candidato)
        except Exception:
            pass
    return None

def llm_json(mensajes, max_tokens, temperature, esfuerzo="low", timeout=30):
    for modelo in MODELOS:
        try:
            r = client.with_options(timeout=timeout).chat.completions.create(
                model=modelo, messages=mensajes, response_format={"type": "json_object"},
                temperature=temperature, max_tokens=max_tokens,
                extra_body={"reasoning_effort": esfuerzo}
            )
            datos = json_defensivo(r.choices[0].message.content or "")
            if datos:
                return datos
        except Exception:
            pass
    return None

def paginas_validas(lista) -> List[Dict[str, str]]:
    out = []
    if isinstance(lista, list):
        for p in lista:
            if isinstance(p, dict) and isinstance(p.get("texto"), str) and p["texto"].strip():
                out.append({"titulo": str(p.get("titulo", "")).strip()[:80], "texto": p["texto"].strip()})
    return out

class Turno(BaseModel):
    emisor: str = Field(max_length=40)
    texto: str = Field(max_length=2500)

class EvalReq(BaseModel):
    curso_nombre: str = Field(max_length=80)
    tema_titulo: str = Field(max_length=160)
    contenido: str = Field(max_length=8000)
    pregunta_actual: str = Field(max_length=400)
    clave: str = Field(max_length=400)
    pista: str = Field("", max_length=400)
    intentos: int = Field(0, ge=0, le=10)
    historial: List[Turno] = Field(default_factory=list, max_length=20)

class NuevoTema(BaseModel):
    curso_nombre: str = Field(max_length=80)
    tema_solicitado: str = Field("", max_length=120)

class Ampliar(BaseModel):
    curso_nombre: str = Field(max_length=80)
    titulo: str = Field(max_length=160)
    contenido: str = Field(max_length=4000)

class LibreReq(BaseModel):
    historial: List[Turno] = Field(min_length=1, max_length=20)

class BienestarReq(BaseModel):
    historial: List[Turno] = Field(min_length=1, max_length=20)

class CuestionarioReq(BaseModel):
    curso_nombre: str = Field(max_length=80)
    tema_titulo: str = Field(max_length=160)
    contenido: str = Field(max_length=8000)

class ErrorUnicoReq(BaseModel):
    curso_nombre: str = Field(max_length=80)
    tema_titulo: str = Field(max_length=160)
    pregunta: str = Field(max_length=400)
    respuesta_dada: str = Field(max_length=400)
    respuesta_correcta: str = Field(max_length=400)
    explicacion: str = Field(max_length=800)

@app.get("/api/cursos")
def cursos() -> Dict[str, Any]:
    return CATALOGO_CURSOS

@app.post("/api/generar-tema")
def generar_tema(s: NuevoTema) -> Dict[str, Any]:
    enfoque = s.tema_solicitado or "un tema trascendente del Currículo Nacional de Secundaria del Perú"
    prompt = f"""Genera una clase completa para secundaria en Perú, curso '{s.curso_nombre}', tema: '{enfoque}'.
Devuelve SOLO JSON con: titulo, area, resumen (1 oración),
paginas: lista de 5 objetos {{"titulo": título corto, "texto": 110-150 palabras}} (se lee como un libro),
preguntas: lista de 3 objetos {{"pregunta","clave","pista"}}. Usa formato LaTeX delimitado por $$ $$ para fórmulas matemáticas si el curso lo requiere.
Usa solo datos que conozcas con certeza."""
    d = llm_json(
        [{"role": "system", "content": "Eres redactor curricular del Minedu. Responde solo JSON válido."},
         {"role": "user", "content": prompt}],
        max_tokens=5000, temperature=0.6, timeout=90)
    paginas = paginas_validas(d.get("paginas")) if d else []
    preguntas = d.get("preguntas") if d else None
    if not (paginas and isinstance(preguntas, list) and preguntas):
        raise HTTPException(502, "No se pudo generar la lección")
    d["paginas"] = paginas
    d["contenido"] = "\n\n".join(p["texto"] for p in paginas)
    d["preguntas"] = [
        {"numero": i + 1, "pregunta": str(p.get("pregunta", "")), "clave": str(p.get("clave", "")),
         "pista": str(p.get("pista", ""))}
        for i, p in enumerate(preguntas) if isinstance(p, dict) and p.get("pregunta")
    ]
    d["id"] = "gen_" + re.sub(r"\W+", "_", str(d.get("titulo", "tema")).lower())[:40]
    d.setdefault("area", s.curso_nombre)
    d.setdefault("resumen", "")
    return d

@app.post("/api/ampliar-tema")
def ampliar_tema(a: Ampliar) -> Dict[str, Any]:
    clave = f"{a.curso_nombre}|{a.titulo}"
    if clave in _CACHE_CLASES:
        return {"paginas": _CACHE_CLASES[clave]}
    prompt = f"""Curso '{a.curso_nombre}'. Lección: '{a.titulo}'.
Texto base: "{a.contenido}"
Convierte esto en una clase completa para secundaria, escrita como un libro de 5 páginas. 
Si hay matemáticas usa delimitadores LaTeX $$ $$. Conserva TODOS los hechos.
Devuelve SOLO JSON: {{"paginas":[{{"titulo":"...","texto":"..."}}]}}"""
    d = llm_json(
        [{"role": "system", "content": "Eres docente redactora del Minedu. Responde solo JSON válido."},
         {"role": "user", "content": prompt}],
        max_tokens=4500, temperature=0.5, timeout=90)
    paginas = paginas_validas(d.get("paginas")) if d else []
    if len(paginas) < 2:
        raise HTTPException(502, "No se pudo ampliar la clase")
    _CACHE_CLASES[clave] = paginas
    return {"paginas": paginas}

@app.get("/api/tts")
async def tts(texto: str = Query(..., min_length=1, max_length=1800)) -> StreamingResponse:
    async def audio():
        async for c in edge_tts.Communicate(texto, VOZ, rate="-3%").stream():
            if c["type"] == "audio":
                yield c["data"]
    return StreamingResponse(audio(), media_type="audio/mpeg")

@app.post("/api/stt")
def stt(file: UploadFile = File(...)) -> Dict[str, str]:
    datos = file.file.read(6_000_000)
    if not datos:
        return {"texto": ""}
    try:
        r = client.audio.transcriptions.create(
            model=STT_MODEL,
            file=(file.filename or "voz.webm", datos),
            language="es",
            temperature=0.0
        )
        return {"texto": r.text.strip()}
    except Exception:
        return {"texto": ""}

INTENCIONES = {"responde", "no_sabe", "pide_explicacion", "confirma", "otra_cosa", "fuera_de_tema"}

@app.post("/api/evaluar")
def evaluar(d: EvalReq) -> Dict[str, Any]:
    dialogo = "\n".join(f"{t.emisor}: {t.texto}" for t in d.historial)
    ultimo = d.intentos + 1 >= MAX_INTENTOS
    extra = ("ESTE ES EL ÚLTIMO INTENTO: si la intención es responde o no_sabe y no acertó, REVELA la respuesta "
             "esperada y explícala con calidez en 3-4 oraciones; no vuelvas a preguntar, pasarán a la siguiente."
             if ultimo else "")
    system = f"""Eres la Profesora Clara, docente de secundaria en Perú, curso {d.curso_nombre}.
Tema: '{d.tema_titulo}'. Contenido de la clase: "{d.contenido}"
Pregunta actual: "{d.pregunta_actual}"
Respuesta esperada: "{d.clave}"
Pista base: "{d.pista}"
Intentos fallidos previos en esta pregunta: {d.intentos}. {extra}

PASO 1 - Clasifica el ÚLTIMO mensaje del estudiante en "intencion":
- responde: intenta contestar la pregunta.
- no_sabe: dice que no sabe.
- pide_explicacion: pide que le expliques mejor.
- confirma: contesta afirmativamente a algo.
- otra_cosa: duda o comentario relacionado con la clase, distinto a la pregunta.
- fuera_de_tema: nada que ver con la clase o inapropiado.

PASO 2 - Responde según la intención:
- responde: evalúa si acerto (true/false). Si el curso es de matemáticas o requiere procedimientos, evalúa el paso a paso algebraico o lógico basándote en la respuesta del alumno; no exijas la respuesta final de inmediato, valida si el procedimiento parcial que propuso es correcto y anímalo a seguir. Si no acertó, da una pista gradual SIN dar la respuesta literal.
- no_sabe / pide_explicacion / otra_cosa: EXPLICA con ejemplos sencillos sin regalar la respuesta exacta; termina con una pregunta guiada. Si hay operaciones matemáticas escribe las fórmulas con $$ $$.
- fuera_de_tema: redirige a la clase con amabilidad.
Nunca digas "incorrecto". Usa "muy bien" SOLO si acerto=true. Máx. 900 caracteres.
Responde solo JSON: {{"intencion": "...", "acerto": true|false, "devolucion": "..."}}"""
    r = llm_json(
        [{"role": "system", "content": system},
         {"role": "user", "content": f"<dialogo>\n{dialogo}\n</dialogo>\nEvalúa el último mensaje del estudiante."}],
        max_tokens=1500, temperature=0.3, esfuerzo="medium")
    if not (r and str(r.get("devolucion", "")).strip()):
        return {"acerto": False, "avanzar": False, "intento_fallido": False, "intencion": "responde",
                "texto_profesora": f"Te escuché con atención. Una pista: {d.pista} ¿Lo intentas otra vez?"}
    intencion = r.get("intencion") if r.get("intencion") in INTENCIONES else "responde"
    acerto = bool(r.get("acerto")) and intencion == "responde"
    fallido = intencion in ("responde", "no_sabe") and not acerto
    return {"acerto": acerto, "intencion": intencion, "intento_fallido": fallido,
            "avanzar": acerto or (fallido and ultimo),
            "texto_profesora": str(r["devolucion"]).strip()}

@app.post("/api/generar-cuestionario")
def generar_cuestionario(c: CuestionarioReq) -> Dict[str, Any]:
    prompt = f"""Genera un cuestionario de evaluación de 3 a 4 preguntas de opción múltiple para secundaria en Perú.
Curso: '{c.curso_nombre}'. Tema: '{c.tema_titulo}'.
Contenido base: "{c.contenido}"

Devuelve SOLO JSON con el siguiente formato exacto:
{{
  "preguntas": [
    {{
      "id": 1,
      "pregunta": "¿Texto claro de la pregunta?",
      "opciones": ["Opción A", "Opción B", "Opción C", "Opción D"],
      "correcta": 0,
      "concepto": "Concepto o competencia evaluada",
      "explicacion": "Explicación breve de por qué es la correcta en 1-2 oraciones."
    }}
  ]
}}
Nota: 'correcta' debe ser un número entero de 0 a 3 (0 para la primera opción, 1 para la segunda, etc.). Usa fórmulas LaTeX con $$ $$ si aplica."""
    d = llm_json(
        [{"role": "system", "content": "Eres un especialista en evaluación formativa del Minedu. Responde solo JSON válido."},
         {"role": "user", "content": prompt}],
        max_tokens=3000, temperature=0.4, timeout=45)
    
    preguntas = d.get("preguntas") if d and isinstance(d.get("preguntas"), list) else []
    validas = []
    for i, p in enumerate(preguntas):
        if isinstance(p, dict) and p.get("pregunta") and isinstance(p.get("opciones"), list) and len(p["opciones"]) >= 2:
            correcta = p.get("correcta", 0)
            if not isinstance(correcta, int) or correcta < 0 or correcta >= len(p["opciones"]):
                correcta = 0
            validas.append({
                "id": i + 1,
                "pregunta": str(p["pregunta"]).strip(),
                "opciones": [str(op).strip() for op in p["opciones"][:4]],
                "correcta": correcta,
                "concepto": str(p.get("concepto", "Comprensión del tema")).strip(),
                "explicacion": str(p.get("explicacion", "")).strip()
            })
    if not validas:
        raise HTTPException(502, "No se pudo generar el cuestionario")
    return {"preguntas": validas}

@app.post("/api/explicar-error")
def explicar_error(e: ErrorUnicoReq) -> Dict[str, Any]:
    system = f"""Eres la Profesora Clara, docente empática de secundaria en Perú.
Curso: '{e.curso_nombre}'. Tema: '{e.tema_titulo}'.
El estudiante se equivocó en esta pregunta: "{e.pregunta}"
Eligió: "{e.respuesta_dada}". La correcta es: "{e.respuesta_correcta}".
Explicación técnica del concepto: {e.explicacion}

TU TAREA:
Explica con calidez y paciencia por qué su respuesta es incorrecta y enséñale el concepto correcto de forma sencilla para que pueda volver a intentarlo.
Máximo 4 oraciones (800 caracteres). Termina animándolo a intentarlo de nuevo. Usa $$ $$ para matemáticas.
Responde SOLO JSON: {{"texto_profesora": "tu explicación aquí"}}"""

    d = llm_json(
        [{"role": "system", "content": system},
         {"role": "user", "content": "Genera la explicación pedagógica en base al error."}],
        max_tokens=800, temperature=0.5, timeout=40)

    if d and str(d.get("texto_profesora", "")).strip():
        return {"texto_profesora": str(d["texto_profesora"]).strip()}
    return {"texto_profesora": "Revisa bien los conceptos de la lectura. ¡Inténtalo de nuevo, tú puedes!"}

@app.post("/api/libre")
def libre(d: LibreReq) -> Dict[str, Any]:
    dialogo = "\n".join(f"{t.emisor}: {t.texto}" for t in d.historial)
    system = """Eres la Profesora Clara, docente virtual de un colegio de secundaria en Perú.
En MODO LIBRE el estudiante puede preguntarte de cualquier tema EDUCATIVO.
Reglas:
1. Solo contenido educativo y apto para menores.
2. Explica en 4-7 oraciones, claro y cálido. Si hay operaciones matemáticas escribe las fórmulas con $$ $$.
3. Termina con una pregunta corta.
4. Si el estudiante menciona violencia, acoso o riesgo para su salud, responde con empatía y anímalo a hablar con un adulto o psicóloga del colegio y líneas 100 y 113.
Responde solo JSON: {"educativo": true|false, "respuesta": "...", "sugerencias": ["3 preguntas de máx. 8 palabras"]}"""
    r = llm_json(
        [{"role": "system", "content": system},
         {"role": "user", "content": f"<dialogo>\n{dialogo}\n</dialogo>\nResponde al último mensaje."}],
        max_tokens=1500, temperature=0.5)
    if r and str(r.get("respuesta", "")).strip():
        sug = [str(s)[:70] for s in r.get("sugerencias", []) if isinstance(s, str)][:3]
        return {"texto_profesora": str(r["respuesta"]).strip()[:1700], "sugerencias": sug,
                "educativo": bool(r.get("educativo", True))}
    return {"texto_profesora": "No pude responder ahora mismo. ¿Me lo preguntas otra vez?",
            "sugerencias": [], "educativo": True}

@app.post("/api/bienestar")
def bienestar(d: BienestarReq) -> Dict[str, Any]:
    dialogo = "\n".join(f"{t.emisor}: {t.texto}" for t in d.historial)
    system = """Eres Clara, orientadora escolar empática y amable en un colegio de secundaria en Perú.
Tu objetivo en este Espacio Seguro es brindar escucha activa, validación emocional y contención.
Reglas:
1. Valida los sentimientos del estudiante sin juzgar.
2. No ofrezcas diagnósticos clínicos ni consejos médicos. Eres un apoyo emocional primario, no terapeuta clínica.
3. Usa un tono extremadamente cálido, calmado, paciente y cercano en 3 a 5 oraciones.
4. Si detectas ideación suicida, abuso, violencia o riesgo grave inminente, responde con extrema empatía, sugiérele hablar urgentemente con un adulto de confianza, tutor o padres, y menciona las líneas de apoyo gratuito 100 y 113 de Perú de forma calmada.
5. Finaliza siempre tu intervención con una pregunta abierta y suave para invitar a seguir conversando, demostrando interés genuino.
Responde solo JSON válido: {"respuesta": "tu respuesta aquí"}"""
    r = llm_json(
        [{"role": "system", "content": system},
         {"role": "user", "content": f"<dialogo>\n{dialogo}\n</dialogo>\nResponde al último mensaje del estudiante priorizando la empatía."}],
        max_tokens=1000, temperature=0.6)
    if r and str(r.get("respuesta", "")).strip():
        return {"texto_profesora": str(r["respuesta"]).strip()[:1700]}
    return {"texto_profesora": "Estoy aquí para escucharte. ¿Puedes contarme un poco más?"}

@app.get("/")
def home() -> FileResponse:
    os.makedirs("static", exist_ok=True)
    return FileResponse(BASE / "static" / "index.html")

app.mount("/static", StaticFiles(directory=BASE / "static"), name="static")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")
