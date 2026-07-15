from flask import Flask, render_template, request, jsonify
import re
import os
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.data.path.append('/tmp/nltk_data')
try:
    nltk.download('punkt', download_dir='/tmp/nltk_data', quiet=True)
    nltk.download('stopwords', download_dir='/tmp/nltk_data', quiet=True)
    nltk.download('wordnet', download_dir='/tmp/nltk_data', quiet=True)
except:
    pass

app = Flask(__name__)

base_conocimiento = [
    {"pregunta": "hola", "respuesta": "¡Hola! Soy INDI 😊 Estoy para explicarte temas y hacerte practicar. ¿Qué quieres aprender hoy?", "accion": None},
    {"pregunta": "buenos dias", "respuesta": "¡Buenos días! ¿En qué te ayudo hoy?", "accion": None},
    {"pregunta": "buenas tardes", "respuesta": "¡Buenas tardes! Cuéntame qué tema te interesa.", "accion": None},
    {"pregunta": "buenas noches", "respuesta": "¡Buenas noches! Aquí estoy para ayudarte.", "accion": None},
    {"pregunta": "quien eres", "respuesta": "Soy INDI, tu asistente de estudios. Te explico temas y te hago cuestionarios cuando tú quieras.", "accion": None},
    {"pregunta": "adios", "respuesta": "¡Hasta luego! Vuelve cuando quieras seguir aprendiendo 😊", "accion": None},

    {"pregunta": "matemáticas", "respuesta": """📚 **Matemáticas**
Estudia cantidades, formas y cambios. Tiene aritmética, geometría, álgebra y estadística.
📚 Fuentes:
🔗 https://es.khanacademy.org/math
🔗 https://www.matematicas18.com/
🔗 https://www.universoformulas.com/
🔗 https://www.educaplay.com/matematicas/
¿Quieres el cuestionario? Di: hazme el cuestionario""",
    "cuestionario": """📋 Cuestionario:
1. ¿Qué estudian las matemáticas?
2. ¿Qué rama estudia las figuras?
3. ¿Cuánto es 8×7?
4. ¿Qué es un número primo?
5. ¿Cuántos lados tiene un cuadrado?""",
    "respuestas": """✅ Respuestas:
1. Cantidades, estructuras, espacio y cambios.
2. Geometría.
3. 56.
4. Solo se divide entre 1 y él mismo.
5. 4 lados."""},

    {"pregunta": "ciencias", "respuesta": """📚 **Ciencias**
Estudia la naturaleza, seres vivos y fenómenos. Hay naturales, sociales y formales.
📚 Fuentes:
🔗 https://es.khanacademy.org/science
🔗 https://www.nationalgeographicla.com/ciencia
🔗 https://www.cienciafacil.com/
🔗 https://www.educ.ar/recursos/ciencias-naturales
¿Quieres practicar? Pídeme el cuestionario""",
    "cuestionario": """📋 Cuestionario:
1. ¿Qué es la fotosíntesis?
2. ¿En qué planeta vivimos?
3. ¿Qué gas sueltan las plantas?
4. ¿Qué bombea la sangre?
5. ¿De qué está hecho el agua?""",
    "respuestas": """✅ Respuestas:
1. Las plantas fabrican su alimento con luz.
2. Tierra.
3. Oxígeno.
4. El corazón.
5. Hidrógeno y oxígeno."""},

    {"pregunta": "historia del Perú", "respuesta": """📚 **Historia del Perú**
Etapas: Prehispánica, Colonial y Republicana.
📚 Fuentes:
🔗 https://historiaperuana.pe/
🔗 https://www.cultura.gob.pe/historia-del-peru
🔗 https://www.universoperu.com/historia
🔗 https://www.bnp.gob.pe/historia-del-peru/
¿Quieres el cuestionario?""",
    "cuestionario": """📋 Cuestionario:
1. ¿Civilización más antigua?
2. ¿Año de independencia?
3. ¿Quién la proclamó en Lima?
4. ¿Nombre del imperio inca?
5. ¿Quién fue San Martín?""",
    "respuestas": """✅ Respuestas:
1. Caral.
2. 1821.
3. José de San Martín.
4. Tahuantinsuyo.
5. El libertador."""},

    {"pregunta": "fracciones", "respuesta": """📚 **Fracciones**
Representan partes de un todo: numerador / denominador.
📚 Fuentes:
🔗 https://www.universoformulas.com/matematicas/aritmetica/fracciones/
🔗 https://es.khanacademy.org/math/arithmetic/fraction-arithmetic
🔗 https://www.matematicas18.com/fracciones/
🔗 https://www.educaplay.com/fracciones/
¿Quieres practicar?""",
    "cuestionario": """📋 Cuestionario:
1. ¿Qué es una fracción?
2. ¿Qué es el numerador?
3. ¿Qué es el denominador?
4. ¿Cuánto es 1/2 + 1/2?
5. ¿Cuánto es la mitad?""",
    "respuestas": """✅ Respuestas:
1. Parte de un todo.
2. Parte que tomamos.
3. Partes en que se divide.
4. 1 entero.
5. 1/2."""},

    {"pregunta": "fotosíntesis", "respuesta": """📚 **Fotosíntesis**
Las plantas usan luz, agua y CO₂ para hacer su alimento y liberar oxígeno.
📚 Fuentes:
🔗 https://www.nationalgeographicla.com/ciencia/que-es-la-fotosintesis
🔗 https://es.khanacademy.org/science/ap-biology/cellular-energetics/photosynthesis
🔗 https://www.cienciafacil.com/fotosintesis.html
🔗 https://www.botanica-uno.com/la-fotosintesis/
¿Cuestionario?""",
    "cuestionario": """📋 Cuestionario:
1. ¿Qué necesitan las plantas?
2. ¿Qué liberan?
3. ¿Dónde ocurre?
4. ¿Qué gas usan?
5. ¿Por qué es importante?""",
    "respuestas": """✅ Respuestas:
1. Luz, agua, dióxido de carbono.
2. Oxígeno.
3. En las hojas.
4. Dióxido de carbono.
5. Nos da aire para respirar."""},

    {"pregunta": "culturas preincas", "respuesta": """📚 **Culturas Preincas**
Antes de los incas: Caral, Chavín, Nazca, Moche, Chimú...
📚 Fuentes:
🔗 https://www.universoperu.com/culturas-preincas
🔗 https://www.cultura.gob.pe/culturas-prehispanicas
🔗 https://historiaperuana.pe/culturas-preincas/
🔗 https://www.museonacional.pe/culturas-preincas
¿Cuestionario?""",
    "cuestionario": """📋 Cuestionario:
1. ¿La más antigua?
2. ¿Las líneas de...?
3. ¿Capital de los Chimú?
4. ¿Cultura de los felinos?
5. ¿Dónde estaba Chavín?""",
    "respuestas": """✅ Respuestas:
1. Caral.
2. Nazca.
3. Chan Chan.
4. Chavín.
5. En los Andes centrales."""},

    {"pregunta": "imperio inca", "respuesta": """📚 **Imperio Inca / Tahuantinsuyo**
Capital Cusco, máximo el Inca, usaban quipus.
📚 Fuentes:
🔗 https://www.machupicchu.gob.pe/historia/
🔗 https://www.cultura.gob.pe/imperio-inca
🔗 https://historiaperuana.pe/imperio-inca/
🔗 https://www.incasdelperu.com/
¿Cuestionario?""",
    "cuestionario": """📋 Cuestionario:
1. ¿Qué significa Tahuantinsuyo?
2. ¿Capital?
3. ¿Máxima autoridad?
4. ¿Cómo contaban?
5. ¿Año de llegada españoles?""",
    "respuestas": """✅ Respuestas:
1. Las 4 regiones unidas.
2. Cusco.
3. El Inca.
4. Con quipus.
5. 1532."""},

    {"pregunta": "cuestionario", "respuesta": "Aquí va el cuestionario del último tema que viste. ¿Quieres las respuestas después?", "accion": "mostrar_cuestionario"},
    {"pregunta": "hazme el cuestionario", "respuesta": "¡Vamos a practicar! Aquí tienes:", "accion": "mostrar_cuestionario"},
    {"pregunta": "ponme el cuestionario", "respuesta": "Claro, aquí está:", "accion": "mostrar_cuestionario"},
    {"pregunta": "respuestas", "respuesta": "Aquí están las respuestas correctas:", "accion": "mostrar_respuestas"},
    {"pregunta": "dame las respuestas", "respuesta": "Claro, aquí van:", "accion": "mostrar_respuestas"},
    {"pregunta": "ver respuestas", "respuesta": "Aquí tienes:", "accion": "mostrar_respuestas"},
    {"pregunta": "si", "respuesta": "¡Perfecto! Aquí tienes las respuestas:", "accion": "mostrar_respuestas"},
    {"pregunta": "sí", "respuesta": "¡Perfecto! Aquí tienes las respuestas:", "accion": "mostrar_respuestas"},
    {"pregunta": "ya termine", "respuesta": "Muy bien, aquí van las respuestas:", "accion": "mostrar_respuestas"},
    {"pregunta": "ya terminé", "respuesta": "Muy bien, aquí van las respuestas:", "accion": "mostrar_respuestas"},
    {"pregunta": "corrigeme", "respuesta": "Claro, aquí están las respuestas:", "accion": "mostrar_respuestas"}
]

lemmatizador = WordNetLemmatizer()
stop_words = set(stopwords.words('spanish'))

def limpiar_texto(texto):
    texto = texto.lower()
    texto = re.sub(r'[^a-záéíóúñ\s]', '', texto)
    palabras = texto.split()
    palabras = [lemmatizador.lemmatize(p) for p in palabras if p not in stop_words]
    return ' '.join(palabras)

ultimo_tema = None

def buscar_respuesta(pregunta_usuario):
    global ultimo_tema
    pregunta_limpia = limpiar_texto(pregunta_usuario)

    variantes = {
        "mate": "matemáticas", "mates": "matemáticas",
        "ciencia": "ciencias", "bio": "ciencias",
        "historia peru": "historia del Perú", "peru": "historia del Perú",
        "incas": "imperio inca", "tahuantinsuyo": "imperio inca",
        "culturas antiguas": "culturas preincas",
        "plantas": "fotosíntesis",
        "parte de un todo": "fracciones",
        "quiero el cuestionario": "hazme el cuestionario",
        "quiero respuestas": "dame las respuestas"
    }
    for var, orig in variantes.items():
        if var in pregunta_limpia:
            pregunta_limpia = pregunta_limpia.replace(var, orig).strip()

    preguntas_limpias = [limpiar_texto(i["pregunta"]) for i in base_conocimiento]
    vec = TfidfVectorizer()
    matriz = vec.fit_transform(preguntas_limpias)
    vec_preg = vec.transform([pregunta_limpia])
    sim = cosine_similarity(vec_preg, matriz)[0]
    idx = sim.argmax()

    if sim[idx] > 0.05 and "accion" in base_conocimiento[idx]:
        acc = base_conocimiento[idx]["accion"]
        txt = base_conocimiento[idx]["respuesta"]
        if acc == "mostrar_cuestionario":
            if ultimo_tema and ultimo_tema.get("cuestionario"):
                return f"{txt}\n\n{ultimo_tema['cuestionario']}"
            return "Primero pregúntame un tema y luego te hago el cuestionario 😊"
        if acc == "mostrar_respuestas":
            if ultimo_tema and ultimo_tema.get("respuestas"):
                return f"{txt}\n\n{ultimo_tema['respuestas']}"
            return "Primero haz el cuestionario de un tema 😊"

    if sim[idx] > 0.05:
        ultimo_tema = base_conocimiento[idx]
        return ultimo_tema["respuesta"]

    return "No te entendí bien. Pregúntame sobre matemáticas, ciencias, historia o temas del Perú 😊"

@app.route('/')
def inicio():
    return render_template('index.html')

@app.route('/consultar', methods=['POST'])
def consultar():
    datos = request.get_json()
    return jsonify({"respuesta": buscar_respuesta(datos.get("pregunta", ""))})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
