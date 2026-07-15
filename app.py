from flask import Flask, render_template, request, jsonify
import re
import os
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.data.path.append('/tmp/nltk_data')
nltk.download('punkt', download_dir='/tmp/nltk_data')
nltk.download('stopwords', download_dir='/tmp/nltk_data')
nltk.download('wordnet', download_dir='/tmp/nltk_data')

app = Flask(__name__)

# ==================== BASE DE CONOCIMIENTO AMPLIADA ====================
base_conocimiento = [
    # Saludos y conversación general
    {"pregunta": "hola", "respuesta": "¡Hola! Soy **INDI**, tu asistente de estudios personalizado 😊 Estoy aquí para explicarte cualquier tema con detalle, responder tus dudas y ayudarte a aprender mejor. ¿Sobre qué te gustaría conversar hoy?", "cuestionario": None},
    {"pregunta": "buenos dias", "respuesta": "¡Buenos días! Un gusto acompañarte en tu estudio. Pregúntame lo que necesites: puedo explicarte conceptos, darte ejemplos, recomendarte fuentes confiables y mucho más. ¿Por dónde empezamos?", "cuestionario": None},
    {"pregunta": "buenas tardes", "respuesta": "¡Buenas tardes! ¿Qué tal tu día? Cuéntame qué tema te interesa, y te lo explico paso a paso sin complicaciones 😊", "cuestionario": None},
    {"pregunta": "buenas noches", "respuesta": "¡Buenas noches! Claro que sí, estoy aquí para ayudarte antes de descansar. ¿Sobre qué tema quieres conversar?", "cuestionario": None},
    {"pregunta": "quien eres", "respuesta": "Soy **INDI**, tu asistente de estudios diseñado especialmente para ayudarte a aprender temas de primaria y secundaria. Te explico todo de forma sencilla, te doy fuentes confiables y cuando tú quieras, te preparo un cuestionario para practicar. ¿En qué te ayudo?", "cuestionario": None},
    {"pregunta": "que puedes hacer", "respuesta": "Puedo hacer muchas cosas por ti:\n✅ Explicarte cualquier tema de forma clara y detallada\n✅ Responder todas tus dudas hasta que quede todo claro\n✅ Recomendarte fuentes confiables para profundizar\n✅ Darte ejemplos prácticos\n✅ Y cuando tú me lo pidas, prepararte un cuestionario para practicar\n\n¿Qué te gustaría hacer primero?", "cuestionario": None},
    {"pregunta": "adios", "respuesta": "¡Hasta luego! Fue un gusto ayudarte. Vuelve cuando quieras seguir aprendiendo, aquí estaré 😊", "cuestionario": None},

    # Matemáticas (información ampliada)
    {"pregunta": "matemáticas", "respuesta": """📚 **¿Qué son las matemáticas?**
Son la ciencia que estudia las cantidades, las estructuras, el espacio y los cambios que ocurren en todo lo que nos rodea. No son solo números: nos ayudan a resolver problemas cotidianos, desde repartir dinero hasta construir edificios o viajar al espacio.

Se dividen en varias ramas principales:
🔢 **Aritmética**: Estudia los números y las operaciones básicas (suma, resta, multiplicación, división).
📐 **Geometría**: Estudia las formas, los tamaños y las posiciones de las figuras.
📊 **Estadística**: Recopila y analiza datos para sacar conclusiones.
🔍 **Álgebra**: Usa letras junto a números para resolver problemas desconocidos.

📌 **¿Por qué son importantes?**
Las usamos todos los días: cuando compramos algo, cuando vemos la hora, cuando cocinamos o cuando calculamos el tiempo que tardamos en llegar a un lugar.

🔗 **Fuentes para profundizar**:
- https://es.khanacademy.org/math → Cursos gratuitos desde lo más básico hasta avanzado, con ejercicios interactivos.
- https://www.universoformulas.com/matematicas/ → Explicaciones claras, fórmulas y ejemplos resueltos paso a paso.
- https://www.ck12.org/c/math/ → Material adaptado para secundaria con actividades sencillas.

¿Te gustaría que te explique alguna rama en especial, o tienes alguna duda concreta sobre las matemáticas? Si quieres practicar, solo dime "hazme el cuestionario" y te lo preparo.""",
    "cuestionario": """
📋 **Cuestionario de Matemáticas**
1. ¿Qué estudian principalmente las matemáticas?
2. ¿Cómo se llama la rama que estudia las formas y figuras?
3. ¿Cuánto es 7 × 9?
4. ¿Qué tipo de número es el 7: primo o compuesto?
5. ¿Cuántos lados tiene un triángulo?"""},

    # Ciencias
    {"pregunta": "ciencias", "respuesta": """📚 **¿Qué son las ciencias?**
Son el conjunto de conocimientos que estudian el mundo natural, los seres vivos y todos los fenómenos que ocurren en la Tierra y el universo. Todo lo que sabemos lo descubrimos observando, preguntando y haciendo experimentos.

Las ciencias se dividen en tres grandes grupos:
🌿 **Ciencias Naturales**: Estudian la naturaleza (Biología, Física, Química, Geología).
👥 **Ciencias Sociales**: Estudian al ser humano y la sociedad (Historia, Geografía, Economía).
🧠 **Ciencias Formales**: Estudian las ideas y el razonamiento (Matemáticas, Lógica).

📌 **¿Por qué son importantes?**
Gracias a la ciencia tenemos medicinas, electricidad, internet, conocemos cómo funciona nuestro cuerpo y cómo se formó el universo.

🔗 **Fuentes para profundizar**:
- https://es.khanacademy.org/science → Lecciones completas con videos y ejercicios prácticos.
- https://www.nationalgeographicla.com/ciencia → Información visual y fiable sobre naturaleza y descubrimientos.
- https://www.britannica.com/es/ciencia → Artículos revisados por especialistas en todas las ramas científicas.

¿Te interesa saber más sobre alguna rama en especial? Si quieres poner a prueba lo aprendido, solo pídeme el cuestionario.""",
    "cuestionario": """
📋 **Cuestionario de Ciencias**
1. ¿Qué es la fotosíntesis?
2. ¿Cuál es el planeta donde vivimos?
3. ¿Qué gas necesitan las plantas para vivir?
4. ¿Qué órgano bombea la sangre en el cuerpo humano?
5. ¿De qué elementos está hecho el agua?"""},

    # Lenguaje
    {"pregunta": "lenguaje", "respuesta": """📚 **¿Qué es el lenguaje?**
Es la capacidad que tenemos los seres humanos para comunicarnos y expresar lo que pensamos, sentimos y sabemos. Estudia las palabras, las oraciones y las reglas para usar bien el idioma español.

📌 **¿Qué aprendemos aquí?**
✅ **Gramática**: Las reglas para formar oraciones correctas.
✅ **Ortografía**: Cómo escribir bien las palabras.
✅ **Vocabulario**: El significado de cada palabra.
✅ **Comprensión**: Cómo entender lo que leemos y expresar lo que queremos decir.

No es solo escribir bien: es saber comunicarte de forma clara para que los demás te entiendan.

🔗 **Fuentes para profundizar**:
- https://dle.rae.es/ → Diccionario oficial y normas correctas del idioma español.
- https://concepto.de/lenguaje/ → Definiciones sencillas sobre palabras y tipos de lenguaje.
- https://www.portaleducativo.net/lenguaje/ → Ejercicios de gramática y ortografía para secundaria.

¿Tienes alguna duda sobre algún tipo de palabra, o quieres que te explique algo más? Si quieres practicar, solo dime "cuestionario".""",
    "cuestionario": """
📋 **Cuestionario de Lenguaje**
1. ¿Qué nombra un sustantivo?
2. ¿Qué expresa un verbo?
3. ¿Cuál es la diferencia entre sustantivo propio y común?
4. ¿Qué signo se usa al terminar una oración que expresa una duda?
5. ¿Qué es una oración?"""},

    # Historia del Perú
    {"pregunta": "historia del Perú", "respuesta": """📚 **¿Qué es la historia del Perú?**
Es el relato de todos los sucesos importantes que han ocurrido en nuestro territorio desde hace miles de años hasta hoy. Se divide en etapas para entenderla mejor:

⏳ **Etapas principales**:
1. **Época Prehispánica**: Antes de la llegada de los españoles, con las culturas antiguas y el Imperio Inca.
2. **Época Colonial**: Cuando el territorio estuvo bajo el dominio de España.
3. **Época Republicana**: Desde que nos independizamos hasta la actualidad.

📌 **¿Por qué es importante conocerla?**
Porque así entendemos de dónde venimos, cómo se formó nuestro país y quiénes somos como peruanos.

🔗 **Fuentes para profundizar**:
- https://historiaperuana.pe/ → Portal especializado con fechas y sucesos clave.
- https://www.gob.pe/ministeriodecultura/historia/ → Información oficial del Ministerio de Cultura.
- https://www.peru.travel/es/descubre-peru/historia → Resumen estructurado de todas las etapas históricas.

¿Te gustaría que te explique alguna etapa en más detalle? Cuando quieras practicar, pídeme el cuestionario.""",
    "cuestionario": """
📋 **Cuestionario de Historia del Perú**
1. ¿Cuál es la civilización más antigua del Perú?
2. ¿En qué año se proclamó la independencia del Perú?
3. ¿Quién proclamó la independencia en Lima?
4. ¿Cómo se llamó el imperio de los incas?
5. ¿Quién fue José de San Martín?"""},

    # Cuestionario a petición
    {"pregunta": "cuestionario", "respuesta": "¡Claro que sí! Aquí tienes el cuestionario para practicar. ¿Te gustaría que te dé las respuestas después?", "cuestionario": "ultimo"},
    {"pregunta": "hazme el cuestionario", "respuesta": "¡Vamos a poner a prueba lo aprendido! Aquí tienes el cuestionario correspondiente. ¿Quieres que te dé las respuestas al terminar?", "cuestionario": "ultimo"},
    {"pregunta": "quiero practicar", "respuesta": "Perfecto, aquí tienes el cuestionario para practicar. ¿Te ayudo con las respuestas luego?", "cuestionario": "ultimo"},
    {"pregunta": "ejercicios", "respuesta": "Claro, aquí tienes los ejercicios para el tema que vimos. ¿Quieres revisar las respuestas después?", "cuestionario": "ultimo"}
]

lemmatizador = WordNetLemmatizer()
stop_words = set(stopwords.words('spanish'))

def limpiar_texto(texto):
    texto = texto.lower()
    texto = re.sub(r'[^a-záéíóúñ\s]', '', texto)
    palabras = texto.split()
    palabras = [lemmatizador.lemmatize(p) for p in palabras if p not in stop_words]
    return ' '.join(palabras)

preguntas_limpias = [limpiar_texto(item["pregunta"]) for item in base_conocimiento]
respuestas = [item["respuesta"] for item in base_conocimiento]
cuestionarios = [item["cuestionario"] for item in base_conocimiento]

vectorizador = TfidfVectorizer()
matriz_tfidf = vectorizador.fit_transform(preguntas_limpias)

# Guardamos el último tema que consultó el usuario
ultimo_tema = None

def buscar_respuesta(pregunta_usuario):
    global ultimo_tema
    pregunta_limpia = limpiar_texto(pregunta_usuario)
    
    variantes = {
        "ciencia": "ciencias",
        "matematica": "matemáticas",
        "mate": "matemáticas",
        "lengua": "lenguaje",
        "historia peru": "historia del Perú",
        "culturas antiguas": "culturas preincas",
        "incas": "Imperio Incaico",
        "tahuantinsuyo": "Imperio Incaico",
        "regiones peru": "regiones naturales",
        "costa sierra selva": "regiones naturales",
        "que es fraccion": "fracciones",
        "plantas comen luz": "fotosíntesis",
        "que es sustantivo": "sustantivos",
        "ponme el cuestionario": "cuestionario",
        "quiero el cuestionario": "cuestionario"
    }

    for var, original in variantes.items():
        if var in pregunta_limpia:
            pregunta_limpia = pregunta_limpia.replace(var, original).strip()

    # Si pide cuestionario, usamos el último tema consultado
    if "cuestionario" in pregunta_limpia or "ejercicios" in pregunta_limpia:
        if ultimo_tema and ultimo_tema in respuestas:
            indice = respuestas.index(ultimo_tema)
            if cuestionarios[indice]:
                return f"📋 **Cuestionario sobre tu último tema**:\n{cuestionarios[indice]}"
        return "Primero consulta un tema y luego te daré el cuestionario para practicar 😊"

    vector_pregunta = vectorizador.transform([pregunta_limpia])
    similitudes = cosine_similarity(vector_pregunta, matriz_tfidf)[0]
    indice_mejor = similitudes.argmax()

    if similitudes[indice_mejor] > 0.03:
        respuesta_final = respuestas[indice_mejor]
        ultimo_tema = respuesta_final
        return respuesta_final
    else:
        return "Puedo ayudarte con temas como matemáticas, ciencias, lenguaje, historia del Perú y mucho más. Cuéntame qué te interesa saber y te lo explico con detalle 😊"

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
