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

# ==================== BASE COMPLETA CON TODOS LOS TEMAS ====================
base_conocimiento = [
    # Saludos y conversación general
    {"pregunta": "hola", "respuesta": "¡Hola! Soy **INDI**, tu asistente de estudios personalizado 😊 Estoy aquí para explicarte cualquier tema con detalle, responder tus dudas y ayudarte a aprender mejor. ¿Sobre qué te gustaría conversar hoy?", "cuestionario": None, "respuestas": None},
    {"pregunta": "buenos dias", "respuesta": "¡Buenos días! Un gusto acompañarte en tu estudio. Pregúntame lo que necesites: puedo explicarte conceptos, darte ejemplos, recomendarte fuentes confiables y mucho más. ¿Por dónde empezamos?", "cuestionario": None, "respuestas": None},
    {"pregunta": "buenas tardes", "respuesta": "¡Buenas tardes! ¿Qué tal tu día? Cuéntame qué tema te interesa, y te lo explico paso a paso sin complicaciones 😊", "cuestionario": None, "respuestas": None},
    {"pregunta": "buenas noches", "respuesta": "¡Buenas noches! Claro que sí, estoy aquí para ayudarte antes de descansar. ¿Sobre qué tema quieres conversar?", "cuestionario": None, "respuestas": None},
    {"pregunta": "quien eres", "respuesta": "Soy **INDI**, tu asistente de estudios diseñado especialmente para ayudarte a aprender temas de primaria y secundaria. Te explico todo de forma sencilla, te doy fuentes confiables y cuando tú quieras, te preparo un cuestionario o encuesta para practicar. ¿En qué te ayudo?", "cuestionario": None, "respuestas": None},
    {"pregunta": "que puedes hacer", "respuesta": "Puedo hacer muchas cosas por ti:\n✅ Explicarte cualquier tema de forma clara y detallada\n✅ Responder todas tus dudas hasta que quede todo claro\n✅ Recomendarte fuentes confiables para profundizar\n✅ Darte ejemplos prácticos\n✅ Y cuando tú me lo pidas, prepararte un cuestionario o encuesta para practicar\n✅ También puedo darte las respuestas al terminar\n\n¿Qué te gustaría hacer primero?", "cuestionario": None, "respuestas": None},
    {"pregunta": "adios", "respuesta": "¡Hasta luego! Fue un gusto ayudarte. Vuelve cuando quieras seguir aprendiendo, aquí estaré 😊", "cuestionario": None, "respuestas": None},

    # Matemáticas
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

¿Te gustaría que te explique alguna rama en especial? Si quieres practicar, solo dime "hazme el cuestionario" o "hazme la encuesta".""",
    "cuestionario": """
📋 **Cuestionario de Matemáticas**
1. ¿Qué estudian principalmente las matemáticas?
2. ¿Cómo se llama la rama que estudia las formas y figuras?
3. ¿Cuánto es 7 × 9?
4. ¿Qué tipo de número es el 7: primo o compuesto?
5. ¿Cuántos lados tiene un triángulo?""",
    "respuestas": """
✅ **Respuestas correctas**:
1. Las cantidades, estructuras, espacio y cambios.
2. Geometría.
3. 63.
4. Número primo.
5. 3 lados."""},

    # Fracciones
    {"pregunta": "fracciones", "respuesta": """📚 **¿Qué son las fracciones?**
Son una forma de representar una parte de un todo. Por ejemplo, si tienes una pizza dividida en 4 partes y te comes 1, has comido 1/4 de la pizza.

📌 **Partes de una fracción**:
- **Numerador**: El número de arriba, indica cuántas partes tomamos.
- **Denominador**: El número de abajo, indica en cuántas partes iguales dividimos el todo.

🔗 **Fuentes para profundizar**:
- https://www.universoformulas.com/matematicas/aritmetica/fracciones/ → Explicaciones y ejemplos.
- https://es.khanacademy.org/math/arithmetic-home/fractions → Lecciones interactivas.

¿Tienes alguna duda? Si quieres practicar, pídeme el cuestionario.""",
    "cuestionario": """
📋 **Cuestionario de Fracciones**
1. ¿Qué representa una fracción?
2. ¿Cómo se llama el número de arriba?
3. ¿Cómo se llama el número de abajo?
4. ¿Cuánto es 1/2 + 1/2?
5. ¿Qué fracción es igual a la mitad?""",
    "respuestas": """
✅ **Respuestas correctas**:
1. Una parte de un todo.
2. Numerador.
3. Denominador.
4. 1 entero.
5. 1/2."""},

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
- https://es.khanacademy.org/science → Lecciones completas con videos y ejercicios.
- https://www.nationalgeographicla.com/ciencia → Información visual y fiable.
- https://www.britannica.com/es/ciencia → Artículos revisados por especialistas.

¿Te interesa saber más sobre alguna rama? Pide el cuestionario cuando quieras practicar.""",
    "cuestionario": """
📋 **Cuestionario de Ciencias**
1. ¿Qué es la fotosíntesis?
2. ¿Cuál es el planeta donde vivimos?
3. ¿Qué gas necesitan las plantas para vivir?
4. ¿Qué órgano bombea la sangre?
5. ¿De qué está hecho el agua?""",
    "respuestas": """
✅ **Respuestas correctas**:
1. El proceso por el cual las plantas fabrican su propio alimento.
2. La Tierra.
3. Dióxido de carbono.
4. El corazón.
5. Hidrógeno y oxígeno."""},

    # Fotosíntesis
    {"pregunta": "fotosíntesis", "respuesta": """📚 **¿Qué es la fotosíntesis?**
Es el proceso por el cual las plantas, algas y algunas bacterias fabrican su propio alimento. Para hacerlo necesitan luz solar, agua y dióxido de carbono.

📌 **¿Por qué es tan importante?**
- Produce el oxígeno que necesitamos para respirar.
- Es la base de toda la cadena alimenticia.
- Ayuda a limpiar el aire del planeta.

🔗 **Fuentes**:
- https://www.nationalgeographicla.com/ciencia/medio-ambiente/que-es-la-fotosintesis → Explicación sencilla.
- https://es.khanacademy.org/science/ap-biology/cellular-energetics/photosynthesis → Lección detallada.

¿Quieres practicar lo aprendido? Pídeme el cuestionario.""",
    "cuestionario": """
📋 **Cuestionario de Fotosíntesis**
1. ¿Qué necesitan las plantas para hacerla?
2. ¿Qué gas liberan en este proceso?
3. ¿En qué parte de la planta ocurre?
4. ¿De qué color necesitan la luz?
5. ¿Por qué es importante para nosotros?""",
    "respuestas": """
✅ **Respuestas correctas**:
1. Luz solar, agua y dióxido de carbono.
2. Oxígeno.
3. En las hojas.
4. Luz solar.
5. Nos da el oxígeno que respiramos."""},

    # Lenguaje
    {"pregunta": "lenguaje", "respuesta": """📚 **¿Qué es el lenguaje?**
Es la capacidad que tenemos los seres humanos para comunicarnos y expresar lo que pensamos, sentimos y sabemos. Estudia las palabras, las oraciones y las reglas para usar bien el español.

📌 **¿Qué aprendemos aquí?**
✅ **Gramática**: Reglas para formar oraciones correctas.
✅ **Ortografía**: Cómo escribir bien las palabras.
✅ **Vocabulario**: El significado de cada palabra.

🔗 **Fuentes**:
- https://dle.rae.es/ → Diccionario oficial de la RAE.
- https://concepto.de/lenguaje/ → Definiciones sencillas.
- https://www.portaleducativo.net/lenguaje/ → Ejercicios prácticos.

¿Tienes alguna duda? Pide el cuestionario cuando quieras.""",
    "cuestionario": """
📋 **Cuestionario de Lenguaje**
1. ¿Qué nombra un sustantivo?
2. ¿Qué expresa un verbo?
3. ¿Cuál es la diferencia entre sustantivo propio y común?
4. ¿Qué signo se usa al terminar una oración?
5. ¿Qué es una oración?""",
    "respuestas": """
✅ **Respuestas correctas**:
1. Personas, animales, cosas, lugares o sentimientos.
2. Acciones o estados.
3. El propio nombra algo único (ej: Lima), el común nombra cosas en general (ej: ciudad).
4. El punto.
5. Un conjunto de palabras con sentido completo."""},

    # Sustantivos
    {"pregunta": "sustantivos", "respuesta": """📚 **¿Qué son los sustantivos?**
Son palabras que sirven para nombrar todo lo que existe: personas, animales, cosas, lugares, sentimientos o ideas.

📌 **Tipos principales**:
- **Propios**: Nombres únicos (ej: Perú, María, Machu Picchu).
- **Comunes**: Nombres generales (ej: país, niña, montaña).
- **Concretos**: Cosas que se pueden tocar o ver.
- **Abstractos**: Sentimientos o ideas que no se pueden tocar.

🔗 **Fuentes**:
- https://concepto.de/sustantivo/ → Clasificación completa.
- https://dle.rae.es/sustantivo → Normas oficiales.

¿Quieres practicar? Pídeme el cuestionario.""",
    "cuestionario": """
📋 **Cuestionario de Sustantivos**
1. ¿Qué nombra un sustantivo?
2. ¿Cuál es la diferencia entre propio y común?
3. ¿Qué tipo es "Lima"?
4. ¿Qué tipo es "amor"?
5. Escribe un sustantivo de animal.""",
    "respuestas": """
✅ **Respuestas correctas**:
1. Personas, animales, cosas, lugares, sentimientos o ideas.
2. El propio es único, el común es general.
3. Sustantivo propio.
4. Sustantivo abstracto.
5. Ejemplo: perro, gato, cóndor."""},

    # Historia del Perú
    {"pregunta": "historia del Perú", "respuesta": """📚 **¿Qué es la historia del Perú?**
Es el relato de todos los sucesos importantes que han ocurrido en nuestro territorio desde hace miles de años hasta hoy. Se divide en etapas para entenderla mejor:

⏳ **Etapas principales**:
1. **Época Prehispánica**: Antes de la llegada de los españoles, con culturas antiguas y el Imperio Inca.
2. **Época Colonial**: Bajo el dominio de España.
3. **Época Republicana**: Desde la independencia hasta la actualidad.

🔗 **Fuentes**:
- https://historiaperuana.pe/ → Portal especializado.
- https://www.gob.pe/ministeriodecultura/historia/ → Información oficial.
- https://www.peru.travel/es/descubre-peru/historia → Resumen detallado.

¿Te gustaría saber más sobre alguna etapa? Pide el cuestionario cuando quieras.""",
    "cuestionario": """
📋 **Cuestionario de Historia del Perú**
1. ¿Cuál es la civilización más antigua del Perú?
2. ¿En qué año se proclamó la independencia?
3. ¿Quién proclamó la independencia en Lima?
4. ¿Cómo se llamó el imperio de los incas?
5. ¿Quién fue José de San Martín?""",
    "respuestas": """
✅ **Respuestas correctas**:
1. Caral.
2. 1821.
3. José de San Martín.
4. Tahuantinsuyo.
5. El libertador que ayudó a independizar el Perú."""},

    # Culturas Preincas
    {"pregunta": "culturas preincas", "respuesta": """📚 **¿Qué son las culturas preincas?**
Son las civilizaciones que se desarrollaron en el territorio peruano antes del Imperio Inca. Las más importantes son: Caral, Chavín, Paracas, Nazca, Moche, Wari, Tiahuanaco y Chimú.

📌 **Algunos datos**:
- **Caral**: La civilización más antigua de América.
- **Nazca**: Famosa por las líneas de Nazca.
- **Chimú**: Construyeron la ciudad de Chan Chan.

🔗 **Fuentes**:
- https://www.museoarqueologicodelima.gob.pe/culturas-prehispanicas/ → Datos oficiales del museo.
- https://www.universoperu.com/culturas-preincas → Información completa.

¿Quieres practicar? Pídeme el cuestionario.""",
    "cuestionario": """
📋 **Cuestionario de Culturas Preincas**
1. ¿Cuál es la cultura más antigua de América?
2. ¿Qué cultura hizo las líneas de Nazca?
3. ¿Dónde se ubicó la cultura Chavín?
4. ¿Qué cultura fue experta en orfebrería?
5. ¿Qué cultura construyó Chan Chan?""",
    "respuestas": """
✅ **Respuestas correctas**:
1. Caral.
2. Cultura Nazca.
3. En los Andes centrales.
4. Cultura Chimú.
5. Cultura Chimú."""},

    # Imperio Incaico
    {"pregunta": "Imperio Incaico", "respuesta": """📚 **¿Qué fue el Imperio Incaico?**
También llamado **Tahuantinsuyo**, fue el imperio más grande de Sudamérica antes de la llegada de los españoles. Se desarrolló entre los años 1200 y 1532 d.C., con capital en Cusco.

📌 **Datos importantes**:
- Se extendió por Perú, Bolivia, Ecuador y partes de Chile, Argentina y Colombia.
- Su máxima autoridad era el Inca.
- Usaban el quipu para contar y registrar datos.

🔗 **Fuentes**:
- https://www.machupicchu.gob.pe/historia/ → Datos oficiales.
- https://www.cuscoperu.com/es/historia/imperio-inca → Detalles de su organización.

¿Quieres practicar? Pídeme el cuestionario.""",
    "cuestionario": """
📋 **Cuestionario del Imperio Incaico**
1. ¿Qué significa "Tahuantinsuyo"?
2. ¿Cuál era su capital?
3. ¿Quién era la máxima autoridad?
4. ¿Qué sistema usaban para contar?
5. ¿En qué año llegaron los españoles?""",
    "respuestas": """
✅ **Respuestas correctas**:
1. "Las cuatro regiones unidas".
2. Cusco.
3. El Inca.
4. El quipu.
5. En 1532."""},

    # Regiones Naturales
    {"pregunta": "regiones naturales", "respuesta": """📚 **¿Qué son las regiones naturales del Perú?**
Nuestro país tiene una gran variedad de climas y paisajes, divididos principalmente en 3 grandes regiones:
🌊 **Costa**: Franja al lado del mar, clima cálido y seco.
⛰️ **Sierra**: La cordillera de los Andes, clima frío y templado.
🌳 **Selva**: Zona de bosques tropicales, clima cálido y lluvioso.

🔗 **Fuentes**:
- https://www.minam.gob.pe/biodiversidad/regiones-naturales/ → Información oficial del Ministerio del Ambiente.
- https://www.peru.travel/es/descubre-peru/geografia → Guía completa.

¿Quieres practicar? Pídeme el cuestionario.""",
    "cuestionario": """
📋 **Cuestionario de Regiones Naturales**
1. ¿Cuáles son las 3 regiones principales?
2. ¿En qué región está el Machu Picchu?
3. ¿Qué clima tiene la costa?
4. ¿En qué región nace el río Amazonas?
5. ¿Cuál es la región más alta?""",
    "respuestas": """
✅ **Respuestas correctas**:
1. Costa, Sierra y Selva.
2. En la Sierra.
3. Cálido y seco.
4. En la Sierra.
5. La Sierra."""},

    # Frases para pedir cuestionario/encuesta y respuestas
    {"pregunta": "cuestionario", "respuesta": "¡Claro que sí! Aquí tienes el cuestionario del último tema que viste. ¿Quieres que te dé las respuestas después?", "accion": "mostrar_cuestionario"},
    {"pregunta": "encuesta", "respuesta": "¡Claro! Aquí tienes la encuesta para practicar el último tema que consultaste. ¿Quieres ver las respuestas al terminar?", "accion": "mostrar_cuestionario"},
    {"pregunta": "hazme el cuestionario", "respuesta": "¡Vamos a poner a prueba lo aprendido! Aquí tienes el cuestionario. ¿Quieres las respuestas luego?", "accion": "mostrar_cuestionario"},
    {"pregunta": "hazme la encuesta", "respuesta": "¡Perfecto! Aquí tienes la encuesta del tema que vimos. ¿Quieres que te dé las respuestas después?", "accion": "mostrar_cuestionario"},
    {"pregunta": "quiero practicar", "respuesta": "¡Muy bien! Aquí tienes el ejercicio para repasar. ¿Necesitas las respuestas?", "accion": "mostrar_cuestionario"},
    {"pregunta": "respuestas", "respuesta": "¡Claro que sí! Aquí tienes las respuestas correctas:", "accion": "mostrar_respuestas"},
    {"pregunta": "si", "respuesta": "¡Perfecto! Aquí tienes las respuestas correctas:", "accion": "mostrar_respuestas"},
    {"pregunta": "dame las respuestas", "respuesta": "Claro que sí, aquí van las respuestas:", "accion": "mostrar_respuestas"}
]

lemmatizador = WordNetLemmatizer()
stop_words = set(stopwords.words('spanish'))

def limpiar_texto(texto):
    texto = texto.lower()
    texto = re.sub(r'[^a-záéíóúñ\s]', '', texto)
    palabras = texto.split()
    palabras = [lemmatizador.lemmatize(p) for p in palabras if p not in stop_words]
    return ' '.join(palabras)

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
        "ponme la encuesta": "encuesta",
        "quiero la encuesta": "encuesta",
        "ejercicios": "cuestionario",
        "ver respuestas": "respuestas"
    }

    for var, original in variantes.items():
        if var in pregunta_limpia:
            pregunta_limpia = pregunta_limpia.replace(var, original).strip()

    # Preparamos los datos para la búsqueda
    preguntas_limpias = [limpiar_texto(item["pregunta"]) for item in base_conocimiento]
    vectorizador = TfidfVectorizer()
    matriz_tfidf = vectorizador.fit_transform(preguntas_limpias)
    vector_pregunta = vectorizador.transform([pregunta_limpia])
    similitudes = cosine_similarity(vector_pregunta, matriz_tfidf)[0]
    indice_mejor = similitudes.argmax()

    # Caso 1: Es una acción (pedir cuestionario o respuestas)
    if similitudes[indice_mejor] > 0.03 and "accion" in base_conocimiento[indice_mejor]:
        accion = base_conocimiento[indice_mejor]["accion"]
        respuesta_base = base_conocimiento[indice_mejor]["respuesta"]

        if accion == "mostrar_cuestionario":
            if ultimo_tema and ultimo_tema.get("cuestionario"):
                return f"{respuesta_base}\n\n{ultimo_tema['cuestionario']}"
            else:
                return "Primero consulta un tema (como matemáticas, ciencias o historia), y luego te daré la encuesta o cuestionario 😊"

        if accion == "mostrar_respuestas":
            if ultimo_tema and ultimo_tema.get("respuestas"):
                return f"{respuesta_base}\n\n{ultimo_tema['respuestas']}"
            else:
                return "Primero pide el cuestionario de un tema, y luego te daré las respuestas 😊"

    # Caso 2: Es un tema normal
    if similitudes[indice_mejor] > 0.03:
        tema_encontrado = base_conocimiento[indice_mejor]
        ultimo_tema = tema_encontrado
        return tema_encontrado["respuesta"]

    # Caso 3: No se encontró el tema
    return "Puedo ayudarte con temas como matemáticas, ciencias, lenguaje, historia del Perú, culturas preincas y mucho más. Cuéntame qué te interesa saber y te lo explico con detalle 😊"

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
