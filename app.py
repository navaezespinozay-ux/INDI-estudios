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

# ==================== BASE CON ENLACES CORREGIDOS Y FUNCIONALES ====================
base_conocimiento = [
    # Saludos
    {"pregunta": "hola", "respuesta": "¡Hola! Soy INDI, tu asistente de estudios. Elige un tema y te daré información, fuentes confiables y un cuestionario para practicar 😊", "cuestionario": None},
    {"pregunta": "buenos dias", "respuesta": "¡Buenos días! ¿Qué tema quieres aprender hoy?", "cuestionario": None},
    {"pregunta": "adios", "respuesta": "¡Hasta luego! Vuelve cuando quieras seguir aprendiendo.", "cuestionario": None},

    # Matemáticas
    {"pregunta": "matemáticas", "respuesta": """📚 **¿Qué son?** Estudia cantidades, estructuras, formas y los cambios en todo lo que nos rodea.

🔗 https://es.khanacademy.org/math
📝 Cursos gratuitos desde nivel básico hasta avanzado, con ejercicios interactivos.

🔗 https://www.universoformulas.com/matematicas/
📝 Explicaciones claras, fórmulas y ejemplos resueltos paso a paso.

🔗 https://www.ck12.org/c/math/
📝 Material adaptado para secundaria con actividades y esquemas sencillos.""",
    "cuestionario": """
📋 **Cuestionario rápido**
1. ¿Qué estudian principalmente las matemáticas?
2. ¿Cómo se llama el número de arriba en una fracción?
3. ¿Cuánto es 7 × 9?
4. ¿Qué tipo de número es el 7: primo o compuesto?
5. ¿Cuántos lados tiene un triángulo?"""},

    # Ciencias
    {"pregunta": "ciencias", "respuesta": """📚 **¿Qué son?** Estudian el mundo natural, los seres vivos y los fenómenos que ocurren en la Tierra y el universo. Incluye biología, física y química.

🔗 https://es.khanacademy.org/science
📝 Lecciones completas con videos y ejercicios prácticos.

🔗 https://www.nationalgeographicla.com/ciencia
📝 Información visual y fiable sobre naturaleza y descubrimientos.

🔗 https://www.britannica.com/es/ciencia
📝 Artículos revisados por especialistas en todas las ramas científicas.""",
    "cuestionario": """
📋 **Cuestionario rápido**
1. ¿Qué es la fotosíntesis?
2. ¿Cuál es el planeta donde vivimos?
3. ¿Qué gas necesitan las plantas para vivir?
4. ¿Qué órgano bombea la sangre?
5. ¿De qué está hecho el agua?"""},

    # Lenguaje
    {"pregunta": "lenguaje", "respuesta": """📚 **¿Qué es?** Es la capacidad de comunicarnos, y estudia las palabras, oraciones y las reglas para usar bien el idioma español.

🔗 https://dle.rae.es/
📝 Diccionario oficial y normas correctas del idioma.

🔗 https://concepto.de/lenguaje/
📝 Definiciones sencillas sobre palabras y tipos de lenguaje.

🔗 https://www.portaleducativo.net/lenguaje/
📝 Ejercicios de gramática y ortografía para secundaria.""",
    "cuestionario": """
📋 **Cuestionario rápido**
1. ¿Qué nombra un sustantivo?
2. ¿Qué expresa un verbo?
3. ¿Cuál es la diferencia entre sustantivo propio y común?
4. ¿Qué signo se usa al terminar una oración?
5. ¿Qué es una oración?"""},

    # Historia del Perú
    {"pregunta": "historia del Perú", "respuesta": """📚 **¿Qué es?** Narra los sucesos del territorio peruano, divididos en: culturas antiguas, Imperio Inca, conquista, virreinato, independencia y república.

🔗 https://historiaperuana.pe/
📝 Portal especializado con fechas y sucesos clave.

🔗 https://www.superprof.pe/blog/peru-proceso-historico/
📝 Información oficial del Ministerio de Cultura.

🔗 https://www.peru.travel/es/descubre-peru/historia
📝 Resumen estructurado de todas las etapas históricas.""",
    "cuestionario": """
📋 **Cuestionario rápido**
1. ¿Cuál es la civilización más antigua del Perú?
2. ¿En qué año se proclamó la independencia?
3. ¿Quién proclamó la independencia en Lima?
4. ¿Cómo se llamó el imperio de los incas?
5. ¿Quién fue José de San Martín?"""},

    # Culturas Preincas
    {"pregunta": "culturas preincas", "respuesta": """📚 **¿Qué son?** Son las civilizaciones que se desarrollaron en el Perú antes del Imperio Inca: Caral, Chavín, Paracas, Nazca, Moche, Wari, Tiahuanaco y Chimú.

🔗 https://www.museoarqueologicodelima.gob.pe/culturas-prehispanicas/
📝 Datos oficiales del Museo Arqueológico del Perú.

🔗 https://www.universoperu.com/culturas-preincas
📝 Información completa de cada civilización.

🔗 https://www.peru.travel/es/descubre-peru/historia/culturas-prehispanicas
📝 Resumen detallado con características y logros.""",
    "cuestionario": """
📋 **Cuestionario rápido**
1. ¿Cuál es la cultura más antigua de América?
2. ¿Qué cultura hizo las famosas líneas de Nazca?
3. ¿Dónde se ubicó la cultura Chavín?
4. ¿Qué cultura fue experta en orfebrería?
5. ¿Qué cultura construyó la ciudad de Chan Chan?"""},

    # Imperio Incaico
    {"pregunta": "Imperio Incaico", "respuesta": """📚 **¿Qué fue?** El imperio más grande de Sudamérica, también llamado Tahuantinsuyo, con capital en Cusco. Se desarrolló entre los años 1200 y 1532 d.C.

🔗 https://www.machupicchu.gob.pe/historia/
📝 Datos oficiales de Machu Picchu y el imperio.

🔗 https://www.cuscoperu.com/es/historia/imperio-inca
📝 Detalles sobre su organización y caminos.

🔗 https://www.perurail.com/es/blog/el-imperio-inca/
📝 Información sobre costumbres y sitios arqueológicos.""",
    "cuestionario": """
📋 **Cuestionario rápido**
1. ¿Qué significa "Tahuantinsuyo"?
2. ¿Cuál era su capital?
3. ¿Quién era la máxima autoridad?
4. ¿Qué sistema usaban para contar?
5. ¿En qué año llegaron los españoles?"""},

    # Regiones Naturales
    {"pregunta": "regiones naturales", "respuesta": """📚 **¿Qué son?** El territorio peruano se divide principalmente en 3 grandes regiones: Costa, Sierra y Selva, cada una con clima y vida propia.

🔗 https://www.minam.gob.pe/biodiversidad/regiones-naturales/
📝 Datos oficiales del Ministerio del Ambiente.

🔗 https://www.gob.pe/institucion/minam/informes-publicaciones/3612934-regiones-del-peru
📝 Información oficial del Estado peruano.

🔗 https://www.peru.travel/es/descubre-peru/geografia
📝 Guía completa con clima y biodiversidad.""",
    "cuestionario": """
📋 **Cuestionario rápido**
1. ¿Cuáles son las 3 regiones principales?
2. ¿En qué región está el Machu Picchu?
3. ¿Qué clima tiene la costa?
4. ¿En qué región nace el río Amazonas?
5. ¿Cuál es la región más alta?"""},

    # Fracciones
    {"pregunta": "fracciones", "respuesta": """📚 **¿Qué son?** Representan una parte de un todo: tienen un número de arriba (numerador) y uno de abajo (denominador).

🔗 https://www.universoformulas.com/matematicas/aritmetica/fracciones/
📝 Explicación detallada y ejemplos resueltos.

🔗 https://es.khanacademy.org/math/arithmetic-home/fractions
📝 Lección interactiva con ejercicios.

🔗 https://www.smartick.es/blog/matematicas/fracciones/que-es-una-fraccion/
📝 Definición sencilla con ejemplos cotidianos.""",
    "cuestionario": """
📋 **Cuestionario rápido**
1. ¿Qué representa una fracción?
2. ¿Cómo se llama el número de arriba?
3. ¿Cómo se llama el número de abajo?
4. ¿Cuánto es 1/2 + 1/2?
5. ¿Qué fracción es igual a la mitad?"""},

    # Fotosíntesis
    {"pregunta": "fotosíntesis", "respuesta": """📚 **¿Qué es?** El proceso por el cual las plantas fabrican su propio alimento usando luz solar, agua y aire.

🔗 https://es.khanacademy.org/science/ap-biology/cellular-energetics/photosynthesis
📝 Lección interactiva con gráficos.

🔗 https://www.nationalgeographicla.com/ciencia/medio-ambiente/que-es-la-fotosintesis
📝 Explicación sencilla y visual.

🔗 https://www.rjb.csic.es/la-fotosintesis/
📝 Información del Real Jardín Botánico.""",
    "cuestionario": """
📋 **Cuestionario rápido**
1. ¿Qué necesitan las plantas para hacerla?
2. ¿Qué gas liberan en este proceso?
3. ¿En qué parte de la planta ocurre?
4. ¿De qué color necesitan la luz?
5. ¿Por qué es importante para nosotros?"""},

    # Sustantivos
    {"pregunta": "sustantivos", "respuesta": """📚 **¿Qué son?** Palabras que sirven para nombrar personas, animales, cosas, lugares, sentimientos o ideas.

🔗 https://concepto.de/sustantivo/
📝 Definición completa y clasificación con ejemplos.

🔗 https://www.portaleducativo.net/gramatica/sustantivos
📝 Ejercicios para reconocerlos fácilmente.

🔗 https://dle.rae.es/sustantivo
📝 Normas oficiales de uso de la RAE.""",
    "cuestionario": """
📋 **Cuestionario rápido**
1. ¿Qué nombra un sustantivo?
2. ¿Cuál es la diferencia entre propio y común?
3. ¿Qué tipo es "Lima"?
4. ¿Qué tipo es "amor"?
5. Escribe un sustantivo de animal."""}
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

def buscar_respuesta(pregunta_usuario):
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
        "que es sustantivo": "sustantivos"
    }

    for var, original in variantes.items():
        if var in pregunta_limpia:
            pregunta_limpia = pregunta_limpia.replace(var, original).strip()

    vector_pregunta = vectorizador.transform([pregunta_limpia])
    similitudes = cosine_similarity(vector_pregunta, matriz_tfidf)[0]
    indice_mejor = similitudes.argmax()

    if similitudes[indice_mejor] > 0.03:
        respuesta_final = respuestas[indice_mejor]
        cuestionario = cuestionarios[indice_mejor]
        if cuestionario:
            respuesta_final += "\n\n" + cuestionario
        return respuesta_final
    else:
        return "Puedo ayudarte con estos temas: matemáticas, ciencias, lenguaje, historia del Perú, culturas preincas, Imperio Incaico, regiones naturales, fracciones, fotosíntesis y sustantivos 😊"

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
