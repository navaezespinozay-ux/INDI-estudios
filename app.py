from flask import Flask, render_template, request, jsonify
import re
import os
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Descargar recursos de NLTK de forma segura para Render
nltk.data.path.append('/tmp/nltk_data')
nltk.download('punkt', download_dir='/tmp/nltk_data')
nltk.download('stopwords', download_dir='/tmp/nltk_data')
nltk.download('wordnet', download_dir='/tmp/nltk_data')

app = Flask(__name__)

# ==================== BASE DE CONOCIMIENTO + CUESTIONARIOS ====================
base_conocimiento = [
    # Saludos
    {"pregunta": "hola", "respuesta": "¡Hola! Soy tu asistente de estudios. Pregúntame sobre temas de secundaria, te daré información, enlaces y un cuestionario para practicar.", "cuestionario": None},
    {"pregunta": "buenos dias", "respuesta": "¡Buenos días! ¿Qué tema quieres aprender hoy?", "cuestionario": None},
    {"pregunta": "adios", "respuesta": "¡Hasta luego! Vuelve cuando quieras seguir aprendiendo.", "cuestionario": None},

    # Categorías generales
    {"pregunta": "matemáticas", "respuesta": """📚 Las matemáticas estudian cantidades, estructuras, formas y sus cambios. Aquí tienes recursos confiables:

🔗 https://es.khanacademy.org/es/matematicas
📝 Descripción: Cursos gratuitos desde nivel básico hasta avanzado, con ejercicios interactivos.

🔗 https://www.universoformulas.com/
📝 Descripción: Explicaciones, fórmulas y ejemplos resueltos de todos los temas.

🔗 https://www.ck12.org/math/
📝 Descripción: Material educativo adaptado para secundaria con actividades prácticas.""",
    "cuestionario": """
📋 **Cuestionario rápido sobre Matemáticas**
1. ¿Qué estudian las matemáticas principalmente?
2. ¿Cómo se llama el número que está arriba en una fracción?
3. ¿Cuál es el resultado de 3 × 8?
4. ¿Qué tipo de número es el 7: primo o compuesto?
5. ¿Cuántos lados tiene un triángulo?
    """},

    {"pregunta": "ciencias", "respuesta": """📚 Las ciencias estudian el mundo natural, sus fenómenos y leyes. Incluye biología, física, química y más:

🔗 https://es.khanacademy.org/es/ciencia
📝 Descripción: Lecciones completas por áreas científicas con videos y ejercicios.

🔗 https://www.nationalgeographic.org/education/
📝 Descripción: Material visual y explicativo sobre naturaleza, espacio y descubrimientos.

🔗 https://www.britannica.com/science
📝 Descripción: Información revisada por especialistas en todas las ramas de la ciencia.""",
    "cuestionario": """
📋 **Cuestionario rápido sobre Ciencias**
1. ¿Qué es la fotosíntesis?
2. ¿Cuál es el planeta más cercano al Sol?
3. ¿De qué está hecha la materia?
4. ¿Qué gas necesitan las plantas para vivir?
5. ¿Qué parte del cuerpo bombea la sangre?
    """},

    {"pregunta": "lenguaje", "respuesta": """📚 El lenguaje es la capacidad de comunicarse, y la lingüística estudia su estructura y uso:

🔗 https://www.rae.es/drae
📝 Descripción: Diccionario oficial y normas de la Real Academia Española.

🔗 https://concepto.de/lenguaje/
📝 Descripción: Definición, tipos y funciones del lenguaje humano.

🔗 https://www.portaleducativo.net/lenguaje/
📝 Descripción: Ejercicios y explicaciones de gramática y literatura para secundaria.""",
    "cuestionario": """
📋 **Cuestionario rápido sobre Lenguaje**
1. ¿Qué es un sustantivo?
2. ¿Cuáles son las partes principales de una oración?
3. ¿Qué es un verbo?
4. ¿Qué tipo de sustantivo es "Lima": propio o común?
5. ¿Qué signo se usa para terminar una oración?
    """},

    # Historia del Perú
    {"pregunta": "historia del Perú", "respuesta": """📚 La historia del Perú se divide en: Etapa Prehispánica, Conquista, Virreinato, Emancipación, República y Época Contemporánea.

🔗 https://www.superprof.pe/blog/peru-proceso-historico/
📝 Descripción: Resumen estructurado de todas las etapas históricas del Perú, con fechas y sucesos clave.

🔗 https://historiaperuana.pe/
📝 Descripción: Portal especializado en historia peruana, con artículos detallados y fuentes confiables.

🔗 https://www.donquijote.org/es/cultura-peruana/historia/
📝 Descripción: Explicación sencilla y organizada de la evolución histórica del país.""",
    "cuestionario": """
📋 **Cuestionario rápido sobre Historia del Perú**
1. ¿Cuál es la civilización más antigua del Perú?
2. ¿En qué año se proclamó la independencia?
3. ¿Quién proclamó la independencia en Lima?
4. ¿Cómo se llamó el imperio de los incas?
5. ¿Cuáles son las etapas principales de la historia peruana?
    """},

    {"pregunta": "culturas preincas", "respuesta": """📚 Las principales culturas preincas son: Caral, Chavín, Paracas, Nazca, Moche, Tiahuanaco, Wari y Chimú.

🔗 https://es.slideshare.net/slideshow/las-culturas-pre-incaicas-21961494/21961494
📝 Descripción: Presentación detallada con características, organización y logros de cada cultura.

🔗 https://culturas-preincas.com/
📝 Descripción: Información completa dedicada exclusivamente a cada civilización prehispánica.

🔗 https://www.studocu.com/pe/document/universidad-nacional-autonoma-de-chota/social-historia/las-culturas-pre-incas-del-peru/118197986
📝 Descripción: Resumen académico con datos precisos sobre las culturas anteriores a los incas.""",
    "cuestionario": """
📋 **Cuestionario rápido sobre Culturas Preincas**
1. ¿Cuál es la civilización más antigua de América?
2. ¿Qué cultura construyó las famosas líneas de Nazca?
3. ¿Dónde se ubicó la cultura Chavín?
4. ¿Qué cultura se desarrolló en la costa norte y fue experta en orfebrería?
5. ¿Cuál fue la cultura que dominó los Andes antes de los incas?
    """},

    {"pregunta": "Imperio Incaico", "respuesta": """📚 El Imperio Incaico se desarrolló entre los años 1200 y 1532 d.C., con capital en Cusco.

🔗 https://www.culturarecreacionydeporte.gov.co/es/principal/noticias/historia-del-imperio-inca
📝 Descripción: Explicación sobre el origen, expansión y caída del Tahuantinsuyo.

🔗 https://www.perurail.com/es/blog/los-incas-del-tahuantinsuyo-el-imperio-mas-grande-de-sudamerica/
📝 Descripción: Detalles sobre la organización social, caminos, arquitectura y legado incaico.

🔗 https://www.chullostravelperu.com/blog/imperio-inca-historia-legado-peru
📝 Descripción: Información clara sobre los líderes, costumbres y sitios arqueológicos del imperio.""",
    "cuestionario": """
📋 **Cuestionario rápido sobre el Imperio Incaico**
1. ¿Cuál era la capital del Tahuantinsuyo?
2. ¿Quién era la máxima autoridad incaica?
3. ¿Qué significa "Tahuantinsuyo"?
4. ¿En qué año llegaron los españoles al imperio inca?
5. ¿Qué sistema usaban para contar y registrar datos?
    """},

    {"pregunta": "independencia del Perú", "respuesta": """📚 Se proclamó el 28 de julio de 1821 por José de San Martín, y se consolidó en 1824 con las batallas de Junín y Ayacucho.

🔗 https://www.lifeder.com/independencia-del-peru/
📝 Descripción: Explicación completa de causas, desarrollo y consecuencias del proceso de emancipación.

🔗 https://estudiosindianos.up.edu.pe/recursos/humanidades-digitales-linea-de-tiempo-del-proceso-de-independencia-del-peru-1814-1824/
📝 Descripción: Línea de tiempo con todos los sucesos clave de la independencia.

🔗 https://www.bcrp.gob.pe/docs/Billetes-Monedas/Conmemorativas/2021/folleto-bicentenario-independencia.pdf
📝 Descripción: Documento oficial del Banco Central con datos y fechas exactas del Bicentenario.""",
    "cuestionario": """
📋 **Cuestionario rápido sobre la Independencia**
1. ¿En qué fecha se proclamó la independencia del Perú?
2. ¿Quién fue el libertador que llegó a Lima en 1821?
3. ¿En qué batalla se consolidó definitivamente la independencia?
4. ¿Qué país ayudó principalmente con tropas y recursos?
5. ¿Quién lideró el ejército en la batalla de Ayacucho?
    """},

    # Geografía
    {"pregunta": "regiones naturales del Perú", "respuesta": """📚 El Perú tiene 3 regiones naturales principales: Costa, Sierra y Selva.

🔗 https://www.grupodocenteperu.com/wp-content/uploads/2022/12/14-12-l-GRUPO-DOCENTE-PERU-l-CIENCIAS-SOCIALES.pdf
📝 Descripción: Guía completa con clasificación, límites, clima y biodiversidad de cada región.

🔗 https://www.ck12.org/search/?q=regiones+naturales+del+peru
📝 Descripción: Material educativo con esquemas, mapas y ejercicios prácticos.

🔗 https://www.gob.pe/institucion/corpac/informes-publicaciones/3612934-regiones-del-peru
📝 Descripción: Información oficial del Estado peruano sobre la división territorial y natural.""",
    "cuestionario": """
📋 **Cuestionario rápido sobre Regiones Naturales**
1. ¿Cuáles son las 3 regiones naturales principales del Perú?
2. ¿En qué región se encuentra el nevado Huascarán?
3. ¿Qué clima predomina en la costa?
4. ¿En qué región nace el río Amazonas?
5. ¿Cuántas regiones naturales propuso el geógrafo Javier Pulgar Vidal?
    """},

    # Temas específicos
    {"pregunta": "fracciones", "respuesta": """📚 Una fracción representa una parte de un todo: tiene numerador (la parte) y denominador (el total).

🔗 https://www.universoformulas.com/matematicas/aritmetica/fracciones/
📝 Descripción: Explicación detallada, tipos y todas las operaciones con ejemplos resueltos.

🔗 https://rea.ceibal.edu.uy/elp/-qu-son-las-fracciones/qu_es_una_fraccin.html
📝 Descripción: Definición sencilla con ejemplos de la vida cotidiana y esquemas claros.

🔗 https://www.superprof.com.ar/blog/temas-basicos-fracciones/
📝 Descripción: Guía para aprender a leer, identificar y usar fracciones correctamente.""",
    "cuestionario": """
📋 **Cuestionario rápido sobre Fracciones**
1. ¿Qué es una fracción?
2. ¿Cómo se llama el número de arriba en una fracción?
3. ¿Cómo se llama el número de abajo?
4. ¿Cuánto es 1/2 + 1/2?
5. ¿Qué tipo de fracción es 3/4: propia o impropia?
    """},

    {"pregunta": "fotosíntesis", "respuesta": """📚 Proceso por el cual las plantas fabrican su propio alimento usando luz solar, agua y dióxido de carbono.

🔗 https://www.fundacionaquae.org/wiki/fotosintesis-plantas/
📝 Descripción: Explicación paso a paso, elementos necesarios y su importancia para la vida en la Tierra.

🔗 https://es.khanacademy.org/science/ap-biology/cellular-energetics/photosynthesis/a/intro-to-photosynthesis
📝 Descripción: Lección interactiva con gráficos y ejercicios para entender el proceso biológico.

🔗 https://museovirtual.csic.es/salas/vida/vida10.htm
📝 Descripción: Recurso educativo con infografías y ejemplos visuales.""",
    "cuestionario": """
📋 **Cuestionario rápido sobre Fotosíntesis**
1. ¿Qué es la fotosíntesis?
2. ¿Qué 3 elementos necesitan las plantas para hacerla?
3. ¿Qué gas liberan las plantas en este proceso?
4. ¿En qué parte de la planta ocurre principalmente?
5. ¿Por qué es importante para los seres vivos?
    """},

    {"pregunta": "sustantivo", "respuesta": """📚 Palabra que nombra personas, animales, cosas, lugares, sentimientos o ideas.

🔗 https://concepto.de/sustantivo/
📝 Descripción: Definición completa, clasificación y ejemplos de cada tipo de sustantivo.

🔗 https://www.superprof.com.ar/blog/sustantivos-identificar/
📝 Descripción: Consejos y ejercicios para reconocer y usar sustantivos en oraciones.

🔗 https://www3.gobiernodecanarias.org/medusa/ecoblog/msuaump/adaptacion-2o-de-eso/2-los-sustantivos/
📝 Descripción: Material escolar con explicaciones adaptadas para secundaria.""",
    "cuestionario": """
📋 **Cuestionario rápido sobre Sustantivos**
1. ¿Qué nombra un sustantivo?
2. ¿Cuál es la diferencia entre sustantivo propio y común?
3. ¿Qué tipo es "casa"?
4. ¿Qué tipo es "María"?
5. ¿Es "amor" un sustantivo concreto o abstracto?
    """}
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
        "quiero aprender sobre": "",
        "quiero saber sobre": "",
        "dime sobre": "",
        "hablame de": "",
        "que es": "",
        "temas de": "",
        "historia peru": "historia del Perú",
        "culturas antiguas": "culturas preincas",
        "incas": "Imperio Incaico",
        "liberacion peru": "independencia del Perú",
        "regiones peru": "regiones naturales del Perú",
        "que es fotosintesis": "fotosíntesis",
        "que es fraccion": "fracciones",
        "comunicacion": "lenguaje",
        "lengua": "lenguaje"
    }

    for var, original in variantes.items():
        if var in pregunta_limpia:
            pregunta_limpia = pregunta_limpia.replace(var, original).strip()

    vector_pregunta = vectorizador.transform([pregunta_limpia])
    similitudes = cosine_similarity(vector_pregunta, matriz_tfidf)[0]
    indice_mejor = similitudes.argmax()

    if similitudes[indice_mejor] > 0.05:
        respuesta_final = respuestas[indice_mejor]
        cuestionario = cuestionarios[indice_mejor]
        if cuestionario:
            respuesta_final += "\n\n" + cuestionario
        return respuesta_final
    else:
        return "No tengo información sobre eso aún. Intenta preguntar sobre historia del Perú, matemáticas, ciencias, lenguaje o comunicación, y te daré información, enlaces y un cuestionario para practicar."

@app.route('/')
def inicio():
    return render_template('index.html')

@app.route('/consultar', methods=['POST'])
def consultar():
    datos = request.get_json()
    return jsonify({"respuesta": buscar_respuesta(datos.get("pregunta", ""))})

# Configuración para Render
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)