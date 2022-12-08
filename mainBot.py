# Importanción de librerías
import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()
import numpy as np
import tflearn
import tensorflow
from tensorflow.python.framework import ops
import json
import random
import pickle

# Lectura de archivo donde se encuentran las preguntas
with open("contenido.json") as archivo:
    datos = json.load(archivo)

# Cuando ya se corré el programa la primera vez, el chat revisa si existe el archivo variables.pickle
# Esto sirve para optimizar la compilación ya que no tiene que generar nuevamente las variables
# En caso de que no exista, lo crea y le agrega los valores correcpondientes
try:
    with open("variables.pickle", "rb") as archivoPickle:
        palabras, tags, entrenamiento, salida = pickle.load(archivoPickle)
except: 

    # Definición de variables base
    palabras = [] # Se almacenarán las palabras por separado
    tags = [] # Separa almacenan los tags
    auxX = [] # Almacenará las palabras del arreglo palabras
    auxY = [] # Almacenará los tags pero este permitirá que se repitan

    # Se recorre el archivo para empezar a separar los datos por del archivo contenido.json
    # Estos datos se separan por tags
    for contenido in datos["contenido"]:
        for patrones in contenido["patrones"]:
            auxPalabra = nltk.word_tokenize(patrones) # Toma una frase y la separa por palabras
            palabras.extend(auxPalabra)
            auxX.append(auxPalabra)
            auxY.append(contenido["tag"])

            if contenido["tag"] not in tags:  # Verificar tags repetidos
                tags.append(contenido["tag"]) 

    # Almacenamiento de datos recogidos en las variables
    palabras = [stemmer.stem(w.lower()) for w in palabras if w!= "?"] # Stem a¡hace un casteo para 
    palabras = sorted(list(set(palabras)))
    tags = sorted(tags)

    entrenamiento = [] # Almacena listas con datos binarios donde indica si una palabra fue encontrada con relación a un tag donde 1 indica que fue encontrada y 0 que no fue encontrada
    salida = [] # Almacena listas con datos binarios donde indica la relación de las palabras encontradas con tags

    salidaVacia = [0 for _ in range(len(tags))]

    for x, documento in enumerate(auxX):
        cubeta = []
        auxPalabra = [stemmer.stem(w.lower()) for w in documento]
        for w in palabras:
            if w in auxPalabra:
                cubeta.append(1)
            else:
                cubeta.append(0)
        filaSalida = salidaVacia[:]
        filaSalida[tags.index(auxY[x])] = 1
        entrenamiento.append(cubeta)
        salida.append(filaSalida)

    entrenamiento = np.array(entrenamiento)
    salida = np.array(salida)
    with open("variables.pickle", "wb") as archivoPickle:
        pickle.dump((palabras, tags, entrenamiento, salida), archivoPickle)

# Creación de la red neuronal del chat
# Reiniciar el espacio  de la red neuronal para que quede en blanco
ops.reset_default_graph()

red = tflearn.input_data(shape=[None, len(entrenamiento[0])]) # anexar las palabras del programa a la red neuronal
red = tflearn.fully_connected(red, 10) # Agregar columna de neuron as a la red neuronal
red = tflearn.fully_connected(red, 10) # Agregar columna de neuronas a la red neuronal
red = tflearn.fully_connected(red, len(salida[0]), activation="softmax") # Agregar columna de neuronas a la red neuronal para la salida
red = tflearn.regression(red) # Permitir obtener probavbilidades en el chat

modelo = tflearn.DNN(red) # Se crea un modelo
# Revisa si el archivo modelo.tflearn existe para optimizar tiempo de compilación, en cado de que no, lo crea
try:
    modelo.load("modelo.tflearn")
except:
    # Se define en el modelo los daots de entrada, calida y la cantidad de veces que revisará la información 
    modelo.fit(entrenamiento, salida, n_epoch=1000, batch_size=10, show_metric=True)
    modelo.save("modelo.tflearn") # Guardar el arvhioc de compilación

# Definir la funciópn que interactua con el usuario
def mainBot():
    seguir = True
    while seguir:
        entrada = input("Tu: ")
        cubeta = [0 for _ in range(len(palabras))]
        entradaProcesada = nltk.word_tokenize(entrada) # Separar entradas de caracteres especiales
        entradaProcesada = [stemmer.stem(palabra.lower()) for palabra in entradaProcesada]
        for palabraIndividual in entradaProcesada:
            for i, palabra in enumerate(palabras):
                if palabra == palabraIndividual:
                    cubeta[i] = 1 # Indicar la posición de una palabra
        resultados = modelo.predict([np.array(cubeta)]) # Agregar posibles resultados de la entrada con un rando del 0 a 1 donde enrre más cercano a 1 significa que es más probable
        resultadosIndices = np.argmax(resultados) # Regresa el indice de los posibles tags de la respuesta
        tag = tags[resultadosIndices]

        for tagAux in datos["contenido"]:
            if tagAux["tag"] == tag:
                seguir = tag != "despedida" # Cerrar el chatbot en caso de que el tag indique la despedida 
                respuesta = tagAux["respuestas"] # Selecciona una de las respuestas pósibles según la relación y la probabilidad que obtuvo 

        print("BOT: ", random.choice(respuesta)) # Imprimir respuesat del chatbot

mainBot()
