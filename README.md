# Modelo de clasificación de imágenes en aplicación Flask
## Reto Deep Learning
### Pre-requisitos
Se utilizó el lenguaje Python 3.7 en Linux para la solución del reto:
- Tener instaladas las librerias predeterminadas de Anaconda3 https://www.anaconda.com/distribution/
- Tener instalado las dependencias de pytorch: https://pytorch.org/get-started/locally/
- Tener instalado la librería Flask: https://flask.palletsprojects.com/en/1.1.x/installation/
- Tener instalado OpenCV: https://docs.opencv.org/3.4/d2/de6/tutorial_py_setup_in_ubuntu.html

Se instaló el programa ngrok: https://ngrok.com/download
Tener activado el entorno anaconda activado
### Pasos
#### 1. Pre-procesamiento:
1.1 Abrir Jupyter Notebook
```
$ jupyter notebook
```
1.2 Ejecutar el archivo ./Preprocesamiento/PreprocessingImages.ipynb para analizar las imágenes y tomar decisiones con respecto a qué técnicas usar para el pre-procesamiento de imágenes.
1.3 Ejecutar el archivo ./Preprocesamiento/SplitTrainAndTest.ipynb para dividir la data en entrenamiento y pruebas para su posterior validación en la fase de entrenamiento
1.4 Ejecutar el archivo ./Preprocesamiento/SortingDatabaseInFolders.ipynb para dar formato a las imágenes y así, durante el entrenamiento, pasarlas a clasificar mediante la clase ImageFolder

#### 2. Entrenamiento:
2.1 Ejecutar el archivo ./Preprocesamiento/train.ipynb para entrenar el modelo propuesto con la data del reto

#### 3. Desplegar servicio Web
3.1 Desplegar el framework Flask mediante el comando:
```
$ flask run
```
3.2 Desplegar ngrok mediante el siguiente comando en la carpeta donde se encuentra instalado ngrok (para evitar abrir otro terminal puedes separar el proceso actual con Ctrl-A+Ctrl-D)
```
$ ./ngrok http 5000
```
3.3 Copiar el enlace web de ngrok
3.4 Para correr el test de prueba, abrir un navegador web e introducir el siguiente enlace: codigo.ngrok.io/API/test2.csv