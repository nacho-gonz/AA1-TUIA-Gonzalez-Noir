# Instrucciones.

## Instalacion 

- *1* **clonar el repositorio**

git clone https://github.com/nacho-gonz/AA1-TUIA-Gonzalez-Noir

- *2* **Instalar las dependencias necesarias**:
  - docker pull python:3.10-slim (en caso de ser necesario para el build de la imagen)

## Ejecución del código:
    
  - Dentro de la carpeta docker del repositorio clonado utilizaremos el comando: ``` docker build -t "nombre-imagen" . ```
  - En la localización que quiera correr el modelo predictor utilize el comando: ``` docker run --rm -v $(pwd)/tucarpeta:/app/files "nombre-imagen" ``` . Donde ``` tucarpeta ``` va a ser una carpeta dentro de tu localización donde vas a ingresar tus archivos a predecir y van a ser generados los archivos con las predicciones.
  - El archivo obligatoriamente tiene que estar en formato ``` .csv ``` y su nombre tiene que ser ```input.csv```.


## Precauciones con los datos de predicción:

  - El archivo obligatoriamente tiene que estar en formato ``` .csv ``` y su nombre tiene que ser ```input.csv```.
  - Los datos a predecir necesitan obligatoriamente tener las variables ```Date``` y ```Location``` para funcionar.


