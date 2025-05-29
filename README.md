# Proyecto - Compresión RLE y Huffman en Clasificación de Minerales

## Origen de los datos:
Las imágenes se obtienen del sitio web de [MinDat](https://mindat.org) 
con ayuda de la lista de URL's que alberga, provista por Oliver Hennigh en su [repositorio](https://github.com/loliverhennigh/MinDat-Mineral-Image-Dataset/blob/master/img_url_list.csv) de GitHub.

## Descripción del modelo previamente estudiado:
Se ha usado una red neuronal secuencial 
compuesta por cuatro capas de convolución y una capa totalmente conectada. 
El modelo, alimentado con 25,000 imágenes de 96x96 pixeles, 
es capaz de clasificar cinco categorías, 
previamente escogidas por su relevancia 
dentro de MinDat y en la comunidad geológica en general.

## Comparación entre RLE y Huffman:
Se aplican ambas técnicas de compresión en las imágenes ya preprocesadas.
RLE se aplica a las máscaras binarias previamente obtenidas en el 
preprocesamiento y almacenadas en disco. Huffman se aplica en la 
representación a blanco y negro, obtenida directamente y al momento de 
la imagen preprocesada.

## Random Forest con las métricas de compresión
Se entrena un modelo de Random Forest con algunas de las métricas 
obtenidas en el paso anterior, combinando RLE y Huffman para poder 
extraer información relevante para clasificación.