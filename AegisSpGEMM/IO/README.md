#  Carpeta Entrada Datos

El Cargador Binario Mapeado en Memoria

Ingesta archivo binario `ag_news-384-euclidean.bin` utiliza formato **little-endian** y 
comienza con una cabecera de 16 bytes que contiene 4 valores int32 (dimension, 
numero_vectores_base numero_consultas y topk). 
A continuación, se estructuran cuatro bloques continuos de datos en float32: train, test,
neighbors y distances. 
Se destaca que el bloque neighbors se almacena intencionalmente como float32 a pesar de 
contener identificadores.
