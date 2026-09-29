#  Carpeta Core

Contendrá los administradores de memoria física. Aquí residirán las clases que solicitan
bloques `storageModeShared` al sistema operativo, asegurando que los punteros sean accedidos
simultáneamente por la CPU y la GPU sin copias intermedias en el bus de memoria.

