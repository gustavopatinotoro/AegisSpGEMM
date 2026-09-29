#  Carpeta Pipeline

El flujo de control que unifica la consulta generada por el Apple Neural Engine (Core ML)
con la ejecución algebraica paralela de la GPU.

Instanciar un orquestador que reciba los datos de entrada, coordine las conversiones 
matemáticas, gestione el ciclo de vida de la memoria UMA y lance la búsqueda paralela


`Pipeline/SearchOrchestrator.swift:` Este módulo es la única interfaz pública que debería
consumir un sistema externo (como un servidor RAG o una API)
